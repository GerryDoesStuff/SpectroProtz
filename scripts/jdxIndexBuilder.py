"""
Indexer + JCAMP headers + Consensus

What it does
- Parses folders of spectra: JCAMP-DX (*.jdx/*.dx) only. Supports multi-spectrum JCAMPs with XYDATA blocks containing multiple Y columns.
- Standardizes axes: X → cm^-1, Y → absorbance for peak finding (keeps raw units too).
- Preprocesses, detects, and fits peaks per spectrum. Stores to DuckDB + Parquet.
- Stores headers into promoted columns and into a key–value table.
- Builds per-file and global consensus peak tables for prioritization in lookups.

Usage
  pip install duckdb pandas numpy scipy scikit-learn pyarrow
  python index_and_consensus.py /path/to/data /path/to/index_dir \
      --prominence 0.02 --min-distance 5 --sg-win 9 --sg-poly 3 \
      --als-lam 1e5 --als-p 0.01 --model Gaussian --min-r2 0.9 \
      --file-min-samples 2 --file-eps-factor 0.5 --file-eps-min 2.0 \
      --global-min-samples 2 --global-eps-abs 4.0

Notes
- JCAMP: uses FIRSTX/DELTAX/NPOINTS when XYDATA present. Applies XFACTOR/YFACTOR. Y from TRANSMITTANCE → absorbance by A=-log10(T).
- Consensus distances are in cm^-1. Weighted medians use weights = area * r2.
"""

from __future__ import annotations
import os, re, json, math, argparse, hashlib, glob, logging, sys
from typing import List, Tuple, Dict, Optional
import numpy as np, pandas as pd, duckdb
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
logger=logging.getLogger(__name__)


class UnsupportedSpectrumError(RuntimeError):
    """Raised when the JCAMP headers do not describe an FTIR spectrum."""

    def __init__(self, descriptor: Optional[str] = None, header_key: Optional[str] = None):
        descriptor_clean = (descriptor or "").strip()
        if not descriptor_clean:
            descriptor_clean = "Unknown spectrum type"
        message = descriptor_clean
        if header_key:
            message = f"{descriptor_clean} (header {header_key})"
        super().__init__(message)
        self.descriptor = descriptor_clean
        self.header_key = header_key


_FTIR_HEADER_KEYS = {
    "DATA TYPE",
    "DATATYPE",
    "CLASS",
    "SPECTROMETER/DATATYPE",
    "SPECTROMETER TYPE",
    "SPECTROMETER",
    "INSTRUMENT",
    "TECHNIQUE",
}


def is_ftir_spectrum(headers: Dict[str, str]) -> bool:
    """Return True when the provided headers describe an FTIR spectrum.

    Parameters
    ----------
    headers:
        Mapping of JCAMP header keys to values.

    Returns
    -------
    bool
        True when any recognised FTIR descriptor is found.

    Raises
    ------
    UnsupportedSpectrumError
        If none of the inspected headers appear to describe an FTIR spectrum.
    """

    candidates: List[Tuple[str, str]] = []
    for key, value in headers.items():
        if not value:
            continue
        key_upper = key.upper()
        if key_upper in _FTIR_HEADER_KEYS or "DATA TYPE" in key_upper or "SPECTROMETER" in key_upper:
            candidates.append((key, value))

    def _normalise(text: str) -> List[str]:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", text).upper()
        tokens = cleaned.split()
        if not tokens:
            return []
        joined = " ".join(tokens)
        # Include joined token variants so patterns like "FT IR" become "FTIR".
        tokens.append(joined.replace(" ", ""))
        return tokens

    for key, value in candidates:
        tokens = _normalise(value)
        if not tokens:
            continue
        if any(tok in {"IR", "INFRARED", "FTIR"} for tok in tokens):
            return True
        if any(tok.startswith("FT") and tok.endswith("IR") for tok in tokens):
            return True
        if any(tok.endswith("INFRARED") for tok in tokens):
            return True
        if any(tok.endswith("IR") and len(tok) <= 4 for tok in tokens):
            return True

    primary_descriptor: Optional[Tuple[str, str]] = None
    for candidate in candidates:
        if candidate[0].upper().startswith("DATA"):
            primary_descriptor = candidate
            break
    if primary_descriptor is None and candidates:
        primary_descriptor = candidates[0]

    descriptor_value = primary_descriptor[1] if primary_descriptor else None
    descriptor_key = primary_descriptor[0] if primary_descriptor else None
    raise UnsupportedSpectrumError(descriptor_value, descriptor_key)

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - fallback for minimal environments
    class DBSCAN:  # type: ignore
        """Minimal 1-D density clustering fallback used when scikit-learn is
        unavailable. The implementation groups sorted values whose successive
        spacing is within ``eps``. Groups that do not meet ``min_samples`` are
        labelled as noise (-1)."""

        def __init__(self, eps: float, min_samples: int, **_: Dict):
            self.eps = float(eps)
            self.min_samples = int(min_samples)

        def fit_predict(self, X):
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 1:
                raise RuntimeError('Fallback DBSCAN only supports 1-D inputs')
            values = arr[:, 0]
            n = len(values)
            if n == 0:
                return np.empty(0, dtype=int)
            order = np.argsort(values)
            labels = np.full(n, -1, dtype=int)
            cluster: List[int] = []
            cluster_id = 0
            last_val = None
            for idx in order:
                val = values[idx]
                if not cluster:
                    cluster = [idx]
                    last_val = val
                    continue
                if last_val is not None and abs(val - last_val) <= self.eps:
                    cluster.append(idx)
                    last_val = val
                    continue
                if len(cluster) >= self.min_samples:
                    for c_idx in cluster:
                        labels[c_idx] = cluster_id
                    cluster_id += 1
                cluster = [idx]
                last_val = val
            if len(cluster) >= self.min_samples:
                for c_idx in cluster:
                    labels[c_idx] = cluster_id
            return labels

PROMOTED_KEYS = {
    'TITLE':'title','DATA TYPE':'data_type','JCAMP-DX':'jcamp_ver','NPOINTS':'npoints_hdr',
    'XUNITS':'x_units_raw','YUNITS':'y_units_raw','XFACTOR':'x_factor','YFACTOR':'y_factor',
    'DELTAX':'deltax_hdr','FIRSTX':'firstx','LASTX':'lastx','FIRSTY':'firsty',
    'MAXX':'maxx','MINX':'minx','MAXY':'maxy','MINY':'miny','RESOLUTION':'resolution',
    'STATE':'state','CLASS':'class','ORIGIN':'origin','OWNER':'owner','DATE':'date',
    'NAMES':'names','CAS REGISTRY NO':'cas','MOLFORM':'molform','$NIST SOURCE':'nist_source'
}
NUMERIC_KEYS={'NPOINTS','XFACTOR','YFACTOR','DELTAX','FIRSTX','LASTX','FIRSTY','MAXX','MINX','MAXY','MINY','RESOLUTION'}
PROMOTED_COLUMN_TYPES={PROMOTED_KEYS[k]:('DOUBLE' if k in NUMERIC_KEYS else 'TEXT') for k in PROMOTED_KEYS}
KEY_RE=re.compile(r'^##\s*([^=]+?)\s*=\s*(.*)$')
COMMENT_RE=re.compile(r'^\s*\$\$')


def file_sha1(path:str,block:int=1<<20)->str:
    h=hashlib.sha1()
    with open(path,'rb') as f:
        while (b:=f.read(block)):
            h.update(b)
    return h.hexdigest()


def parse_jdx_headers(path:str)->Dict[str,str]:
    headers={};cur=None;buf=[]
    with open(path,'r',errors='ignore') as f:
        for raw in f:
            line=raw.rstrip('\n')
            if COMMENT_RE.match(line):
                continue
            m=KEY_RE.match(line)
            if m:
                k=m.group(1).strip().upper();v=m.group(2).strip()
                if cur is not None:
                    headers[cur]='\n'.join(buf).strip();buf=[]
                cur=k;buf=[v];continue
            if line.upper().startswith('##XYDATA='):
                break
            if cur is not None:
                buf.append(line)
    if cur is not None and cur not in headers:
        headers[cur]='\n'.join(buf).strip()
    return headers


def _parse_numeric(value:Optional[str])->Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError,ValueError):
        if isinstance(value,str):
            m=re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',value)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return None
    return None


def _expand_xpp_block(lines:List[List[float]],headers:Dict[str,str],xfactor:float,yfactor:float)->Tuple[np.ndarray,List[np.ndarray]]:
    if not lines:
        raise ValueError('Empty XYDATA block')
    start_vals=[ln[0] for ln in lines if len(ln)>=1]
    counts=[max(0,len(ln)-1) for ln in lines]
    y_vals=[]
    for ln in lines:
        if len(ln)>=2:
            y_vals.extend(ln[1:])
    if not y_vals:
        raise ValueError('No Y data found in XYDATA block')
    firstx=_parse_numeric(headers.get('FIRSTX'))
    deltax=_parse_numeric(headers.get('DELTAX'))
    npoints_raw=headers.get('NPOINTS')
    npoints=None
    if npoints_raw is not None:
        try:
            npoints=int(round(float(npoints_raw)))
        except (TypeError,ValueError):
            pass
    if firstx is None and start_vals:
        firstx=start_vals[0]
    if deltax is None and len(start_vals)>1:
        deltas=[]
        for i in range(len(start_vals)-1):
            cnt=counts[i]
            if cnt>0:
                deltas.append((start_vals[i+1]-start_vals[i])/cnt)
        if deltas:
            deltax=float(np.median(deltas))
    if npoints is None:
        npoints=len(y_vals)
    if deltax is None:
        lastx=_parse_numeric(headers.get('LASTX'))
        if lastx is not None and firstx is not None and npoints>1:
            deltax=(lastx-firstx)/(npoints-1)
    if deltax is None:
        minx=_parse_numeric(headers.get('MINX'))
        maxx=_parse_numeric(headers.get('MAXX'))
        if firstx is not None and minx is not None and maxx is not None and npoints>1:
            span=maxx-minx
            if abs(span)>0:
                # choose direction matching firstx proximity
                asc=firstx<=maxx
                target=maxx if asc else minx
                deltax=(target-firstx)/(npoints-1)
    if deltax is None:
        raise ValueError('Unable to determine DELTAX for compressed XYDATA block')
    if firstx is None:
        raise ValueError('Unable to determine FIRSTX for compressed XYDATA block')
    if len(y_vals)<npoints:
        raise ValueError('Y data shorter than expected NPOINTS in XYDATA block')
    if len(y_vals)>npoints:
        y_vals=y_vals[:npoints]
    x=np.asarray(firstx+np.arange(npoints)*deltax,dtype=float)*xfactor
    y=np.asarray(y_vals,dtype=float)*yfactor
    return x,[y]


def _parse_column_block(lines:List[List[float]],xfactor:float,yfactor:float)->Tuple[np.ndarray,List[np.ndarray]]:
    if not lines:
        raise ValueError('Empty column block')

    max_cols=max(len(r) for r in lines)
    x_vals:List[float]=[]
    column_values:List[List[float]]=[[] for _ in range(max_cols-1)]

    for row in lines:
        if not row:
            continue
        x_vals.append(row[0]*xfactor)
        for col_idx in range(1,len(row)):
            column_values[col_idx-1].append(row[col_idx]*yfactor)

    x=np.asarray(x_vals,dtype=float)
    spectra=[np.asarray(vals,dtype=float) for vals in column_values if vals]
    return x,spectra


def parse_jcamp_multispec(path:str)->Tuple[np.ndarray,List[np.ndarray],Dict]:
    headers=parse_jdx_headers(path)
    xfactor=float(headers.get('XFACTOR','1'))
    yfactor=float(headers.get('YFACTOR','1'))
    xunits=headers.get('XUNITS','')
    blocks=[]
    current=None
    with open(path,'r',errors='ignore') as f:
        for raw in f:
            line=raw.strip()
            if not line:
                continue
            upper=line.upper()
            if upper.startswith('##XYDATA='):
                descriptor=line.split('=',1)[1].strip() if '=' in line else ''
                current={'descriptor':descriptor,'lines':[]}
                blocks.append(current)
                continue
            if upper.startswith('##'):
                current=None
                continue
            if current is None:
                continue
            nums=[float(n) for n in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',line)]
            if nums:
                current['lines'].append(nums)
    if not blocks:
        raise ValueError('No XYDATA content')
    x_ref=None
    spectra=[]
    for block in blocks:
        lines=block['lines']
        if not lines:
            continue
        descriptor=block.get('descriptor','').upper()
        if 'X++' in descriptor:
            x_block,block_spectra=_expand_xpp_block(lines,headers,xfactor,yfactor)
        else:
            x_block,block_spectra=_parse_column_block(lines,xfactor,yfactor)
        if x_ref is None:
            x_ref=x_block
        else:
            if len(x_ref)!=len(x_block) or not np.allclose(x_ref,x_block,rtol=0,atol=1e-6):
                raise ValueError('Inconsistent X axes between XYDATA blocks')
        spectra.extend(block_spectra)
    if x_ref is None or not spectra:
        raise ValueError('No spectra parsed from JCAMP file')
    x_cm1=standardize_x(x_ref,xunits)
    return x_cm1,spectra,headers


def standardize_x(x:np.ndarray,units:str)->np.ndarray:
    u=units.strip().lower()
    if u in {'1/cm','cm-1','1/cm-1','wavenumbers','cm^-1'}:
        return x
    if u in {'micrometers','micrometer','microns','micron','um','µm'}:
        return 10000.0/np.maximum(x,1e-12)
    return x

def transmittance_to_abs(y:np.ndarray,percent:bool=False)->np.ndarray:
    """Convert a transmittance trace into absorbance.

    Parameters
    ----------
    y : np.ndarray
        Transmittance values expressed either as a fraction (0-1) or percent
        (0-100).
    percent : bool, optional
        When True, interpret the input values as percent transmittance.

    Returns
    -------
    np.ndarray
        Absorbance values computed as ``A = -log10(T)``.
    """

    y_arr=np.asarray(y,dtype=float)
    if percent:
        y_arr=y_arr/100.0
    T=np.clip(y_arr,1e-6,1.0)
    return -np.log10(T)


def _transmittance_mode(y_units:str)->Optional[str]:
    """Detect whether the provided Y units denote transmittance.

    Returns ``'fraction'`` for 0-1 transmittance, ``'percent'`` for %T, or
    ``None`` when no conversion is required.
    """

    if not y_units:
        return None
    u=y_units.strip().lower()
    if not u:
        return None
    compact=re.sub(r'\s+','',u)
    if 'transmittance' not in u and '%t' not in compact and 't%' not in compact:
        return None
    if '%' in u or 'percent' in u or 'pct' in u or '%t' in compact or 't%' in compact:
        return 'percent'
    return 'fraction'


def convert_y_for_processing(y:np.ndarray,y_units:str)->np.ndarray:
    """Return a copy of *y* converted to absorbance when required."""

    mode=_transmittance_mode(y_units)
    if mode is None:
        return np.asarray(y,dtype=float).copy()
    return transmittance_to_abs(y,percent=(mode=='percent'))


def sanitize_xy(x:np.ndarray,y:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    """Return finite, aligned ``(x, y)`` arrays for downstream processing."""

    x_arr=np.asarray(x,dtype=float)
    y_arr=np.asarray(y,dtype=float)

    n=min(len(x_arr),len(y_arr))
    if n==0:
        return x_arr[:0],y_arr[:0]

    x_arr=x_arr[:n].copy()
    y_arr=y_arr[:n].astype(float,copy=True)

    if not np.all(np.isfinite(y_arr)):
        finite_idx=np.flatnonzero(np.isfinite(y_arr))
        if finite_idx.size==0:
            return x_arr[:0],y_arr[:0]
        last=finite_idx[-1]+1
        x_arr=x_arr[:last]
        y_arr=y_arr[:last]

    finite_mask=np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(finite_mask):
        x_arr=x_arr[finite_mask]
        y_arr=y_arr[finite_mask]

    return x_arr,y_arr

def _nan_safe(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        return values
    idx = np.arange(values.size)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.zeros_like(values)
    filled = values.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], values[mask])
    return filled


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = _nan_safe(y)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L), format="csc")
    laplacian = (D.T @ D).tocsc()
    w = np.ones(L)
    for _ in range(max(1, int(niter))):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + lam * laplacian
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def airpls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    niter: int = 20,
    tol: float = 1e-6,
    weight_floor: float = 1e-3,
) -> np.ndarray:
    y = _nan_safe(y)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L), format="csc")
    H = lam * (D.T @ D).tocsc()
    w = np.ones(L)
    for idx in range(1, max(1, int(niter)) + 1):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        neg = d[d < 0]
        if neg.size == 0:
            break
        mean = float(np.mean(neg))
        std = float(np.std(neg))
        if std <= 1e-12:
            break
        if abs(mean) / std < tol:
            break
        scaled = np.clip((idx * d) / (abs(mean) + 1e-12), -50.0, 50.0)
        weights = np.exp(scaled)
        w = np.where(d >= 0, weight_floor, weights)
        w[0] = weight_floor
        w[-1] = weight_floor
    return z


def arpls_baseline(y: np.ndarray, lam: float = 1e5, niter: int = 20, tol: float = 1e-6) -> np.ndarray:
    return airpls_baseline(y, lam=lam, niter=niter, tol=tol)


def _parse_piecewise_ranges(ranges: str | None) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    parsed: List[Tuple[float, float]] = []
    for block in ranges.split(","):
        parts = block.replace("–", "-").split("-")
        if len(parts) != 2:
            continue
        try:
            start = float(parts[0])
            stop = float(parts[1])
        except ValueError:
            continue
        parsed.append((start, stop))
    return parsed


def _apply_baseline(
    y: np.ndarray,
    method: str,
    lam: float,
    p: float,
    niter: int,
) -> np.ndarray:
    method = (method or "").lower()
    if method == "asls":
        return als_baseline(y, lam=lam, p=p, niter=niter)
    if method == "arpls":
        return arpls_baseline(y, lam=lam, niter=niter)
    if method == "airpls":
        return airpls_baseline(y, lam=lam, niter=niter)
    raise ValueError(f"Unsupported baseline method: {method}")


def _baseline_piecewise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    lam: float,
    p: float,
    niter: int,
    ranges: List[Tuple[float, float]],
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.zeros_like(y)
    covered = np.zeros_like(y, dtype=bool)
    for start, stop in ranges:
        low = min(start, stop)
        high = max(start, stop)
        mask = (x >= low) & (x <= high)
        if not np.any(mask):
            continue
        baseline[mask] = _apply_baseline(y[mask], method, lam, p, niter)
        covered |= mask
    if not np.all(covered):
        fallback = _apply_baseline(y, method, lam, p, niter)
        baseline[~covered] = fallback[~covered]
    return baseline


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    als_lam: float,
    als_p: float,
    *,
    baseline_method: str = "airpls",
    baseline_niter: int = 20,
    baseline_piecewise: bool = False,
    baseline_ranges: str | None = None,
) -> np.ndarray:
    y2 = np.asarray(y, dtype=float).copy()
    n = len(y2)
    if sg_win and sg_win % 2 == 1 and sg_win > 2 and n >= 3:
        win = min(sg_win, n if n % 2 == 1 else n - 1)
        if win >= 3:
            poly = min(max(sg_poly, 0), win - 1)
            if win > poly:
                y2 = savgol_filter(y2, win, poly)
    if als_lam > 0:
        if n >= 3:
            ranges = _parse_piecewise_ranges(baseline_ranges)
            if baseline_piecewise and ranges:
                baseline = _baseline_piecewise(
                    x,
                    y2,
                    method=baseline_method,
                    lam=als_lam,
                    p=als_p,
                    niter=baseline_niter,
                    ranges=ranges,
                )
            else:
                baseline = _apply_baseline(y2, baseline_method, als_lam, als_p, baseline_niter)
            y2 = y2 - baseline
        elif n:
            y2 = y2 - np.mean(y2)
    m = np.max(np.abs(y2))
    if m > 0:
        y2 /= m
    return y2

def gaussian(x,A,x0,s,C): return A*np.exp(-0.5*((x-x0)/s)**2)+C

def fit_peak(x,y,idx,model,window):
    n=len(x);i0=max(0,idx-window);i1=min(n,idx+window);xs=x[i0:i1];ys=y[i0:i1]
    if len(xs)<5: return None
    try:
        A=np.max(ys)-np.median(ys);x0=x[idx];C=np.median(ys);sig=3*np.median(np.diff(xs))
        popt,_=curve_fit(gaussian,xs,ys,p0=[A,x0,sig,C],maxfev=6000)
        A,x0,sig,C=popt;yfit=gaussian(xs,*popt)
        fwhm=2.3548*abs(sig)
        r2=1-np.sum((ys-yfit)**2)/(np.sum((ys-np.mean(ys))**2)+1e-12)
        order=np.argsort(xs)
        xs_sorted=xs[order]
        y_sorted=yfit[order]
        area=float(np.trapz(y_sorted-C,xs_sorted))
        return dict(center=x0,fwhm=fwhm,amplitude=A,area=area,r2=r2)
    except: return None

def merge_peak_indices(y_proc: np.ndarray, pos_idxs: np.ndarray, neg_idxs: np.ndarray, min_distance_pts: int) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int, float]] = []
    if pos_idxs is not None:
        for idx in np.asarray(pos_idxs, dtype=int):
            if 0 <= idx < len(y_proc):
                candidates.append((int(idx), 1, float(abs(y_proc[int(idx)]))))
    if neg_idxs is not None:
        for idx in np.asarray(neg_idxs, dtype=int):
            if 0 <= idx < len(y_proc):
                candidates.append((int(idx), -1, float(abs(y_proc[int(idx)]))))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    min_distance_pts = max(int(min_distance_pts), 0)
    for idx, polarity, score in candidates:
        if any(abs(idx - prev_idx) <= min_distance_pts for prev_idx, _, _ in selected):
            continue
        selected.append((idx, polarity, score))
    selected.sort(key=lambda item: item[0])
    return [(idx, polarity) for idx, polarity, _ in selected]

def init_db(outdir:str):
    os.makedirs(outdir,exist_ok=True)
    con=duckdb.connect(os.path.join(outdir,'peaks.duckdb'))
    con.execute('''CREATE TABLE IF NOT EXISTS spectra(
        file_id TEXT PRIMARY KEY,
        path TEXT,
        n_points INT,
        n_spectra INT,
        meta_json TEXT
    );''')
    for column,sql_type in PROMOTED_COLUMN_TYPES.items():
        con.execute(f'ALTER TABLE spectra ADD COLUMN IF NOT EXISTS {column} {sql_type}')
    con.execute('''CREATE TABLE IF NOT EXISTS peaks(file_id TEXT,spectrum_id INT,peak_id INT,polarity INT,center DOUBLE,fwhm DOUBLE,amplitude DOUBLE,area DOUBLE,r2 DOUBLE,PRIMARY KEY(file_id,spectrum_id,peak_id));''')
    con.execute('ALTER TABLE peaks ADD COLUMN IF NOT EXISTS polarity INT')
    con.execute('''CREATE TABLE IF NOT EXISTS file_consensus(file_id TEXT,cluster_id INT,polarity INT,center DOUBLE,fwhm DOUBLE,support INT,PRIMARY KEY(file_id,cluster_id,polarity));''')
    con.execute('ALTER TABLE file_consensus ADD COLUMN IF NOT EXISTS polarity INT')
    con.execute('''CREATE TABLE IF NOT EXISTS global_consensus(cluster_id INT,polarity INT,center DOUBLE,support INT,PRIMARY KEY(cluster_id,polarity));''')
    con.execute('ALTER TABLE global_consensus ADD COLUMN IF NOT EXISTS polarity INT')
    con.execute('''CREATE TABLE IF NOT EXISTS ingest_errors(
        file_path TEXT PRIMARY KEY,
        error TEXT,
        occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );''')
    return con

def store_headers(con,file_id:str,headers:Dict[str,str]):
    meta_json=json.dumps(headers,ensure_ascii=False)
    columns=['file_id','path','n_points','n_spectra','meta_json']
    values=[file_id,'',0,0,meta_json]
    for key,column in PROMOTED_KEYS.items():
        raw=headers.get(key)
        if raw is None:
            coerced=None
        elif key in NUMERIC_KEYS:
            num=_parse_numeric(raw)
            coerced=float(num) if num is not None else None
        else:
            coerced=raw
        columns.append(column)
        values.append(coerced)
    placeholders=','.join(['?']*len(values))
    col_clause=','.join(columns)
    con.execute(
        f'INSERT OR REPLACE INTO spectra ({col_clause}) VALUES ({placeholders})',
        values
    )

def index_file(path:str,con,args,failed_files=None):
    def record_failure(message: str) -> None:
        if failed_files is not None:
            failed_files.append((path, message))

    headers=parse_jdx_headers(path)
    try:
        is_ftir_spectrum(headers)
    except UnsupportedSpectrumError as exc:
        message=f"Unsupported spectrum type: {exc.descriptor}"
        logger.info("Skipping non-FTIR spectrum %s: %s", path, exc)
        record_failure(message)
        try:
            con.execute(
                """
                INSERT INTO ingest_errors (file_path,error)
                VALUES (?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    error=excluded.error,
                    occurred_at=now()
                """,
                [path, message]
            )
        except Exception:
            logger.debug("Failed to record ingest error for %s", path, exc_info=True)
        if getattr(args,'strict',False) or getattr(args,'_collect_skips',False):
            raise
        return 0,0
    try:
        x,Y,headers=parse_jcamp_multispec(path)
    except Exception as exc:
        logger.error("Failed to parse JCAMP file %s: %s", path, exc)
        record_failure(str(exc))
        try:
            con.execute(
                """
                INSERT INTO ingest_errors (file_path,error)
                VALUES (?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    error=excluded.error,
                    occurred_at=now()
                """,
                [path, str(exc)]
            )
        except Exception:
            logger.debug("Failed to record ingest error for %s", path, exc_info=True)
        if getattr(args,'strict',False):
            raise
        return 0,0
    file_id=file_sha1(path)
    total_peaks=0
    processed_spectra=0
    max_points=0
    y_units=headers.get('YUNITS','')

    con.execute('DELETE FROM ingest_errors WHERE file_path=?',[path])

    con.execute('BEGIN TRANSACTION')
    try:
        store_headers(con,file_id,headers)
        con.execute('DELETE FROM peaks WHERE file_id=?',[file_id])

        for sid,y in enumerate(Y):
            x_clean,y_clean=sanitize_xy(x,y)
            if x_clean.size==0 or y_clean.size==0:
                continue

            processed_spectra+=1
            max_points=max(max_points,len(x_clean))

            y_for_processing=convert_y_for_processing(y_clean,y_units)
            y_proc=preprocess(
                x_clean,
                y_for_processing,
                args.sg_win,
                args.sg_poly,
                args.als_lam,
                args.als_p,
                baseline_method=args.baseline_method,
                baseline_niter=args.baseline_niter,
                baseline_piecewise=args.baseline_piecewise,
                baseline_ranges=args.baseline_ranges,
            )
            diffs=np.diff(x_clean)
            if diffs.size:
                step=np.nanmedian(np.abs(diffs))
            else:
                step=np.nan
            if not np.isfinite(step) or step<=0:
                span=np.nanmax(x_clean)-np.nanmin(x_clean) if x_clean.size else np.nan
                if np.isfinite(span) and span>0 and len(x_clean)>1:
                    step=span/(len(x_clean)-1)
                else:
                    step=float(args.min_distance) if args.min_distance>0 else 1.0
            dist_pts=max(1,int(np.ceil(float(args.min_distance)/max(step,1e-9))))
            idxs_pos,_=find_peaks(y_proc,prominence=args.prominence,distance=dist_pts)
            idxs_neg=np.array([],dtype=int)
            if getattr(args,'detect_negative_peaks',False):
                idxs_neg,_=find_peaks(-y_proc,prominence=args.prominence,distance=dist_pts)
            peak_candidates=merge_peak_indices(y_proc,idxs_pos,idxs_neg,dist_pts)
            pid=0
            for i,polarity in peak_candidates:
                fit_y=-y_proc if polarity<0 else y_proc
                fit=fit_peak(x_clean,fit_y,i,args.model,args.fit_window_pts)
                if not fit or fit['r2']<args.min_r2: continue
                if polarity<0:
                    fit=dict(fit)
                    fit['amplitude']*=-1
                    fit['area']*=-1
                con.execute(
                    'INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?,?,?,?,?)',
                    [
                        file_id,
                        sid,
                        pid,
                        int(polarity),
                        fit['center'],
                        fit['fwhm'],
                        fit['amplitude'],
                        fit['area'],
                        fit['r2'],
                    ],
                )
                pid+=1
            total_peaks+=pid
        con.execute('UPDATE spectra SET path=?,n_points=?,n_spectra=? WHERE file_id=?',[path,max_points,processed_spectra,file_id])
        con.execute('COMMIT')
    except Exception:
        con.execute('ROLLBACK')
        raise
    return processed_spectra,total_peaks

def weighted_median(x,w):
    o=np.argsort(x);x=x[o];w=w[o];c=np.cumsum(w)/(np.sum(w)+1e-12)
    return x[min(np.searchsorted(c,0.5),len(x)-1)]

def build_file_consensus(con,args):
    df=con.execute('SELECT file_id,COALESCE(polarity,1) AS polarity,center,fwhm,area,r2 FROM peaks').fetch_df()
    if df.empty: return pd.DataFrame()
    out=[]
    for (fid, polarity), g in df.groupby(['file_id','polarity']):
        c=g['center'].to_numpy();f=g['fwhm'].to_numpy();w=(np.abs(g['area'])*g['r2']).to_numpy()
        eps=max(args.file_eps_factor*np.nanmedian(f),args.file_eps_min)
        labels=DBSCAN(eps=eps,min_samples=args.file_min_samples).fit_predict(c.reshape(-1,1))
        for cl in set(labels):
            if cl==-1: continue
            m=labels==cl
            cx=c[m]
            cf=f[m]
            cw=w[m]
            valid=np.isfinite(cw) & (cw>0)
            if not np.any(valid):
                continue
            cx=cx[valid]
            cf=cf[valid]
            cw=cw[valid]
            out.append(
                dict(
                    file_id=fid,
                    polarity=int(polarity),
                    center=weighted_median(cx,cw),
                    fwhm=np.nanmedian(cf),
                    support=int(np.sum(valid)),
                )
            )
    fc=pd.DataFrame(out)
    if fc.empty:
        return fc
    fc=fc.sort_values(['file_id','polarity','center']).reset_index(drop=True)
    fc['cluster_id']=fc.groupby(['file_id']).cumcount()
    cols=['file_id','cluster_id','polarity','center','fwhm','support']
    return fc[cols]

def build_global_consensus(con,fc,args):
    if fc.empty: return pd.DataFrame()
    out=[]
    for polarity, group in fc.groupby('polarity'):
        c=group['center'].to_numpy()
        labels=DBSCAN(eps=args.global_eps_abs,min_samples=args.global_min_samples).fit_predict(c.reshape(-1,1))
        for cl in set(labels):
            if cl==-1: continue
            m=labels==cl;cx=c[m]
            out.append(dict(polarity=int(polarity), center=float(np.median(cx)), support=int(np.sum(m))))
    gc=pd.DataFrame(out)
    if gc.empty:
        return gc
    gc=gc.sort_values(['polarity','center']).reset_index(drop=True)
    gc['cluster_id']=np.arange(len(gc),dtype=int)
    cols=['cluster_id','polarity','center','support']
    return gc[cols]

def persist_consensus(con,fc,gc):
    con.execute('DELETE FROM file_consensus')
    con.execute('DELETE FROM global_consensus')
    if not fc.empty:
        con.executemany(
            'INSERT INTO file_consensus (file_id,cluster_id,polarity,center,fwhm,support) VALUES (?,?,?,?,?,?)',
            fc[['file_id','cluster_id','polarity','center','fwhm','support']].itertuples(index=False,name=None)
        )
    if not gc.empty:
        con.executemany(
            'INSERT INTO global_consensus (cluster_id,polarity,center,support) VALUES (?,?,?,?)',
            gc[['cluster_id','polarity','center','support']].itertuples(index=False,name=None)
        )

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap=argparse.ArgumentParser()
    ap.add_argument(
        'data_dir',
        nargs='?',
        default=script_dir,
        help='Directory containing JCAMP-DX files (defaults to script location)',
    )
    ap.add_argument(
        'index_dir',
        nargs='?',
        default=script_dir,
        help='Output directory for DuckDB/Parquet files (defaults to script location)',
    )
    ap.add_argument('--prominence',type=float,default=0.01);ap.add_argument('--min-distance',type=float,default=3.0,dest='min_distance')
    ap.add_argument('--sg-win',type=int,default=7,dest='sg_win');ap.add_argument('--sg-poly',type=int,default=3,dest='sg_poly')
    ap.add_argument('--als-lam',type=float,default=5e4,dest='als_lam');ap.add_argument('--als-p',type=float,default=0.01,dest='als_p')
    ap.add_argument('--baseline-method',choices=['arpls','asls','airpls'],default='airpls',dest='baseline_method')
    ap.add_argument('--baseline-niter',type=int,default=20,dest='baseline_niter')
    ap.add_argument('--baseline-piecewise',action='store_true',dest='baseline_piecewise')
    ap.add_argument(
        '--baseline-ranges',
        default='4000-2500,2500-1800,1800-900,900-450',
        dest='baseline_ranges',
        help='Comma-separated wavenumber ranges for piecewise baseline (e.g. 4000-2500,2500-1800).',
    )
    ap.add_argument('--model',choices=['Gaussian'],default='Gaussian');ap.add_argument('--min-r2',type=float,default=0.85,dest='min_r2')
    ap.add_argument('--fit-window-pts',type=int,default=70,dest='fit_window_pts')
    ap.add_argument('--file-min-samples',type=int,default=2);ap.add_argument('--file-eps-factor',type=float,default=0.5);ap.add_argument('--file-eps-min',type=float,default=2.0)
    ap.add_argument('--global-min-samples',type=int,default=2);ap.add_argument('--global-eps-abs',type=float,default=4.0)
    ap.add_argument('--detect-negative-peaks',action='store_true',dest='detect_negative_peaks',help='Also detect negative peaks by searching inverted spectra.')
    ap.add_argument('--strict',action='store_true',help='Raise exceptions during indexing instead of skipping files')
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    con=init_db(args.index_dir)
    files=[p for p in glob.glob(os.path.join(args.data_dir,'**','*'),recursive=True) if os.path.isfile(p) and re.search(r'\.(jdx|dx)$',p,re.I)]
    total_files=len(files)
    print(f'Found {total_files} JCAMP file(s). Starting indexing...')
    total_specs=0;total_peaks=0
    for idx,p in enumerate(files,1):
        print(f'[{idx}/{total_files}] Indexing {p}', file=sys.stdout, flush=True)
        try:
            n_s,n_p=index_file(p,con,args)
        except UnsupportedSpectrumError:
            if getattr(args,'strict',False):
                raise
            n_s,n_p=0,0
        total_specs+=n_s;total_peaks+=n_p
    print('Building file-level consensus clusters...', file=sys.stdout, flush=True)
    fc=build_file_consensus(con,args)
    print('Building global consensus clusters...', file=sys.stdout, flush=True)
    gc=build_global_consensus(con,fc,args)
    persist_consensus(con,fc,gc)
    print(f'Indexed spectra: {total_specs} | Peaks: {total_peaks} | File clusters: {len(fc)} | Global: {len(gc)}')
    con.close()
if __name__=='__main__': main()
