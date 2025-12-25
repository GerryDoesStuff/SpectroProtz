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
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.special import wofz
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


MAD_SCALE = 1.4826


def _estimate_spacing(x: np.ndarray) -> float:
    diffs = np.diff(np.asarray(x, dtype=float))
    if diffs.size == 0:
        return float("nan")
    spacing = float(np.nanmedian(np.abs(diffs)))
    return spacing if np.isfinite(spacing) and spacing > 0 else float("nan")


def _resolve_sg_window_points(
    x: np.ndarray,
    sg_win: int | None,
    sg_window_cm: float | None,
    n: int,
) -> int | None:
    window: int | None = None
    if sg_window_cm is not None and sg_window_cm > 0:
        spacing = _estimate_spacing(x)
        if np.isfinite(spacing) and spacing > 0:
            points = int(np.round(float(sg_window_cm) / spacing)) + 1
            if points < 3:
                points = 3
            if points % 2 == 0:
                points += 1
            window = points
    if window is None and sg_win:
        window = int(sg_win)
        if window % 2 == 0:
            window += 1
    if window is None:
        return None
    if n < 3:
        return None
    max_window = n if n % 2 == 1 else n - 1
    return min(window, max_window) if max_window >= 3 else None


def _coerce_sg_polyorder(value: int) -> int:
    poly = int(value)
    if poly < 2:
        return 2
    if poly > 3:
        return 3
    return poly


def _apply_sg_smoothing(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    sg_window_cm: float | None,
) -> np.ndarray:
    y2 = np.asarray(y, dtype=float).copy()
    n = y2.size
    if n < 3:
        return y2
    window = _resolve_sg_window_points(x, sg_win, sg_window_cm, n)
    if window is None or window < 3:
        return y2
    poly = _coerce_sg_polyorder(sg_poly)
    if window <= poly:
        poly = max(1, window - 1)
    if window <= poly or poly < 1:
        return y2
    return savgol_filter(y2, window, poly)


def estimate_noise_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    sg_window_cm: float | None = None,
) -> float:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size < 3:
        return float("nan")
    smooth = _apply_sg_smoothing(x, y_arr, sg_win, sg_poly, sg_window_cm)
    residual = y_arr - smooth
    median = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - median)))
    if not np.isfinite(mad):
        return float("nan")
    return MAD_SCALE * mad


def _append_processing_step(
    registry: List[Dict[str, object]] | None,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    metadata: Optional[Dict[str, object]] = None,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    if registry is None:
        return
    step_metadata = dict(metadata or {})
    if extra_metadata:
        step_metadata.update(extra_metadata)
    registry.append(
        {
            "label": label,
            "x": np.asarray(x, dtype=float).copy(),
            "y": np.asarray(y, dtype=float).copy(),
            "metadata": step_metadata,
        }
    )


def preprocess_with_noise(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    als_lam: float,
    als_p: float,
    *,
    sg_window_cm: float | None = None,
    baseline_method: str = "airpls",
    baseline_niter: int = 20,
    baseline_piecewise: bool = False,
    baseline_ranges: str | None = None,
    step_registry: Optional[List[Dict[str, object]]] = None,
    step_metadata: Optional[Dict[str, object]] = None,
) -> tuple[np.ndarray, float]:
    y2 = np.asarray(y, dtype=float).copy()
    n = len(y2)
    noise_sigma = estimate_noise_sigma(x, y2, sg_win, sg_poly, sg_window_cm)
    y2 = _apply_sg_smoothing(x, y2, sg_win, sg_poly, sg_window_cm)
    _append_processing_step(step_registry, "smoothed", x, y2, step_metadata)
    baseline = np.zeros_like(y2)
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
            baseline = np.full_like(y2, float(np.mean(y2)))
            y2 = y2 - baseline
    _append_processing_step(step_registry, "baseline_estimate", x, baseline, step_metadata)
    _append_processing_step(step_registry, "baseline_corrected", x, y2, step_metadata)
    m = np.max(np.abs(y2))
    if m > 0:
        y2 /= m
        if np.isfinite(noise_sigma):
            noise_sigma /= m
    _append_processing_step(
        step_registry,
        "preprocessed",
        x,
        y2,
        step_metadata,
        extra_metadata={"normalization_factor": float(m) if np.isfinite(m) else float("nan")},
    )
    return y2, noise_sigma


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    sg_win: int,
    sg_poly: int,
    als_lam: float,
    als_p: float,
    *,
    sg_window_cm: float | None = None,
    baseline_method: str = "airpls",
    baseline_niter: int = 20,
    baseline_piecewise: bool = False,
    baseline_ranges: str | None = None,
) -> np.ndarray:
    y2, _ = preprocess_with_noise(
        x,
        y,
        sg_win,
        sg_poly,
        als_lam,
        als_p,
        sg_window_cm=sg_window_cm,
        baseline_method=baseline_method,
        baseline_niter=baseline_niter,
        baseline_piecewise=baseline_piecewise,
        baseline_ranges=baseline_ranges,
    )
    return y2

def gaussian(x, A, x0, s, C):
    return A * np.exp(-0.5 * ((x - x0) / s) ** 2) + C


def lorentzian(x, A, x0, gamma, C):
    return A / (1.0 + ((x - x0) / gamma) ** 2) + C


def voigt_profile(x, A, x0, sigma, gamma, C):
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return A * profile + C


def _ensure_odd_window(window: int, max_len: int) -> int:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > max_len:
        window = max_len if max_len % 2 == 1 else max_len - 1
    return max(window, 3)


def _resolve_step_from_x(xs: np.ndarray) -> float:
    step = np.median(np.abs(np.diff(xs))) if xs.size > 1 else 0.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    return float(step)


def _compute_second_derivative(xs: np.ndarray, ys: np.ndarray, distance_pts: int) -> np.ndarray:
    if ys.size < 5:
        return np.zeros_like(ys)
    window = _ensure_odd_window(max(7, distance_pts * 2 + 1), ys.size)
    poly = 3 if window >= 5 else 2
    step = _resolve_step_from_x(xs)
    try:
        return savgol_filter(ys, window_length=window, polyorder=poly, deriv=2, delta=step)
    except Exception:
        first = np.gradient(ys, step)
        return np.gradient(first, step)


def _fit_peak_spline(xs: np.ndarray, ys: np.ndarray, x0_guess: float) -> Optional[Dict[str, float]]:
    if xs.size < 5:
        return None
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    smoothing = max(0.0, 0.001 * np.nanvar(ys_sorted) * xs_sorted.size)
    spline = UnivariateSpline(xs_sorted, ys_sorted, k=3, s=smoothing)
    dense_x = np.linspace(xs_sorted[0], xs_sorted[-1], xs_sorted.size * 8)
    dense_y = spline(dense_x)
    local_mask = np.abs(dense_x - x0_guess) <= (xs_sorted[-1] - xs_sorted[0]) * 0.5
    if not np.any(local_mask):
        local_mask = np.ones_like(dense_x, dtype=bool)
    local_idx = int(np.argmax(dense_y[local_mask]))
    local_x = dense_x[local_mask][local_idx]
    peak_y = float(dense_y[local_mask][local_idx])
    baseline = float(np.median(ys_sorted))
    half_height = baseline + (peak_y - baseline) * 0.5
    left_candidates = np.where(dense_x <= local_x)[0]
    right_candidates = np.where(dense_x >= local_x)[0]
    left_idx = left_candidates[np.where(dense_y[left_candidates] <= half_height)[0]]
    right_idx = right_candidates[np.where(dense_y[right_candidates] <= half_height)[0]]
    if left_idx.size == 0 or right_idx.size == 0:
        return None
    left_x = float(dense_x[left_idx[-1]])
    right_x = float(dense_x[right_idx[0]])
    fwhm = float(abs(right_x - left_x))
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(np.maximum(dense_y - baseline, 0.0), dense_x))
    else:
        area = float(np.trapz(np.maximum(dense_y - baseline, 0.0), dense_x))
    return dict(center=float(local_x), fwhm=fwhm, amplitude=float(peak_y - baseline), area=area, r2=1.0)


def _fit_peak_single(
    xs: np.ndarray,
    ys: np.ndarray,
    x0_guess: float,
    model: str,
    *,
    center_bounds: tuple[float, float] | None = None,
) -> Optional[Dict[str, float]]:
    if xs.size < 5:
        return None
    model = model.lower()
    if model == "spline":
        return _fit_peak_spline(xs, ys, x0_guess)
    local_idx = int(np.argmin(np.abs(xs - x0_guess)))
    step = _resolve_step_from_x(xs)
    C = float(np.median(ys))
    A = max(float(ys[local_idx] - C), 1e-6)
    width_guess = max(step * 2.0, np.abs(xs[-1] - xs[0]) / 6.0)
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    x0 = float(x0_guess)
    x0_min = float(np.min(xs_sorted))
    x0_max = float(np.max(xs_sorted))
    if center_bounds is not None:
        bound_min = float(min(center_bounds))
        bound_max = float(max(center_bounds))
        x0_min = max(x0_min, bound_min)
        x0_max = min(x0_max, bound_max)
        if x0_min >= x0_max:
            x0_min = float(np.min(xs_sorted))
            x0_max = float(np.max(xs_sorted))
    if model == "lorentzian":
        bounds = (
            [0.0, x0_min, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf],
        )
        popt, _ = curve_fit(
            lorentzian,
            xs_sorted,
            ys_sorted,
            p0=[A, x0, width_guess, C],
            maxfev=12000,
            bounds=bounds,
        )
        A, x0, gamma, C = popt
        yfit = lorentzian(xs_sorted, *popt)
        fwhm = 2.0 * abs(gamma)
        component = lorentzian(xs_sorted, A, x0, gamma, 0.0)
    elif model == "voigt":
        bounds = (
            [0.0, x0_min, step * 0.25, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf, np.inf],
        )
        popt, _ = curve_fit(
            voigt_profile,
            xs_sorted,
            ys_sorted,
            p0=[A, x0, width_guess, width_guess, C],
            maxfev=20000,
            bounds=bounds,
        )
        A, x0, sigma, gamma, C = popt
        yfit = voigt_profile(xs_sorted, *popt)
        fwhm = 0.5346 * (2.0 * abs(gamma)) + math.sqrt(
            0.2166 * (2.0 * abs(gamma)) ** 2 + (2.3548 * abs(sigma)) ** 2
        )
        component = voigt_profile(xs_sorted, A, x0, sigma, gamma, 0.0)
    else:
        bounds = (
            [0.0, x0_min, step * 0.25, -np.inf],
            [np.inf, x0_max, np.inf, np.inf],
        )
        popt, _ = curve_fit(
            gaussian,
            xs_sorted,
            ys_sorted,
            p0=[A, x0, width_guess, C],
            maxfev=12000,
            bounds=bounds,
        )
        A, x0, sigma, C = popt
        yfit = gaussian(xs_sorted, *popt)
        fwhm = 2.3548 * abs(sigma)
        component = gaussian(xs_sorted, A, x0, sigma, 0.0)
    r2 = 1 - np.sum((ys_sorted - yfit) ** 2) / (np.sum((ys_sorted - np.mean(ys_sorted)) ** 2) + 1e-12)
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(component, xs_sorted))
    else:
        area = float(np.trapz(component, xs_sorted))
    return dict(center=float(x0), fwhm=float(fwhm), amplitude=float(A), area=area, r2=float(r2))


def _fit_multi_peak(
    xs: np.ndarray,
    ys: np.ndarray,
    centers: List[float],
    model: str,
) -> Optional[Tuple[List[Dict[str, float]], float]]:
    if xs.size < 5 or not centers:
        return None
    model = model.lower()
    if model not in {"gaussian", "lorentzian", "voigt"}:
        return None
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    step = _resolve_step_from_x(xs_sorted)
    C0 = float(np.median(ys_sorted))
    span = float(np.abs(xs_sorted[-1] - xs_sorted[0]))
    width_guess = max(step * 2.0, span / max(len(centers) * 3.0, 3.0))
    params = []
    lower = []
    upper = []
    for center in centers:
        local_idx = int(np.argmin(np.abs(xs_sorted - center)))
        A0 = max(float(ys_sorted[local_idx] - C0), 1e-6)
        params.append(A0)
        params.append(float(center))
        params.append(width_guess)
        if model == "voigt":
            params.append(width_guess)
            lower.extend([0.0, float(np.min(xs_sorted)), step * 0.25, step * 0.25])
            upper.extend([np.inf, float(np.max(xs_sorted)), np.inf, np.inf])
        else:
            lower.extend([0.0, float(np.min(xs_sorted)), step * 0.25])
            upper.extend([np.inf, float(np.max(xs_sorted)), np.inf])
    params.append(C0)
    lower.append(-np.inf)
    upper.append(np.inf)

    def _sum_gaussian(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += gaussian(x, p[idx], p[idx + 1], p[idx + 2], 0.0)
            idx += 3
        total += p[-1]
        return total

    def _sum_lorentzian(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += lorentzian(x, p[idx], p[idx + 1], p[idx + 2], 0.0)
            idx += 3
        total += p[-1]
        return total

    def _sum_voigt(x, *p):
        total = np.zeros_like(x, dtype=float)
        idx = 0
        for _ in centers:
            total += voigt_profile(x, p[idx], p[idx + 1], p[idx + 2], p[idx + 3], 0.0)
            idx += 4
        total += p[-1]
        return total

    fit_fn = _sum_gaussian if model == "gaussian" else _sum_lorentzian if model == "lorentzian" else _sum_voigt
    popt, _ = curve_fit(
        fit_fn,
        xs_sorted,
        ys_sorted,
        p0=params,
        bounds=(lower, upper),
        maxfev=40000,
    )
    yfit = fit_fn(xs_sorted, *popt)
    r2 = 1 - np.sum((ys_sorted - yfit) ** 2) / (np.sum((ys_sorted - np.mean(ys_sorted)) ** 2) + 1e-12)
    results: List[Dict[str, float]] = []
    idx = 0
    for _ in centers:
        A = float(popt[idx])
        x0 = float(popt[idx + 1])
        if model == "voigt":
            sigma = float(popt[idx + 2])
            gamma = float(popt[idx + 3])
            fwhm = 0.5346 * (2.0 * abs(gamma)) + math.sqrt(
                0.2166 * (2.0 * abs(gamma)) ** 2 + (2.3548 * abs(sigma)) ** 2
            )
            component = voigt_profile(xs_sorted, A, x0, sigma, gamma, 0.0)
            idx += 4
        elif model == "lorentzian":
            gamma = float(popt[idx + 2])
            fwhm = 2.0 * abs(gamma)
            component = lorentzian(xs_sorted, A, x0, gamma, 0.0)
            idx += 3
        else:
            sigma = float(popt[idx + 2])
            fwhm = 2.3548 * abs(sigma)
            component = gaussian(xs_sorted, A, x0, sigma, 0.0)
            idx += 3
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(component, xs_sorted))
        else:
            area = float(np.trapz(component, xs_sorted))
        results.append(
            dict(center=x0, fwhm=float(fwhm), amplitude=A, area=area, r2=float(r2))
        )
    return results, float(r2)


def _fit_peak_window(x: np.ndarray, y: np.ndarray, idx: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    i0 = max(0, idx - window)
    i1 = min(n, idx + window)
    return x[i0:i1], y[i0:i1]


def fit_peak(
    x,
    y,
    idx,
    model,
    window,
    *,
    center_bounds: tuple[float, float] | None = None,
    x0_guess: float | None = None,
):
    xs, ys = _fit_peak_window(x, y, idx, window)
    if len(xs) < 5:
        return None
    if x0_guess is None:
        x0_guess = float(x[idx])
    step = _resolve_step_from_x(xs)
    max_delta = step * window * 0.5
    local_mask = np.abs(xs - x0_guess) <= max_delta
    if np.count_nonzero(local_mask) >= 5:
        xs = xs[local_mask]
        ys = ys[local_mask]
    try:
        return _fit_peak_single(xs, ys, x0_guess, model, center_bounds=center_bounds)
    except Exception:
        return None

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


def _get_param(args: object, name: str, default):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _resolve_step(x: np.ndarray, fallback: float) -> float:
    diffs = np.diff(x)
    if diffs.size:
        step = float(np.nanmedian(np.abs(diffs)))
    else:
        step = float("nan")
    if not np.isfinite(step) or step <= 0:
        span = float(np.nanmax(x) - np.nanmin(x)) if x.size else float("nan")
        if np.isfinite(span) and span > 0 and len(x) > 1:
            step = span / (len(x) - 1)
        else:
            step = fallback if fallback > 0 else 1.0
    return step


def _resolve_width_bounds_cm(
    width_min_cm: float,
    width_max_cm: float,
    *,
    step_cm: float,
    resolution_cm: float | None,
) -> tuple[float, float]:
    width_min_cm = float(width_min_cm)
    width_max_cm = float(width_max_cm)
    if width_min_cm <= 0:
        if resolution_cm is not None and resolution_cm > 0:
            width_min_cm = max(float(resolution_cm), step_cm)
        else:
            width_min_cm = step_cm
    if width_max_cm <= 0:
        width_max_cm = max(width_min_cm * 6.0, width_min_cm + step_cm * 4.0)
    if width_max_cm < width_min_cm:
        width_max_cm = width_min_cm
    return width_min_cm, width_max_cm


def _cluster_indices_by_tolerance(
    indices: np.ndarray,
    *,
    x: np.ndarray,
    scores: np.ndarray,
    tolerance_cm: float,
) -> List[int]:
    if indices.size == 0:
        return []
    order = np.argsort(x[indices])
    indices = indices[order]
    scores = scores[order]
    groups: List[List[int]] = []
    current: List[int] = [int(indices[0])]
    for idx in indices[1:]:
        if abs(float(x[int(idx)]) - float(x[current[-1]])) <= tolerance_cm:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)
    representatives: List[int] = []
    for group in groups:
        group_scores = [scores[np.where(indices == g)[0][0]] for g in group]
        best_idx = group[int(np.argmax(group_scores))]
        representatives.append(best_idx)
    return representatives


def _build_cwt_widths(
    *,
    width_min_cm: float,
    width_max_cm: float,
    width_step_cm: float,
    step_cm: float,
) -> np.ndarray:
    if width_step_cm <= 0:
        width_step_cm = max(step_cm, (width_max_cm - width_min_cm) / 8.0 if width_max_cm > width_min_cm else step_cm)
    widths_cm = np.arange(width_min_cm, width_max_cm + width_step_cm * 0.5, width_step_cm)
    widths_pts = np.unique(np.clip(np.round(widths_cm / step_cm), 1, None)).astype(int)
    return widths_pts


def detect_peak_candidates(
    x: np.ndarray,
    y_proc: np.ndarray,
    noise_sigma: float,
    args: object,
    *,
    resolution_cm: float | None = None,
) -> List[Dict[str, object]]:
    min_distance_cm = float(_get_param(args, "min_distance", 1.0))
    step_cm = _resolve_step(x, min_distance_cm)
    distance_pts = max(1, int(np.ceil(min_distance_cm / max(step_cm, 1e-9))))

    prominence_floor = float(_get_param(args, "prominence", 0.01))
    sigma_multiplier = float(_get_param(args, "noise_sigma_multiplier", 3.0))
    if np.isfinite(noise_sigma) and noise_sigma > 0:
        effective_prominence = sigma_multiplier * float(noise_sigma)
    else:
        effective_prominence = prominence_floor

    width_min_cm, width_max_cm = _resolve_width_bounds_cm(
        float(_get_param(args, "peak_width_min", 0.0)),
        float(_get_param(args, "peak_width_max", 0.0)),
        step_cm=step_cm,
        resolution_cm=resolution_cm,
    )
    width_min_pts = max(1, int(np.ceil(width_min_cm / max(step_cm, 1e-9))))
    width_max_pts = max(width_min_pts, int(np.ceil(width_max_cm / max(step_cm, 1e-9))))
    width_bounds = (width_min_pts, width_max_pts)

    detect_negative = bool(_get_param(args, "detect_negative_peaks", False))
    cwt_enabled = bool(_get_param(args, "cwt_enabled", False))
    cwt_width_min_cm = float(_get_param(args, "cwt_width_min", 0.0))
    cwt_width_max_cm = float(_get_param(args, "cwt_width_max", 0.0))
    cwt_width_step_cm = float(_get_param(args, "cwt_width_step", 0.0))
    cwt_cluster_tolerance_cm = float(_get_param(args, "cwt_cluster_tolerance", 8.0))
    merge_tolerance_cm = float(_get_param(args, "merge_tolerance", 8.0))
    if min_distance_cm > 0:
        merge_tolerance_cm = min(merge_tolerance_cm, min_distance_cm)

    candidates: List[Dict[str, object]] = []

    def _add_candidate(index: int, polarity: int, source: str, metadata: Dict[str, object]):
        candidates.append(
            {
                "index": int(index),
                "polarity": int(polarity),
                "score": float(abs(y_proc[int(index)])),
                "detections": [metadata | {"source": source}],
                "sources": {source},
            }
        )

    idxs_pos, _ = find_peaks(
        y_proc, prominence=effective_prominence, distance=distance_pts, width=width_bounds
    )
    for idx in np.asarray(idxs_pos, dtype=int):
        if 0 <= idx < len(y_proc):
            _add_candidate(idx, 1, "prominence", {"prominence": effective_prominence, "width_pts": width_bounds})

    plateau_min_points = max(2, int(_get_param(args, "plateau_min_points", 3)))
    plateau_prominence_factor = float(_get_param(args, "plateau_prominence_factor", 1.0))
    plateau_prominence = max(effective_prominence * plateau_prominence_factor, 0.0)
    if plateau_min_points >= 2:
        plateau_idxs, plateau_props = find_peaks(
            y_proc,
            prominence=plateau_prominence,
            distance=distance_pts,
            plateau_size=plateau_min_points,
        )
        if plateau_idxs.size:
            left_edges = plateau_props.get("left_edges", [])
            right_edges = plateau_props.get("right_edges", [])
            sizes = plateau_props.get("plateau_sizes", [])
            for idx, left, right, size in zip(plateau_idxs, left_edges, right_edges, sizes):
                left = int(left)
                right = int(right)
                center = int(round((left + right) / 2))
                if 0 <= center < len(y_proc):
                    _add_candidate(
                        center,
                        1,
                        "plateau",
                        {
                            "plateau_left": left,
                            "plateau_right": right,
                            "plateau_size": int(size),
                            "prominence": plateau_prominence,
                        },
                    )

    if detect_negative:
        idxs_neg, _ = find_peaks(
            -y_proc, prominence=effective_prominence, distance=distance_pts, width=width_bounds
        )
        for idx in np.asarray(idxs_neg, dtype=int):
            if 0 <= idx < len(y_proc):
                _add_candidate(idx, -1, "prominence", {"prominence": effective_prominence, "width_pts": width_bounds})

    if cwt_enabled:
        cwt_width_min_cm, cwt_width_max_cm = _resolve_width_bounds_cm(
            cwt_width_min_cm or width_min_cm,
            cwt_width_max_cm or width_max_cm,
            step_cm=step_cm,
            resolution_cm=resolution_cm,
        )
        widths_pts = _build_cwt_widths(
            width_min_cm=cwt_width_min_cm,
            width_max_cm=cwt_width_max_cm,
            width_step_cm=cwt_width_step_cm,
            step_cm=step_cm,
        )
        cwt_pos = np.asarray(find_peaks_cwt(y_proc, widths_pts), dtype=int)
        if cwt_pos.size:
            scores = np.abs(y_proc[cwt_pos])
            clustered = _cluster_indices_by_tolerance(
                cwt_pos, x=x, scores=scores, tolerance_cm=cwt_cluster_tolerance_cm
            )
            for idx in clustered:
                _add_candidate(
                    idx,
                    1,
                    "cwt",
                    {"widths_pts": widths_pts.tolist(), "cluster_tolerance": cwt_cluster_tolerance_cm},
                )
        if detect_negative:
            cwt_neg = np.asarray(find_peaks_cwt(-y_proc, widths_pts), dtype=int)
            if cwt_neg.size:
                scores = np.abs(y_proc[cwt_neg])
                clustered = _cluster_indices_by_tolerance(
                    cwt_neg, x=x, scores=scores, tolerance_cm=cwt_cluster_tolerance_cm
                )
                for idx in clustered:
                    _add_candidate(
                        idx,
                        -1,
                        "cwt",
                        {"widths_pts": widths_pts.tolist(), "cluster_tolerance": cwt_cluster_tolerance_cm},
                    )

    if not candidates:
        return []

    def _extract_plateau_bounds(detections: List[Dict[str, object]]):
        lefts = []
        rights = []
        for detection in detections:
            if detection.get("source") != "plateau":
                continue
            left = detection.get("plateau_left")
            right = detection.get("plateau_right")
            if left is None or right is None:
                continue
            lefts.append(int(left))
            rights.append(int(right))
        if not lefts or not rights:
            return None
        return min(lefts), max(rights)

    merged: List[Dict[str, object]] = []
    candidates.sort(key=lambda c: c["score"], reverse=True)
    for candidate in candidates:
        idx = int(candidate["index"])
        polarity = int(candidate["polarity"])
        match = None
        for existing in merged:
            if int(existing["polarity"]) != polarity:
                continue
            if abs(float(x[idx]) - float(x[int(existing["index"])])) <= merge_tolerance_cm:
                match = existing
                break
        if match is None:
            merged.append(candidate)
            continue
        match["sources"] = set(match["sources"]) | set(candidate["sources"])
        match["detections"] = list(match["detections"]) + list(candidate["detections"])
        if float(candidate["score"]) > float(match["score"]):
            match["index"] = idx
            match["score"] = float(candidate["score"])

    merged.sort(key=lambda c: c["index"])
    for candidate_id, candidate in enumerate(merged):
        candidate["candidate_id"] = int(candidate_id)
        plateau_bounds = _extract_plateau_bounds(candidate.get("detections", []))
        if plateau_bounds is not None:
            candidate["plateau_bounds"] = plateau_bounds
    return merged


def _cluster_candidates_by_window(
    candidates: List[Dict[str, object]],
    *,
    window: int,
) -> List[List[Dict[str, object]]]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda c: int(c["index"]))
    clusters: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = [sorted_candidates[0]]
    for candidate in sorted_candidates[1:]:
        if int(candidate["index"]) - int(current[-1]["index"]) <= window:
            current.append(candidate)
        else:
            clusters.append(current)
            current = [candidate]
    clusters.append(current)
    return clusters


def _curvature_seed_centers(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    distance_pts: int,
    max_peaks: int,
) -> List[float]:
    curvature = -_compute_second_derivative(xs, ys, distance_pts)
    if curvature.size == 0:
        return []
    prominence = np.nanmedian(np.abs(curvature))
    if not np.isfinite(prominence) or prominence <= 0:
        prominence = np.nanmax(np.abs(curvature)) * 0.1 if curvature.size else 0.0
    prominence = max(prominence, 1e-9)
    idxs, _ = find_peaks(curvature, prominence=prominence, distance=max(1, distance_pts))
    if idxs.size == 0:
        return []
    scores = curvature[idxs]
    order = np.argsort(scores)[::-1]
    idxs = idxs[order][:max_peaks]
    return [float(xs[int(idx)]) for idx in idxs]


def refine_peak_candidates(
    x: np.ndarray,
    y_proc: np.ndarray,
    candidates: List[Dict[str, object]],
    args: object,
    *,
    y_abs: np.ndarray | None = None,
) -> List[Dict[str, object]]:
    if not candidates:
        return []
    model = str(_get_param(args, "model", "Gaussian") or "Gaussian")
    fit_window = max(3, int(_get_param(args, "fit_window_pts", 70)))
    min_r2 = float(_get_param(args, "min_r2", 0.85))
    results: List[Dict[str, object]] = []

    for polarity in (1, -1):
        group = [c for c in candidates if int(c["polarity"]) == polarity]
        if not group:
            continue
        fit_y = -y_proc if polarity < 0 else y_proc
        fit_y_abs = None
        if y_abs is not None and polarity > 0:
            fit_y_abs = np.asarray(y_abs, dtype=float)
        clusters = _cluster_candidates_by_window(group, window=fit_window)
        for cluster in clusters:
            has_plateau = any(c.get("plateau_bounds") is not None for c in cluster)
            cluster_indices = [int(c["index"]) for c in cluster]
            span_start = max(min(cluster_indices) - fit_window, 0)
            span_end = min(max(cluster_indices) + fit_window, len(x))
            xs_cluster = x[span_start:span_end]
            ys_cluster = fit_y[span_start:span_end]
            if len(cluster) > 1 and model.lower() in {"gaussian", "lorentzian", "voigt"} and not has_plateau:
                candidate_centers = [float(x[int(c["index"])]) for c in cluster]
                curvature_centers = _curvature_seed_centers(
                    xs_cluster,
                    ys_cluster,
                    distance_pts=max(1, int(fit_window / max(len(cluster), 1))),
                    max_peaks=max(len(cluster), 2),
                )
                step = _resolve_step_from_x(xs_cluster)
                extra_centers = [
                    c
                    for c in curvature_centers
                    if all(abs(c - base) > step * 0.75 for base in candidate_centers)
                ]
                centers = candidate_centers + extra_centers
                try:
                    fitted = _fit_multi_peak(xs_cluster, ys_cluster, centers, model)
                except Exception:
                    fitted = None
                if fitted is not None:
                    fitted_peaks, _ = fitted
                    for candidate, fit_result in zip(cluster, fitted_peaks[: len(candidate_centers)]):
                        if not fit_result or fit_result.get("r2", 0.0) < min_r2:
                            continue
                        result = dict(fit_result)
                        result["candidate_id"] = int(candidate["candidate_id"])
                        result["index"] = int(candidate["index"])
                        result["polarity"] = int(polarity)
                        result["sources"] = list(candidate.get("sources", []))
                        if polarity < 0:
                            result["amplitude"] = float(result.get("amplitude", 0.0)) * -1
                            result["area"] = float(result.get("area", 0.0)) * -1
                        results.append(result)
                    continue
            for candidate in cluster:
                idx = int(candidate["index"])
                plateau_bounds = candidate.get("plateau_bounds")
                x0_guess = float(x[idx])
                center_bounds = None
                use_absorbance = fit_y_abs is not None and plateau_bounds is not None
                if plateau_bounds is not None:
                    left_idx, right_idx = plateau_bounds
                    left_x = float(x[left_idx])
                    right_x = float(x[right_idx])
                    center_bounds = (min(left_x, right_x), max(left_x, right_x))
                    x0_guess = (center_bounds[0] + center_bounds[1]) / 2.0
                fit_source = fit_y_abs if use_absorbance else fit_y
                fit = fit_peak(
                    x,
                    fit_source,
                    idx,
                    model,
                    fit_window,
                    center_bounds=center_bounds,
                    x0_guess=x0_guess,
                )
                if not fit:
                    if plateau_bounds is None:
                        continue
                    xs, ys = _fit_peak_window(x, fit_source, idx, fit_window)
                    if xs.size < 3:
                        continue
                    local_idx = int(np.argmin(np.abs(xs - float(x[idx]))))
                    baseline = float(np.median(ys))
                    peak_y = float(ys[local_idx])
                    plateau_width = abs(float(x[plateau_bounds[1]]) - float(x[plateau_bounds[0]]))
                    fwhm = max(plateau_width, _resolve_step_from_x(xs) * 2.0)
                    if hasattr(np, "trapezoid"):
                        area = float(np.trapezoid(np.maximum(ys - baseline, 0.0), xs))
                    else:
                        area = float(np.trapz(np.maximum(ys - baseline, 0.0), xs))
                    fit = dict(
                        center=float(x0_guess),
                        fwhm=float(fwhm),
                        amplitude=float(max(peak_y - baseline, 0.0)),
                        area=area,
                        r2=float(min_r2),
                    )
                if plateau_bounds is None and fit.get("r2", 0.0) < min_r2:
                    continue
                result = dict(fit)
                result["candidate_id"] = int(candidate["candidate_id"])
                result["index"] = int(candidate["index"])
                result["polarity"] = int(polarity)
                result["sources"] = list(candidate.get("sources", []))
                if polarity < 0:
                    result["amplitude"] = float(result.get("amplitude", 0.0)) * -1
                    result["area"] = float(result.get("area", 0.0)) * -1
                results.append(result)

    results.sort(key=lambda r: (int(r.get("index", 0)), int(r.get("candidate_id", 0))))
    return results

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

def index_file(path:str,con,args,failed_files=None,step_registry_collector: Optional[List[Dict[str, object]]] = None):
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
    resolution_cm=_parse_numeric(headers.get('RESOLUTION')) if headers else None

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
            step_metadata = {
                "file_id": file_id,
                "spectrum_id": sid,
                "y_units": y_units,
                "processing": {
                    "sg_win": int(args.sg_win),
                    "sg_poly": int(args.sg_poly),
                    "sg_window_cm": float(args.sg_window_cm or 0.0),
                    "als_lam": float(args.als_lam),
                    "als_p": float(args.als_p),
                    "baseline_method": str(args.baseline_method),
                    "baseline_niter": int(args.baseline_niter),
                    "baseline_piecewise": bool(args.baseline_piecewise),
                    "baseline_ranges": args.baseline_ranges,
                },
            }
            step_registry: List[Dict[str, object]] = []
            _append_processing_step(step_registry, "raw", x_clean, y_clean, step_metadata)
            _append_processing_step(
                step_registry,
                "absorbance_converted",
                x_clean,
                y_for_processing,
                step_metadata,
            )
            y_proc, noise_sigma = preprocess_with_noise(
                x_clean,
                y_for_processing,
                args.sg_win,
                args.sg_poly,
                args.als_lam,
                args.als_p,
                sg_window_cm=args.sg_window_cm,
                baseline_method=args.baseline_method,
                baseline_niter=args.baseline_niter,
                baseline_piecewise=args.baseline_piecewise,
                baseline_ranges=args.baseline_ranges,
                step_registry=step_registry,
                step_metadata=step_metadata,
            )
            peak_candidates=detect_peak_candidates(
                x_clean,
                y_proc,
                noise_sigma,
                args,
                resolution_cm=resolution_cm,
            )
            peak_overlay = [
                {
                    "index": int(candidate["index"]),
                    "x": float(x_clean[int(candidate["index"])]),
                    "y": float(y_proc[int(candidate["index"])]),
                    "polarity": int(candidate.get("polarity", 1)),
                    "score": float(candidate.get("score", 0.0)),
                    "sources": sorted(candidate.get("sources", [])),
                }
                for candidate in peak_candidates
                if 0 <= int(candidate["index"]) < len(x_clean)
            ]
            _append_processing_step(
                step_registry,
                "candidate_peaks_overlay",
                x_clean,
                y_proc,
                step_metadata,
                extra_metadata={"candidates": peak_overlay},
            )
            refined = refine_peak_candidates(
                x_clean,
                y_proc,
                peak_candidates,
                args,
                y_abs=y_for_processing,
            )
            pid = 0
            for fit in refined:
                con.execute(
                    'INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?,?,?,?,?)',
                    [
                        file_id,
                        sid,
                        int(fit.get("candidate_id", pid)),
                        int(fit.get("polarity", 1)),
                        fit["center"],
                        fit["fwhm"],
                        fit["amplitude"],
                        fit["area"],
                        fit["r2"],
                    ],
                )
                pid += 1
            total_peaks += pid
            if step_registry_collector is not None:
                step_registry_collector.append(
                    {
                        "file_id": file_id,
                        "spectrum_id": sid,
                        "steps": step_registry,
                    }
                )
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
    ap.add_argument('--prominence',type=float,default=0.01)
    ap.add_argument('--min-distance',type=float,default=3.0,dest='min_distance')
    ap.add_argument('--noise-sigma-multiplier',type=float,default=3.0,dest='noise_sigma_multiplier')
    ap.add_argument('--peak-width-min',type=float,default=0.0,dest='peak_width_min')
    ap.add_argument('--peak-width-max',type=float,default=0.0,dest='peak_width_max')
    ap.add_argument('--cwt-enabled',action='store_true',dest='cwt_enabled')
    ap.add_argument('--cwt-width-min',type=float,default=0.0,dest='cwt_width_min')
    ap.add_argument('--cwt-width-max',type=float,default=0.0,dest='cwt_width_max')
    ap.add_argument('--cwt-width-step',type=float,default=0.0,dest='cwt_width_step')
    ap.add_argument('--cwt-cluster-tolerance',type=float,default=8.0,dest='cwt_cluster_tolerance')
    ap.add_argument('--plateau-min-points',type=int,default=3,dest='plateau_min_points')
    ap.add_argument('--plateau-prominence-factor',type=float,default=1.0,dest='plateau_prominence_factor')
    ap.add_argument('--merge-tolerance',type=float,default=8.0,dest='merge_tolerance')
    ap.add_argument('--sg-win',type=int,default=7,dest='sg_win');ap.add_argument('--sg-poly',type=int,default=3,dest='sg_poly')
    ap.add_argument('--sg-window-cm',type=float,default=0.0,dest='sg_window_cm')
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
    ap.add_argument(
        '--model',
        choices=['Gaussian', 'Lorentzian', 'Voigt', 'Spline'],
        default='Gaussian',
    )
    ap.add_argument('--min-r2',type=float,default=0.85,dest='min_r2')
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
    step_registry_collector: List[Dict[str, object]] = []
    for idx,p in enumerate(files,1):
        print(f'[{idx}/{total_files}] Indexing {p}', file=sys.stdout, flush=True)
        try:
            n_s,n_p=index_file(p,con,args,step_registry_collector=step_registry_collector)
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
