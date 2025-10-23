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
import os, re, json, math, argparse, hashlib, glob
from typing import List, Tuple, Dict, Optional
import numpy as np, pandas as pd, duckdb
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.cluster import DBSCAN

PROMOTED_KEYS = {
    'TITLE':'title','DATA TYPE':'data_type','JCAMP-DX':'jcamp_ver','NPOINTS':'npoints_hdr',
    'XUNITS':'x_units_raw','YUNITS':'y_units_raw','XFACTOR':'x_factor','YFACTOR':'y_factor',
    'DELTAX':'deltax_hdr','FIRSTX':'firstx','LASTX':'lastx','FIRSTY':'firsty',
    'MAXX':'maxx','MINX':'minx','MAXY':'maxy','MINY':'miny','RESOLUTION':'resolution',
    'STATE':'state','CLASS':'class','ORIGIN':'origin','OWNER':'owner','DATE':'date',
    'NAMES':'names','CAS REGISTRY NO':'cas','MOLFORM':'molform','$NIST SOURCE':'nist_source'
}
NUMERIC_KEYS={'NPOINTS','XFACTOR','YFACTOR','DELTAX','FIRSTX','LASTX','FIRSTY','MAXX','MINX','MAXY','MINY','RESOLUTION'}
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
    maxlen=max(len(r) for r in lines)
    mat=np.full((len(lines),maxlen),np.nan)
    for i,r in enumerate(lines):
        mat[i,:len(r)]=r
    x=mat[:,0]*xfactor
    Y=mat[:,1:]*yfactor
    spectra=[Y[:,i] for i in range(Y.shape[1]) if not np.all(np.isnan(Y[:,i]))]
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

def als_baseline(y:np.ndarray,lam=1e5,p=0.01,niter=10)->np.ndarray:
    L=len(y);D=sparse.diags([1,-2,1],[0,-1,-2],shape=(L,L-2));w=np.ones(L)
    for _ in range(niter):
        W=sparse.spdiags(w,0,L,L);Z=W+lam*(D@D.T);z=spsolve(Z,w*y)
        w=p*(y>z)+(1-p)*(y<z)
    return z

def preprocess(y:np.ndarray,sg_win:int,sg_poly:int,als_lam:float,als_p:float)->np.ndarray:
    y2=np.asarray(y,dtype=float).copy()
    n=len(y2)
    if sg_win and sg_win%2==1 and sg_win>2 and n>=3:
        win=min(sg_win,n if n%2==1 else n-1)
        if win>=3:
            poly=min(max(sg_poly,0),win-1)
            if win>poly:
                y2=savgol_filter(y2,win,poly)
    if als_lam>0:
        if n>=3:
            y2=y2-als_baseline(y2,lam=als_lam,p=als_p)
        elif n:
            y2=y2-np.mean(y2)
    m=np.max(np.abs(y2))
    if m>0: y2/=m
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
        area=float(np.trapz(yfit-C,xs))
        return dict(center=x0,fwhm=fwhm,amplitude=A,area=area,r2=r2)
    except: return None

def init_db(outdir:str):
    os.makedirs(outdir,exist_ok=True)
    con=duckdb.connect(os.path.join(outdir,'peaks.duckdb'))
    con.execute('''CREATE TABLE IF NOT EXISTS spectra(file_id TEXT PRIMARY KEY,path TEXT,n_points INT,n_spectra INT,meta_json TEXT);''')
    con.execute('''CREATE TABLE IF NOT EXISTS peaks(file_id TEXT,spectrum_id INT,peak_id INT,center DOUBLE,fwhm DOUBLE,amplitude DOUBLE,area DOUBLE,r2 DOUBLE,PRIMARY KEY(file_id,spectrum_id,peak_id));''')
    return con

def store_headers(con,file_id:str,headers:Dict[str,str]):
    meta_json=json.dumps(headers,ensure_ascii=False)
    con.execute(
        'INSERT OR REPLACE INTO spectra (file_id,path,n_points,n_spectra,meta_json) VALUES (?,?,?,?,?)',
        [file_id,'',0,0,meta_json]
    )

def index_file(path:str,con,args):
    headers=parse_jdx_headers(path)
    try:
        x,Y,headers=parse_jcamp_multispec(path)
    except: return 0,0
    file_id=file_sha1(path)
    store_headers(con,file_id,headers)
    total_peaks=0
    y_units=headers.get('YUNITS','')
    for sid,y in enumerate(Y):
        y_for_processing=convert_y_for_processing(y,y_units)
        y_proc=preprocess(y_for_processing,args.sg_win,args.sg_poly,args.als_lam,args.als_p)
        diffs=np.diff(x)
        if diffs.size:
            step=np.nanmedian(np.abs(diffs))
        else:
            step=np.nan
        if not np.isfinite(step) or step<=0:
            span=np.nanmax(x)-np.nanmin(x) if x.size else np.nan
            if np.isfinite(span) and span>0 and len(x)>1:
                step=span/(len(x)-1)
            else:
                step=float(args.min_distance) if args.min_distance>0 else 1.0
        dist_pts=max(1,int(np.ceil(float(args.min_distance)/max(step,1e-9))))
        idxs,_=find_peaks(y_proc,prominence=args.prominence,distance=dist_pts)
        pid=0
        for i in idxs:
            fit=fit_peak(x,y_proc,i,args.model,args.fit_window_pts)
            if not fit or fit['r2']<args.min_r2: continue
            con.execute('INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?,?,?,?)',[file_id,sid,pid,fit['center'],fit['fwhm'],fit['amplitude'],fit['area'],fit['r2']])
            pid+=1
        total_peaks+=pid
    con.execute('UPDATE spectra SET path=?,n_points=?,n_spectra=? WHERE file_id=?',[path,len(x),len(Y),file_id])
    return len(Y),total_peaks

def weighted_median(x,w):
    o=np.argsort(x);x=x[o];w=w[o];c=np.cumsum(w)/(np.sum(w)+1e-12)
    return x[min(np.searchsorted(c,0.5),len(x)-1)]

def build_file_consensus(con,args):
    df=con.execute('SELECT file_id,center,fwhm,area,r2 FROM peaks').fetch_df()
    if df.empty: return pd.DataFrame()
    out=[]
    for fid,g in df.groupby('file_id'):
        c=g['center'].to_numpy();f=g['fwhm'].to_numpy();w=(g['area']*g['r2']).to_numpy()
        eps=max(args.file_eps_factor*np.nanmedian(f),args.file_eps_min)
        labels=DBSCAN(eps=eps,min_samples=args.file_min_samples).fit_predict(c.reshape(-1,1))
        for cl in set(labels):
            if cl==-1: continue
            m=labels==cl;cx=c[m];cf=f[m];cw=w[m]
            out.append(dict(file_id=fid,center=weighted_median(cx,cw),fwhm=np.nanmedian(cf),support=np.sum(m)))
    return pd.DataFrame(out)

def build_global_consensus(con,fc,args):
    if fc.empty: return pd.DataFrame()
    c=fc['center'].to_numpy();labels=DBSCAN(eps=args.global_eps_abs,min_samples=args.global_min_samples).fit_predict(c.reshape(-1,1))
    out=[]
    for cl in set(labels):
        if cl==-1: continue
        m=labels==cl;cx=c[m];out.append(dict(center=np.median(cx),support=np.sum(m)))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('data_dir');ap.add_argument('index_dir')
    ap.add_argument('--prominence',type=float,default=0.02);ap.add_argument('--min-distance',type=float,default=5.0,dest='min_distance')
    ap.add_argument('--sg-win',type=int,default=9,dest='sg_win');ap.add_argument('--sg-poly',type=int,default=3,dest='sg_poly')
    ap.add_argument('--als-lam',type=float,default=1e5,dest='als_lam');ap.add_argument('--als-p',type=float,default=0.01,dest='als_p')
    ap.add_argument('--model',choices=['Gaussian'],default='Gaussian');ap.add_argument('--min-r2',type=float,default=0.9,dest='min_r2')
    ap.add_argument('--fit-window-pts',type=int,default=50,dest='fit_window_pts')
    ap.add_argument('--file-min-samples',type=int,default=2);ap.add_argument('--file-eps-factor',type=float,default=0.5);ap.add_argument('--file-eps-min',type=float,default=2.0)
    ap.add_argument('--global-min-samples',type=int,default=2);ap.add_argument('--global-eps-abs',type=float,default=4.0)
    args=ap.parse_args()
    con=init_db(args.index_dir)
    files=[p for p in glob.glob(os.path.join(args.data_dir,'**','*'),recursive=True) if os.path.isfile(p) and re.search(r'\.(jdx|dx)$',p,re.I)]
    total_specs=0;total_peaks=0
    for p in files:
        n_s,n_p=index_file(p,con,args);total_specs+=n_s;total_peaks+=n_p
    fc=build_file_consensus(con,args);gc=build_global_consensus(con,fc,args)
    print(f'Indexed spectra: {total_specs} | Peaks: {total_peaks} | File clusters: {len(fc)} | Global: {len(gc)}')
if __name__=='__main__': main()
