#!/usr/bin/env python3
"""
ftir_pdf_digitize.py

Single-file CLI tool to digitize FTIR spectra from a PDF into an Excel workbook.

Core features (per requirements):
- Digitized spectrum curves are mandatory.
- Axis breaks are detected (tick discontinuities) and missing regions are imputed by default (C1 Hermite bridge).
- Transmittance is converted to absorbance by default (still stores transmittance too).
- Metadata per spectrum entry (plus source title/author/doi/isbn) stored in XLSX.
- Optional: save plot crop images and overlays to disk.
- Informative CLI logging + progress; continue on failures.

Notes:
- Many library figures have only an X-axis and vertically-offset spectra (no Y scale).
  In those cases the script outputs a normalized "transmittance_rel" derived from pixel y,
  and the absorbance is computed from that normalized transmittance. This is flagged in QC.

Dependencies:
  pip install pymupdf opencv-python scikit-image pandas openpyxl tqdm pytesseract pillow matplotlib
System dependency:
  tesseract-ocr must be installed and on PATH for OCR.

Example:
  python ftir_pdf_digitize.py --pdf input.pdf --out out.xlsx --pages 12-20 --dpi 600 --save-images
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import fitz  # PyMuPDF
import cv2
from skimage.morphology import skeletonize
from tqdm import tqdm

from PIL import Image

import pytesseract

import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCEL_MAX_ROWS = 1_048_576  # includes header row

# Excel / openpyxl rejects certain control characters in cell strings.
# This regex matches ASCII control chars except TAB(\x09), LF(\x0A), CR(\x0D).
_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def _sanitize_excel_str(s: str) -> str:
    if s is None:
        return ""
    s2 = _ILLEGAL_XLSX_RE.sub(" ", str(s))
    return re.sub(r"[ \t]+", " ", s2).strip()

def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_object_dtype(df2[col]) or pd.api.types.is_string_dtype(df2[col]):
            df2[col] = df2[col].map(lambda x: _sanitize_excel_str(x) if isinstance(x, str) else x)
    return df2

def _build_xydata_payload(
    x: np.ndarray,
    y: np.ndarray,
    *,
    points_per_line: int = 1,
) -> str:
    """Return a JCAMP-style XYDATA payload using (X,Y) or X++(Y..Y) lines."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(len(x_arr), len(y_arr))
    if n == 0:
        return ""
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        return ""
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0:
        return ""

    step = max(1, int(points_per_line))

    def _fmt(value: float) -> str:
        return f"{value:.10g}"

    lines: List[str] = []
    for idx in range(0, len(x_arr), step):
        start_x = x_arr[idx]
        y_slice = y_arr[idx : idx + step]
        parts = [_fmt(start_x)]
        parts.extend(_fmt(val) for val in y_slice)
        lines.append(" ".join(parts))
    return "\n".join(lines)
DEFAULT_DPI = 600

QUALIFIER_MEANINGS = {
    "s": "strong band",
    "w": "weak band",
    "sh": "shoulder",
    None: "",
    "": "",
}

NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


# Mineral formula heuristic: matches typical chemical formula strings with element symbols and bracketed groups.
FORMULA_RE = re.compile(r"(?:(?:\b[A-Z][a-z]?\d*(?:\([^\)]+\)\d*)?){2,}|\b[A-Z][a-z]?\d*(?:\[[^\]]+\]\d*)+)(?:[·\.\u00b7]\s*[A-Z][a-z]?\d*(?:\[[^\]]+\])?\d*)*")

def sanitize_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-\.\s]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "spectrum"
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ISBN_RE = re.compile(r"\b(?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d\b")

def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("ftir_pdf_digitize")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def parse_pages_spec(spec: Optional[str], n_pages: int) -> List[int]:
    if not spec:
        return list(range(n_pages))
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a); b = int(b)
            if a < 0 or b < 0:
                raise ValueError("pages must be >= 0 (use 1-based in CLI; see --pages-help)")
            # user likely provides 1-based; accept both by detecting 0
            # if user uses 1-based, allow "1-3" => pages 0..2
            if a >= 1 and b >= 1:
                rng = range(a-1, b)
            else:
                rng = range(a, b+1)
            pages.extend(list(rng))
        else:
            p = int(part)
            pages.append(p-1 if p >= 1 else p)
    pages = [p for p in pages if 0 <= p < n_pages]
    return sorted(set(pages))

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGB" if pix.n < 4 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _save_digitized_plot_png_impl(df_curve: pd.DataFrame, out_png: Path) -> None:
    """Save a PNG plot of transmittance vs wavenumber from the digitized curve.
    Imputed points are plotted as a separate dashed line.
    """
    if df_curve is None or df_curve.empty:
        return
    fig = plt.figure(figsize=(7.2, 3.6), dpi=150)
    ax = fig.add_subplot(111)
    d = df_curve.sort_values('wavenumber_cm1')
    if 'imputed' in d.columns:
        m = d['imputed'].astype(bool).to_numpy()
    else:
        m = np.zeros(len(d), dtype=bool)
    x = d['wavenumber_cm1'].to_numpy(dtype=float)
    y = d['transmittance'].to_numpy(dtype=float)
    if (~m).any():
        ax.plot(x[~m], y[~m], linewidth=1.0)
    if m.any():
        ax.plot(x[m], y[m], linewidth=1.0, linestyle='--')
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Transmittance')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

def _save_digitized_plot_png(df_curve: pd.DataFrame, out_png: Path) -> None:
    """Public wrapper used by worker."""
    return _save_digitized_plot_png_impl(df_curve, out_png)

def _save_digitized_plot_png(df_curve: pd.DataFrame, out_png: Path) -> None:
    """Backward-compatible alias."""
    return _save_digitized_plot_png_impl(df_curve, out_png)

@dataclass
class OCRToken:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    block: int
    line: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

def ocr_tokens(img_bgr: np.ndarray, psm: int = 6, whitelist: Optional[str] = None) -> List[OCRToken]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Slight upscale helps OCR on small tick labels
    scale = 2
    gray = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    # binarize
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT, config=cfg)
    out: List[OCRToken] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        # scale back coordinates
        x = int(data["left"][i] / scale)
        y = int(data["top"][i] / scale)
        w = int(data["width"][i] / scale)
        h = int(data["height"][i] / scale)
        out.append(OCRToken(
            text=txt,
            conf=conf,
            x=x, y=y, w=w, h=h,
            block=int(data.get("block_num", [0]*n)[i]),
            line=int(data.get("line_num", [0]*n)[i]),
        ))
    return out

@dataclass
class AxisModel:
    # value = m * pixel + b
    m: float
    b: float

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.m * p + self.b

def fit_linear(pixels: Sequence[float], values: Sequence[float]) -> Optional[AxisModel]:
    if len(pixels) < 2:
        return None
    x = np.asarray(pixels, dtype=float)
    y = np.asarray(values, dtype=float)
    # robust-ish: ordinary least squares
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return AxisModel(float(m), float(b))
    except Exception:
        return None

@dataclass
class BreakInfo:
    present: bool
    x_split_pix: Optional[float]
    gap_lo: Optional[float]
    gap_hi: Optional[float]
    deltas: List[float]
    score: float
    marker_present: bool = False
    marker_x_pix: Optional[float] = None

def detect_axis_break(
    x_ticks: List[Tuple[float, float]],
    marker_x_pix: Optional[float] = None,
) -> BreakInfo:
    """
    x_ticks: list of (x_center_pix, tick_value) sorted by x.
    Break detection uses outlier delta in tick_value sequence, plus optional
    break markers (e.g., //) in the x-axis band.
    """
    marker_present = marker_x_pix is not None
    if len(x_ticks) < 2:
        return BreakInfo(False, None, None, None, [], 0.0, marker_present, marker_x_pix)
    xs = np.array([t[0] for t in x_ticks], dtype=float)
    vs = np.array([t[1] for t in x_ticks], dtype=float)

    marker_gap_lo = None
    marker_gap_hi = None
    marker_split = None
    if marker_present:
        left_idx = np.where(xs < marker_x_pix)[0]
        right_idx = np.where(xs > marker_x_pix)[0]
        if left_idx.size == 0 or right_idx.size == 0:
            eq_idx = np.where(xs == marker_x_pix)[0]
            if eq_idx.size > 0:
                idx = int(eq_idx[0])
                if left_idx.size == 0 and idx > 0:
                    left_idx = np.array([idx - 1])
                if right_idx.size == 0 and idx + 1 < len(xs):
                    right_idx = np.array([idx + 1])
        if left_idx.size > 0 and right_idx.size > 0:
            li = int(left_idx[-1])
            ri = int(right_idx[0])
            if li != ri:
                marker_gap_lo = float(min(vs[li], vs[ri]))
                marker_gap_hi = float(max(vs[li], vs[ri]))
                marker_split = float(marker_x_pix)

    if len(x_ticks) < 4:
        if marker_gap_lo is not None and marker_gap_hi is not None:
            return BreakInfo(True, marker_split, marker_gap_lo, marker_gap_hi, [], 0.0, True, marker_x_pix)
        return BreakInfo(False, None, None, None, [], 0.0, marker_present, marker_x_pix)

    deltas = np.diff(vs)
    ad = np.abs(deltas)
    med = float(np.median(ad)) if len(ad) else 0.0
    if med <= 0:
        if marker_gap_lo is not None and marker_gap_hi is not None:
            return BreakInfo(True, marker_split, marker_gap_lo, marker_gap_hi, deltas.tolist(), 0.0, True, marker_x_pix)
        return BreakInfo(False, None, None, None, deltas.tolist(), 0.0, marker_present, marker_x_pix)
    # candidate break: largest delta that's much larger than typical
    idx = int(np.argmax(ad))
    score = float(ad[idx] / med)
    # also require absolute delta to be meaningfully big
    if score >= 2.3 and ad[idx] >= max(300.0, 2.0 * med):
        x_split = float((xs[idx] + xs[idx+1]) / 2.0)
        gap_lo = float(min(vs[idx], vs[idx+1]))
        gap_hi = float(max(vs[idx], vs[idx+1]))
        if marker_split is not None and (xs.min() <= marker_split <= xs.max()):
            x_split = marker_split
        return BreakInfo(True, x_split, gap_lo, gap_hi, deltas.tolist(), score, marker_present, marker_x_pix)
    if marker_gap_lo is not None and marker_gap_hi is not None:
        return BreakInfo(True, marker_split, marker_gap_lo, marker_gap_hi, deltas.tolist(), score, True, marker_x_pix)
    return BreakInfo(False, None, None, None, deltas.tolist(), score, marker_present, marker_x_pix)

def estimate_local_slope(x: np.ndarray, y: np.ndarray, tail: bool, n: int = 25) -> float:
    if len(x) < 2:
        return 0.0
    if n > len(x):
        n = len(x)
    if tail:
        xx = x[-n:]
        yy = y[-n:]
    else:
        xx = x[:n]
        yy = y[:n]
    xx0 = xx - xx.mean()
    denom = float(np.dot(xx0, xx0))
    if denom == 0.0:
        return 0.0
    return float(np.dot(xx0, yy - yy.mean()) / denom)

def hermite_bridge(x0: float, x1: float, y0: float, y1: float, m0: float, m1: float, x_new: np.ndarray) -> np.ndarray:
    L = x1 - x0
    if L == 0:
        return np.full_like(x_new, y0, dtype=float)
    t = (x_new - x0) / L
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*y0 + h10*(L*m0) + h01*y1 + h11*(L*m1)

def transmittance_to_absorbance(T: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # If looks like percent, convert
    T = np.asarray(T, dtype=float)
    if np.nanmax(T) > 1.5:
        T = T / 100.0
    T = np.clip(T, eps, 1.0)
    return -np.log10(T)

@dataclass
class PlotAxes:
    # plot interior bbox in pixels (x0, y0, x1, y1)
    x0: int
    y0: int
    x1: int
    y1: int
    x_axis_y: int
    y_axis_x: Optional[int]
    ok: bool = False

def detect_plot_axes(img_bgr: np.ndarray) -> PlotAxes:
    """
    Robust axis detection on plot crops that may include caption/paragraph text below the plot.

    Strategy:
      1) Edge + HoughLinesP to find long horizontal/vertical lines.
      2) Morphological line extraction as fallback and as a sanity check.
      3) Choose x-axis as the lowest strong horizontal line *above* caption text when present.
         Heuristic: if many horizontals exist, ignore the bottom 20% of the image (often paragraph).
      4) y-axis chosen as the leftmost strong vertical line in the plot region.

    Returns PlotAxes with ok=True only if a strong horizontal axis line was found.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binary for morphology (invert: ink = 1)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # --- Hough candidates ---
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=110,
                            minLineLength=int(0.55 * w), maxLineGap=10)

    horiz: List[Tuple[int, int, int, int, int]] = []
    vert: List[Tuple[int, int, int, int, int]] = []
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = int(math.hypot(dx, dy))
            if dy <= 5 and dx >= int(0.45 * w):
                horiz.append((x1, y1, x2, y2, length))
            if dx <= 5 and dy >= int(0.35 * h):
                vert.append((x1, y1, x2, y2, length))

    # --- Morphological line extraction ---
    # Horizontal lines: long run kernel
    k_h = max(40, w // 6)
    k_v = max(40, h // 6)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))

    horiz_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    # Row "strength" for horizontal lines
    row_strength = horiz_lines.mean(axis=1)  # 0..255
    col_strength = vert_lines.mean(axis=0)

    # Candidate y positions for horizontal axes (rows with strong line presence)
    strong_rows = np.where(row_strength > (0.06 * 255))[0]  # tuned
    strong_cols = np.where(col_strength > (0.06 * 255))[0]

    # Decide if caption text is present: bottom 20% has lots of ink but lacks long horizontals
    # We avoid selecting axes from bottom 20% unless no other option exists.
    ignore_bottom = int(0.80 * h)

    def choose_x_axis_y() -> Tuple[int, bool]:
        # Prefer Hough horizontals; pick the lowest long line above ignore_bottom if possible.
        if horiz:
            cand = sorted(horiz, key=lambda t: (t[4], t[1]), reverse=True)
            above = [t for t in cand if t[1] < ignore_bottom]
            chosen = above[0] if above else cand[0]
            return int(np.median([chosen[1], chosen[3]])), True

        # Morphology fallback: pick max-strength row near bottom but above ignore_bottom if possible
        if len(strong_rows) > 0:
            rows_above = strong_rows[strong_rows < ignore_bottom]
            if len(rows_above) > 0:
                return int(rows_above.max()), True
            return int(strong_rows.max()), True

        # Last resort: use projection in a restricted band (avoid bottom caption)
        band_start = int(0.35 * h)
        band_end = int(0.80 * h)
        proj = (thr[band_start:band_end, :] > 0).mean(axis=1)
        if proj.size > 0:
            return int(band_start + int(np.argmax(proj))), False
        return int(0.70 * h), False

    def choose_y_axis_x() -> Optional[int]:
        # Prefer Hough verticals: leftmost long line
        if vert:
            best = sorted(vert, key=lambda t: (min(t[0], t[2]), -t[4]))[0]
            return int(np.median([best[0], best[2]]))
        # Morphology: use strong columns; pick leftmost in left half
        if len(strong_cols) > 0:
            cols = strong_cols[strong_cols < int(0.55 * w)]
            if len(cols) > 0:
                return int(cols.min())
            return int(strong_cols.min())
        return None

    x_axis_y, ok = choose_x_axis_y()
    y_axis_x = choose_y_axis_x()

    # Interior bbox: margins relative to detected axes
    margin = int(max(10, 0.01 * w))
    x0 = (y_axis_x + margin) if y_axis_x is not None else margin
    x1 = w - margin
    y1 = max(margin, x_axis_y - margin)
    y0 = margin

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    return PlotAxes(x0=x0, y0=y0, x1=x1, y1=y1, x_axis_y=x_axis_y, y_axis_x=y_axis_x, ok=ok)

def crop_region(img_bgr: np.ndarray, rect: Tuple[int,int,int,int]) -> np.ndarray:
    x0,y0,x1,y1 = rect
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img_bgr.shape[1], x1); y1 = min(img_bgr.shape[0], y1)
    return img_bgr[y0:y1, x0:x1].copy()

def preprocess_for_curve(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binary image (255=curve-like ink) for connected-component extraction.

    Key steps:
    - Otsu binarization (dark ink on white background)
    - border-line suppression (axes/borders/ticks) restricted to plot margins
    - small-noise cleanup
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.medianBlur(thr, 3)

    h, w = thr.shape[:2]
    if h < 10 or w < 10:
        return thr

    # Suppress long horizontal/vertical border lines near plot margins (axes/borders).
    # This reduces failure modes where an axis line is mistaken for a spectrum component.
    margin_y = max(6, int(0.10 * h))
    margin_x = max(6, int(0.10 * w))

    # Detect long horizontal segments
    hk = max(25, int(0.20 * w))
    vk = max(25, int(0.20 * h))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    lines = cv2.bitwise_or(horiz, vert)

    border_mask = np.zeros_like(thr)
    border_mask[:margin_y, :] = 255
    border_mask[h - margin_y:, :] = 255
    border_mask[:, :margin_x] = 255
    border_mask[:, w - margin_x:] = 255
    lines_border = cv2.bitwise_and(lines, border_mask)
    thr = cv2.bitwise_and(thr, cv2.bitwise_not(lines_border))

    # Final cleanup: remove tiny speckles
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return thr

def component_is_axis_like(comp: "CurveComponent", band_shape: Tuple[int, int]) -> bool:
    """Heuristic to reject components that look like axes/borders instead of a spectrum."""
    h, w = band_shape
    if comp.pixels.size == 0:
        return True
    ys = comp.pixels[:, 0]
    xs = comp.pixels[:, 1]
    x_span = float(xs.max() - xs.min()) if xs.size else 0.0
    y_span = float(ys.max() - ys.min()) if ys.size else 0.0

    # Very flat in y but long in x => likely axis/border line
    if x_span > 0.65 * w and y_span < max(6.0, 0.03 * h):
        return True
    # Very flat in x but long in y => likely y-axis/border line
    if y_span > 0.65 * h and x_span < max(6.0, 0.03 * w):
        return True
    # Tiny fragments
    if x_span < 0.12 * w and y_span < 0.12 * h:
        return True
    return False

def despike_vertical_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious near-vertical artifact runs (often seen as 'infill' noise in plots).

    This targets runs of extremely steep local slopes. It is conservative: it only removes
    points when steep slopes persist for several consecutive steps.
    """
    if df is None or df.empty:
        return df
    if "wavenumber_cm1" not in df.columns or "transmittance" not in df.columns:
        return df
    x = df["wavenumber_cm1"].to_numpy(dtype=float)
    y = df["transmittance"].to_numpy(dtype=float)
    if len(x) < 10:
        return df

    dx = np.diff(x)
    dy = np.diff(y)
    good = dx > 0
    if not np.any(good):
        return df
    slope = np.zeros_like(dy)
    slope[good] = np.abs(dy[good] / dx[good])

    thr = 0.15  # transmittance per cm-1; tuned to catch axis-artifacts rather than genuine peaks
    bad = slope > thr

    # Find consecutive runs of bad slopes
    bad_idx = np.where(bad)[0]
    if bad_idx.size == 0:
        return df

    drop = np.zeros(len(x), dtype=bool)
    run_start = bad_idx[0]
    prev = bad_idx[0]
    for idx in bad_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        # close run
        if (prev - run_start + 1) >= 5:
            # mark points that participate in this run (edges included)
            drop[run_start:prev + 2] = True
        run_start = idx
        prev = idx
    # last run
    if (prev - run_start + 1) >= 5:
        drop[run_start:prev + 2] = True

    if drop.mean() > 0.25:
        # Something went wrong; avoid destroying the curve.
        return df

    if np.any(drop):
        out = df.loc[~drop].copy()
        out = out.sort_values("wavenumber_cm1").reset_index(drop=True)
        return out
    return df

def _axis_filter_mask(
    df: pd.DataFrame,
    *,
    axes: PlotAxes,
    interior_shape: Tuple[int, int],
    border_px: int = 3,
    axis_px: int = 3,
) -> np.ndarray:
    if df is None or df.empty:
        return np.zeros(0, dtype=bool)
    if "x_pix" not in df.columns or "y_pix" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    if border_px <= 0 and axis_px <= 0:
        return np.zeros(len(df), dtype=bool)

    w_int, h_int = interior_shape[1], interior_shape[0]
    x_pix = df["x_pix"].to_numpy(dtype=float)
    y_pix = df["y_pix"].to_numpy(dtype=float)
    full_x = x_pix + float(axes.x0)
    full_y = y_pix + float(axes.y0)

    mask = np.zeros(len(df), dtype=bool)
    if border_px > 0:
        mask |= (x_pix <= border_px) | (x_pix >= (w_int - border_px))
        mask |= (y_pix <= border_px) | (y_pix >= (h_int - border_px))
    if axis_px > 0:
        mask |= np.abs(full_y - float(axes.x_axis_y)) <= axis_px
        if axes.y_axis_x is not None:
            mask |= np.abs(full_x - float(axes.y_axis_x)) <= axis_px
    return mask

def filter_points_near_axes(
    df: pd.DataFrame,
    *,
    axes: PlotAxes,
    interior_shape: Tuple[int, int],
    border_px: int = 3,
    axis_px: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter points that lie near the plot border or axis lines."""
    if df is None or df.empty:
        return df, {"axis_filter_applied": False, "axis_filter_removed": 0}
    mask = _axis_filter_mask(
        df,
        axes=axes,
        interior_shape=interior_shape,
        border_px=border_px,
        axis_px=axis_px,
    )
    if mask.size == 0 or not np.any(mask):
        return df, {"axis_filter_applied": False, "axis_filter_removed": 0}
    filtered = df.loc[~mask].copy()
    filtered = filtered.sort_values("wavenumber_cm1").reset_index(drop=True)
    return filtered, {"axis_filter_applied": True, "axis_filter_removed": int(mask.sum())}

def collapse_duplicate_wavenumbers(
    df: pd.DataFrame,
    *,
    min_bin_width: float = 0.2,
    representative: str = "median",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin nearby x positions and take robust summaries to stabilize dense traces."""
    if df is None or df.empty:
        return df, {
            "collapsed_points": False,
            "collapse_bins": 0,
            "collapse_bin_width": None,
            "collapse_removed": 0,
        }
    if "wavenumber_cm1" not in df.columns or "transmittance" not in df.columns:
        return df, {
            "collapsed_points": False,
            "collapse_bins": 0,
            "collapse_bin_width": None,
            "collapse_removed": 0,
        }

    df_sorted = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    x = df_sorted["wavenumber_cm1"].to_numpy(dtype=float)
    if len(x) < 2:
        return df_sorted, {
            "collapsed_points": False,
            "collapse_bins": len(df_sorted),
            "collapse_bin_width": None,
            "collapse_removed": 0,
        }

    diffs = np.diff(x)
    pos = diffs[diffs > 0]
    med_dx = float(np.median(pos)) if pos.size else 0.0
    bin_width = max(float(min_bin_width), 0.5 * med_dx) if med_dx > 0 else float(min_bin_width)
    if bin_width <= 0:
        return df_sorted, {
            "collapsed_points": False,
            "collapse_bins": len(df_sorted),
            "collapse_bin_width": None,
            "collapse_removed": 0,
        }

    x0 = float(x[0])
    bin_id = np.floor((x - x0) / bin_width).astype(int)

    representative = (representative or "median").lower()
    use_top = representative in {"top", "top-most", "topmost", "min_y"}
    agg_map: Dict[str, Any] = {
        "wavenumber_cm1": "median",
        "transmittance": "median",
    }

    def _mode_int(series: pd.Series) -> int:
        mode_vals = series.mode()
        if not mode_vals.empty:
            return int(mode_vals.iloc[0])
        return int(series.iloc[0])

    for col in ("segment_id", "component_index"):
        if col in df_sorted.columns:
            agg_map[col] = _mode_int
    if "imputed" in df_sorted.columns:
        agg_map["imputed"] = "max"
    if "x_pix" in df_sorted.columns:
        agg_map["x_pix"] = "median"
    if "y_pix" in df_sorted.columns:
        agg_map["y_pix"] = "median"

    grouped = df_sorted.assign(_bin_id=bin_id).groupby("_bin_id", sort=True, as_index=False)
    collapsed = grouped.agg(agg_map)
    if use_top and "y_pix" in df_sorted.columns:
        idx = grouped["y_pix"].idxmin()
        top_rows = df_sorted.loc[idx, ["_bin_id", "transmittance"]].rename(columns={"transmittance": "_top_T"})
        collapsed = collapsed.merge(top_rows, on="_bin_id", how="left")
        collapsed["transmittance"] = collapsed["_top_T"].fillna(collapsed["transmittance"])
        collapsed = collapsed.drop(columns=["_top_T"])
    collapsed = collapsed.drop(columns=["_bin_id"]).sort_values("wavenumber_cm1").reset_index(drop=True)

    collapse_removed = int(len(df_sorted) - len(collapsed))
    info = {
        "collapsed_points": len(collapsed) < len(df_sorted),
        "collapse_bins": int(len(collapsed)),
        "collapse_bin_width": float(bin_width),
        "collapse_removed": collapse_removed,
    }
    return collapsed, info

def apply_rolling_median_if_oscillatory(
    df: pd.DataFrame,
    *,
    window: int = 5,
    osc_ratio: float = 4.0,
    osc_min_amplitude: float = 0.08,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply a rolling median filter if large oscillations are detected."""
    if df is None or df.empty:
        return df, {"rolling_median_applied": False}
    if "transmittance" not in df.columns:
        return df, {"rolling_median_applied": False}
    y = df["transmittance"].to_numpy(dtype=float)
    if len(y) < max(3, window):
        return df, {"rolling_median_applied": False}
    diffs = np.abs(np.diff(y))
    if diffs.size == 0:
        return df, {"rolling_median_applied": False}
    med = float(np.median(diffs))
    p90 = float(np.percentile(diffs, 90))
    if med <= 0:
        osc = p90 >= osc_min_amplitude
    else:
        osc = (p90 / med) >= osc_ratio and p90 >= osc_min_amplitude
    if not osc:
        return df, {"rolling_median_applied": False}

    y_series = pd.Series(y)
    smooth = (
        y_series.rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    out = df.copy()
    out["transmittance"] = smooth
    return out, {"rolling_median_applied": True}

@dataclass
class CurveComponent:
    comp_id: int
    bbox: Tuple[int,int,int,int]
    pixels: np.ndarray  # Nx2 (y,x) indices within plot interior

def extract_curve_components(plot_bin: np.ndarray, min_area: int) -> List[CurveComponent]:
    """
    plot_bin: binary image (255 = curve-like pixels)
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((plot_bin > 0).astype(np.uint8), connectivity=8)
    comps: List[CurveComponent] = []
    for cid in range(1, n_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(labels == cid)
        pix = np.column_stack([ys, xs])
        comps.append(CurveComponent(comp_id=cid, bbox=(x, y, x+w, y+h), pixels=pix))
    # Largest first
    comps.sort(key=lambda c: c.pixels.shape[0], reverse=True)
    return comps

def curve_points_from_component(comp: CurveComponent) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return x_pix, y_pix arrays within plot interior coordinate system.
    Uses median y per x to collapse thickness.
    """
    ys = comp.pixels[:, 0]
    xs = comp.pixels[:, 1]
    # group by x
    order = np.argsort(xs)
    xs = xs[order]; ys = ys[order]
    uniq_x, idx_start = np.unique(xs, return_index=True)
    y_med = np.zeros_like(uniq_x, dtype=float)
    for i, xval in enumerate(uniq_x):
        start = idx_start[i]
        end = idx_start[i+1] if i+1 < len(idx_start) else len(xs)
        y_med[i] = float(np.median(ys[start:end]))
    return uniq_x.astype(float), y_med

def get_tick_candidates_from_pdf_text(page: fitz.Page, clip_rect: fitz.Rect) -> List[Tuple[str, fitz.Rect]]:
    """
    Extract text spans within clip_rect. Returns list of (text, bbox_rect) in page coordinates.
    """
    d = page.get_text("dict")
    out: List[Tuple[str, fitz.Rect]] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                txt = (sp.get("text") or "").strip()
                if not txt:
                    continue
                bbox = sp.get("bbox")
                if not bbox:
                    continue
                r = fitz.Rect(bbox)
                if r.intersects(clip_rect):
                    out.append((txt, r))
    return out

def parse_numeric(text: str) -> Optional[float]:
    t = text.strip()
    # remove common OCR/PDF artifacts
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace(",", ".")
    t = re.sub(r"[^\d\.\-\+]", "", t)
    if not t:
        return None
    if NUM_RE.match(t):
        return safe_float(t)
    return None

def detect_break_marker_in_xband(x_img: np.ndarray) -> Tuple[Optional[float], int]:
    """
    Detect visual axis-break markers (e.g., "//") in the x-axis band image.
    Returns (marker_x_in_x_img, count_of_marker_lines).
    """
    if x_img is None or x_img.size == 0:
        return None, 0
    if x_img.ndim == 3:
        gray = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = x_img.copy()
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return None, 0
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(6, int(0.02 * w))
    max_len = max(min_len + 2, int(0.2 * w))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, int(0.2 * min(h, w))),
        minLineLength=min_len,
        maxLineGap=3,
    )
    if lines is None:
        return None, 0
    candidates: List[Tuple[float, float, float]] = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len or length > max_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) < 20 or abs(angle) > 70:
            continue
        midx = (x1 + x2) / 2.0
        midy = (y1 + y2) / 2.0
        candidates.append((midx, midy, angle))
    if len(candidates) < 2:
        return None, 0

    candidates.sort(key=lambda c: c[0])
    x_tol = max(10.0, 0.04 * w)
    y_tol = max(10.0, 0.08 * h)
    ang_tol = 15.0
    for i in range(len(candidates) - 1):
        c1 = candidates[i]
        c2 = candidates[i + 1]
        if abs(c1[0] - c2[0]) <= x_tol and abs(c1[1] - c2[1]) <= y_tol and abs(c1[2] - c2[2]) <= ang_tol:
            return float((c1[0] + c2[0]) / 2.0), 2

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1 = candidates[i]
            c2 = candidates[j]
            if abs(c1[0] - c2[0]) <= x_tol and abs(c1[1] - c2[1]) <= y_tol and abs(c1[2] - c2[2]) <= ang_tol:
                return float((c1[0] + c2[0]) / 2.0), 2
    return None, 0

def extract_ticks(img_bgr: np.ndarray, axes: PlotAxes, logger: logging.Logger) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]], Dict[str, Any]]:
    """
    Returns:
      x_ticks: list of (x_pix_interior, value)
      y_ticks: list of (y_pix_interior, value) where y_pix_interior is pixel y within interior
    """
    h, w = img_bgr.shape[:2]
    # Regions for OCR relative to full image
    # x ticks: strip below x-axis
    x_start = min(h-1, axes.x_axis_y + max(2, int(0.01*h)))
    x_end = min(h, axes.x_axis_y + int(0.22*h))
    # If plot sits above caption/paragraph text, cap the tick OCR band to avoid paragraph.
    if axes.x_axis_y < int(0.75*h):
        x_end = min(x_end, int(0.85*h))
    xreg = (axes.x0, x_start, axes.x1, max(x_start+1, x_end))
    # y ticks: strip left of y-axis (or left margin)
    left_edge = axes.y_axis_x if axes.y_axis_x is not None else axes.x0
    yreg = (max(0, left_edge - int(0.15*w)), axes.y0, max(0, axes.x0 - 5), axes.y1)

    meta: Dict[str, Any] = {
        "xreg": xreg,
        "yreg": yreg,
        "x_break_marker_present": False,
        "x_break_marker_pix": None,
        "x_break_marker_count": 0,
    }

    x_img = crop_region(img_bgr, xreg)
    y_img = crop_region(img_bgr, yreg) if (yreg[2] > yreg[0] + 5) else None

    marker_x = None
    marker_count = 0
    axis_band_half = max(3, int(0.03 * h))
    axis_y0 = max(0, axes.x_axis_y - axis_band_half)
    axis_y1 = min(h, axes.x_axis_y + axis_band_half)
    axis_xreg = (axes.x0, axis_y0, axes.x1, max(axis_y0 + 1, axis_y1))
    axis_img = crop_region(img_bgr, axis_xreg)
    if axis_img is not None and axis_img.size > 0:
        marker_x, marker_count = detect_break_marker_in_xband(axis_img)
        if marker_x is not None:
            marker_x_full = axis_xreg[0] + marker_x
            meta["x_break_marker_pix"] = float(marker_x_full - axes.x0)
            meta["x_break_marker_count"] = int(marker_count)
            meta["x_break_marker_present"] = True
    if marker_x is None and x_img is not None and x_img.size > 0:
        marker_x, marker_count = detect_break_marker_in_xband(x_img)
        if marker_x is not None:
            marker_x_full = xreg[0] + marker_x
            meta["x_break_marker_pix"] = float(marker_x_full - axes.x0)
            meta["x_break_marker_count"] = int(marker_count)
            meta["x_break_marker_present"] = True
    if marker_x is None:
        meta["x_break_marker_pix"] = None
        meta["x_break_marker_count"] = 0
        meta["x_break_marker_present"] = False

    # OCR x ticks
    x_tokens = ocr_tokens(x_img, psm=6, whitelist="0123456789.-")
    x_vals: List[Tuple[float,float,float]] = []  # (x_center_in_full, val, conf)
    for tok in x_tokens:
        if tok.conf < 40:
            continue
        val = parse_numeric(tok.text)
        if val is None:
            continue
        # plausible wavenumbers
        if not (50 <= val <= 5000):
            continue
        x_center_full = xreg[0] + tok.cx
        x_vals.append((x_center_full, float(val), float(tok.conf)))
    # de-duplicate by value keeping best conf
    best_by_val: Dict[float, Tuple[float,float]] = {}
    for xc, v, conf in x_vals:
        if v not in best_by_val or conf > best_by_val[v][1]:
            best_by_val[v] = (xc, conf)
    x_ticks = [(xc - axes.x0, v) for v, (xc, conf) in best_by_val.items()]
    x_ticks.sort(key=lambda t: t[0])
    meta["x_ticks_n"] = len(x_ticks)

    # OCR y ticks
    y_ticks: List[Tuple[float,float]] = []
    y_mode = "missing"
    if y_img is not None and y_img.size > 0:
        y_tokens = ocr_tokens(y_img, psm=6, whitelist="0123456789.-")
        y_vals: List[Tuple[float,float,float]] = []  # (y_center_full, val, conf)
        for tok in y_tokens:
            if tok.conf < 40:
                continue
            val = parse_numeric(tok.text)
            if val is None:
                continue
            # plausible transmittance range: 0-1.5 or 0-100
            if not ((0 <= val <= 1.5) or (0 <= val <= 110)):
                continue
            y_center_full = yreg[1] + tok.cy
            y_vals.append((y_center_full, float(val), float(tok.conf)))
        # de-duplicate by value keeping best
        best_by_val_y: Dict[float, Tuple[float,float]] = {}
        for yc, v, conf in y_vals:
            if v not in best_by_val_y or conf > best_by_val_y[v][1]:
                best_by_val_y[v] = (yc, conf)
        y_ticks = [(yc - axes.y0, v) for v, (yc, conf) in best_by_val_y.items()]
        # sort by pixel y
        y_ticks.sort(key=lambda t: t[0])
        meta["y_ticks_n"] = len(y_ticks)
        if len(y_ticks) >= 2:
            y_mode = "calibrated"
    meta["y_mode"] = y_mode
    return x_ticks, y_ticks, meta

def digitize_single_component(
    comp: CurveComponent,
    axes: PlotAxes,
    x_model_left: AxisModel,
    x_model_right: Optional[AxisModel],
    break_info: BreakInfo,
    y_model: Optional[AxisModel],
    y_mode: str,
    interior_shape: Tuple[int,int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns CurvePoints dataframe rows for one component (one spectrum) + qc dict.
    """
    w_int, h_int = interior_shape[1], interior_shape[0]
    x_pix, y_pix = curve_points_from_component(comp)
    # Convert to full image coordinate (within interior)
    # x_pix, y_pix are within interior already (0..w_int, 0..h_int)
    # Map to wavenumber
    if break_info.present and break_info.x_split_pix is not None and x_model_right is not None:
        x_split = break_info.x_split_pix - axes.x0  # to interior coordinates
        left_mask = x_pix < x_split
        wn = np.empty_like(x_pix, dtype=float)
        wn[left_mask] = x_model_left(x_pix[left_mask])
        wn[~left_mask] = x_model_right(x_pix[~left_mask])
        segment_id = np.where(left_mask, 0, 1)
    else:
        wn = x_model_left(x_pix)
        segment_id = np.zeros_like(x_pix, dtype=int)

    # Map to transmittance (calibrated mode only; global normalization applied later)
    if y_model is not None and y_mode == "calibrated":
        # y_model expects pixel in interior y; but we fit with y ticks relative to interior
        T = y_model(y_pix)
        # keep in 0..1
        T = np.clip(T, 1e-6, 1.0)
        A = transmittance_to_absorbance(T)
    else:
        T = np.full_like(y_pix, np.nan, dtype=float)
        A = np.full_like(y_pix, np.nan, dtype=float)

    df = pd.DataFrame({
        "wavenumber_cm1": wn,
        "transmittance": T,
        "absorbance": A,
        "segment_id": segment_id,
        "imputed": False,
        "x_pix": x_pix,
        "y_pix": y_pix,
    })

    # Sort by wavenumber for later imputation and consistent output
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)

    if y_mode == "calibrated":
        y_min = float(np.nanmin(T)) if np.isfinite(T).any() else None
        y_max = float(np.nanmax(T)) if np.isfinite(T).any() else None
    else:
        y_min = None
        y_max = None

    qc = {
        "n_points": int(len(df)),
        "y_mode": y_mode,
        "y_normalization": "calibrated" if y_mode == "calibrated" else "global",
        "y_min": y_min,
        "y_max": y_max,
    }
    return df, qc

def impute_gap(
    df: pd.DataFrame,
    gap_lo: float,
    gap_hi: float,
    dx: Optional[float] = None,
    side_n: int = 25,
) -> pd.DataFrame:
    """
    Adds imputed points across (gap_lo, gap_hi). Input df sorted by wavenumber.
    """
    if gap_lo is None or gap_hi is None:
        return df
    if gap_hi <= gap_lo:
        return df
    x = df["wavenumber_cm1"].to_numpy(dtype=float)
    yT = df["transmittance"].to_numpy(dtype=float)

    left_mask = x <= gap_lo
    right_mask = x >= gap_hi
    if not left_mask.any() or not right_mask.any():
        return df

    xL = x[left_mask]; yL = yT[left_mask]
    xR = x[right_mask]; yR = yT[right_mask]

    y0 = float(yL[-1]); y1 = float(yR[0])
    m0 = estimate_local_slope(xL, yL, tail=True, n=side_n)
    m1 = estimate_local_slope(xR, yR, tail=False, n=side_n)

    if dx is None:
        dxL = np.median(np.diff(xL)) if len(xL) > 2 else np.nan
        dxR = np.median(np.diff(xR)) if len(xR) > 2 else np.nan
        dx_candidates = [d for d in [dxL, dxR] if np.isfinite(d) and d > 0]
        dx = float(min(dx_candidates)) if dx_candidates else float((gap_hi-gap_lo)/200.0)
    dx = max(dx, (gap_hi-gap_lo)/2000.0)

    x_gap = np.arange(gap_lo + dx, gap_hi, dx, dtype=float)
    if len(x_gap) == 0:
        return df
    y_gap = hermite_bridge(gap_lo, gap_hi, y0, y1, m0, m1, x_gap)
    y_gap = np.clip(y_gap, 1e-6, 1.0)
    a_gap = transmittance_to_absorbance(y_gap)

    df_gap = pd.DataFrame({
        "wavenumber_cm1": x_gap,
        "transmittance": y_gap,
        "absorbance": a_gap,
        "segment_id": -1,
        "imputed": True,
    })
    out = pd.concat([df, df_gap], ignore_index=True).sort_values("wavenumber_cm1").reset_index(drop=True)
    return out

def merged_component_ranges(df: pd.DataFrame) -> List[Tuple[float, float]]:
    if df is None or df.empty:
        return []
    if "component_index" in df.columns:
        ranges = df.groupby("component_index")["wavenumber_cm1"].agg(["min", "max"]).reset_index()
        spans = [(float(r["min"]), float(r["max"])) for _, r in ranges.iterrows()]
    else:
        spans = [(float(df["wavenumber_cm1"].min()), float(df["wavenumber_cm1"].max()))]

    spans.sort(key=lambda t: t[0])
    if len(spans) <= 1:
        return spans

    xvals = df["wavenumber_cm1"].to_numpy(dtype=float)
    diffs = np.diff(np.sort(xvals))
    med = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
    tol = max(2.0 * med, 5.0) if med > 0 else 5.0

    merged: List[List[float]] = []
    for start, end in spans:
        if not merged:
            merged.append([start, end])
            continue
        if start <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]

def overlay_curve(img_bgr: np.ndarray, axes: PlotAxes, curve_df: pd.DataFrame, out_path: Path) -> None:
    """
    Creates an overlay image: plot crop + digitized points (in pixel space approximation).
    We cannot invert calibration reliably here without storing models, so overlay uses
    the pixel curve extracted before calibration only when available. This function is
    mainly used when we still have a plot interior mask; implemented via plotting in data
    space instead: absorbance vs wavenumber, not overlay on original image.
    """
    # Data-space plot
    plt.figure(figsize=(10, 3))
    plt.plot(curve_df["wavenumber_cm1"], curve_df["absorbance"], linewidth=1.0)
    plt.gca().invert_xaxis()  # FTIR convention (optional); can be removed
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def parse_peak_list(text: str) -> Tuple[str, List[Tuple[float, Optional[str], str]]]:
    """
    Parse 'Wavenumbers (cm−1): ...' style lists, returning (raw, parsed list).
    Parsed list items: (wavenumber, qualifier, meaning)
    """
    raw = ""
    peaks: List[Tuple[float, Optional[str], str]] = []
    if not text:
        return raw, peaks
    m = re.search(r"Wavenumbers?.{0,30}?:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return raw, peaks
    raw = m.group(1).strip()
    # stop at first newline that looks like end of list (double newline or 'References' etc.)
    raw = re.split(r"\n\s*\n|References|Locality|Formula", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # tokenization
    tokens = re.split(r"[,\s]+", raw)
    for t in tokens:
        if not t:
            continue
        t = t.strip().strip(";.")
        # qualifier may be appended
        q = None
        if t.lower().endswith("sh"):
            q = "sh"; num = t[:-2]
        elif t.lower().endswith("s"):
            q = "s"; num = t[:-1]
        elif t.lower().endswith("w"):
            q = "w"; num = t[:-1]
        else:
            num = t
        num = re.sub(r"[^\d\.]", "", num)
        if not num:
            continue
        val = safe_float(num)
        if val is None:
            continue
        peaks.append((float(val), q, QUALIFIER_MEANINGS.get(q, "")))
    return raw, peaks


def parse_mineral_metadata(text: str, fallback_name: str = "") -> Dict[str, str]:
    """
    Extract mineral name + chemical formula from nearby PDF text.
    Returns dict with keys: mineral_name, formula
    """
    out = {"mineral_name": "", "formula": ""}
    t = (text or "").strip()

    # Mineral name: prefer first non-empty line that contains letters and isn't "Fig."
    name = ""
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("fig"):
            continue
        if re.search(r"[A-Za-z]", s):
            name = s
            break
    if not name:
        name = (fallback_name or "").strip()
    out["mineral_name"] = name

    # Formula: first match of formula-like token
    fm = FORMULA_RE.search(t.replace("\u2212", "-").replace("\u00b7", "·"))
    if fm:
        out["formula"] = fm.group(0).strip()
    return out

def parse_spectrum_name_from_text(text: str, fallback_name: str = "") -> str:
    """
    Extract spectrum name from text above the graph, preferably the line above the formula.
    """
    if not text:
        return (fallback_name or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return (fallback_name or "").strip()
    formula_idx = None
    for idx, line in enumerate(lines):
        if FORMULA_RE.search(line.replace("\u2212", "-").replace("\u00b7", "·")):
            formula_idx = idx
            break
    if formula_idx is not None:
        for j in range(formula_idx - 1, -1, -1):
            s = lines[j]
            if not s:
                continue
            if s.lower().startswith("fig"):
                continue
            if FORMULA_RE.search(s):
                continue
            if re.search(r"[A-Za-z]", s):
                return s
    for line in lines:
        if line.lower().startswith("fig"):
            continue
        if FORMULA_RE.search(line):
            continue
        if re.search(r"[A-Za-z]", line):
            return line
    return (fallback_name or "").strip()

def parse_description_from_text(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*Description\s*:\s*(.*)", line, flags=re.IGNORECASE)
        if not m:
            continue
        desc = (m.group(1) or "").strip()
        if desc:
            return desc
        tail: List[str] = []
        for nxt in lines[i+1:]:
            if not nxt.strip():
                break
            if re.match(r"\s*\w[\w\s]{0,20}:\s*", nxt):
                break
            tail.append(nxt.strip())
            if len(tail) >= 2:
                break
        return " ".join(tail).strip()
    return ""
def extract_entry_text_near_image(page: fitz.Page, img_rect: fitz.Rect) -> str:
    """
    Heuristic: collect text blocks whose vertical position is near the image.
    """
    d = page.get_text("dict")
    blocks = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = b.get("bbox")
        if not bbox:
            continue
        r = fitz.Rect(bbox)
        # near image: within a band above/around it
        if r.y1 <= img_rect.y0 + 30 and r.y1 >= img_rect.y0 - 250:
            blocks.append((r.y0, b))
        # sometimes metadata is directly below
        if r.y0 >= img_rect.y1 - 30 and r.y0 <= img_rect.y1 + 120:
            blocks.append((r.y0, b))
    blocks.sort(key=lambda t: t[0])
    texts: List[str] = []
    for _, b in blocks:
        for ln in b.get("lines", []):
            line_txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if line_txt:
                texts.append(line_txt)
    return "\n".join(texts)


def extract_entry_text_around_image(page: fitz.Page, img_rect: fitz.Rect, pad_top: float = 320.0, pad_bottom: float = 680.0) -> str:
    """
    Extract text around an image using a clip rectangle expanded above/below.
    This is usually better than "near" heuristics for capturing mineral name/formula/peaks.
    """
    pr = page.rect
    clip = fitz.Rect(pr.x0, max(pr.y0, img_rect.y0 - pad_top), pr.x1, min(pr.y1, img_rect.y1 + pad_bottom))
    try:
        t = page.get_text("text", clip=clip) or ""
    except Exception:
        t = ""
    t = t.strip()
    if t:
        return t
    return extract_entry_text_near_image(page, img_rect)

def extract_entry_text_bands(
    page: fitz.Page,
    img_rect: fitz.Rect,
    *,
    pad_top: float = 320.0,
    pad_bottom: float = 680.0,
) -> Tuple[str, str]:
    """
    Extract text above and below an image using expanded clip rectangles.
    Returns (above_text, below_text).
    """
    pr = page.rect
    above = fitz.Rect(pr.x0, max(pr.y0, img_rect.y0 - pad_top), pr.x1, img_rect.y0)
    below = fitz.Rect(pr.x0, img_rect.y1, pr.x1, min(pr.y1, img_rect.y1 + pad_bottom))
    try:
        above_text = (page.get_text("text", clip=above) or "").strip()
    except Exception:
        above_text = ""
    try:
        below_text = (page.get_text("text", clip=below) or "").strip()
    except Exception:
        below_text = ""
    return above_text, below_text
def detect_labels_for_components(img_bgr: np.ndarray, comps: List[CurveComponent], axes: PlotAxes) -> Dict[int, str]:
    """
    OCR mineral names written near each curve, assign by nearest line of text.
    Returns dict comp_id -> label string.
    """
    tokens = ocr_tokens(img_bgr, psm=6, whitelist=None)
    # Build lines
    lines: Dict[Tuple[int,int], List[OCRToken]] = {}
    for t in tokens:
        if t.conf < 45:
            continue
        if not re.search(r"[A-Za-z]", t.text):
            continue
        key = (t.block, t.line)
        lines.setdefault(key, []).append(t)
    line_items: List[Tuple[str, Tuple[int,int,int,int], float]] = []
    for _, toks in lines.items():
        toks.sort(key=lambda z: z.x)
        txt = " ".join([z.text for z in toks]).strip()
        if len(txt) < 3:
            continue
        x0 = min(z.x for z in toks)
        y0 = min(z.y for z in toks)
        x1 = max(z.x+z.w for z in toks)
        y1 = max(z.y+z.h for z in toks)
        cx = (x0+x1)/2
        cy = (y0+y1)/2
        line_items.append((txt, (x0,y0,x1,y1), cx))


@dataclass
class LabelLine:
    text: str
    bbox: Tuple[int,int,int,int]
    cy: float

def detect_label_lines(img_bgr: np.ndarray, axes: PlotAxes, min_conf: float = 45.0) -> List[LabelLine]:
    """
    Detect spectrum labels (mineral names) inside the plot area, typically on the right.
    Returns label lines sorted by vertical position.
    """
    tokens = ocr_tokens(img_bgr, psm=6, whitelist=None)
    lines: Dict[Tuple[int,int], List[OCRToken]] = {}
    for t in tokens:
        if t.conf < min_conf:
            continue
        if not re.search(r"[A-Za-z]", t.text):
            continue
        if not (axes.y0 <= t.cy <= img_bgr.shape[0]):
            continue
        if t.cx < axes.x0 + 0.55*(axes.x1-axes.x0):
            continue
        key = (t.block, t.line)
        lines.setdefault(key, []).append(t)

    label_lines: List[LabelLine] = []
    for _, toks in lines.items():
        toks.sort(key=lambda z: z.x)
        txt = " ".join([z.text for z in toks]).strip()
        if len(txt) < 3:
            continue
        if txt.lower().startswith("fig"):
            continue
        x0 = min(z.x for z in toks)
        y0 = min(z.y for z in toks)
        x1 = max(z.x+z.w for z in toks)
        y1 = max(z.y+z.h for z in toks)
        cy = (y0+y1)/2
        label_lines.append(LabelLine(text=txt, bbox=(x0,y0,x1,y1), cy=cy))

    label_lines.sort(key=lambda L: L.cy)
    merged: List[LabelLine] = []
    for L in label_lines:
        if not merged:
            merged.append(L); continue
        if abs(L.cy - merged[-1].cy) < 10:
            if L.text not in merged[-1].text:
                merged[-1] = LabelLine(text=(merged[-1].text + " " + L.text).strip(),
                                       bbox=merged[-1].bbox,
                                       cy=merged[-1].cy)
        else:
            merged.append(L)
    return merged

def label_bands_from_lines(label_lines: List[LabelLine], axes: PlotAxes, pad: int = 6) -> List[Tuple[LabelLine, Tuple[int,int]]]:
    """
    Convert label line y positions into demarcated y-bands in plot interior coordinates.
    Returns list of (LabelLine, (y0_int, y1_int)).
    """
    if not label_lines:
        return []
    ys = [L.cy for L in label_lines]
    bounds = []
    for i, L in enumerate(label_lines):
        y_mid_prev = (ys[i-1] + ys[i]) / 2 if i > 0 else axes.y0
        y_mid_next = (ys[i] + ys[i+1]) / 2 if i+1 < len(ys) else axes.y1
        y0 = int(max(axes.y0, y_mid_prev - pad))
        y1 = int(min(axes.y1, y_mid_next + pad))
        y0_int = max(0, y0 - axes.y0)
        y1_int = max(y0_int + 1, y1 - axes.y0)
        bounds.append((L, (y0_int, y1_int)))
    return bounds
    labels: Dict[int, str] = {}
    for comp in comps:
        x0,y0,x1,y1 = comp.bbox
        # bbox in interior; convert to full image coords
        bx0 = axes.x0 + x0
        by0 = axes.y0 + y0
        bx1 = axes.x0 + x1
        by1 = axes.y0 + y1
        bc_x = (bx0+bx1)/2
        bc_y = (by0+by1)/2
        best = None
        best_d = 1e18
        for txt, (lx0,ly0,lx1,ly1), lc_x in line_items:
            # prefer labels to the right of curve
            if lc_x < bx1 - 10:
                continue
            lc_y = (ly0+ly1)/2
            d = abs(lc_y - bc_y) + 0.3*abs(lc_x - bc_x)
            if d < best_d:
                best_d = d
                best = txt
        if best:
            labels[comp.comp_id] = best
    return labels

def remap_label_lines_to_axes(
    label_lines: List[LabelLine],
    source_axes: PlotAxes,
    target_axes: PlotAxes,
) -> List[LabelLine]:
    if not label_lines:
        return []
    src_span = float(max(1, source_axes.y1 - source_axes.y0))
    tgt_span = float(max(1, target_axes.y1 - target_axes.y0))
    remapped: List[LabelLine] = []
    for L in label_lines:
        frac = (L.cy - source_axes.y0) / src_span
        cy = target_axes.y0 + frac * tgt_span
        remapped.append(LabelLine(text=L.text, bbox=(target_axes.x1, int(cy), target_axes.x1, int(cy)), cy=cy))
    remapped.sort(key=lambda L: L.cy)
    return remapped


def write_single_spectrum_xlsx(
    out_xlsx: Path,
    run_df: pd.DataFrame,
    entry_row: Dict[str, Any],
    curve_rows: List[Dict[str, Any]],
    peaks_rows: List[Dict[str, Any]],
    qc_rows: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    entry_df = pd.DataFrame([entry_row])
    curve_df = pd.DataFrame(curve_rows)
    peaks_df = pd.DataFrame(peaks_rows)
    qc_df = pd.DataFrame(qc_rows)
    sheets = {
        "RunInfo": run_df,
        "Entry": entry_df,
        "CurvePoints": curve_df,
        "PeaksText": peaks_df,
        "QC": qc_df,
    }
    write_excel_with_splitting(out_xlsx, sheets, logger)

def _prepare_jdx_xydata(curve_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if not curve_rows:
        return np.array([]), np.array([]), 0.0, 0.0
    df = pd.DataFrame(curve_rows)
    if df.empty or "wavenumber_cm1" not in df.columns or "transmittance" not in df.columns:
        return np.array([]), np.array([]), 0.0, 0.0
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    x = df["wavenumber_cm1"].to_numpy(dtype=float)
    t = df["transmittance"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(t)
    if not np.any(finite_mask):
        return np.array([]), np.array([]), 0.0, 0.0
    x = x[finite_mask]
    t = t[finite_mask]
    if x.size == 0:
        return np.array([]), np.array([]), 0.0, 0.0
    absorbance = transmittance_to_absorbance(t)
    finite_a = np.isfinite(absorbance)
    if not np.any(finite_a):
        return np.array([]), np.array([]), 0.0, 0.0
    x = x[finite_a]
    absorbance = absorbance[finite_a]
    if x.size == 0:
        return np.array([]), np.array([]), 0.0, 0.0
    y_min = float(np.min(absorbance))
    y_max = float(np.max(absorbance))
    if y_max > y_min:
        y = (absorbance - y_min) / (y_max - y_min)
    else:
        y = np.zeros_like(absorbance)
    diffs = np.diff(x)
    diffs = diffs[diffs > 0]
    deltax = float(np.median(diffs)) if diffs.size else 0.0
    firstx = float(x[0]) if x.size else 0.0
    return x, y, firstx, deltax

def _append_header(lines: List[str], key: str, value: Optional[str]) -> None:
    value_str = _sanitize_jdx_str(value)
    if not value_str:
        return
    lines.append(f"##{key}={value_str}")

def _sanitize_jdx_str(value: Optional[str]) -> str:
    if value is None:
        return ""
    value_str = _ILLEGAL_XLSX_RE.sub(" ", str(value))
    value_str = re.sub(r"[ \t]+", " ", value_str).strip()
    return value_str

def _first_nonempty(*values: Optional[str]) -> str:
    for value in values:
        value_str = _sanitize_jdx_str(value)
        if value_str:
            return value_str
    return ""

def _build_jdx_headers(
    entry_row: Dict[str, Any],
    *,
    npoints: int,
    firstx: float,
    deltax: float,
    title_override: Optional[str] = None,
) -> List[str]:
    lines = [
        "##JCAMP-DX=5.01",
        "##DATA TYPE=FTIR",
        "##XUNITS=1/CM",
        "##YUNITS=ABSORBANCE",
        f"##NPOINTS={npoints}",
        f"##FIRSTX={firstx:.10g}",
        f"##DELTAX={deltax:.10g}",
    ]
    title = _first_nonempty(
        entry_row.get("entry_name"),
        title_override,
        entry_row.get("label_ocr"),
        entry_row.get("entry_label"),
        entry_row.get("entry_id"),
    )
    description = _first_nonempty(entry_row.get("description"), entry_row.get("entry_description"))
    _append_header(lines, "TITLE", title)
    _append_header(lines, "NOTES", description)
    _append_header(lines, "ORIGIN", _first_nonempty(entry_row.get("source_title"), entry_row.get("origin")))
    _append_header(lines, "OWNER", _first_nonempty(entry_row.get("source_author"), entry_row.get("owner")))
    _append_header(lines, "DATE", _first_nonempty(entry_row.get("date"), entry_row.get("timestamp"), entry_row.get("run_timestamp")))
    names = _first_nonempty(entry_row.get("mineral_name"), entry_row.get("label_ocr"), entry_row.get("entry_label"))
    _append_header(lines, "NAMES", names)
    _append_header(lines, "CAS REGISTRY NO", _first_nonempty(entry_row.get("cas_registry_no"), entry_row.get("cas")))
    _append_header(lines, "MOLFORM", entry_row.get("formula"))
    return lines

def write_single_spectrum_jdx(
    out_jdx: Path,
    entry_row: Dict[str, Any],
    curve_rows: List[Dict[str, Any]],
    logger: logging.Logger,
    *,
    points_per_line: int = 6,
) -> bool:
    x, y, firstx, deltax = _prepare_jdx_xydata(curve_rows)
    if x.size == 0 or y.size == 0:
        logger.warning(f"Skipping JDX for {entry_row.get('entry_id', '')}: no finite curve data.")
        return False
    headers = _build_jdx_headers(entry_row, npoints=int(x.size), firstx=firstx, deltax=deltax)
    xydata_payload = _build_xydata_payload(x, y, points_per_line=points_per_line)
    if not xydata_payload:
        logger.warning(f"Skipping JDX for {entry_row.get('entry_id', '')}: no XYDATA payload.")
        return False
    out_jdx.parent.mkdir(parents=True, exist_ok=True)
    lines = headers + ["##XYDATA=(X++(Y..Y))", xydata_payload, "##END="]
    out_jdx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote JDX: {out_jdx}")
    return True

def write_multi_spectrum_jdx(
    out_jdx: Path,
    entries: Dict[str, Dict[str, Any]],
    curve_by: Dict[str, List[Dict[str, Any]]],
    logger: logging.Logger,
    *,
    points_per_line: int = 6,
) -> bool:
    spectra: List[Tuple[str, Dict[str, Any], np.ndarray, np.ndarray, float, float]] = []
    for entry_id, entry_row in entries.items():
        curve_rows = curve_by.get(entry_id, [])
        x, y, firstx, deltax = _prepare_jdx_xydata(curve_rows)
        if x.size == 0 or y.size == 0:
            logger.warning(f"Skipping master JDX entry {entry_id}: no finite curve data.")
            continue
        spectra.append((entry_id, entry_row, x, y, firstx, deltax))

    if not spectra:
        logger.warning("Skipping master JDX: no spectra with usable curve data.")
        return False

    base_x = spectra[0][2]
    base_firstx = spectra[0][4]
    base_deltax = spectra[0][5]
    for entry_id, _, x, _, _, _ in spectra[1:]:
        if x.size != base_x.size or not np.allclose(x, base_x, rtol=1e-6, atol=1e-6):
            logger.warning(f"Skipping master JDX: X-axis mismatch for entry {entry_id}.")
            return False

    blocks: List[str] = []
    for entry_id, entry_row, x, y, firstx, deltax in spectra:
        headers = _build_jdx_headers(
            entry_row,
            npoints=int(x.size),
            firstx=firstx,
            deltax=deltax,
            title_override=f"{entry_id} {entry_row.get('label_ocr', '')}".strip(),
        )
        xydata_payload = _build_xydata_payload(x, y, points_per_line=points_per_line)
        if not xydata_payload:
            logger.warning(f"Skipping master JDX entry {entry_id}: no XYDATA payload.")
            continue
        blocks.extend(headers)
        blocks.append("##XYDATA=(X++(Y..Y))")
        blocks.append(xydata_payload)
        blocks.append("##END=")

    if not blocks:
        logger.warning("Skipping master JDX: no XYDATA payloads.")
        return False

    out_jdx.parent.mkdir(parents=True, exist_ok=True)
    out_jdx.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    logger.info(f"Wrote JDX: {out_jdx}")
    return True

def write_excel_with_splitting(
    out_path: Path,
    sheets: Dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            df = sanitize_dataframe_for_excel(df)
            if len(df) <= EXCEL_MAX_ROWS - 1:
                df.to_excel(writer, sheet_name=name[:31], index=False)
            else:
                # split
                chunk_size = EXCEL_MAX_ROWS - 1
                n_chunks = int(math.ceil(len(df) / chunk_size))
                logger.info(f"Sheet '{name}' has {len(df):,} rows; splitting into {n_chunks} sheets.")
                for i in range(n_chunks):
                    chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
                    sheet_name = f"{name}_{i+1:02d}"[:31]
                    chunk.to_excel(writer, sheet_name=sheet_name, index=False)

def infer_source_meta(pdf_path: Path, cli: Dict[str, Optional[str]]) -> Dict[str, str]:
    fname = pdf_path.name
    title = cli.get("source_title") or ""
    author = cli.get("source_author") or ""
    doi = cli.get("doi") or ""
    isbn = cli.get("isbn") or ""

    if not title:
        # crude title: everything before first '['
        title = fname.split("[", 1)[0].replace("_", " ").strip()
    if not doi:
        m = DOI_RE.search(fname)
        if m:
            doi = m.group(0)
    if not isbn:
        m2 = ISBN_RE.search(fname)
        if m2:
            isbn = m2.group(0)
    # author likely not in metadata; leave as provided
    return {"source_title": title, "source_author": author, "doi": doi, "isbn": isbn}

def _set_tesseract_cmd(tesseract_cmd: Optional[str], logger: logging.Logger) -> None:
    cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD") or ""
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        logger.info(f"Using tesseract_cmd: {cmd}")

def _expand_clip_for_axes(page_rect: fitz.Rect, rect: fitz.Rect) -> fitz.Rect:
    """
    Asymmetric expansion around embedded image rect to capture full plot incl. axes and tick labels.
    Tuned for this book layout: more padding on left and bottom.
    """
    w = rect.width
    h = rect.height
    left = 0.28 * w
    bottom = 0.30 * h
    right = 0.08 * w
    top = 0.08 * h
    clip = fitz.Rect(rect.x0 - left, rect.y0 - top, rect.x1 + right, rect.y1 + bottom)
    clip = clip & page_rect
    return clip

def _process_page_worker(args: Tuple) -> Dict[str, Any]:
    (
        pdf_path_str,
        pno,
        dpi,
        save_graph_images,
        graph_dir_str,
        save_digitized_plots,
        digitized_dir_str,
        verbose,
        source_meta,
        max_images_per_page,
        bin_representative,
        axis_filter_px,
        border_filter_px,
        rolling_median_window,
        label_lookahead_pages,
    ) = args
    logs: List[Tuple[str, str]] = []  # (level, msg)
    def log(level: str, msg: str):
        logs.append((level, msg))

    pdf_path = Path(pdf_path_str)
    graph_dir = Path(graph_dir_str) if graph_dir_str else None
    digitized_dir = Path(digitized_dir_str) if digitized_dir_str else None

    entries_rows: List[Dict[str, Any]] = []
    peaks_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    written_entry_ids: set = set()

    try:
        doc = fitz.open(str(pdf_path))
        page = doc[pno]
        images = page.get_images(full=True)
        if len(images) == 0:
            log("INFO", f"Page {pno+1}: no embedded images; skipping.")
            doc.close()
            return {"page_index": pno, "entries_rows": entries_rows, "peaks_rows": peaks_rows, "qc_rows": qc_rows, "curve_rows": curve_rows, "logs": logs}

        if len(images) > max_images_per_page:
            log("WARN", f"Page {pno+1}: {len(images)} images > cap {max_images_per_page}; truncating.")
            images = images[:max_images_per_page]

        log("INFO", f"Page {pno+1}: {len(images)} image(s)")

        page_rect = page.rect

        def _render_image_crop_for_page(target_page: fitz.Page, target_rect: fitz.Rect) -> np.ndarray:
            clip = _expand_clip_for_axes(target_page.rect, target_rect)
            pix = target_page.get_pixmap(clip=clip, dpi=dpi)
            pil = pixmap_to_pil(pix)
            return pil_to_bgr(pil)

        def _find_lookahead_labels() -> Tuple[List[LabelLine], int]:
            if label_lookahead_pages <= 0:
                return [], 0
            for offset in range(1, label_lookahead_pages + 1):
                next_idx = pno + offset
                if next_idx >= len(doc):
                    break
                next_page = doc[next_idx]
                next_images = next_page.get_images(full=True)
                if img_idx >= len(next_images):
                    continue
                next_xref = next_images[img_idx][0]
                next_rects = next_page.get_image_rects(next_xref)
                if not next_rects:
                    continue
                next_rect = next_rects[0]
                next_bgr = _render_image_crop_for_page(next_page, next_rect)
                next_axes = detect_plot_axes(next_bgr)
                next_label_lines = detect_label_lines(next_bgr, next_axes)
                if next_label_lines:
                    remapped = remap_label_lines_to_axes(next_label_lines, next_axes, axes)
                    if remapped:
                        return remapped, offset
            return [], 0

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            entry_text = extract_entry_text_around_image(page, rect)
            entry_text_above, entry_text_below = extract_entry_text_bands(page, rect)
            entry_description = parse_description_from_text(entry_text_below)

            # Render crop with larger asymmetric padding so ticks/axes are included
            clip = _expand_clip_for_axes(page_rect, rect)
            pix = page.get_pixmap(clip=clip, dpi=dpi)
            pil = pixmap_to_pil(pix)
            bgr = pil_to_bgr(pil)

            image_path = ""
            axes = detect_plot_axes(bgr)
            # Save a graph snapshot cropped around detected axes (with padding for tick labels).
            if save_graph_images and graph_dir is not None:
                graph_dir.mkdir(parents=True, exist_ok=True)
                image_path = str(graph_dir / f"p{pno+1:04d}_img{img_idx:02d}_graph.png")
                h_img, w_img = bgr.shape[:2]

                pad_x = max(10, int(0.06 * w_img))
                pad_top = max(10, int(0.06 * h_img))
                pad_bot = max(14, int(0.10 * h_img))
                x0s = max(0, axes.x0 - pad_x)
                x1s = min(w_img, axes.x1 + pad_x)
                y0s = max(0, axes.y0 - pad_top)
                y1s = min(h_img, axes.y1 + pad_bot)

                snap_w = max(0, x1s - x0s)
                snap_h = max(0, y1s - y0s)

                # If axis detection is wrong, the crop can collapse to a thin strip.
                # Fall back to saving the full rendered clip to avoid over-cropping.
                if snap_w < int(0.55 * w_img) or snap_h < int(0.35 * h_img):
                    snap = bgr
                else:
                    snap = crop_region(bgr, (x0s, y0s, x1s, y1s))

                bgr_to_pil(snap).save(image_path)
            plot_interior = crop_region(bgr, (axes.x0, axes.y0, axes.x1, axes.y1))
            if plot_interior.size == 0:
                log("ERROR", f"Page {pno+1} img {img_idx}: empty plot interior crop.")
                continue

            # Label-driven demarcation: only extract spectra that have OCR-detectable labels.
            label_lines = detect_label_lines(bgr, axes)
            label_page_offset = 0
            if not label_lines:
                label_lines, label_page_offset = _find_lookahead_labels()
                if label_lines:
                    log(
                        "WARN",
                        f"Page {pno+1} img {img_idx}: no labels detected; using labels from page {pno+1+label_page_offset}.",
                    )
            if not label_lines:
                log("WARN", f"Page {pno+1} img {img_idx}: no labels detected; skipping extraction per rule.")
                continue

            plot_bin = preprocess_for_curve(plot_interior)
            skel = skeletonize((plot_bin > 0)).astype(np.uint8) * 255
            bands = label_bands_from_lines(label_lines, axes)

            # For each label band, collect 1..N curve components (to support split spectra).
            # Each item: (band_index, label_line, [components])
            band_components: List[Tuple[int, LabelLine, List[CurveComponent]]] = []
            for bi, (L, (y0b, y1b)) in enumerate(bands):
                band_skel = skel[y0b:y1b, :]
                if band_skel.size == 0:
                    continue
                min_area = max(60, int(0.0009 * band_skel.shape[0] * max(1, band_skel.shape[1])))

                comps_b = extract_curve_components(band_skel, min_area=min_area)
                if not comps_b:
                    band_bin = plot_bin[y0b:y1b, :]
                    comps_b = extract_curve_components(band_bin, min_area=min_area)
                if not comps_b:
                    continue

                # Filter out axis-like components and keep top few by (x_span, area)
                h_band, w_band = band_skel.shape[:2]
                filtered: List[Tuple[float, CurveComponent]] = []
                for c in comps_b:
                    if component_is_axis_like(c, (h_band, w_band)):
                        continue
                    ys = c.pixels[:, 0]; xs = c.pixels[:, 1]
                    x_span = float(xs.max() - xs.min()) if xs.size else 0.0
                    score = x_span * 10.0 + float(c.pixels.shape[0])
                    filtered.append((score, c))
                if not filtered:
                    continue
                filtered.sort(key=lambda t: t[0], reverse=True)
                keep = [c for _, c in filtered[:3]]

                shifted: List[CurveComponent] = []
                for k, c0 in enumerate(keep):
                    pix = c0.pixels.copy()
                    pix[:, 0] += y0b
                    bbox = (c0.bbox[0], c0.bbox[1] + y0b, c0.bbox[2], c0.bbox[3] + y0b)
                    shifted.append(CurveComponent(comp_id=(bi + 1) * 10 + k, bbox=bbox, pixels=pix))
                band_components.append((bi, L, shifted))

            if not band_components:
                log("WARN", f"Page {pno+1} img {img_idx}: labels detected but no curve components found in label bands.")
                continue

            # OCR ticks (v1 behavior)
            x_ticks, y_ticks, tick_meta = extract_ticks(bgr, axes, setup_logger(verbose))
            if len(x_ticks) < 2:
                log("ERROR", f"Page {pno+1} img {img_idx}: insufficient x ticks ({len(x_ticks)}).")
                # record failure rows for each labeled band (no-label => no extraction)
                for s_idx, (bi, L, _) in enumerate(band_components):
                    label_text = (L.text.strip() if L is not None else "").strip()
                    if not label_text:
                        log("WARN", f"Page {pno+1} img {img_idx} spec {s_idx}: empty label; skipping.")
                        continue
                    entry_id = f"p{pno+1:04d}_img{img_idx:02d}_spec{s_idx:02d}"
                    meta_md = parse_mineral_metadata(entry_text, fallback_name=label_text)
                    entry_name = parse_spectrum_name_from_text(entry_text_above, fallback_name=label_text)
                    entries_rows.append({
                        "entry_id": entry_id,
                        "page_index": pno,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "label_ocr": label_text,
                        "label_page_offset": label_page_offset,
                        "entry_name": entry_name,
                        "entry_description": entry_description,
                        "description": entry_description,
                        "mineral_name": meta_md.get("mineral_name", ""),
                        "formula": meta_md.get("formula", ""),
                        "entry_text_raw": entry_text,
                        "wavenumbers_raw": "",
                        "axis_break_present": False,
                        "gap_lo_cm1": None,
                        "gap_hi_cm1": None,
                        "x_orientation": "",
                        "y_axis_type": tick_meta.get("y_mode", ""),
                        "digitize_status": "failed",
                        "qc_flag": True,
                        "qc_notes": f"Insufficient x ticks for calibration (found {len(x_ticks)})",
                        "image_path": image_path,
                        **source_meta,
                    })
                    qc_rows.append({
                        "entry_id": entry_id,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "stage": "x_calibration",
                        "status": "failed",
                        "notes": f"x_ticks={len(x_ticks)}",
                    })
                continue

            # Build x models
            break_info = detect_axis_break(x_ticks, tick_meta.get("x_break_marker_pix"))
            x_model_left = None
            x_model_right = None
            if break_info.present and break_info.x_split_pix is not None:
                x_split_int = break_info.x_split_pix - axes.x0
                left = [(xp, v) for (xp, v) in x_ticks if xp < x_split_int]
                right = [(xp, v) for (xp, v) in x_ticks if xp >= x_split_int]
                if len(left) >= 2:
                    x_model_left = fit_linear([a for a, _ in left], [b for _, b in left])
                if len(right) >= 2:
                    x_model_right = fit_linear([a for a, _ in right], [b for _, b in right])
            else:
                x_model_left = fit_linear([t[0] for t in x_ticks], [t[1] for t in x_ticks])

            if x_model_left is None:
                log("ERROR", f"Page {pno+1} img {img_idx}: x calibration fit failed.")
                continue

            x_orientation = "increasing" if x_model_left.m > 0 else "decreasing" if x_model_left.m < 0 else ""

            # y model
            y_model = None
            y_mode = tick_meta.get("y_mode", "missing")
            if len(y_ticks) >= 2:
                y_model = fit_linear([t[0] for t in y_ticks], [t[1] for t in y_ticks])

            # Peak list text parse
            w_raw, peaks = parse_peak_list(entry_text)            # Digitize spectra per label band (1 spectrum per label)
            for s_idx, (bi, L, comps_for_label) in enumerate(band_components):
                label_text = (L.text.strip() if L is not None else "").strip()
                if not label_text:
                    log("WARN", f"Page {pno+1} img {img_idx} spec {s_idx}: empty label; skipping.")
                    continue

                entry_id = f"p{pno+1:04d}_img{img_idx:02d}_spec{s_idx:02d}"
                meta_md = parse_mineral_metadata(entry_text, fallback_name=label_text)
                entry_name = parse_spectrum_name_from_text(entry_text_above, fallback_name=label_text)

                # Digitize all components for this label (supports split spectra).
                dfs: List[pd.DataFrame] = []
                qc_notes_parts: List[str] = []
                y_norm_mode = "calibrated" if y_mode == "calibrated" else "global"
                for ci, comp in enumerate(comps_for_label):
                    try:
                        df_seg, qc_seg = digitize_single_component(
                            comp=comp,
                            axes=axes,
                            x_model_left=x_model_left,
                            x_model_right=x_model_right,
                            break_info=break_info,
                            y_model=y_model,
                            y_mode=y_mode,
                            interior_shape=plot_interior.shape[:2],
                        )
                        df_seg["component_index"] = ci
                        dfs.append(df_seg)
                    except Exception as e:
                        qc_notes_parts.append(f"component {ci} failed: {e}")

                if not dfs:
                    entries_rows.append({
                        "entry_id": entry_id,
                        "page_index": pno,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "label_ocr": label_text,
                        "label_page_offset": label_page_offset,
                        "entry_name": entry_name,
                        "entry_description": entry_description,
                        "description": entry_description,
                        "mineral_name": meta_md.get("mineral_name", ""),
                        "formula": meta_md.get("formula", ""),
                        "entry_text_raw": entry_text,
                        "wavenumbers_raw": "",
                        "axis_break_present": bool(break_info.present),
                        "gap_lo_cm1": break_info.gap_lo,
                        "gap_hi_cm1": break_info.gap_hi,
                        "x_orientation": x_orientation,
                        "y_axis_type": y_mode,
                        "y_normalization": y_norm_mode,
                        "digitize_status": "failed",
                        "qc_flag": True,
                        "qc_notes": "no curve components digitized; " + "; ".join(qc_notes_parts),
                        "image_path": image_path,
                        **source_meta,
                    })
                    qc_rows.append({
                        "entry_id": entry_id,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "stage": "digitize",
                        "status": "failed",
                        "y_normalization": y_norm_mode,
                        "notes": "; ".join(qc_notes_parts)[:250],
                    })
                continue

                df_curve = pd.concat(dfs, ignore_index=True)
                df_curve = df_curve.sort_values("wavenumber_cm1").reset_index(drop=True)

                if y_mode != "calibrated":
                    if "y_pix" not in df_curve.columns:
                        qc_notes_parts.append("missing y_pix for global normalization")
                    else:
                        y_pix_all = df_curve["y_pix"].to_numpy(dtype=float)
                        if y_pix_all.size:
                            y_min = float(np.min(y_pix_all))
                            y_max = float(np.max(y_pix_all))
                            if y_max - y_min < 1e-6:
                                T = np.full_like(y_pix_all, 0.5, dtype=float)
                            else:
                                T = 1.0 - (y_pix_all - y_min) / (y_max - y_min)
                            T = np.clip(T, 1e-6, 1.0)
                            df_curve["transmittance"] = T
                            df_curve["absorbance"] = transmittance_to_absorbance(T)
                        else:
                            df_curve["transmittance"] = np.array([], dtype=float)
                            df_curve["absorbance"] = np.array([], dtype=float)

                axis_filter_applied = False
                rolling_median_applied = False
                axis_removed = 0
                original_points = len(df_curve)

                if (axis_filter_px > 0) or (border_filter_px > 0):
                    df_curve, axis_info = filter_points_near_axes(
                        df_curve,
                        axes=axes,
                        interior_shape=plot_interior.shape[:2],
                        border_px=border_filter_px,
                        axis_px=axis_filter_px,
                    )
                    axis_filter_applied = bool(axis_info.get("axis_filter_applied", False))
                    axis_removed = int(axis_info.get("axis_filter_removed", 0))

                df_curve, collapse_info = collapse_duplicate_wavenumbers(
                    df_curve,
                    representative=bin_representative,
                )
                if original_points > 0:
                    log(
                        "INFO",
                        (
                            f"Page {pno+1} img {img_idx} spec {s_idx}: de-jitter "
                            f"axis_removed={axis_removed} "
                            f"collapse_removed={collapse_info.get('collapse_removed', 0)} "
                            f"bins={collapse_info.get('collapse_bins', len(df_curve))} "
                            f"bin_width={collapse_info.get('collapse_bin_width')}"
                        ),
                    )

                # Remove obvious vertical 'infill' artifacts (conservative)
                df_curve = despike_vertical_artifacts(df_curve)

                df_curve, roll_info = apply_rolling_median_if_oscillatory(
                    df_curve,
                    window=rolling_median_window,
                )
                rolling_median_applied = bool(roll_info.get("rolling_median_applied", False))

                # Drop pixel coordinates now that post-processing is done.
                for col in ("x_pix", "y_pix"):
                    if col in df_curve.columns:
                        df_curve = df_curve.drop(columns=[col])

                # Reject near-flat traces (common failure: digitizing an axis line instead of a spectrum)
                try:
                    y_rng = float(df_curve["transmittance"].max() - df_curve["transmittance"].min())
                except Exception:
                    y_rng = 0.0
                if y_rng < 0.03:
                    entries_rows.append({
                        "entry_id": entry_id,
                        "page_index": pno,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "label_ocr": label_text,
                        "label_page_offset": label_page_offset,
                        "entry_name": entry_name,
                        "entry_description": entry_description,
                        "description": entry_description,
                        "mineral_name": meta_md.get("mineral_name", ""),
                        "formula": meta_md.get("formula", ""),
                        "entry_text_raw": entry_text,
                        "wavenumbers_raw": "",
                        "axis_break_present": bool(break_info.present),
                        "gap_lo_cm1": break_info.gap_lo,
                        "gap_hi_cm1": break_info.gap_hi,
                        "x_orientation": x_orientation,
                        "y_axis_type": y_mode,
                        "y_normalization": y_norm_mode,
                        "digitize_status": "failed",
                        "axis_filter_applied": axis_filter_applied,
                        "rolling_median_applied": rolling_median_applied,
                        "bin_width_cm1": collapse_info.get("collapse_bin_width"),
                        "qc_flag": True,
                        "qc_notes": f"Rejected near-flat trace (y_range={y_rng:.4f}); likely axis/border extracted",
                        "image_path": image_path,
                        **source_meta,
                    })
                    qc_rows.append({
                        "entry_id": entry_id,
                        "page_number_1based": pno+1,
                        "image_index": img_idx,
                        "spectrum_index": s_idx,
                        "stage": "digitize",
                        "status": "failed",
                        "y_normalization": y_norm_mode,
                        "notes": (
                            f"flat_trace y_range={y_rng:.4f} "
                            f"axis_filter={axis_filter_applied} rolling_median={rolling_median_applied}"
                        ),
                    })
                continue

                # Gap imputation: (a) explicit axis-break gap, (b) any large gap between digitized segments
                if break_info.present and break_info.gap_lo is not None and break_info.gap_hi is not None:
                    df_curve = impute_gap(df_curve, break_info.gap_lo, break_info.gap_hi)

                # Auto-detect additional large gaps (e.g., split spectra where components were disconnected)
                xvals = df_curve["wavenumber_cm1"].to_numpy(dtype=float)
                if len(xvals) > 5:
                    diffs = np.diff(xvals)
                    med = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
                    if med > 0:
                        thr_gap = max(60.0, 12.0 * med)
                        if med < 5.0:
                            thr_gap = max(30.0, 8.0 * med)
                    else:
                        thr_gap = 120.0
                    if break_info.marker_present:
                        if med > 0:
                            thr_gap = min(thr_gap, max(30.0, 6.0 * med))
                        else:
                            thr_gap = min(thr_gap, 80.0)

                    ranges = merged_component_ranges(df_curve)
                    gap_count = 0
                    for ri in range(len(ranges) - 1):
                        g0 = float(ranges[ri][1])
                        g1 = float(ranges[ri + 1][0])
                        if g1 - g0 <= thr_gap:
                            continue
                        df_curve = impute_gap(df_curve, g0, g1)
                        gap_count += 1
                        if gap_count >= 2:
                            break

                # Keep transmittance only (drop absorbance) to match requirement
                if "absorbance" in df_curve.columns:
                    df_curve = df_curve.drop(columns=["absorbance"])

                # Peaks list from entry text (if present)
                w_raw, peaks = parse_peak_list(entry_text)
                for (wn, q, meaning) in peaks:
                    peaks_rows.append({
                        "entry_id": entry_id,
                        "wavenumber_cm1": wn,
                        "qualifier": q or "",
                        "qualifier_meaning": meaning,
                        "parse_confidence": 1.0,
                    })

                # Save digitized curve rows
                for _, row in df_curve.iterrows():
                    curve_rows.append({
                        "entry_id": entry_id,
                        "wavenumber_cm1": float(row["wavenumber_cm1"]),
                        "transmittance": float(row["transmittance"]),
                        "segment_id": int(row.get("segment_id", 0)),
                        "imputed": bool(row.get("imputed", False)),
                    })

                # Save digitized plot image (from extracted data)
                digitized_plot_path = ""
                if save_digitized_plots:
                    try:
                        digitized_plot_path = str(digitized_dir / f"{entry_id}.png")
                        _save_digitized_plot_png(df_curve, Path(digitized_plot_path))
                    except Exception as e:
                        log("WARN", f"Page {pno+1} img {img_idx} spec {s_idx}: failed to save digitized plot: {e}")

                # Record entry metadata
                qc_notes = "; ".join([p for p in qc_notes_parts if p])[:400]
                entries_rows.append({
                    "entry_id": entry_id,
                    "page_index": pno,
                    "page_number_1based": pno+1,
                    "image_index": img_idx,
                    "spectrum_index": s_idx,
                    "label_ocr": label_text,
                    "label_page_offset": label_page_offset,
                    "entry_name": entry_name,
                    "entry_description": entry_description,
                    "description": entry_description,
                    "mineral_name": meta_md.get("mineral_name", ""),
                    "formula": meta_md.get("formula", ""),
                    "entry_text_raw": entry_text,
                    "wavenumbers_raw": w_raw,
                    "axis_break_present": bool(break_info.present),
                    "gap_lo_cm1": break_info.gap_lo,
                    "gap_hi_cm1": break_info.gap_hi,
                    "x_orientation": x_orientation,
                    "y_axis_type": y_mode,
                    "y_normalization": y_norm_mode,
                    "digitize_status": "ok",
                    "axis_filter_applied": axis_filter_applied,
                    "rolling_median_applied": rolling_median_applied,
                    "bin_width_cm1": collapse_info.get("collapse_bin_width"),
                    "collapsed_points": bool(collapse_info.get("collapsed_points", False)),
                    "collapse_bins": int(collapse_info.get("collapse_bins", len(df_curve))),
                    "collapse_bin_width_cm1": collapse_info.get("collapse_bin_width"),
                    "qc_flag": False,
                    "qc_notes": qc_notes,
                    "image_path": image_path,
                    "digitized_plot_path": digitized_plot_path,
                    **source_meta,
                })
                qc_rows.append({
                    "entry_id": entry_id,
                    "page_number_1based": pno+1,
                    "image_index": img_idx,
                    "spectrum_index": s_idx,
                    "stage": "digitize",
                    "status": "ok",
                    "y_normalization": y_norm_mode,
                    "notes": (
                        f"points={len(df_curve)} components={len(comps_for_label)} y_mode={y_mode} "
                        f"collapsed_points={collapse_info.get('collapsed_points', False)} "
                        f"collapse_bins={collapse_info.get('collapse_bins', len(df_curve))} "
                        f"axis_filter={axis_filter_applied} rolling_median={rolling_median_applied}"
                    ),
                })
    except Exception as e:
        log("ERROR", f"Page {pno+1}: exception: {e}")
        log("ERROR", traceback.format_exc())
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return {
        "page_index": pno,
        "entries_rows": entries_rows,
        "peaks_rows": peaks_rows,
        "qc_rows": qc_rows,
        "curve_rows": curve_rows,
        "logs": logs,
    }

def main() -> int:
    ap = argparse.ArgumentParser(description="Digitize FTIR spectra from a PDF into an Excel workbook (multiprocessing, transmittance).")
    ap.add_argument("--pdf", required=True, type=str, help="Input PDF path")
    ap.add_argument("--out", required=True, type=str, help="Output XLSX path")
    ap.add_argument("--pages", type=str, default=None, help='Page selection, e.g. "1-10,12,50-60" (1-based). Default: all.')
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Render DPI (default {DEFAULT_DPI})")
    ap.add_argument("--save-images", action="store_true", help="Save graph crop PNGs (including axes/ticks).")
    ap.add_argument("--image-dir", type=str, default=None, help="Directory for saved images (default: alongside xlsx).")
    ap.add_argument("--graph-dir", type=str, default=None, help="Directory for saved graph snapshot crops (default: <out>_graphs).")
    ap.add_argument("--digitized-plot-dir", type=str, default=None, help="Directory for saved plots generated from digitized data (default: <out>_digitized_plots).")
    ap.add_argument("--save-digitized-plots", action="store_true", help="Save plots generated from digitized data (in addition to graph crops).")
    ap.add_argument("--source-title", type=str, default=None, help="Source title override")
    ap.add_argument("--source-author", type=str, default=None, help="Source author override")
    ap.add_argument("--doi", type=str, default=None, help="DOI override")
    ap.add_argument("--isbn", type=str, default=None, help="ISBN override")
    ap.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract.exe (optional if on PATH).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of worker processes (1 disables multiprocessing).")
    ap.add_argument("--chunksize", type=int, default=1, help="Pool imap chunksize.")
    ap.add_argument("--max-images-per-page", type=int, default=50, help="Safety cap for images per page")
    ap.add_argument("--bin-representative", type=str, default="median", choices=["median", "top"], help="Representative y per bin (median or top).")
    ap.add_argument("--axis-filter-px", type=int, default=3, help="Pixel distance from axis lines to discard (0 disables).")
    ap.add_argument("--border-filter-px", type=int, default=3, help="Pixel distance from plot border to discard (0 disables).")
    ap.add_argument("--rolling-median-window", type=int, default=5, help="Window size for oscillation-triggered rolling median filter.")
    ap.add_argument("--label-lookahead-pages", type=int, default=1, help="Pages to scan ahead for labels when current page lacks them.")
    ap.add_argument("--per-spectrum-xlsx", dest="per_spectrum_xlsx", action="store_true", default=True, help="Write one XLSX per spectrum entry_id (default: enabled).")
    ap.add_argument("--no-per-spectrum-xlsx", dest="per_spectrum_xlsx", action="store_false", help="Disable per-spectrum XLSX output.")
    ap.add_argument("--per-spectrum-jdx", dest="per_spectrum_jdx", action="store_true", default=True, help="Write one JDX per spectrum entry_id (default: enabled).")
    ap.add_argument("--no-per-spectrum-jdx", dest="per_spectrum_jdx", action="store_false", help="Disable per-spectrum JDX output.")
    ap.add_argument("--out-jdx", type=str, default=None, help="Output path for a multi-spectrum JDX file (optional).")
    ap.add_argument("--spectrum-outdir", type=str, default=None, help="Directory for per-spectrum XLSX files (default: <out>_spectra_xlsx).")
    ap.add_argument("--spectrum-jdx-outdir", type=str, default=None, help="Directory for per-spectrum JDX files (default: <out>_spectra_jdx).")
    ap.add_argument("--no-master-xlsx", action="store_true", help="Do not write the combined master XLSX (only per-spectrum XLSX).")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    logger = setup_logger(args.verbose)

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return 2

    _set_tesseract_cmd(args.tesseract_cmd, logger)

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        logger.error("Tesseract OCR not available. Install Tesseract and ensure it's on PATH, or pass --tesseract-cmd.")
        logger.error(str(e))
        return 2

    logger.info(f"Opening PDF for page count: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    doc.close()

    pages = parse_pages_spec(args.pages, n_pages)
    logger.info(f"Pages selected: {len(pages)} / {n_pages}")

    sha = file_sha256(pdf_path)
    source_meta = infer_source_meta(pdf_path, {
        "source_title": args.source_title,
        "source_author": args.source_author,
        "doi": args.doi,
        "isbn": args.isbn,
    })
    # Output image directories
    out_stem = out_path.stem
    base_dir = out_path.parent
    graph_dir = Path(args.graph_dir).expanduser().resolve() if args.graph_dir else base_dir / (out_stem + "_graphs")
    digitized_dir = Path(args.digitized_plot_dir).expanduser().resolve() if args.digitized_plot_dir else base_dir / (out_stem + "_digitized_plots")
    if args.save_images:
        graph_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving graph snapshots to: {graph_dir}")
    # Save digitized plots if explicitly requested, or default-on when --save-images is used
    save_digitized_plots = bool(args.save_digitized_plots or args.save_images)
    if save_digitized_plots:
        digitized_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving digitized plots to: {digitized_dir}")

    spectrum_outdir = Path(args.spectrum_outdir).expanduser().resolve() if args.spectrum_outdir else base_dir / (out_stem + "_spectra_xlsx")
    spectrum_jdx_outdir = (
        Path(args.spectrum_jdx_outdir).expanduser().resolve()
        if args.spectrum_jdx_outdir
        else base_dir / (out_stem + "_spectra_jdx")
    )
    if args.per_spectrum_xlsx:
        spectrum_outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Per-spectrum XLSX output to: {spectrum_outdir}")
    if args.per_spectrum_jdx:
        spectrum_jdx_outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Per-spectrum JDX output to: {spectrum_jdx_outdir}")

    out_jdx_path = Path(args.out_jdx).expanduser().resolve() if args.out_jdx else None
    if out_jdx_path:
        logger.info(f"Master JDX output to: {out_jdx_path}")

    run_info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pdf_path": str(pdf_path),
        "pdf_sha256": sha,
        "pages_spec": args.pages or "",
        "dpi": args.dpi,
        "save_images": args.save_images,
        "save_digitized_plots": bool(save_digitized_plots),
        "workers": args.workers,
        "chunksize": args.chunksize,
        "bin_representative": args.bin_representative,
        "axis_filter_px": args.axis_filter_px,
        "border_filter_px": args.border_filter_px,
        "rolling_median_window": args.rolling_median_window,
        "label_lookahead_pages": args.label_lookahead_pages,
        **source_meta,
        "qualifiers": "s=strong band; w=weak band; sh=shoulder",
        "y_output": "transmittance (percent if calibrated; relative otherwise)",
        "x_output": "ascending (low -> high)",
        "script": Path(__file__).name if "__file__" in globals() else "pdfSpectraDigitizerFTIR-v6_from_v1_mp_transmittance.py",
    }
    run_df = pd.DataFrame([run_info])

    # Prepare tasks
    worker_args = []
    for pno in pages:
        worker_args.append((
            str(pdf_path), pno, args.dpi,
            bool(args.save_images), str(graph_dir) if args.save_images else "",
            bool(save_digitized_plots), str(digitized_dir) if save_digitized_plots else "",
            args.verbose, source_meta, args.max_images_per_page,
            args.bin_representative, args.axis_filter_px, args.border_filter_px, args.rolling_median_window,
            args.label_lookahead_pages,
        ))

    entries_rows: List[Dict[str, Any]] = []
    peaks_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    written_entry_ids: set = set()
    per_spectrum_jdx_written = 0
    per_spectrum_jdx_skipped = 0
    master_jdx_written = False

    if args.workers <= 1:
        logger.info("Running in single-process mode (--workers 1).")
        for a in tqdm(worker_args, desc="Pages", unit="page"):
            res = _process_page_worker(a)
            for lvl, msg in res.get("logs", []):
                if lvl == "ERROR":
                    logger.error(msg)
                elif lvl == "WARN":
                    logger.warning(msg)
                elif lvl == "DEBUG":
                    logger.debug(msg)
                else:
                    logger.info(msg)
            page_entries = res.get("entries_rows", [])
            page_peaks = res.get("peaks_rows", [])
            page_qc = res.get("qc_rows", [])
            page_curve = res.get("curve_rows", [])
            entries_rows.extend(page_entries)
            peaks_rows.extend(page_peaks)
            qc_rows.extend(page_qc)
            curve_rows.extend(page_curve)

            if args.per_spectrum_xlsx or args.per_spectrum_jdx:
                by_entry: Dict[str, Dict[str, Any]] = {}
                for er in page_entries:
                    eid = er.get("entry_id", "")
                    if not eid or er.get("digitize_status") != "ok":
                        continue
                    if eid in written_entry_ids:
                        continue
                    by_entry[eid] = er
                if by_entry:
                    peaks_by: Dict[str, List[Dict[str, Any]]] = {}
                    for pr in page_peaks:
                        peaks_by.setdefault(pr.get("entry_id", ""), []).append(pr)
                    qc_by: Dict[str, List[Dict[str, Any]]] = {}
                    for qr in page_qc:
                        qc_by.setdefault(qr.get("entry_id", ""), []).append(qr)
                    curve_by: Dict[str, List[Dict[str, Any]]] = {}
                    for cr in page_curve:
                        curve_by.setdefault(cr.get("entry_id", ""), []).append(cr)
                    for eid, er in by_entry.items():
                        label = sanitize_filename(er.get("label_ocr", ""))
                        base = f"{eid}__{label}" if label else eid
                        if args.per_spectrum_xlsx:
                            out_xlsx = spectrum_outdir / (base + ".xlsx")
                            write_single_spectrum_xlsx(out_xlsx, run_df, er, curve_by.get(eid, []), peaks_by.get(eid, []), qc_by.get(eid, []), logger)
                        if args.per_spectrum_jdx:
                            out_jdx = spectrum_jdx_outdir / (base + ".jdx")
                            if write_single_spectrum_jdx(out_jdx, er, curve_by.get(eid, []), logger):
                                per_spectrum_jdx_written += 1
                            else:
                                per_spectrum_jdx_skipped += 1
                        written_entry_ids.add(eid)
    else:
        logger.info(f"Running multiprocessing: workers={args.workers} chunksize={args.chunksize}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            it = pool.imap_unordered(_process_page_worker, worker_args, chunksize=args.chunksize)
            for res in tqdm(it, total=len(worker_args), desc="Pages", unit="page"):
                for lvl, msg in res.get("logs", []):
                    if lvl == "ERROR":
                        logger.error(msg)
                    elif lvl == "WARN":
                        logger.warning(msg)
                    else:
                        logger.info(msg)
                page_entries = res.get("entries_rows", [])
                page_peaks = res.get("peaks_rows", [])
                page_qc = res.get("qc_rows", [])
                page_curve = res.get("curve_rows", [])
                entries_rows.extend(page_entries)
                peaks_rows.extend(page_peaks)
                qc_rows.extend(page_qc)
                curve_rows.extend(page_curve)

                if args.per_spectrum_xlsx or args.per_spectrum_jdx:
                    by_entry: Dict[str, Dict[str, Any]] = {}
                    for er in page_entries:
                        eid = er.get("entry_id", "")
                        if not eid or er.get("digitize_status") != "ok":
                            continue
                        if eid in written_entry_ids:
                            continue
                        by_entry[eid] = er
                    if by_entry:
                        peaks_by: Dict[str, List[Dict[str, Any]]] = {}
                        for pr in page_peaks:
                            peaks_by.setdefault(pr.get("entry_id", ""), []).append(pr)
                        qc_by: Dict[str, List[Dict[str, Any]]] = {}
                        for qr in page_qc:
                            qc_by.setdefault(qr.get("entry_id", ""), []).append(qr)
                        curve_by: Dict[str, List[Dict[str, Any]]] = {}
                        for cr in page_curve:
                            curve_by.setdefault(cr.get("entry_id", ""), []).append(cr)
                        for eid, er in by_entry.items():
                            label = sanitize_filename(er.get("label_ocr", ""))
                            base = f"{eid}__{label}" if label else eid
                            if args.per_spectrum_xlsx:
                                out_xlsx = spectrum_outdir / (base + ".xlsx")
                                write_single_spectrum_xlsx(out_xlsx, run_df, er, curve_by.get(eid, []), peaks_by.get(eid, []), qc_by.get(eid, []), logger)
                            if args.per_spectrum_jdx:
                                out_jdx = spectrum_jdx_outdir / (base + ".jdx")
                                if write_single_spectrum_jdx(out_jdx, er, curve_by.get(eid, []), logger):
                                    per_spectrum_jdx_written += 1
                                else:
                                    per_spectrum_jdx_skipped += 1
                            written_entry_ids.add(eid)

    entries_df = pd.DataFrame(entries_rows)
    peaks_df = pd.DataFrame(peaks_rows)
    qc_df = pd.DataFrame(qc_rows)
    curve_df = pd.DataFrame(curve_rows)

    # Ensure curve columns
    if curve_df.empty:
        curve_df = pd.DataFrame(columns=["entry_id", "wavenumber_cm1", "transmittance", "segment_id", "imputed"])
    else:
        curve_df = curve_df.sort_values(["entry_id", "wavenumber_cm1"]).reset_index(drop=True)

    if not entries_df.empty:
        entries_df = entries_df.sort_values(["page_index", "image_index", "spectrum_index"]).reset_index(drop=True)

    sheets = {
        "RunInfo": run_df,
        "Entries": entries_df,
        "CurvePoints": curve_df,
        "PeaksText": peaks_df,
        "QC": qc_df,
    }

    if not args.no_master_xlsx:
        logger.info(f"Writing XLSX: {out_path}")
        write_excel_with_splitting(out_path, sheets, logger)
    else:
        logger.info("Skipping master XLSX (--no-master-xlsx).")

    if out_jdx_path:
        entries_by_id = {
            er["entry_id"]: er
            for er in entries_rows
            if er.get("entry_id") and er.get("digitize_status") == "ok"
        }
        curve_by: Dict[str, List[Dict[str, Any]]] = {}
        for cr in curve_rows:
            curve_by.setdefault(cr.get("entry_id", ""), []).append(cr)
        master_jdx_written = write_multi_spectrum_jdx(out_jdx_path, entries_by_id, curve_by, logger)
    if args.per_spectrum_jdx or out_jdx_path:
        summary_parts = []
        if args.per_spectrum_jdx:
            summary_parts.append(f"per-spectrum wrote {per_spectrum_jdx_written}, skipped {per_spectrum_jdx_skipped}")
        if out_jdx_path:
            summary_parts.append(f"master {'wrote 1' if master_jdx_written else 'skipped'}")
        logger.info(f"JDX summary: {', '.join(summary_parts)}")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
