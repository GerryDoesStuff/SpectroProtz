from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Tuple
import numpy as np

@dataclass
class Spectrum:
    wavelength: np.ndarray          # or wavenumber for IR/Raman
    intensity: np.ndarray           # absorbance/reflectance/counts
    meta: Dict[str, Any]

@dataclass
class BatchResult:
    processed: List[Spectrum]
    qc_table: List[Dict[str, Any]]
    figures: Dict[str, bytes]       # PNG/SVG bytes
    audit: List[str]
    report_text: Optional[str] = None

class SpectroscopyPlugin:
    id: str = "base"
    label: str = "Base"
    xlabel: str = "x"

    def detect(self, paths: Iterable[str]) -> bool:
        return False

    def load(self, paths: Iterable[str]) -> List[Spectrum]:
        raise NotImplementedError

    def validate(self, specs: List[Spectrum], recipe: Dict[str, Any]) -> List[str]:
        return []

    def preprocess(self, specs: List[Spectrum], recipe: Dict[str, Any]) -> List[Spectrum]:
        return specs

    def analyze(self, specs: List[Spectrum], recipe: Dict[str, Any]) -> Tuple[List[Spectrum], List[Dict[str, Any]]]:
        return specs, []

    def export(self, specs: List[Spectrum], qc: List[Dict[str, Any]], recipe: Dict[str, Any]) -> BatchResult:
        return BatchResult(processed=specs, qc_table=qc, figures={}, audit=[])
