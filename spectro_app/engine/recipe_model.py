from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Recipe:
    module: str = "uvvis"
    params: Dict[str, Any] = field(default_factory=dict)
    version: str = "0.1.0"

    def validate(self) -> list[str]:
        errs = []
        # Example validation
        s = self.params.get("smoothing", {})
        win = s.get("window", 15)
        poly = s.get("polyorder", 3)
        if win and win <= poly*2:
            errs.append("Savitzky–Golay window must be > 2*polyorder")
        return errs
