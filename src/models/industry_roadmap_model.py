"""
Industry roadmap consensus model for CRQC emergence predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class IndustryRoadmapModel:
    """Combines industry roadmap projections into a single estimate."""

    roadmaps: Dict[str, Dict[int, int]] = None

    def __post_init__(self) -> None:
        if self.roadmaps is None:
            self.roadmaps = {
                "IBM": {2025: 1386, 2026: 2000, 2027: 4000, 2028: 10000},
                "Google": {2025: 1000, 2026: 2000, 2027: 4000, 2028: 8000},
                "IonQ": {2025: 64, 2026: 128, 2027: 256, 2028: 512},
            }

    def predict(self) -> Dict[str, Any]:
        medians = []
        ci_bounds = []

        for name, roadmap in self.roadmaps.items():
            years = sorted(roadmap.keys())
            qubits = np.array([roadmap[year] for year in years], dtype=float)
            try:
                idx = np.where(qubits >= 1631)[0][0]
                median_year = years[idx]
            except IndexError:
                median_year = years[-1] + 2
            medians.append(median_year)
            ci_bounds.append((years[0], years[-1] + 2))

        median = float(np.median(medians))
        ci_low = float(np.percentile([low for low, _ in ci_bounds], 25))
        ci_high = float(np.percentile([high for _, high in ci_bounds], 75))

        return {
            "median": median,
            "ci_95": (ci_low, ci_high),
        }
