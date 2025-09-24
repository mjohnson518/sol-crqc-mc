"""
Simple exponential growth model extrapolating CRQC emergence from historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class ExponentialGrowthModel:
    """Extrapolates CRQC emergence using exponential growth assumptions."""

    growth_rate: float = 1.45
    current_qubits: int = 433
    required_qubits: int = 1631

    def predict(self) -> Dict[str, Any]:
        years_needed = np.log(self.required_qubits / self.current_qubits) / np.log(self.growth_rate)
        median_year = 2025 + years_needed

        ci_width = max(2.0, years_needed * 0.3)
        ci_95 = (median_year - ci_width, median_year + ci_width)

        return {
            "median": float(median_year),
            "ci_95": ci_95,
        }
