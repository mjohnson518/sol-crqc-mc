"""
Ensemble prediction utilities for combining multiple CRQC models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class EnsembleMemberResult:
    name: str
    median: float
    ci_95: tuple[float, float]
    weight: float


class EnsembleCRQC:
    """Combine multiple model predictions with uncertainty propagation."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self.weights = weights

    def combine_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        results: List[EnsembleMemberResult] = []
        for name, info in predictions.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            median = info.get("median")
            ci_95 = info.get("ci_95")
            if median is None or ci_95 is None:
                continue
            results.append(EnsembleMemberResult(name, float(median), tuple(ci_95), weight))

        if not results:
            raise ValueError("No valid predictions supplied to EnsembleCRQC")

        total_weight = sum(member.weight for member in results)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        normalized_weights = [member.weight / total_weight for member in results]

        weighted_median = float(
            np.dot([member.median for member in results], normalized_weights)
        )

        lower_bounds = [member.ci_95[0] for member in results]
        upper_bounds = [member.ci_95[1] for member in results]
        weighted_lower = float(np.dot(lower_bounds, normalized_weights))
        weighted_upper = float(np.dot(upper_bounds, normalized_weights))

        return {
            "median": weighted_median,
            "ci_95": (weighted_lower, weighted_upper),
            "members": [member.__dict__ for member in results],
        }
