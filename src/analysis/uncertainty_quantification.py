"""
Advanced uncertainty quantification utilities for CRQC prediction models.

This module separates and quantifies different types of uncertainty:
- Aleatory: inherent randomness in quantum progress
- Epistemic: knowledge limitations and data sparsity
- Model: structural uncertainty in models
- Deep: scenarios representing unknown unknowns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Optional, Any
from math import isfinite
import numpy as np


@dataclass
class UncertaintyInterval:
    """Represents a confidence or credibility interval."""

    level: float
    lower: float
    upper: float
    width: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "level": self.level,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
        }


@dataclass
class UncertaintyComponent:
    """Breakdown of uncertainty contributions."""

    name: str
    value: float
    contribution: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "contribution": self.contribution,
            "description": self.description,
        }


@dataclass
class ScenarioImpact:
    """Impact of stress-test scenarios on the prediction."""

    name: str
    median_shift: float
    confidence_shift: float
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "median_shift": self.median_shift,
            "confidence_shift": self.confidence_shift,
            "notes": self.notes,
        }


@dataclass
class UncertaintyReport:
    """Full report of uncertainty analysis results."""

    intervals: List[UncertaintyInterval] = field(default_factory=list)
    components: List[UncertaintyComponent] = field(default_factory=list)
    scenarios: List[ScenarioImpact] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervals": [interval.to_dict() for interval in self.intervals],
            "components": [comp.to_dict() for comp in self.components],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "notes": list(self.notes),
        }


class UncertaintyAnalysis:
    """Compute advanced uncertainty metrics for CRQC predictions."""

    DEFAULT_CONFIDENCE_LEVELS = (0.50, 0.80, 0.95)

    def __init__(self, confidence_levels: Sequence[float] | None = None) -> None:
        self.confidence_levels = tuple(confidence_levels or self.DEFAULT_CONFIDENCE_LEVELS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        model_predictions: Dict[str, Dict[str, Any]],
        ensemble_prediction: Any,
    ) -> UncertaintyReport:
        """Generate a comprehensive uncertainty report."""

        report = UncertaintyReport()

        intervals = self._compute_intervals(ensemble_prediction.probability_by_year)
        report.intervals.extend(intervals)

        components = self._compute_components(model_predictions, ensemble_prediction, intervals)
        report.components.extend(components)

        scenario_impacts, notes = self._stress_test_scenarios(ensemble_prediction, model_predictions)
        report.scenarios.extend(scenario_impacts)
        report.notes.extend(notes)

        return report

    # ------------------------------------------------------------------
    # Interval calculations
    # ------------------------------------------------------------------
    def _compute_intervals(self, probability_by_year: Dict[int, float]) -> List[UncertaintyInterval]:
        if not probability_by_year:
            return []

        years, probabilities = self._sanitize_distribution(probability_by_year)
        intervals = []

        for level in self.confidence_levels:
            lower_quantile = (1 - level) / 2
            upper_quantile = 1 - lower_quantile

            lower = self._quantile(years, probabilities, lower_quantile)
            upper = self._quantile(years, probabilities, upper_quantile)

            if lower is None or upper is None:
                continue

            intervals.append(
                UncertaintyInterval(
                    level=level,
                    lower=lower,
                    upper=upper,
                    width=upper - lower,
                )
            )

        return intervals

    def _sanitize_distribution(self, distribution: Dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        sorted_items = sorted(distribution.items())
        years = np.array([item[0] for item in sorted_items], dtype=float)
        probabilities = np.array([item[1] for item in sorted_items], dtype=float)

        probabilities = np.clip(probabilities, 0.0, 1.0)
        if probabilities[-1] < 1.0:
            probabilities[-1] = 1.0
        if probabilities[0] > 0.0:
            probabilities[0] = 0.0

        return years, probabilities

    def _quantile(self, years: np.ndarray, probabilities: np.ndarray, q: float) -> Optional[float]:
        if q <= 0:
            return float(years[0])
        if q >= 1:
            return float(years[-1])

        indices = np.searchsorted(probabilities, q, side="left")
        idx = min(indices, len(probabilities) - 1)

        if probabilities[idx] == q or idx == 0:
            return float(years[idx])

        prev_idx = idx - 1
        p0, p1 = probabilities[prev_idx], probabilities[idx]
        y0, y1 = years[prev_idx], years[idx]

        if p1 == p0:
            return float(y1)

        fraction = (q - p0) / (p1 - p0)
        return float(y0 + fraction * (y1 - y0))

    # ------------------------------------------------------------------
    # Component breakdown
    # ------------------------------------------------------------------
    def _compute_components(
        self,
        model_predictions: Dict[str, Dict[str, Any]],
        ensemble_prediction: Any,
        intervals: Sequence[UncertaintyInterval],
    ) -> List[UncertaintyComponent]:
        components: List[UncertaintyComponent] = []

        if not model_predictions:
            return components

        medians = []
        interval_widths = []

        for prediction in model_predictions.values():
            median = prediction.get("median_year") or prediction.get("median")
            if median is None:
                continue
            medians.append(median)

            prob_map = prediction.get("probability_by_year")
            if prob_map:
                individual_intervals = self._compute_intervals(prob_map)
                width_95 = next((ival.width for ival in individual_intervals if abs(ival.level - 0.95) < 1e-6), None)
                if width_95 is not None and isfinite(width_95):
                    interval_widths.append(width_95)

        if len(medians) >= 2:
            epistemic = float(np.std(medians))
        else:
            epistemic = 0.0

        aleatory = float(np.mean(interval_widths)) if interval_widths else 0.0

        ensemble_95 = next((ival for ival in intervals if abs(ival.level - 0.95) < 1e-6), None)
        ensemble_width = ensemble_95.width if ensemble_95 else aleatory + epistemic

        model_uncertainty = max(ensemble_width - aleatory, 0.0)
        deep_uncertainty = max(ensemble_width - (aleatory + epistemic + model_uncertainty), 0.0)

        raw_values = [aleatory, epistemic, model_uncertainty, deep_uncertainty]
        total = sum(raw_values) or 1.0
        contributions = [value / total for value in raw_values]

        component_info = [
            ("aleatory", aleatory, contributions[0], "Natural randomness in quantum progress trajectories."),
            ("epistemic", epistemic, contributions[1], "Knowledge limitations and data sparsity across technologies."),
            ("model", model_uncertainty, contributions[2], "Structural and methodological uncertainty between models."),
            ("deep", deep_uncertainty, contributions[3], "Hard-to-quantify factors captured via scenario stress tests."),
        ]

        for name, value, contribution, description in component_info:
            components.append(
                UncertaintyComponent(
                    name=name,
                    value=value,
                    contribution=contribution,
                    description=description,
                )
            )

        return components

    # ------------------------------------------------------------------
    # Scenario stress testing
    # ------------------------------------------------------------------
    def _stress_test_scenarios(
        self,
        ensemble_prediction: Any,
        model_predictions: Dict[str, Dict[str, Any]],
    ) -> tuple[List[ScenarioImpact], List[str]]:
        scenarios: List[ScenarioImpact] = []
        notes: List[str] = []

        base_median = ensemble_prediction.median_year
        base_confidence = ensemble_prediction.confidence_score if hasattr(ensemble_prediction, "confidence_score") else 0.0

        scenario_defs = {
            "quantum_winter": {
                "shift": +6.0,
                "confidence_shift": -0.10,
                "notes": "Fundamental barriers or funding cuts slow progress considerably.",
            },
            "breakthrough": {
                "shift": -5.0,
                "confidence_shift": +0.05,
                "notes": "Major algorithmic breakthrough or hardware leap accelerates CRQC.",
            },
            "china_surprise": {
                "shift": -3.0,
                "confidence_shift": +0.08,
                "notes": "Unannounced national-level advance reveals sudden capability leap.",
            },
            "silicon_photonics": {
                "shift": -2.5,
                "confidence_shift": +0.04,
                "notes": "Scalability breakthrough in photonic or integrated platforms.",
            },
        }

        for name, cfg in scenario_defs.items():
            scenarios.append(
                ScenarioImpact(
                    name=name,
                    median_shift=cfg["shift"],
                    confidence_shift=cfg["confidence_shift"],
                    notes=cfg["notes"],
                )
            )

        disagreement = self._model_disagreement(model_predictions)
        if disagreement is not None:
            notes.append(
                f"Model disagreement span (min/max median): {disagreement['min']:.1f} – {disagreement['max']:.1f}"
            )

        notes.append(
            f"Scenario-adjusted range: {base_median + min(cfg['shift'] for cfg in scenario_defs.values()):.1f} – "
            f"{base_median + max(cfg['shift'] for cfg in scenario_defs.values()):.1f}"
        )
        notes.append(
            f"Confidence sensitivity: {base_confidence + min(cfg['confidence_shift'] for cfg in scenario_defs.values()):.1%} – "
            f"{base_confidence + max(cfg['confidence_shift'] for cfg in scenario_defs.values()):.1%}"
        )

        return scenarios, notes

    def _model_disagreement(self, predictions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
        medians = []
        for pred in predictions.values():
            median = pred.get("median_year") or pred.get("median")
            if median is not None:
                medians.append(float(median))

        if not medians:
            return None

        return {
            "min": min(medians),
            "max": max(medians),
            "span": max(medians) - min(medians),
        }
