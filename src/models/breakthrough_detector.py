"""
Breakthrough detection system for early warning of quantum technology leaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque
import math


@dataclass
class SignalState:
    """Tracks the latest value and z-score for a breakthrough signal."""

    name: str
    value: float
    mean: float
    std: float
    z_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "value": self.value,
            "mean": self.mean,
            "std": self.std,
            "z_score": self.z_score,
        }


@dataclass
class BreakthroughAssessment:
    """Result of evaluating breakthrough risk."""

    composite_score: float
    level: str
    signal_states: List[SignalState]
    alerts: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "composite_score": self.composite_score,
            "level": self.level,
            "signal_states": [state.to_dict() for state in self.signal_states],
            "alerts": list(self.alerts),
        }


class BreakthroughDetector:
    """Detects potential breakthrough activity using multiple signals."""

    SIGNAL_WEIGHTS: Dict[str, float] = {
        "paper_velocity": 0.20,
        "citation_burst": 0.25,
        "patent_clusters": 0.15,
        "researcher_movement": 0.20,
        "conference_buzz": 0.10,
        "investment_spike": 0.10,
    }

    THRESHOLDS = {
        "low": 1.5,
        "medium": 2.5,
        "high": 3.5,
    }

    BASELINES = {
        "paper_velocity": (8.0, 3.0),
        "citation_burst": (5.0, 2.0),
        "patent_clusters": (2.0, 1.0),
        "researcher_movement": (3.0, 1.5),
        "conference_buzz": (2.0, 1.0),
        "investment_spike": (150.0, 75.0),  # Millions USD
    }

    HISTORY_LENGTH = 30

    def __init__(self) -> None:
        self.history: Dict[str, deque] = {
            key: deque(maxlen=self.HISTORY_LENGTH)
            for key in self.SIGNAL_WEIGHTS
        }
        self._last_assessment: BreakthroughAssessment | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_signals(self, metrics: Dict[str, float]) -> BreakthroughAssessment:
        signal_states: List[SignalState] = []

        for signal, weight in self.SIGNAL_WEIGHTS.items():
            raw_value = float(metrics.get(signal, 0.0))
            mean, std = self._estimate_baseline(signal)
            z_score = self._compute_z_score(raw_value, mean, std)
            state = SignalState(signal, raw_value, mean, std, z_score)
            signal_states.append(state)
            self.history[signal].append(raw_value)

        composite_score = self._composite_score(signal_states)
        level = self._classify_level(composite_score)
        alerts = self._generate_alerts(signal_states, composite_score)

        assessment = BreakthroughAssessment(
            composite_score=composite_score,
            level=level,
            signal_states=signal_states,
            alerts=alerts,
        )
        self._last_assessment = assessment
        return assessment

    def latest_assessment(self) -> BreakthroughAssessment | None:
        return self._last_assessment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _estimate_baseline(self, signal: str) -> Tuple[float, float]:
        history = self.history[signal]
        if len(history) >= 5:
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / max(len(history) - 1, 1)
            std = math.sqrt(max(variance, 1e-6))
            return mean, std

        baseline = self.BASELINES.get(signal, (1.0, 0.5))
        return baseline

    def _compute_z_score(self, value: float, mean: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return (value - mean) / (std + 1e-6)

    def _composite_score(self, signal_states: List[SignalState]) -> float:
        total_weight = sum(self.SIGNAL_WEIGHTS.values()) or 1.0
        weighted_sum = 0.0
        for state in signal_states:
            weight = self.SIGNAL_WEIGHTS[state.name]
            weighted_sum += weight * state.z_score
        return weighted_sum / total_weight

    def _classify_level(self, score: float) -> str:
        if score >= self.THRESHOLDS["high"]:
            return "high"
        if score >= self.THRESHOLDS["medium"]:
            return "medium"
        if score >= self.THRESHOLDS["low"]:
            return "low"
        return "baseline"

    def _generate_alerts(self, signal_states: List[SignalState], score: float) -> List[str]:
        alerts: List[str] = []

        for state in signal_states:
            if state.z_score >= self.THRESHOLDS["medium"]:
                alerts.append(
                    f"{state.name.replace('_', ' ').title()} anomaly (z={state.z_score:.2f})"
                )

        level = self._classify_level(score)
        if level in {"medium", "high"}:
            alerts.insert(0, f"Breakthrough risk {level.upper()} (score={score:.2f})")

        return alerts
