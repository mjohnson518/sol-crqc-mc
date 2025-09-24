"""
Geopolitical and economic factors model for CRQC emergence predictions.

This module models how national investments, competition, cooperation,
export controls, and economic indicators influence the timeline for
cryptographically relevant quantum computers (CRQC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

import numpy as np


@dataclass
class NationalProfile:
    """Represents a nation's investment profile."""

    name: str
    annual_investment: float  # USD per year
    growth_rate: float  # multiplicative factor (>1 means growing)
    baseline_capability: float  # normalized capability score
    collaboration_factor: float  # 0-1, higher when more cooperative
    export_control_exposure: float  # 0-1, higher when restricted


class GeopoliticalModel:
    """Quantify geopolitical and economic factors affecting CRQC timelines."""

    def __init__(self) -> None:
        self.national_profiles: Dict[str, NationalProfile] = {
            "usa": NationalProfile("USA", 1.2e9, 1.15, 0.85, 0.7, 0.4),
            "china": NationalProfile("China", 3.0e9, 1.25, 0.90, 0.3, 0.2),
            "eu": NationalProfile("European Union", 1.0e9, 1.10, 0.70, 0.8, 0.5),
            "uk": NationalProfile("United Kingdom", 0.5e9, 1.08, 0.65, 0.6, 0.4),
            "canada": NationalProfile("Canada", 0.36e9, 1.12, 0.55, 0.7, 0.3),
            "japan": NationalProfile("Japan", 0.5e9, 1.10, 0.60, 0.75, 0.4),
            "australia": NationalProfile("Australia", 0.1e9, 1.20, 0.45, 0.65, 0.3),
        }

        self.competition_factors = {
            "arms_race_acceleration": 1.30,
            "collaboration_efficiency": 0.80,
            "export_control_impact": 0.70,
        }

        self._cached_metrics: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_geopolitical_adjustment(self, years: int = 10) -> Dict[str, float]:
        """Compute adjustments to CRQC timeline based on geopolitical dynamics."""

        investment_growth = self._aggregate_investment_growth(years)
        competition_score = self._compute_competition_pressure()
        cooperation_score = self._compute_cooperation_benefit()
        export_penalty = self._compute_export_controls_penalty()

        acceleration = self._investment_to_progress(investment_growth)
        acceleration *= (1 + competition_score)
        acceleration *= max(0.1, cooperation_score)
        acceleration *= max(0.1, 1 - export_penalty)

        timeline_adjustment_years = self._progress_to_time_shift(acceleration)

        self._cached_metrics = {
            "investment_growth": investment_growth,
            "competition_score": competition_score,
            "cooperation_score": cooperation_score,
            "export_penalty": export_penalty,
            "acceleration_factor": acceleration,
            "timeline_adjustment": timeline_adjustment_years,
        }

        return self._cached_metrics

    def get_cached_metrics(self) -> Optional[Dict[str, float]]:
        """Return cached metrics from the last computation."""

        return self._cached_metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_investment_growth(self, years: int) -> float:
        cumulative = 0.0
        for profile in self.national_profiles.values():
            growth_multiplier = (profile.growth_rate ** years - 1) / math.log(profile.growth_rate)
            cumulative += profile.annual_investment * max(growth_multiplier, 1.0)
        return cumulative

    def _compute_competition_pressure(self) -> float:
        capability_scores = [profile.baseline_capability for profile in self.national_profiles.values()]
        variance = float(np.var(capability_scores))
        normalized_variance = min(variance / 0.05, 1.0)
        return normalized_variance * (self.competition_factors["arms_race_acceleration"] - 1)

    def _compute_cooperation_benefit(self) -> float:
        collaboration_levels = [profile.collaboration_factor for profile in self.national_profiles.values()]
        mean_collaboration = np.mean(collaboration_levels)
        return max(0.2, min(1.0, mean_collaboration * self.competition_factors["collaboration_efficiency"]))

    def _compute_export_controls_penalty(self) -> float:
        exposures = [profile.export_control_exposure for profile in self.national_profiles.values()]
        mean_exposure = np.mean(exposures)
        return min(1.0, mean_exposure * self.competition_factors["export_control_impact"])

    def _investment_to_progress(self, cumulative_investment: float) -> float:
        alpha = 0.12
        beta = 0.75
        log_investment = math.log(max(cumulative_investment, 1.0))
        return max(0.1, alpha * log_investment + beta)

    def _progress_to_time_shift(self, acceleration_factor: float) -> float:
        reference_acceleration = 1.8
        shift = reference_acceleration - acceleration_factor
        return float(np.clip(shift, -5.0, 5.0))

    def summarize(self) -> Dict[str, Any]:
        if self._cached_metrics is None:
            self.compute_geopolitical_adjustment()
        return dict(self._cached_metrics or {})
