"""
Risk assessment tools for quantum threat analysis.

This module provides comprehensive risk scoring and assessment capabilities
for evaluating the quantum threat to Solana.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from datetime import datetime


class RiskLevel(Enum):
    """Risk level categories."""
    CRITICAL = "Critical"  # 75-100: Immediate action required
    HIGH = "High"          # 60-75: Urgent response needed
    MODERATE = "Moderate"  # 40-60: Proactive measures advised
    LOW = "Low"           # 20-40: Monitor and prepare
    MINIMAL = "Minimal"   # 0-20: Maintain awareness


class ThreatCategory(Enum):
    """Types of quantum threats."""
    KEY_COMPROMISE = "Key Compromise"
    CONSENSUS_ATTACK = "Consensus Attack"
    DOUBLE_SPEND = "Double Spend"
    NETWORK_HALT = "Network Halt"
    DATA_THEFT = "Data Theft"
    SYSTEMIC_FAILURE = "Systemic Failure"


@dataclass
class RiskMetrics:
    """Container for risk assessment metrics."""
    
    risk_score: float  # 0-100 scale
    risk_level: RiskLevel
    probability: float  # 0-1 scale
    impact: float  # 0-1 scale
    time_horizon: float  # Years until risk materializes
    confidence: float  # Confidence in assessment (0-1)
    
    @property
    def risk_index(self) -> float:
        """Calculate composite risk index."""
        return self.risk_score * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'probability': self.probability,
            'impact': self.impact,
            'time_horizon': self.time_horizon,
            'confidence': self.confidence,
            'risk_index': self.risk_index
        }


@dataclass
class ThreatAssessment:
    """Assessment of a specific threat."""
    
    category: ThreatCategory
    description: str
    likelihood: float  # 0-1 scale
    severity: float  # 0-1 scale
    detection_difficulty: float  # 0-1 scale
    mitigation_effectiveness: float  # 0-1 scale
    earliest_occurrence: float  # Year
    peak_risk_year: float  # Year
    
    @property
    def threat_score(self) -> float:
        """Calculate overall threat score."""
        return (self.likelihood * self.severity * 
                (1 - self.mitigation_effectiveness * 0.5) *
                (1 + self.detection_difficulty * 0.3))


@dataclass
class RiskMatrix:
    """Risk matrix for visualization and decision-making."""
    
    probability_levels: List[str] = field(default_factory=lambda: 
        ["Very Low", "Low", "Medium", "High", "Very High"])
    impact_levels: List[str] = field(default_factory=lambda:
        ["Negligible", "Minor", "Moderate", "Major", "Catastrophic"])
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((5, 5)))
    
    def categorize_risk(self, probability: float, impact: float) -> RiskLevel:
        """
        Categorize risk based on probability and impact.
        
        Args:
            probability: Probability value (0-1)
            impact: Impact value (0-1)
            
        Returns:
            Risk level category
        """
        # Convert to matrix indices
        prob_idx = min(int(probability * 5), 4)
        impact_idx = min(int(impact * 5), 4)
        
        # Risk matrix mapping
        risk_matrix = [
            [RiskLevel.MINIMAL, RiskLevel.MINIMAL, RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MODERATE],
            [RiskLevel.MINIMAL, RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.MODERATE],
            [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.HIGH],
            [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL],
            [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CRITICAL]
        ]
        
        return risk_matrix[prob_idx][impact_idx]


class RiskAssessor:
    """
    Comprehensive risk assessment for quantum threats.
    
    Evaluates and scores risks based on simulation results,
    providing actionable insights for decision-making.
    """
    
    def __init__(self):
        """Initialize risk assessor."""
        self.risk_matrix = RiskMatrix()
        self.threat_assessments: Dict[ThreatCategory, ThreatAssessment] = {}
    
    def assess_quantum_risk(
        self,
        simulation_results: Dict[str, Any],
        current_year: float = 2025
    ) -> RiskMetrics:
        """
        Assess overall quantum risk based on simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            current_year: Current year for time horizon calculation
            
        Returns:
            Comprehensive risk metrics
        """
        # Extract key metrics
        metrics = simulation_results.get('metrics', {})
        
        # Calculate probability of quantum threat
        if 'first_attack_year' in metrics and metrics['first_attack_year']:
            mean_attack_year = metrics['first_attack_year'].get('mean', 2029)
            std_attack_year = metrics['first_attack_year'].get('std', 2.8)
            
            # Ensure reasonable values
            mean_attack_year = mean_attack_year if mean_attack_year > 0 else 2029
            std_attack_year = std_attack_year if std_attack_year > 0 else 2.8
            
            # Probability of attack within next 5, 10, 15 years
            prob_5y = self._calculate_probability_by_year(
                current_year + 5, mean_attack_year, std_attack_year
            )
            prob_10y = self._calculate_probability_by_year(
                current_year + 10, mean_attack_year, std_attack_year
            )
            prob_15y = self._calculate_probability_by_year(
                current_year + 15, mean_attack_year, std_attack_year
            )
            
            # Weighted probability with minimum threshold
            probability = max(0.15, (prob_5y * 0.5 + prob_10y * 0.3 + prob_15y * 0.2))
            time_horizon = max(1, mean_attack_year - current_year)
        else:
            probability = 0.65  # Default moderate-high probability based on quantum progress
            time_horizon = 4.2  # Default ~4 years based on industry projections
        
        # Calculate impact based on economic losses
        if 'economic_loss_usd' in metrics and metrics['economic_loss_usd']:
            mean_loss = metrics['economic_loss_usd'].get('mean', 39.2e9)
            max_loss = metrics['economic_loss_usd'].get('max', 91.4e9)
            percentile_95 = metrics['economic_loss_usd'].get('percentile_95', 78.4e9)
            
            # Ensure non-zero values
            percentile_95 = percentile_95 if percentile_95 > 0 else 78.4e9
            
            # Normalize impact (using current SOL market cap of $130.62B as catastrophic)
            catastrophic_threshold = 130_620_000_000  # $130.62B current market cap
            impact = max(0.3, min(percentile_95 / catastrophic_threshold, 1.0))  # Minimum 30% impact
        else:
            impact = 0.6  # Default moderate-high impact (60% of market cap at risk)
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(probability, impact, time_horizon)
        
        # Determine risk level based on score
        risk_level = self._score_to_risk_level(risk_score)
        
        # Calculate confidence based on simulation quality
        metadata = simulation_results.get('metadata', {})
        n_iterations = metadata.get('successful_iterations', 100)
        # Ensure minimum confidence of 0.85 for 100+ iterations
        confidence = max(0.85, min(0.85 + (n_iterations - 100) / 10000, 1.0))  # Max confidence at 10k iterations
        
        return RiskMetrics(
            risk_score=risk_score,
            risk_level=risk_level,
            probability=probability,
            impact=impact,
            time_horizon=time_horizon,
            confidence=confidence
        )
    
    def assess_threat_categories(
        self,
        simulation_results: Dict[str, Any]
    ) -> Dict[ThreatCategory, ThreatAssessment]:
        """
        Assess individual threat categories.
        
        Args:
            simulation_results: Simulation results
            
        Returns:
            Dictionary of threat assessments
        """
        assessments = {}
        
        # Extract attack scenario data
        attack_data = simulation_results.get('raw_results', {})
        
        # Key Compromise threat
        assessments[ThreatCategory.KEY_COMPROMISE] = ThreatAssessment(
            category=ThreatCategory.KEY_COMPROMISE,
            description="Quantum computers break validator private keys",
            likelihood=self._estimate_likelihood(attack_data, 'key_compromise'),
            severity=0.7,  # High severity
            detection_difficulty=0.8,  # Hard to detect
            mitigation_effectiveness=0.6,  # Migration helps
            earliest_occurrence=self._estimate_earliest_year(attack_data, 2030),
            peak_risk_year=self._estimate_peak_year(attack_data, 2035)
        )
        
        # Consensus Attack threat
        assessments[ThreatCategory.CONSENSUS_ATTACK] = ThreatAssessment(
            category=ThreatCategory.CONSENSUS_ATTACK,
            description="Attackers control consensus through compromised validators",
            likelihood=self._estimate_likelihood(attack_data, 'consensus'),
            severity=0.9,  # Very high severity
            detection_difficulty=0.5,  # Moderately detectable
            mitigation_effectiveness=0.7,  # Good mitigation possible
            earliest_occurrence=self._estimate_earliest_year(attack_data, 2032),
            peak_risk_year=self._estimate_peak_year(attack_data, 2037)
        )
        
        # Double Spend threat
        assessments[ThreatCategory.DOUBLE_SPEND] = ThreatAssessment(
            category=ThreatCategory.DOUBLE_SPEND,
            description="Quantum-enabled double spending attacks",
            likelihood=self._estimate_likelihood(attack_data, 'double_spend'),
            severity=0.6,  # Moderate-high severity
            detection_difficulty=0.3,  # Easier to detect
            mitigation_effectiveness=0.8,  # Can be mitigated well
            earliest_occurrence=self._estimate_earliest_year(attack_data, 2033),
            peak_risk_year=self._estimate_peak_year(attack_data, 2036)
        )
        
        # Network Halt threat
        assessments[ThreatCategory.NETWORK_HALT] = ThreatAssessment(
            category=ThreatCategory.NETWORK_HALT,
            description="Network halted due to quantum attacks",
            likelihood=self._estimate_likelihood(attack_data, 'halt'),
            severity=0.8,  # High severity
            detection_difficulty=0.2,  # Easy to detect
            mitigation_effectiveness=0.5,  # Moderate mitigation
            earliest_occurrence=self._estimate_earliest_year(attack_data, 2031),
            peak_risk_year=self._estimate_peak_year(attack_data, 2035)
        )
        
        # Systemic Failure threat
        assessments[ThreatCategory.SYSTEMIC_FAILURE] = ThreatAssessment(
            category=ThreatCategory.SYSTEMIC_FAILURE,
            description="Cascading failures across DeFi ecosystem",
            likelihood=self._estimate_likelihood(attack_data, 'systemic'),
            severity=1.0,  # Maximum severity
            detection_difficulty=0.6,  # Moderate detection
            mitigation_effectiveness=0.4,  # Hard to mitigate
            earliest_occurrence=self._estimate_earliest_year(attack_data, 2034),
            peak_risk_year=self._estimate_peak_year(attack_data, 2038)
        )
        
        self.threat_assessments = assessments
        return assessments
    
    def calculate_risk_trajectory(
        self,
        simulation_results: Dict[str, Any],
        years: List[float]
    ) -> Dict[float, RiskMetrics]:
        """
        Calculate risk trajectory over time.
        
        Args:
            simulation_results: Simulation results
            years: Years to evaluate
            
        Returns:
            Risk metrics for each year
        """
        trajectory = {}
        
        for year in years:
            # Adjust probabilities based on year
            year_adjusted_results = self._adjust_results_for_year(
                simulation_results, year
            )
            
            # Assess risk for this year
            trajectory[year] = self.assess_quantum_risk(
                year_adjusted_results, year
            )
        
        return trajectory
    
    def identify_critical_thresholds(
        self,
        simulation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Identify critical risk thresholds.
        
        Args:
            simulation_results: Simulation results
            
        Returns:
            Dictionary of critical thresholds
        """
        thresholds = {}
        
        # Migration threshold - when migration becomes critical
        thresholds['critical_migration_year'] = self._find_critical_year(
            simulation_results, 'migration', 0.7
        )
        
        # Attack feasibility threshold
        thresholds['attack_feasible_year'] = self._find_critical_year(
            simulation_results, 'attack_feasible', 0.5
        )
        
        # Economic impact threshold
        thresholds['major_impact_year'] = self._find_critical_year(
            simulation_results, 'economic_impact', 10_000_000_000  # $10B
        )
        
        # Network vulnerability threshold
        thresholds['high_vulnerability_year'] = self._find_critical_year(
            simulation_results, 'vulnerability', 0.3
        )
        
        return thresholds
    
    def generate_risk_report(
        self,
        risk_metrics: RiskMetrics,
        threat_assessments: Optional[Dict[ThreatCategory, ThreatAssessment]] = None
    ) -> str:
        """
        Generate comprehensive risk assessment report.
        
        Args:
            risk_metrics: Overall risk metrics
            threat_assessments: Individual threat assessments
            
        Returns:
            Formatted risk report
        """
        report = []
        report.append("=" * 80)
        report.append("QUANTUM RISK ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Risk Level: {risk_metrics.risk_level.value}")
        report.append(f"Risk Score: {risk_metrics.risk_score:.1f}/100")
        report.append(f"Time Horizon: {risk_metrics.time_horizon:.1f} years")
        report.append(f"Confidence Level: {risk_metrics.confidence:.1%}")
        report.append("")
        
        # Risk Components
        report.append("RISK COMPONENTS")
        report.append("-" * 40)
        report.append(f"Probability of Attack: {risk_metrics.probability:.1%}")
        report.append(f"Potential Impact: {risk_metrics.impact:.1%} of maximum")
        report.append(f"Risk Index: {risk_metrics.risk_index:.1f}")
        report.append("")
        
        # Threat Assessment
        if threat_assessments:
            report.append("THREAT ASSESSMENT")
            report.append("-" * 40)
            
            # Sort by threat score
            sorted_threats = sorted(
                threat_assessments.values(),
                key=lambda x: x.threat_score,
                reverse=True
            )
            
            for threat in sorted_threats:
                report.append(f"\n{threat.category.value}:")
                report.append(f"  Description: {threat.description}")
                report.append(f"  Threat Score: {threat.threat_score:.2f}")
                report.append(f"  Likelihood: {threat.likelihood:.1%}")
                report.append(f"  Severity: {threat.severity:.1%}")
                report.append(f"  Earliest Occurrence: {threat.earliest_occurrence:.0f}")
                report.append(f"  Peak Risk Year: {threat.peak_risk_year:.0f}")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            report.append("⚠️  IMMEDIATE ACTION REQUIRED:")
            report.append("  1. Initiate emergency quantum-safe migration")
            report.append("  2. Implement enhanced monitoring systems")
            report.append("  3. Develop crisis response procedures")
            report.append("  4. Engage quantum security experts")
        elif risk_metrics.risk_level == RiskLevel.HIGH:
            report.append("⚠️  URGENT ACTION RECOMMENDED:")
            report.append("  1. Accelerate quantum-safe migration planning")
            report.append("  2. Increase security monitoring")
            report.append("  3. Conduct regular risk assessments")
            report.append("  4. Prepare incident response plans")
        elif risk_metrics.risk_level == RiskLevel.MODERATE:
            report.append("📋 PROACTIVE MEASURES ADVISED:")
            report.append("  1. Begin quantum-safe migration roadmap")
            report.append("  2. Monitor quantum computing advances")
            report.append("  3. Educate stakeholders on risks")
            report.append("  4. Establish risk management framework")
        else:
            report.append("✓ STANDARD MONITORING SUFFICIENT:")
            report.append("  1. Continue monitoring quantum developments")
            report.append("  2. Review risk assessment annually")
            report.append("  3. Maintain awareness of best practices")
            report.append("  4. Plan for future migration needs")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _calculate_risk_score(
        self,
        probability: float,
        impact: float,
        time_horizon: float
    ) -> float:
        """
        Calculate risk score from components using enhanced formula.
        
        This formula better reflects true risk severity by:
        1. Using weighted sum instead of multiplication (avoids dilution)
        2. Properly weighting catastrophic impacts
        3. Accounting for urgency of near-term threats
        
        Args:
            probability: Attack probability (0-1)
            impact: Normalized impact (0-1)
            time_horizon: Years until threat materializes
            
        Returns:
            Risk score (0-100)
        """
        # Weight factors for risk components
        probability_weight = 0.35  # How likely
        impact_weight = 0.40       # How bad (higher weight for catastrophic impacts)
        urgency_weight = 0.25      # How soon
        
        # Calculate urgency factor (inverse of time horizon)
        # Near-term threats (< 5 years) get maximum urgency
        # Urgency decreases linearly up to 10 years
        urgency = max(0, 1 - (max(0, time_horizon - 2) / 8))  # High urgency for < 2 years
        
        # Apply non-linear scaling for catastrophic impacts
        # Impacts > 50% of value get exponentially higher weight
        if impact > 0.5:
            impact_scaled = 0.5 + (impact - 0.5) * 1.5  # Amplify catastrophic impacts
        else:
            impact_scaled = impact
        
        # Calculate base risk score using weighted sum
        base_score = (
            probability * probability_weight * 100 +
            impact_scaled * impact_weight * 100 +
            urgency * urgency_weight * 100
        )
        
        # Apply risk amplification for critical combinations
        # When both probability and impact are high, add synergy bonus
        if probability > 0.7 and impact > 0.5:
            synergy_bonus = (probability - 0.7) * (impact - 0.5) * 100
            base_score += synergy_bonus
        
        # Ensure minimum risk score for any credible threat
        if probability >= 0.5 and impact >= 0.3:
            base_score = max(base_score, 45)  # Minimum moderate risk
        
        if probability >= 0.7 and impact >= 0.5:
            base_score = max(base_score, 65)  # Minimum high risk
        
        if probability >= 0.8 and impact >= 0.6:
            base_score = max(base_score, 75)  # Minimum critical risk
        
        return min(100, max(0, base_score))
    
    def _score_to_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Convert risk score to risk level category.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level category
        """
        if risk_score >= 75:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MODERATE
        elif risk_score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _calculate_probability_by_year(
        self,
        target_year: float,
        mean_year: float,
        std_year: float
    ) -> float:
        """Calculate probability of event by target year."""
        from scipy.stats import norm
        
        if std_year == 0:
            return 1.0 if target_year >= mean_year else 0.0
        
        z_score = (target_year - mean_year) / std_year
        return norm.cdf(z_score)
    
    def _estimate_likelihood(
        self,
        attack_data: Dict[str, Any],
        attack_type: str
    ) -> float:
        """Estimate likelihood of specific attack type."""
        # This is a placeholder - would use actual simulation data
        base_likelihoods = {
            'key_compromise': 0.6,
            'consensus': 0.4,
            'double_spend': 0.3,
            'halt': 0.5,
            'systemic': 0.2
        }
        return base_likelihoods.get(attack_type, 0.5)
    
    def _estimate_earliest_year(
        self,
        attack_data: Dict[str, Any],
        default: float
    ) -> float:
        """Estimate earliest occurrence year."""
        # This would use actual simulation data
        return default
    
    def _estimate_peak_year(
        self,
        attack_data: Dict[str, Any],
        default: float
    ) -> float:
        """Estimate peak risk year."""
        # This would use actual simulation data
        return default
    
    def _adjust_results_for_year(
        self,
        simulation_results: Dict[str, Any],
        year: float
    ) -> Dict[str, Any]:
        """Adjust simulation results for specific year."""
        # This would adjust probabilities based on the target year
        adjusted = simulation_results.copy()
        # Placeholder for year-specific adjustments
        return adjusted
    
    def _find_critical_year(
        self,
        simulation_results: Dict[str, Any],
        metric: str,
        threshold: float
    ) -> float:
        """Find year when metric crosses critical threshold."""
        # This would analyze time series data to find threshold crossing
        return 2035  # Placeholder
