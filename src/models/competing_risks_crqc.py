"""
Competing Risks Survival Analysis for CRQC Emergence.

This module models multiple competing pathways to achieving CRQC capability,
accounting for the fact that different routes (Shor's algorithm improvements,
hybrid attacks, unexpected breakthroughs) compete to be the first to break cryptography.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import weibull_min, expon, lognorm
from scipy.special import expit

logger = logging.getLogger(__name__)

# Try to import lifelines for survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter, AalenAdditiveFitter
    from lifelines.statistics import multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.info("lifelines not available - using simplified survival analysis")


@dataclass
class CRQCPathway:
    """Represents a specific pathway to achieving CRQC capability."""
    
    name: str
    description: str
    base_qubits_required: int
    current_qubits: int
    improvement_potential: float  # Factor by which requirements might decrease
    hazard_function: Callable  # Function defining instantaneous risk
    covariates: Dict[str, float] = field(default_factory=dict)
    
    def time_to_capability(self, growth_rate: float) -> float:
        """Calculate time until this pathway achieves CRQC."""
        effective_requirement = self.base_qubits_required * self.improvement_potential
        
        if self.current_qubits >= effective_requirement:
            return 0.0
        
        # Log growth model
        years = np.log(effective_requirement / self.current_qubits) / np.log(growth_rate)
        return max(0, years)
    
    def hazard_at_time(self, t: float) -> float:
        """Calculate hazard (instantaneous risk) at time t."""
        return self.hazard_function(t, self.covariates)


@dataclass
class CompetingEvent:
    """Represents an event that could prevent or delay CRQC."""
    
    name: str
    description: str
    probability_function: Callable  # P(event occurs before time t)
    impact_on_crqc: float  # Multiplier on timeline (>1 means delay)


@dataclass 
class SurvivalPrediction:
    """Results from competing risks survival analysis."""
    
    pathway_probabilities: Dict[str, float]
    cumulative_incidence: Dict[str, np.ndarray]
    median_times: Dict[str, float]
    overall_crqc_probability: np.ndarray
    time_points: np.ndarray
    winning_pathway: str
    confidence_intervals: Dict[str, Tuple[float, float]]


class CompetingRisksCRQC:
    """
    Models CRQC emergence as a competing risks problem where multiple
    pathways race to be the first to achieve cryptographically relevant quantum computing.
    """
    
    def __init__(self):
        """Initialize competing risks model with pathways and events."""
        self.pathways = self._initialize_pathways()
        self.competing_events = self._initialize_competing_events()
        self.time_horizon = 30  # Years
        self.time_points = np.linspace(0, self.time_horizon, 361)  # Monthly resolution
        
    def _initialize_pathways(self) -> Dict[str, CRQCPathway]:
        """Define different pathways to CRQC capability."""
        
        pathways = {
            "shor_traditional": CRQCPathway(
                name="Traditional Shor's Algorithm",
                description="Standard implementation of Shor's algorithm with current techniques",
                base_qubits_required=2330,
                current_qubits=1180,  # Current best
                improvement_potential=1.0,  # No improvement expected
                hazard_function=self._weibull_hazard,
                covariates={"shape": 2.5, "scale": 8.0}
            ),
            
            "shor_optimized": CRQCPathway(
                name="Optimized Shor's Algorithm", 
                description="Improved implementation with windowing, active volume compilation, etc.",
                base_qubits_required=2330,
                current_qubits=1180,
                improvement_potential=0.7,  # 30% reduction from optimizations
                hazard_function=self._weibull_hazard,
                covariates={"shape": 2.0, "scale": 6.0}
            ),
            
            "regev_algorithm": CRQCPathway(
                name="Regev's Algorithm (2024)",
                description="New factoring approach with reduced gate depth",
                base_qubits_required=2000,
                current_qubits=1180,
                improvement_potential=0.85,
                hazard_function=self._lognormal_hazard,
                covariates={"mu": 1.8, "sigma": 0.4}
            ),
            
            "hybrid_quantum_classical": CRQCPathway(
                name="Hybrid Quantum-Classical Attack",
                description="Combines quantum subroutines with massive classical computation",
                base_qubits_required=1000,
                current_qubits=1180,
                improvement_potential=1.0,  # Already achievable with current qubits
                hazard_function=self._exponential_hazard,
                covariates={"rate": 0.15}  # Could happen anytime
            ),
            
            "unexpected_breakthrough": CRQCPathway(
                name="Unexpected Algorithmic Breakthrough",
                description="Revolutionary new approach not yet discovered",
                base_qubits_required=500,  # Hypothetical major improvement
                current_qubits=1180,
                improvement_potential=1.0,
                hazard_function=self._constant_hazard,
                covariates={"rate": 0.05}  # 5% annual chance
            ),
            
            "topological_quantum": CRQCPathway(
                name="Topological Quantum Computing",
                description="Microsoft's approach if successful",
                base_qubits_required=1500,  # More efficient if it works
                current_qubits=0,  # Not yet demonstrated
                improvement_potential=1.0,
                hazard_function=self._threshold_hazard,
                covariates={"threshold_year": 7, "post_threshold_rate": 0.3}
            ),
            
            "photonic_quantum": CRQCPathway(
                name="Photonic Quantum Computing",
                description="PsiQuantum's million-qubit approach",
                base_qubits_required=10000,  # Less efficient but massively parallel
                current_qubits=216,
                improvement_potential=0.5,  # Could improve with better algorithms
                hazard_function=self._weibull_hazard,
                covariates={"shape": 1.8, "scale": 10.0}
            )
        }
        
        return pathways
    
    def _initialize_competing_events(self) -> Dict[str, CompetingEvent]:
        """Define events that could prevent or delay CRQC."""
        
        events = {
            "post_quantum_migration": CompetingEvent(
                name="Post-Quantum Migration",
                description="Widespread adoption of quantum-resistant cryptography",
                probability_function=lambda t: 1 - np.exp(-0.1 * t),  # Gradual adoption
                impact_on_crqc=0.5  # Reduces urgency/impact
            ),
            
            "quantum_winter": CompetingEvent(
                name="Quantum Winter",
                description="Funding dries up due to lack of progress",
                probability_function=lambda t: 0.02 * t if t > 5 else 0,  # Risk after 5 years
                impact_on_crqc=2.0  # Doubles timeline
            ),
            
            "physical_barrier": CompetingEvent(
                name="Fundamental Physical Barrier",
                description="Discovery of insurmountable technical obstacle",
                probability_function=lambda t: 1 - np.exp(-0.02 * t),
                impact_on_crqc=5.0  # Major delay
            ),
            
            "regulatory_ban": CompetingEvent(
                name="International Regulatory Ban",
                description="Global agreement to limit quantum computing",
                probability_function=lambda t: 0.01 * min(t, 10),  # Unlikely but possible
                impact_on_crqc=3.0
            )
        }
        
        return events
    
    # Hazard functions for different pathway types
    
    def _weibull_hazard(self, t: float, params: Dict[str, float]) -> float:
        """Weibull hazard function - increasing/decreasing hazard."""
        shape = params.get("shape", 2.0)
        scale = params.get("scale", 10.0)
        
        if t <= 0:
            return 0
        
        return (shape / scale) * (t / scale) ** (shape - 1)
    
    def _exponential_hazard(self, t: float, params: Dict[str, float]) -> float:
        """Exponential hazard - constant hazard rate."""
        rate = params.get("rate", 0.1)
        return rate
    
    def _lognormal_hazard(self, t: float, params: Dict[str, float]) -> float:
        """Log-normal hazard - initially increasing then decreasing."""
        mu = params.get("mu", 2.0)
        sigma = params.get("sigma", 0.5)
        
        if t <= 0:
            return 0
        
        # Hazard = pdf / (1 - cdf)
        dist = lognorm(s=sigma, scale=np.exp(mu))
        pdf = dist.pdf(t)
        sf = dist.sf(t)  # Survival function
        
        return pdf / sf if sf > 0 else 0
    
    def _constant_hazard(self, t: float, params: Dict[str, float]) -> float:
        """Constant hazard rate."""
        return params.get("rate", 0.1)
    
    def _threshold_hazard(self, t: float, params: Dict[str, float]) -> float:
        """Zero hazard until threshold, then constant."""
        threshold = params.get("threshold_year", 5)
        post_rate = params.get("post_threshold_rate", 0.2)
        
        return 0 if t < threshold else post_rate
    
    def calculate_cumulative_incidence(self, 
                                     pathway: CRQCPathway,
                                     time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate cumulative incidence function for a specific pathway.
        This is the probability that this pathway achieves CRQC by time t,
        accounting for competition from other pathways.
        """
        if time_points is None:
            time_points = self.time_points
        
        cumulative_incidence = np.zeros_like(time_points)
        
        for i, t in enumerate(time_points):
            if i == 0:
                cumulative_incidence[i] = 0
                continue
            
            # Calculate hazard for this pathway
            hazard = pathway.hazard_at_time(t)
            
            # Calculate overall survival (no CRQC from any pathway)
            overall_survival = self._calculate_overall_survival(t)
            
            # Cumulative incidence = integral of (hazard * overall_survival)
            dt = time_points[i] - time_points[i-1]
            cumulative_incidence[i] = cumulative_incidence[i-1] + hazard * overall_survival * dt
        
        return cumulative_incidence
    
    def _calculate_overall_survival(self, t: float) -> float:
        """Calculate probability that no pathway has achieved CRQC by time t."""
        
        cumulative_hazard = 0
        
        for pathway in self.pathways.values():
            cumulative_hazard += self._integrate_hazard(pathway, 0, t)
        
        return np.exp(-cumulative_hazard)
    
    def _integrate_hazard(self, pathway: CRQCPathway, t1: float, t2: float) -> float:
        """Integrate hazard function from t1 to t2."""
        
        # Simple trapezoidal integration
        n_steps = 100
        t_vals = np.linspace(t1, t2, n_steps)
        hazard_vals = [pathway.hazard_at_time(t) for t in t_vals]
        
        return np.trapz(hazard_vals, t_vals)
    
    def simulate_first_crqc(self, 
                           n_simulations: int = 10000,
                           include_competing_events: bool = True) -> SurvivalPrediction:
        """
        Simulate which pathway achieves CRQC first using competing risks framework.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            include_competing_events: Whether to model events that could prevent CRQC
            
        Returns:
            SurvivalPrediction with results
        """
        pathway_wins = {name: 0 for name in self.pathways.keys()}
        crqc_times = []
        
        for _ in range(n_simulations):
            # Sample time to CRQC for each pathway
            pathway_times = {}
            
            for name, pathway in self.pathways.items():
                # Sample from survival distribution based on hazard
                if pathway.hazard_function == self._exponential_hazard:
                    time = np.random.exponential(1.0 / pathway.covariates.get("rate", 0.1))
                elif pathway.hazard_function == self._weibull_hazard:
                    time = weibull_min.rvs(
                        c=pathway.covariates.get("shape", 2),
                        scale=pathway.covariates.get("scale", 10)
                    )
                else:
                    # General sampling using inverse transform
                    u = np.random.uniform()
                    time = self._inverse_survival(pathway, u)
                
                # Adjust for current progress
                growth_rate = np.random.lognormal(np.log(1.5), 0.2)
                min_time = pathway.time_to_capability(growth_rate)
                time = max(time, min_time)
                
                pathway_times[name] = time
            
            # Check competing events
            if include_competing_events:
                for event in self.competing_events.values():
                    if np.random.uniform() < event.probability_function(min(pathway_times.values())):
                        # Event occurs - adjust all times
                        pathway_times = {k: v * event.impact_on_crqc 
                                       for k, v in pathway_times.items()}
            
            # Find winning pathway
            winner = min(pathway_times, key=pathway_times.get)
            pathway_wins[winner] += 1
            crqc_times.append(pathway_times[winner])
        
        # Calculate statistics
        pathway_probabilities = {k: v/n_simulations for k, v in pathway_wins.items()}
        
        # Calculate cumulative incidence for each pathway
        cumulative_incidence = {}
        for name, pathway in self.pathways.items():
            cumulative_incidence[name] = self.calculate_cumulative_incidence(pathway)
        
        # Overall CRQC probability over time
        overall_crqc = 1 - np.array([self._calculate_overall_survival(t) for t in self.time_points])
        
        # Find winning pathway
        winning_pathway = max(pathway_probabilities, key=pathway_probabilities.get)
        
        # Calculate confidence intervals
        crqc_times_array = np.array(crqc_times)
        current_year = datetime.now().year
        confidence_intervals = {
            "median": (
                current_year + np.quantile(crqc_times_array, 0.5),
                current_year + np.quantile(crqc_times_array, 0.5)
            ),
            "ci_80": (
                current_year + np.quantile(crqc_times_array, 0.1),
                current_year + np.quantile(crqc_times_array, 0.9)
            ),
            "ci_95": (
                current_year + np.quantile(crqc_times_array, 0.025),
                current_year + np.quantile(crqc_times_array, 0.975)
            )
        }
        
        # Median times for each pathway
        median_times = {}
        for name in self.pathways.keys():
            pathway_specific_times = [
                crqc_times[i] for i in range(n_simulations)
                if list(pathway_wins.keys())[list(pathway_wins.values()).index(max(pathway_wins.values()))] == name
            ]
            if pathway_specific_times:
                median_times[name] = current_year + np.median(pathway_specific_times)
            else:
                median_times[name] = current_year + 20  # Default if never wins
        
        return SurvivalPrediction(
            pathway_probabilities=pathway_probabilities,
            cumulative_incidence=cumulative_incidence,
            median_times=median_times,
            overall_crqc_probability=overall_crqc,
            time_points=self.time_points + current_year,
            winning_pathway=winning_pathway,
            confidence_intervals=confidence_intervals
        )
    
    def _inverse_survival(self, pathway: CRQCPathway, u: float) -> float:
        """
        Inverse survival function for sampling.
        Uses bisection method to find t such that S(t) = u.
        """
        def survival(t):
            return np.exp(-self._integrate_hazard(pathway, 0, t))
        
        # Bisection
        low, high = 0, 100
        while high - low > 0.01:
            mid = (low + high) / 2
            if survival(mid) > u:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
    
    def fine_gray_model(self, 
                       observed_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Implement Fine-Gray model for competing risks regression.
        This accounts for the fact that occurrence of one event precludes others.
        """
        if not LIFELINES_AVAILABLE:
            logger.warning("lifelines not available - returning simplified analysis")
            return self._simplified_fine_gray()
        
        # This would implement the full Fine-Gray competing risks regression
        # For now, return simplified version
        return self._simplified_fine_gray()
    
    def _simplified_fine_gray(self) -> Dict[str, Any]:
        """Simplified version of Fine-Gray model."""
        
        results = {}
        
        for name, pathway in self.pathways.items():
            # Calculate subdistribution hazard
            cumulative_incidence = self.calculate_cumulative_incidence(pathway)
            
            # Find median time (when cumulative incidence = 0.5)
            median_idx = np.where(cumulative_incidence >= 0.5)[0]
            if len(median_idx) > 0:
                median_time = self.time_points[median_idx[0]]
            else:
                median_time = self.time_horizon
            
            results[name] = {
                "cumulative_incidence_at_10y": cumulative_incidence[120] if len(cumulative_incidence) > 120 else 0,  # 10 years
                "median_time": median_time,
                "max_incidence": np.max(cumulative_incidence)
            }
        
        return results


def test_competing_risks():
    """Test the competing risks model."""
    
    print("=" * 50)
    print("COMPETING RISKS CRQC MODEL TEST")
    print("=" * 50)
    
    model = CompetingRisksCRQC()
    
    # Run simulation
    print("\nSimulating competing pathways to CRQC...")
    results = model.simulate_first_crqc(n_simulations=5000)
    
    print(f"\n=== Pathway Probabilities ===")
    for pathway, prob in sorted(results.pathway_probabilities.items(), 
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"{pathway:25s}: {prob:6.1%} {bar}")
    
    print(f"\n=== Winning Pathway ===")
    print(f"Most likely: {results.winning_pathway}")
    print(f"Probability: {results.pathway_probabilities[results.winning_pathway]:.1%}")
    
    print(f"\n=== Timeline ===")
    print(f"Median CRQC year: {results.confidence_intervals['median'][0]:.0f}")
    print(f"80% CI: {results.confidence_intervals['ci_80'][0]:.0f} - {results.confidence_intervals['ci_80'][1]:.0f}")
    print(f"95% CI: {results.confidence_intervals['ci_95'][0]:.0f} - {results.confidence_intervals['ci_95'][1]:.0f}")
    
    print(f"\n=== Fine-Gray Analysis ===")
    fine_gray = model.fine_gray_model()
    for pathway, metrics in fine_gray.items():
        print(f"\n{pathway}:")
        print(f"  10-year cumulative incidence: {metrics['cumulative_incidence_at_10y']:.1%}")
        print(f"  Median time to CRQC: {metrics['median_time']:.1f} years")


if __name__ == "__main__":
    test_competing_risks()
