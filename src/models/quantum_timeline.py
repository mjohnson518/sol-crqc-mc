"""
Quantum computing development timeline model.

This module models the evolution of quantum computing capabilities over time,
including when cryptographically relevant quantum computers (CRQC) might emerge.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.config import QuantumParameters

logger = logging.getLogger(__name__)


class QuantumThreat(Enum):
    """Quantum threat levels based on capabilities."""
    NONE = "none"  # No practical threat
    EMERGING = "emerging"  # Early quantum computers, limited threat
    MODERATE = "moderate"  # Can break some cryptography
    HIGH = "high"  # Can break Ed25519 efficiently
    CRITICAL = "critical"  # Can break all current cryptography


@dataclass
class QuantumCapability:
    """Represents quantum computing capability at a point in time."""
    
    year: float
    logical_qubits: int
    physical_qubits: int
    gate_fidelity: float
    coherence_time_ms: float
    threat_level: QuantumThreat
    
    @property
    def can_break_ed25519(self) -> bool:
        """Check if capable of breaking Ed25519."""
        return self.logical_qubits >= 2330 and self.gate_fidelity >= 0.99
    
    @property
    def estimated_break_time_hours(self) -> float:
        """Estimate time to break one Ed25519 key."""
        if not self.can_break_ed25519:
            return float('inf')
        
        # Base time with minimum required qubits
        base_time = 24.0  # hours
        
        # Advantage from extra qubits (diminishing returns)
        qubit_advantage = min(self.logical_qubits / 2330, 10)
        
        # Advantage from better gate fidelity
        fidelity_factor = 1.0 / (self.gate_fidelity ** 10)
        
        return base_time / qubit_advantage * fidelity_factor


@dataclass
class QuantumTimeline:
    """Complete quantum development timeline."""
    
    capabilities: List[QuantumCapability]
    crqc_year: float  # Year when CRQC emerges
    breakthrough_years: List[float]  # Years with major breakthroughs
    projection_method: str  # Method used for projection
    confidence: float  # Confidence in projection
    
    def get_capability_at_year(self, year: float) -> QuantumCapability:
        """Get quantum capability at specific year."""
        # Find closest year
        for cap in self.capabilities:
            if cap.year >= year:
                return cap
        return self.capabilities[-1]
    
    def years_until_threat_level(self, level: QuantumThreat) -> float:
        """Calculate years until reaching threat level."""
        for cap in self.capabilities:
            if cap.threat_level.value >= level.value:
                return cap.year - 2025
        return float('inf')


class QuantumDevelopmentModel:
    """
    Models the development of quantum computing capabilities over time.
    
    Combines multiple projection methods:
    1. Industry roadmap extrapolation
    2. Expert survey aggregation
    3. Breakthrough probability modeling
    4. Historical growth curve fitting
    """
    
    def __init__(self, params: Optional[QuantumParameters] = None):
        """
        Initialize quantum development model.
        
        Args:
            params: Quantum parameters configuration
        """
        self.params = params or QuantumParameters()
        
        # Historical data points (logical qubits)
        self.historical_data = {
            2019: 53,    # Google Sycamore
            2020: 65,    # IBM
            2021: 127,   # IBM Eagle
            2022: 433,   # IBM Osprey
            2023: 1121,  # IBM Condor
            2024: 1180,  # IBM (projected)
            2025: 1386   # Extrapolated
        }
        
        # Industry roadmaps (optimistic projections)
        self.industry_roadmaps = {
            'IBM': {
                2025: 1386,
                2026: 2000,
                2027: 4000,
                2028: 10000,
                2029: 100000
            },
            'Google': {
                2025: 1000,
                2026: 2000,
                2027: 5000,
                2028: 10000,
                2029: 1000000
            },
            'IonQ': {
                2025: 256,
                2026: 1024,
                2027: 4096,
                2028: 16384
            }
        }
        
        # Expert survey data (probability distributions)
        self.expert_predictions = {
            'optimistic': {'mean': 2030, 'std': 2},
            'moderate': {'mean': 2033, 'std': 3},
            'conservative': {'mean': 2038, 'std': 4}
        }
    
    def sample(self, rng: np.random.RandomState) -> QuantumTimeline:
        """
        Sample a complete quantum development timeline.
        
        Args:
            rng: Random number generator
            
        Returns:
            QuantumTimeline instance
        """
        # Choose projection method
        method_weights = [0.3, 0.4, 0.2, 0.1]  # Industry, Expert, Breakthrough, Historical
        method = rng.choice(
            ['industry', 'expert', 'breakthrough', 'historical'],
            p=method_weights
        )
        
        # Generate timeline based on method
        if method == 'industry':
            timeline = self._sample_industry_projection(rng)
        elif method == 'expert':
            timeline = self._sample_expert_projection(rng)
        elif method == 'breakthrough':
            timeline = self._sample_breakthrough_projection(rng)
        else:
            timeline = self._sample_historical_projection(rng)
        
        timeline.projection_method = method
        
        # Add confidence based on method
        confidence_map = {
            'industry': 0.6,
            'expert': 0.7,
            'breakthrough': 0.5,
            'historical': 0.8
        }
        timeline.confidence = confidence_map[method] * (0.8 + 0.4 * rng.random())
        
        return timeline
    
    def _sample_industry_projection(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample based on industry roadmaps."""
        # Select a roadmap
        company = rng.choice(list(self.industry_roadmaps.keys()))
        roadmap = self.industry_roadmaps[company]
        
        # Add noise and delays
        delay_factor = rng.lognormal(mean=0, sigma=0.3)  # Log-normal delay
        noise_factor = 1 + rng.normal(0, 0.2)  # Gaussian noise
        
        capabilities = []
        crqc_year = None
        
        for year in range(2025, 2046):
            # Interpolate/extrapolate from roadmap
            if year in roadmap:
                base_qubits = roadmap[year]
            else:
                # Exponential extrapolation
                years_ahead = year - 2025
                growth_rate = self.params.qubit_growth_rate * noise_factor
                base_qubits = self.params.initial_qubits * (growth_rate ** years_ahead)
            
            # Apply delay and noise
            logical_qubits = int(base_qubits / (1 + delay_factor) * (0.7 + 0.6 * rng.random()))
            
            # Physical qubits (with error correction overhead)
            physical_qubits = logical_qubits * self.params.physical_to_logical_ratio
            
            # Gate fidelity improves over time
            years_from_now = year - 2025
            gate_fidelity = min(0.999, 0.99 + 0.0005 * years_from_now * rng.random())
            
            # Coherence time improves
            coherence_time = 0.1 * (1.2 ** years_from_now) * (0.5 + rng.random())
            
            # Determine threat level
            threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
            
            capability = QuantumCapability(
                year=year,
                logical_qubits=logical_qubits,
                physical_qubits=physical_qubits,
                gate_fidelity=gate_fidelity,
                coherence_time_ms=coherence_time,
                threat_level=threat_level
            )
            
            capabilities.append(capability)
            
            # Check for CRQC emergence
            if crqc_year is None and capability.can_break_ed25519:
                crqc_year = year + rng.uniform(0, 1)  # Add fractional year
        
        if crqc_year is None:
            crqc_year = 2045 + rng.exponential(scale=5)  # Beyond timeline
        
        # Sample breakthrough years
        breakthrough_years = self._sample_breakthroughs(rng, 2025, int(crqc_year))
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method='industry',
            confidence=0.6
        )
    
    def _sample_expert_projection(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample based on expert surveys."""
        # Weight expert opinions
        weights = [0.2, 0.6, 0.2]  # Optimistic, Moderate, Conservative
        selected = rng.choice(['optimistic', 'moderate', 'conservative'], p=weights)
        
        prediction = self.expert_predictions[selected]
        
        # Sample CRQC year from distribution
        crqc_year = rng.normal(prediction['mean'], prediction['std'])
        crqc_year = max(2026, crqc_year)  # Not before 2026
        
        # Generate capability trajectory
        capabilities = []
        
        for year in range(2025, 2046):
            # S-curve growth toward CRQC
            t = (year - 2025) / (crqc_year - 2025) if crqc_year > 2025 else 1
            t = min(1.5, max(0, t))  # Clamp to reasonable range
            
            # Logistic growth
            logical_qubits = int(
                self.params.logical_qubits_for_ed25519 * 2 / (1 + np.exp(-6 * (t - 0.5)))
            )
            
            # Add noise
            logical_qubits = int(logical_qubits * (0.8 + 0.4 * rng.random()))
            
            # Other parameters
            physical_qubits = logical_qubits * self.params.physical_to_logical_ratio
            gate_fidelity = 0.99 + 0.009 * t
            coherence_time = 0.1 * np.exp(2 * t)
            
            threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
            
            capabilities.append(QuantumCapability(
                year=year,
                logical_qubits=logical_qubits,
                physical_qubits=physical_qubits,
                gate_fidelity=gate_fidelity,
                coherence_time_ms=coherence_time,
                threat_level=threat_level
            ))
        
        breakthrough_years = self._sample_breakthroughs(rng, 2025, int(crqc_year))
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method='expert',
            confidence=0.7
        )
    
    def _sample_breakthrough_projection(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample based on breakthrough probability model."""
        capabilities = []
        crqc_year = None
        breakthrough_years = []
        
        # Current state
        logical_qubits = self.params.initial_qubits
        gate_fidelity = 0.99
        
        for year in range(2025, 2046):
            # Check for breakthrough
            breakthrough_prob = self.params.breakthrough_timeline.get(
                year,
                self.params.breakthrough_probability_annual
            )
            
            if rng.random() < breakthrough_prob:
                # Breakthrough occurred!
                breakthrough_years.append(year)
                
                # Jump in capabilities
                jump_factor = rng.uniform(2, 5)
                logical_qubits = int(logical_qubits * jump_factor)
                gate_fidelity = min(0.9999, gate_fidelity + rng.uniform(0.001, 0.005))
            else:
                # Normal progress
                growth = 1 + rng.normal(0.3, 0.1)  # ~30% annual growth
                logical_qubits = int(logical_qubits * growth)
                gate_fidelity = min(0.9999, gate_fidelity + 0.0001)
            
            # Physical qubits
            physical_qubits = logical_qubits * self.params.physical_to_logical_ratio
            
            # Coherence time
            coherence_time = 0.1 * (1.15 ** (year - 2025))
            
            threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
            
            capability = QuantumCapability(
                year=year,
                logical_qubits=logical_qubits,
                physical_qubits=physical_qubits,
                gate_fidelity=gate_fidelity,
                coherence_time_ms=coherence_time,
                threat_level=threat_level
            )
            
            capabilities.append(capability)
            
            if crqc_year is None and capability.can_break_ed25519:
                crqc_year = year + rng.uniform(0, 1)
        
        if crqc_year is None:
            crqc_year = 2045 + rng.exponential(scale=10)
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method='breakthrough',
            confidence=0.5
        )
    
    def _sample_historical_projection(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample based on historical growth patterns."""
        # Fit exponential curve to historical data
        years = np.array(list(self.historical_data.keys()))
        qubits = np.array(list(self.historical_data.values()))
        
        # Log-linear regression
        log_qubits = np.log(qubits)
        coeffs = np.polyfit(years - 2019, log_qubits, 1)
        
        # Extract growth rate
        annual_growth = np.exp(coeffs[0])
        
        # Add uncertainty
        growth_rate = annual_growth * rng.lognormal(0, 0.1)
        
        capabilities = []
        crqc_year = None
        
        for year in range(2025, 2046):
            years_from_2025 = year - 2025
            
            # Exponential projection with noise
            logical_qubits = int(
                self.params.initial_qubits * (growth_rate ** years_from_2025) *
                rng.lognormal(0, 0.2)
            )
            
            # Physical qubits
            physical_qubits = logical_qubits * self.params.physical_to_logical_ratio
            
            # Gate fidelity improvement
            gate_fidelity = min(0.9999, 0.99 + 0.0002 * years_from_2025)
            
            # Coherence time
            coherence_time = 0.1 * (1.1 ** years_from_2025)
            
            threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
            
            capability = QuantumCapability(
                year=year,
                logical_qubits=logical_qubits,
                physical_qubits=physical_qubits,
                gate_fidelity=gate_fidelity,
                coherence_time_ms=coherence_time,
                threat_level=threat_level
            )
            
            capabilities.append(capability)
            
            if crqc_year is None and capability.can_break_ed25519:
                crqc_year = year + rng.uniform(0, 1)
        
        if crqc_year is None:
            crqc_year = 2045 + rng.exponential(scale=7)
        
        breakthrough_years = self._sample_breakthroughs(rng, 2025, min(2045, int(crqc_year)))
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method='historical',
            confidence=0.8
        )
    
    def _assess_threat_level(self, logical_qubits: int, gate_fidelity: float) -> QuantumThreat:
        """
        Assess quantum threat level based on capabilities.
        
        Args:
            logical_qubits: Number of logical qubits
            gate_fidelity: Gate fidelity rate
            
        Returns:
            QuantumThreat level
        """
        # Handle None values
        if logical_qubits is None:
            logical_qubits = 0
        if gate_fidelity is None:
            gate_fidelity = 0.0
            
        if logical_qubits < 100:
            return QuantumThreat.NONE
        elif logical_qubits < 1000:
            return QuantumThreat.EMERGING
        elif logical_qubits < 2330:
            return QuantumThreat.MODERATE
        elif gate_fidelity >= 0.99:
            if logical_qubits >= 10000:
                return QuantumThreat.CRITICAL
            return QuantumThreat.HIGH
        else:
            return QuantumThreat.MODERATE
    
    def _sample_breakthroughs(
        self,
        rng: np.random.RandomState,
        start_year: int,
        end_year: int
    ) -> List[float]:
        """
        Sample years when breakthroughs occur.
        
        Args:
            rng: Random number generator
            start_year: Start of period
            end_year: End of period
            
        Returns:
            List of breakthrough years
        """
        breakthroughs = []
        
        for year in range(start_year, end_year + 1):
            prob = self.params.breakthrough_timeline.get(
                year,
                self.params.breakthrough_probability_annual
            )
            
            if rng.random() < prob:
                # Add with fractional component
                breakthroughs.append(year + rng.random())
        
        return breakthroughs
    
    def create_consensus_projection(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create consensus projection from multiple samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed
            
        Returns:
            Dictionary with consensus statistics
        """
        rng = np.random.RandomState(seed)
        
        crqc_years = []
        threat_years = {level: [] for level in QuantumThreat}
        
        for _ in range(n_samples):
            timeline = self.sample(rng)
            crqc_years.append(timeline.crqc_year)
            
            for level in QuantumThreat:
                years = timeline.years_until_threat_level(level)
                if years < float('inf'):
                    threat_years[level].append(2025 + years)
        
        # Calculate statistics
        consensus = {
            'crqc_emergence': {
                'mean': np.mean(crqc_years),
                'median': np.median(crqc_years),
                'std': np.std(crqc_years),
                'percentiles': {
                    5: np.percentile(crqc_years, 5),
                    25: np.percentile(crqc_years, 25),
                    50: np.percentile(crqc_years, 50),
                    75: np.percentile(crqc_years, 75),
                    95: np.percentile(crqc_years, 95)
                }
            },
            'threat_levels': {}
        }
        
        for level in QuantumThreat:
            if threat_years[level]:
                years = threat_years[level]
                consensus['threat_levels'][level.value] = {
                    'mean': np.mean(years),
                    'probability': len(years) / n_samples
                }
        
        return consensus


def test_quantum_model():
    """Test the quantum development model."""
    model = QuantumDevelopmentModel()
    rng = np.random.RandomState(42)
    
    # Test single timeline
    timeline = model.sample(rng)
    
    print("Quantum Timeline Sample:")
    print(f"  CRQC Year: {timeline.crqc_year:.1f}")
    print(f"  Method: {timeline.projection_method}")
    print(f"  Confidence: {timeline.confidence:.2f}")
    print(f"  Breakthroughs: {len(timeline.breakthrough_years)}")
    
    # Test capability at specific year
    cap_2030 = timeline.get_capability_at_year(2030)
    print(f"\nCapability in 2030:")
    print(f"  Logical Qubits: {cap_2030.logical_qubits}")
    print(f"  Threat Level: {cap_2030.threat_level.value}")
    print(f"  Can break Ed25519: {cap_2030.can_break_ed25519}")
    
    # Test consensus projection
    consensus = model.create_consensus_projection(n_samples=100, seed=42)
    print(f"\nConsensus Projection (100 samples):")
    print(f"  Mean CRQC Year: {consensus['crqc_emergence']['mean']:.1f}")
    print(f"  Std Dev: {consensus['crqc_emergence']['std']:.1f}")
    
    print("\n✓ Quantum model test passed")


if __name__ == "__main__":
    test_quantum_model()
