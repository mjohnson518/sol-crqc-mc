"""
Quantum computing development timeline model.

This module models the evolution of quantum computing capabilities over time,
including when cryptographically relevant quantum computers (CRQC) might emerge.

Enhanced features (controlled by config flags):
- Cox proportional hazards model for time-dependent covariates
- Multimodal distributions for competing qubit technologies
- Grover's algorithm modeling for hash attacks
- Live data fetching from quantum computing APIs
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, lognorm, expon
from scipy.special import expit
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import lifelines for Cox model if available
try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logging.info("lifelines not available - Cox models will use fallback implementation")

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
class GroverCapability:
    """Represents Grover's algorithm capability for hash attacks."""
    
    year: float
    logical_qubits: int
    can_attack_sha256: bool
    speedup_factor: float
    attack_time_hours: float
    poh_vulnerability: bool  # Solana-specific PoH vulnerability
    
    @property
    def effective_security_bits(self) -> int:
        """Calculate effective security bits after Grover speedup."""
        if self.can_attack_sha256:
            return 128  # SHA-256 reduced from 256 to 128 bits
        return 256


@dataclass
class QuantumTechnology:
    """Represents a specific quantum computing technology path."""
    
    name: str
    current_qubits: int
    growth_rate: float
    gate_fidelity: float
    coherence_time_ms: float
    advantages: List[str]
    disadvantages: List[str]
    probability_weight: float  # Weight in multimodal distribution


@dataclass
class QuantumTimeline:
    """Complete quantum development timeline."""
    
    capabilities: List[QuantumCapability]
    crqc_year: float  # Year when CRQC emerges
    breakthrough_years: List[float]  # Years with major breakthroughs
    projection_method: str  # Method used for projection
    confidence: float  # Confidence in projection
    grover_capabilities: Optional[List[GroverCapability]] = None  # Grover timeline
    winning_technology: Optional[str] = None  # For multimodal sampling
    
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
    
    def get_grover_capability_at_year(self, year: float) -> Optional[GroverCapability]:
        """Get Grover capability at specific year."""
        if not self.grover_capabilities:
            return None
        for cap in self.grover_capabilities:
            if cap.year >= year:
                return cap
        return self.grover_capabilities[-1] if self.grover_capabilities else None


class QuantumDevelopmentModel:
    """
    Models the development of quantum computing capabilities over time.
    
    Combines multiple projection methods:
    1. Industry roadmap extrapolation
    2. Expert survey aggregation
    3. Breakthrough probability modeling
    4. Historical growth curve fitting
    """
    
    def __init__(self, params: Optional[QuantumParameters] = None,
                 enable_live_data: bool = False,
                 enable_grover: bool = False,
                 use_advanced_models: bool = False):
        """
        Initialize quantum development model.
        
        Args:
            params: Quantum parameters configuration
            enable_live_data: Whether to fetch live quantum data
            enable_grover: Whether to model Grover's algorithm
            use_advanced_models: Whether to use Cox/multimodal models
        """
        self.params = params or QuantumParameters()
        self.enable_live_data = enable_live_data
        self.enable_grover = enable_grover
        self.use_advanced_models = use_advanced_models
        
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
        
        # Initialize advanced features if enabled
        if self.use_advanced_models:
            self.technologies = self._initialize_technologies()
            self.cox_covariates = {
                'rd_investment': 1.0,
                'error_rate_progress': 1.0,
                'algorithm_efficiency': 1.0,
                'hardware_availability': 1.0
            }
        
        # Initialize data cache for live data
        if self.enable_live_data:
            self.data_cache = {}
            self.cache_duration = timedelta(hours=1)
    
    def sample(self, rng: np.random.RandomState) -> QuantumTimeline:
        """
        Sample a complete quantum development timeline.
        
        Args:
            rng: Random number generator
            
        Returns:
            QuantumTimeline instance
        """
        # Use advanced models if enabled
        if self.use_advanced_models:
            # Choose between Cox and multimodal methods
            if rng.random() < 0.5 and LIFELINES_AVAILABLE:
                timeline = self._sample_cox_hazards(rng)
            else:
                timeline = self._sample_multimodal_technology(rng)
        else:
            # Choose standard projection method
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
        
        # Add Grover capabilities if enabled
        if self.enable_grover and getattr(self.params, 'enable_grover_modeling', False):
            timeline.grover_capabilities = self._sample_grover_timeline(rng)
        
        # Update with live data if enabled
        if self.enable_live_data:
            self._update_with_live_data(timeline)
        
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
    
    def _initialize_technologies(self) -> List[QuantumTechnology]:
        """Initialize competing quantum computing technologies."""
        return [
            QuantumTechnology(
                name="Superconducting",
                current_qubits=1180,
                growth_rate=1.8,
                gate_fidelity=0.995,
                coherence_time_ms=0.1,
                advantages=["Fast gates", "Mature technology"],
                disadvantages=["Requires extreme cooling"],
                probability_weight=0.4
            ),
            QuantumTechnology(
                name="Trapped Ion",
                current_qubits=56,
                growth_rate=1.5,
                gate_fidelity=0.998,
                coherence_time_ms=10.0,
                advantages=["High fidelity", "Long coherence"],
                disadvantages=["Slower gates"],
                probability_weight=0.25
            ),
            QuantumTechnology(
                name="Topological",
                current_qubits=0,
                growth_rate=0.0,
                gate_fidelity=0.9999,
                coherence_time_ms=1000.0,
                advantages=["Inherent error correction"],
                disadvantages=["Not yet demonstrated"],
                probability_weight=0.1
            ),
            QuantumTechnology(
                name="Photonic",
                current_qubits=216,
                growth_rate=2.0,
                gate_fidelity=0.99,
                coherence_time_ms=float('inf'),
                advantages=["Room temperature", "No decoherence"],
                disadvantages=["Probabilistic gates"],
                probability_weight=0.15
            ),
            QuantumTechnology(
                name="Neutral Atom",
                current_qubits=256,
                growth_rate=2.2,
                gate_fidelity=0.993,
                coherence_time_ms=1.0,
                advantages=["Scalable", "Flexible"],
                disadvantages=["Loading efficiency"],
                probability_weight=0.1
            )
        ]
    
    def _sample_cox_hazards(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample timeline using Cox proportional hazards model."""
        # Generate baseline hazard (Weibull distribution)
        scale = 10.0  # Years
        shape = 2.0  # Shape parameter > 1 means increasing hazard
        
        # Calculate hazard ratio from covariates
        beta_coefficients = {
            'rd_investment': 0.5,
            'error_rate_progress': 0.8,
            'algorithm_efficiency': 0.3,
            'hardware_availability': 0.4
        }
        
        hazard_ratio = np.exp(sum(
            beta * self.cox_covariates[var]
            for var, beta in beta_coefficients.items()
        ))
        
        # Adjust scale based on hazard ratio
        adjusted_scale = scale / hazard_ratio
        
        # Sample CRQC emergence time
        crqc_time = rng.weibull(shape) * adjusted_scale
        crqc_year = 2025 + crqc_time
        
        # Generate capability trajectory with Cox influence
        capabilities = []
        for year in range(2025, 2046):
            progress = (year - 2025) / max(1, crqc_time)
            
            # Logistic growth influenced by hazard ratio
            logical_qubits = int(
                self.params.logical_qubits_for_ed25519 * 2 *
                hazard_ratio / (1 + np.exp(-6 * (progress - 0.5)))
            )
            
            # Add noise
            logical_qubits = int(logical_qubits * rng.lognormal(0, 0.15))
            
            # Other parameters
            physical_qubits = logical_qubits * self.params.physical_to_logical_ratio
            gate_fidelity = min(0.9999, 0.99 + 0.01 * progress * hazard_ratio)
            coherence_time = 0.1 * np.exp(3 * progress)
            
            threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
            
            capabilities.append(QuantumCapability(
                year=year,
                logical_qubits=logical_qubits,
                physical_qubits=physical_qubits,
                gate_fidelity=gate_fidelity,
                coherence_time_ms=coherence_time,
                threat_level=threat_level
            ))
        
        breakthrough_years = self._sample_breakthroughs(rng, 2025, min(2045, int(crqc_year)))
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method='cox_hazards',
            confidence=0.75
        )
    
    def _sample_multimodal_technology(self, rng: np.random.RandomState) -> QuantumTimeline:
        """Sample timeline using multimodal distribution of competing technologies."""
        # Sample which technology wins the race
        tech_probs = [t.probability_weight for t in self.technologies]
        tech_probs = np.array(tech_probs) / sum(tech_probs)  # Normalize
        
        winning_tech = rng.choice(self.technologies, p=tech_probs)
        
        # Generate trajectory for winning technology
        capabilities = []
        crqc_year = None
        
        # Handle special case for topological qubits (breakthrough-based)
        if winning_tech.name == "Topological":
            # Topological qubits emerge suddenly after breakthrough
            breakthrough_year = 2025 + rng.exponential(scale=12)  # Mean ~2037
            
            for year in range(2025, 2046):
                if year < breakthrough_year:
                    logical_qubits = 0
                    gate_fidelity = 0
                else:
                    years_since = year - breakthrough_year
                    logical_qubits = int(100 * (3 ** years_since))
                    gate_fidelity = winning_tech.gate_fidelity
                
                physical_qubits = logical_qubits * 100  # Much lower overhead
                coherence_time = winning_tech.coherence_time_ms if logical_qubits > 0 else 0
                threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
                
                capabilities.append(QuantumCapability(
                    year=year,
                    logical_qubits=logical_qubits,
                    physical_qubits=physical_qubits,
                    gate_fidelity=gate_fidelity,
                    coherence_time_ms=coherence_time,
                    threat_level=threat_level
                ))
                
                if crqc_year is None and logical_qubits >= self.params.logical_qubits_for_ed25519:
                    crqc_year = year
        else:
            # Standard growth for other technologies
            current_qubits = winning_tech.current_qubits
            
            for year in range(2025, 2046):
                years_ahead = year - 2025
                noise = rng.lognormal(0, 0.2)
                
                logical_qubits = int(
                    current_qubits * (winning_tech.growth_rate ** years_ahead) * noise
                )
                
                # Technology-specific parameters
                physical_ratio = 1000 if winning_tech.name == "Superconducting" else 500
                fidelity_improvement = 0.0005 * years_ahead
                
                physical_qubits = logical_qubits * physical_ratio
                gate_fidelity = min(0.9999, winning_tech.gate_fidelity + fidelity_improvement)
                coherence_time = winning_tech.coherence_time_ms * (1.1 ** years_ahead)
                
                threat_level = self._assess_threat_level(logical_qubits, gate_fidelity)
                
                capabilities.append(QuantumCapability(
                    year=year,
                    logical_qubits=logical_qubits,
                    physical_qubits=physical_qubits,
                    gate_fidelity=gate_fidelity,
                    coherence_time_ms=coherence_time,
                    threat_level=threat_level
                ))
                
                if crqc_year is None and logical_qubits >= self.params.logical_qubits_for_ed25519:
                    crqc_year = year + rng.uniform(0, 1)
        
        if crqc_year is None:
            crqc_year = 2045 + rng.exponential(scale=10)
        
        breakthrough_years = self._sample_breakthroughs(rng, 2025, min(2045, int(crqc_year)))
        
        return QuantumTimeline(
            capabilities=capabilities,
            crqc_year=crqc_year,
            breakthrough_years=breakthrough_years,
            projection_method=f'multimodal_{winning_tech.name.lower()}',
            confidence=0.7,
            winning_technology=winning_tech.name
        )
    
    def _sample_grover_timeline(self, rng: np.random.RandomState) -> List[GroverCapability]:
        """Sample timeline for Grover's algorithm capability development."""
        grover_capabilities = []
        
        # Grover emergence as log-normal distribution
        grover_emergence_year = rng.lognormal(
            np.log(self.params.grover_emergence_median_year - 2025),
            0.3
        ) + 2025
        
        for year in range(2025, 2046):
            if year < grover_emergence_year - 5:
                # No Grover capability yet
                grover_cap = GroverCapability(
                    year=year,
                    logical_qubits=0,
                    can_attack_sha256=False,
                    speedup_factor=1.0,
                    attack_time_hours=float('inf'),
                    poh_vulnerability=False
                )
            else:
                # Grover capability emerging/present
                years_since = max(0, year - (grover_emergence_year - 5))
                
                # Exponential growth in Grover-capable qubits
                grover_qubits = int(1000 * (2 ** years_since) * rng.lognormal(0, 0.2))
                
                # Can attack SHA-256 if enough qubits
                can_attack = grover_qubits >= self.params.grover_qubits_sha256
                
                if can_attack:
                    # Speedup improves with more qubits
                    qubit_ratio = min(10, grover_qubits / self.params.grover_qubits_sha256)
                    speedup = self.params.grover_speedup_factor * np.sqrt(qubit_ratio)
                    
                    # Attack time in hours
                    operations = 2 ** 128
                    ops_per_second = self.params.grover_gate_speed_mhz * 1e6
                    attack_time = (operations / ops_per_second) / 3600 / speedup
                    
                    # Solana PoH vulnerability
                    poh_vulnerable = attack_time < 24
                else:
                    speedup = 1.0
                    attack_time = float('inf')
                    poh_vulnerable = False
                
                grover_cap = GroverCapability(
                    year=year,
                    logical_qubits=grover_qubits,
                    can_attack_sha256=can_attack,
                    speedup_factor=speedup,
                    attack_time_hours=attack_time,
                    poh_vulnerability=poh_vulnerable
                )
            
            grover_capabilities.append(grover_cap)
        
        return grover_capabilities
    
    def _update_with_live_data(self, timeline: QuantumTimeline):
        """Update timeline with live quantum data if available."""
        # This is a placeholder for live data integration
        # In production, this would fetch from quantum computing APIs
        # and adjust the timeline based on recent developments
        pass


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
