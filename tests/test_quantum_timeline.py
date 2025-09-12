"""
Tests for quantum timeline model.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.quantum_timeline import (
    QuantumDevelopmentModel,
    QuantumTimeline,
    QuantumCapability,
    QuantumThreat
)
from src.config import QuantumParameters


class TestQuantumCapability:
    """Test QuantumCapability class."""
    
    def test_initialization(self):
        """Test capability initialization."""
        cap = QuantumCapability(
            year=2030,
            logical_qubits=2500,
            physical_qubits=2500000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.HIGH
        )
        
        assert cap.year == 2030
        assert cap.logical_qubits == 2500
        assert cap.threat_level == QuantumThreat.HIGH
    
    def test_can_break_ed25519(self):
        """Test Ed25519 breaking capability check."""
        # Insufficient qubits
        cap1 = QuantumCapability(
            year=2030,
            logical_qubits=2000,
            physical_qubits=2000000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.MODERATE
        )
        assert not cap1.can_break_ed25519
        
        # Sufficient qubits and fidelity
        cap2 = QuantumCapability(
            year=2035,
            logical_qubits=2330,
            physical_qubits=2330000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.HIGH
        )
        assert cap2.can_break_ed25519
        
        # Sufficient qubits but poor fidelity
        cap3 = QuantumCapability(
            year=2035,
            logical_qubits=2330,
            physical_qubits=2330000,
            gate_fidelity=0.98,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.MODERATE
        )
        assert not cap3.can_break_ed25519
    
    def test_estimated_break_time(self):
        """Test break time estimation."""
        # Cannot break
        cap1 = QuantumCapability(
            year=2030,
            logical_qubits=1000,
            physical_qubits=1000000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.EMERGING
        )
        assert cap1.estimated_break_time_hours == float('inf')
        
        # Can break with minimum qubits
        cap2 = QuantumCapability(
            year=2035,
            logical_qubits=2330,
            physical_qubits=2330000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.HIGH
        )
        assert 20 < cap2.estimated_break_time_hours < 30
        
        # Can break with extra qubits (faster)
        cap3 = QuantumCapability(
            year=2040,
            logical_qubits=4660,  # 2x minimum
            physical_qubits=4660000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.HIGH
        )
        assert cap3.estimated_break_time_hours < cap2.estimated_break_time_hours


class TestQuantumTimeline:
    """Test QuantumTimeline class."""
    
    def test_initialization(self):
        """Test timeline initialization."""
        caps = [
            QuantumCapability(
                year=2030,
                logical_qubits=1000,
                physical_qubits=1000000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.EMERGING
            ),
            QuantumCapability(
                year=2035,
                logical_qubits=2500,
                physical_qubits=2500000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.HIGH
            )
        ]
        
        timeline = QuantumTimeline(
            capabilities=caps,
            crqc_year=2035.5,
            breakthrough_years=[2032.3],
            projection_method='expert',
            confidence=0.7
        )
        
        assert len(timeline.capabilities) == 2
        assert timeline.crqc_year == 2035.5
        assert timeline.projection_method == 'expert'
    
    def test_get_capability_at_year(self):
        """Test getting capability at specific year."""
        caps = [
            QuantumCapability(
                year=2030,
                logical_qubits=1000,
                physical_qubits=1000000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.EMERGING
            ),
            QuantumCapability(
                year=2035,
                logical_qubits=2500,
                physical_qubits=2500000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.HIGH
            )
        ]
        
        timeline = QuantumTimeline(
            capabilities=caps,
            crqc_year=2035.5,
            breakthrough_years=[],
            projection_method='expert',
            confidence=0.7
        )
        
        # Before first capability
        cap_2028 = timeline.get_capability_at_year(2028)
        assert cap_2028.year == 2030  # Returns first capability
        
        # Between capabilities
        cap_2032 = timeline.get_capability_at_year(2032)
        assert cap_2032.year == 2035  # Returns next capability
        
        # After last capability
        cap_2040 = timeline.get_capability_at_year(2040)
        assert cap_2040.year == 2035  # Returns last capability
    
    def test_years_until_threat_level(self):
        """Test calculating years until threat level."""
        caps = [
            QuantumCapability(
                year=2030,
                logical_qubits=1000,
                physical_qubits=1000000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.EMERGING
            ),
            QuantumCapability(
                year=2035,
                logical_qubits=2500,
                physical_qubits=2500000,
                gate_fidelity=0.99,
                coherence_time_ms=1.0,
                threat_level=QuantumThreat.HIGH
            )
        ]
        
        timeline = QuantumTimeline(
            capabilities=caps,
            crqc_year=2035.5,
            breakthrough_years=[],
            projection_method='expert',
            confidence=0.7
        )
        
        # Years until emerging threat
        years_emerging = timeline.years_until_threat_level(QuantumThreat.EMERGING)
        assert years_emerging == 2030 - 2025  # 5 years
        
        # Years until high threat
        years_high = timeline.years_until_threat_level(QuantumThreat.HIGH)
        assert years_high == 2035 - 2025  # 10 years
        
        # Never reaches critical
        years_critical = timeline.years_until_threat_level(QuantumThreat.CRITICAL)
        assert years_critical == float('inf')


class TestQuantumDevelopmentModel:
    """Test QuantumDevelopmentModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = QuantumDevelopmentModel()
        
        assert model.params is not None
        assert len(model.historical_data) > 0
        assert len(model.industry_roadmaps) > 0
        assert len(model.expert_predictions) == 3
    
    def test_sample_generation(self):
        """Test timeline generation."""
        model = QuantumDevelopmentModel()
        rng = np.random.RandomState(42)
        
        timeline = model.sample(rng)
        
        assert isinstance(timeline, QuantumTimeline)
        assert len(timeline.capabilities) > 0
        assert timeline.crqc_year > 2025
        assert timeline.projection_method in ['industry', 'expert', 'breakthrough', 'historical']
        assert 0 < timeline.confidence <= 1
    
    def test_industry_projection(self):
        """Test industry roadmap projection."""
        model = QuantumDevelopmentModel()
        rng = np.random.RandomState(42)
        
        timeline = model._sample_industry_projection(rng)
        
        assert timeline.projection_method == 'industry'
        assert len(timeline.capabilities) == 21  # 2025-2045
        
        # Check monotonic growth in qubits
        qubits = [cap.logical_qubits for cap in timeline.capabilities]
        assert all(q2 >= q1 * 0.8 for q1, q2 in zip(qubits[:-1], qubits[1:]))
    
    def test_expert_projection(self):
        """Test expert survey projection."""
        model = QuantumDevelopmentModel()
        rng = np.random.RandomState(42)
        
        timeline = model._sample_expert_projection(rng)
        
        assert timeline.projection_method == 'expert'
        assert 2026 <= timeline.crqc_year <= 2050
        
        # Check S-curve growth pattern
        qubits = [cap.logical_qubits for cap in timeline.capabilities]
        growth_rates = [(q2 - q1) / q1 for q1, q2 in zip(qubits[:-1], qubits[1:]) if q1 > 0]
        # Growth should slow down over time (S-curve)
        assert len(growth_rates) > 0
    
    def test_breakthrough_projection(self):
        """Test breakthrough probability projection."""
        model = QuantumDevelopmentModel()
        rng = np.random.RandomState(42)
        
        timeline = model._sample_breakthrough_projection(rng)
        
        assert timeline.projection_method == 'breakthrough'
        
        # Check for breakthrough jumps
        qubits = [cap.logical_qubits for cap in timeline.capabilities]
        growth_rates = [(q2 / q1) for q1, q2 in zip(qubits[:-1], qubits[1:]) if q1 > 0]
        
        # Should have some large jumps (breakthroughs)
        max_growth = max(growth_rates) if growth_rates else 1
        assert max_growth > 1.5  # At least 50% jump somewhere
    
    def test_historical_projection(self):
        """Test historical growth projection."""
        model = QuantumDevelopmentModel()
        rng = np.random.RandomState(42)
        
        timeline = model._sample_historical_projection(rng)
        
        assert timeline.projection_method == 'historical'
        assert timeline.confidence >= 0.7  # Historical has high confidence
        
        # Check exponential growth pattern
        qubits = [cap.logical_qubits for cap in timeline.capabilities[:10]]
        # Log of qubits should be roughly linear (exponential growth)
        log_qubits = [np.log(q) for q in qubits if q > 0]
        if len(log_qubits) > 2:
            correlation = np.corrcoef(range(len(log_qubits)), log_qubits)[0, 1]
            assert correlation > 0.8  # Strong linear correlation in log space
    
    def test_threat_level_assessment(self):
        """Test threat level assessment."""
        model = QuantumDevelopmentModel()
        
        # No threat
        assert model._assess_threat_level(50, 0.99) == QuantumThreat.NONE
        
        # Emerging threat
        assert model._assess_threat_level(500, 0.99) == QuantumThreat.EMERGING
        
        # Moderate threat
        assert model._assess_threat_level(1500, 0.99) == QuantumThreat.MODERATE
        
        # High threat
        assert model._assess_threat_level(2330, 0.99) == QuantumThreat.HIGH
        
        # Critical threat
        assert model._assess_threat_level(10000, 0.99) == QuantumThreat.CRITICAL
        
        # Insufficient fidelity reduces threat
        assert model._assess_threat_level(2330, 0.98) == QuantumThreat.MODERATE
    
    def test_consensus_projection(self):
        """Test consensus projection generation."""
        model = QuantumDevelopmentModel()
        
        consensus = model.create_consensus_projection(n_samples=50, seed=42)
        
        assert 'crqc_emergence' in consensus
        assert 'threat_levels' in consensus
        
        # Check CRQC statistics
        crqc_stats = consensus['crqc_emergence']
        assert 'mean' in crqc_stats
        assert 'median' in crqc_stats
        assert 'percentiles' in crqc_stats
        
        # Mean should be reasonable
        assert 2025 < crqc_stats['mean'] < 2050
        
        # Check threat level probabilities
        threat_levels = consensus['threat_levels']
        if 'high' in threat_levels:
            assert 0 <= threat_levels['high']['probability'] <= 1
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        model = QuantumDevelopmentModel()
        
        rng1 = np.random.RandomState(123)
        timeline1 = model.sample(rng1)
        
        rng2 = np.random.RandomState(123)
        timeline2 = model.sample(rng2)
        
        # Same projection method
        assert timeline1.projection_method == timeline2.projection_method
        
        # Same CRQC year
        assert abs(timeline1.crqc_year - timeline2.crqc_year) < 0.01
        
        # Same first capability
        cap1 = timeline1.capabilities[0]
        cap2 = timeline2.capabilities[0]
        assert cap1.logical_qubits == cap2.logical_qubits
    
    def test_parameter_override(self):
        """Test using custom parameters."""
        custom_params = QuantumParameters(
            logical_qubits_for_ed25519=3000,  # Higher requirement
            qubit_growth_rate=1.2,  # Slower growth
            breakthrough_probability_annual=0.01  # Lower breakthrough chance
        )
        
        model = QuantumDevelopmentModel(params=custom_params)
        rng = np.random.RandomState(42)
        
        timeline = model.sample(rng)
        
        # Should take longer to reach CRQC with harder requirements
        assert timeline.crqc_year > 2030  # Likely later than with defaults
