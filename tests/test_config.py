"""
Tests for configuration module.
"""

import pytest
import json
from pathlib import Path
import tempfile
import os
from src.config import (
    SimulationParameters,
    QuantumParameters,
    NetworkParameters,
    EconomicParameters,
    get_default_config,
    get_test_config
)


class TestSimulationParameters:
    """Test SimulationParameters class."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        config = SimulationParameters()
        
        assert config.n_iterations == 10_000
        assert config.random_seed == 42
        assert config.start_year == 2025
        assert config.end_year == 2045
        assert config.confidence_level == 0.95
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        config = SimulationParameters(
            n_iterations=5000,
            random_seed=123,
            start_year=2026
        )
        
        assert config.n_iterations == 5000
        assert config.random_seed == 123
        assert config.start_year == 2026
    
    def test_nested_parameters(self):
        """Test nested parameter dataclasses."""
        config = SimulationParameters()
        
        # Test quantum parameters
        assert isinstance(config.quantum, QuantumParameters)
        assert config.quantum.logical_qubits_for_ed25519 == 2330
        assert config.quantum.qubit_growth_rate == 1.5
        
        # Test network parameters
        assert isinstance(config.network, NetworkParameters)
        assert config.network.n_validators == 1032
        assert config.network.total_stake_sol == 400_000_000
        
        # Test economic parameters
        assert isinstance(config.economic, EconomicParameters)
        assert config.economic.sol_price_usd == 185.0
        assert config.economic.total_value_locked_usd == 12_200_000_000
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = SimulationParameters(
                n_iterations=1000,
                output_dir=Path(tmpdir)
            )
            
            # Save config
            filepath = config.save()
            assert filepath.exists()
            
            # Load config
            loaded_config = SimulationParameters.load(filepath)
            
            # Verify loaded values
            assert loaded_config.n_iterations == 1000
            assert loaded_config.random_seed == config.random_seed
            assert loaded_config.quantum.logical_qubits_for_ed25519 == 2330
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration."""
        config = SimulationParameters()
        assert config.validate() is True
    
    def test_validation_invalid_iterations(self):
        """Test validation with invalid iterations."""
        config = SimulationParameters(n_iterations=-100)
        
        with pytest.raises(ValueError, match="n_iterations must be positive"):
            config.validate()
    
    def test_validation_invalid_confidence(self):
        """Test validation with invalid confidence level."""
        config = SimulationParameters(confidence_level=1.5)
        
        with pytest.raises(ValueError, match="confidence_level must be between"):
            config.validate()
    
    def test_validation_invalid_years(self):
        """Test validation with invalid year range."""
        config = SimulationParameters(start_year=2030, end_year=2025)
        
        with pytest.raises(ValueError, match="start_year must be before end_year"):
            config.validate()
    
    def test_summary_generation(self):
        """Test summary string generation."""
        config = SimulationParameters()
        summary = config.summary()
        
        assert "Solana Quantum Monte Carlo Simulation" in summary
        assert "10,000 iterations" in summary
        assert "2025 - 2045" in summary
        assert "Ed25519 Logical Qubits: 2,330" in summary
    
    def test_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("SQMC_N_ITERATIONS", "5000")
        monkeypatch.setenv("SQMC_RANDOM_SEED", "999")
        monkeypatch.setenv("SQMC_VERBOSE", "true")
        monkeypatch.setenv("SQMC_SOL_PRICE", "200.5")
        
        # Create config from environment
        config = SimulationParameters.from_env()
        
        assert config.n_iterations == 5000
        assert config.random_seed == 999
        assert config.verbose_logging is True
        assert config.economic.sol_price_usd == 200.5


class TestQuantumParameters:
    """Test QuantumParameters class."""
    
    def test_default_values(self):
        """Test default quantum parameter values."""
        quantum = QuantumParameters()
        
        assert quantum.logical_qubits_for_ed25519 == 2330
        assert quantum.physical_to_logical_ratio == 1000
        assert quantum.gate_speed_hz == 1e6
        assert quantum.success_probability == 0.9
    
    def test_breakthrough_timeline(self):
        """Test breakthrough timeline structure."""
        quantum = QuantumParameters()
        
        assert 2030 in quantum.breakthrough_timeline
        assert quantum.breakthrough_timeline[2030] == 0.12
        assert all(0 <= p <= 1 for p in quantum.breakthrough_timeline.values())


class TestNetworkParameters:
    """Test NetworkParameters class."""
    
    def test_default_values(self):
        """Test default network parameter values."""
        network = NetworkParameters()
        
        assert network.n_validators == 1032
        assert network.stake_gini_coefficient == 0.84
        assert network.validator_growth_rate == 1.15
    
    def test_geographic_distribution(self):
        """Test geographic distribution sums to 1."""
        network = NetworkParameters()
        
        total = sum(network.geographic_distribution.values())
        assert abs(total - 1.0) < 1e-6
    
    def test_consensus_thresholds(self):
        """Test consensus thresholds are valid."""
        network = NetworkParameters()
        
        assert network.consensus_thresholds['halt'] == 0.333
        assert network.consensus_thresholds['control'] == 0.667
        assert network.consensus_thresholds['halt'] < network.consensus_thresholds['control']


class TestEconomicParameters:
    """Test EconomicParameters class."""
    
    def test_default_values(self):
        """Test default economic parameter values."""
        economic = EconomicParameters()
        
        assert economic.sol_price_usd == 185.0
        assert economic.total_value_locked_usd == 12_200_000_000
        assert economic.daily_volume_usd == 2_500_000_000
    
    def test_defi_protocols(self):
        """Test DeFi protocol breakdown."""
        economic = EconomicParameters()
        
        total_defi = sum(economic.defi_protocols.values())
        assert total_defi == economic.total_value_locked_usd
    
    def test_risk_metrics(self):
        """Test risk metric parameters."""
        economic = EconomicParameters()
        
        assert 0 < economic.var_confidence_level < 1
        assert 0 < economic.cvar_confidence_level < 1
        assert economic.attack_market_impact_multiplier > 1


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        
        assert isinstance(config, SimulationParameters)
        assert config.n_iterations == 10_000
    
    def test_get_test_config(self):
        """Test get_test_config function."""
        config = get_test_config()
        
        assert isinstance(config, SimulationParameters)
        assert config.n_iterations == 100
        assert config.n_cores == 2
        assert config.verbose_logging is True
