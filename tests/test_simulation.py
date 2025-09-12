"""
Tests for the core simulation engine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from src.core.simulation import (
    MonteCarloSimulation,
    SimulationResult,
    SimulationState
)
from src.core.random_engine import RandomEngine, RandomStreamManager
from src.core.results_collector import ResultsCollector, ResultsStatistics
from src.config import SimulationParameters, get_test_config


class TestSimulationResult:
    """Test SimulationResult class."""
    
    def test_initialization(self):
        """Test SimulationResult initialization."""
        result = SimulationResult(
            iteration_id=1,
            quantum_timeline={'crqc_year': 2035},
            network_state={'validators': 5000},
            attack_results={'attacks_successful': 1},
            economic_impact={'total_loss_usd': 1e10},
            first_attack_year=2035.5,
            runtime_seconds=0.1
        )
        
        assert result.iteration_id == 1
        assert result.first_attack_year == 2035.5
        assert result.runtime_seconds == 0.1
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SimulationResult(
            iteration_id=1,
            quantum_timeline={'crqc_year': 2035},
            network_state={'validators': 5000},
            attack_results={'attacks_successful': 1},
            economic_impact={'total_loss_usd': 1e10}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['iteration_id'] == 1
        assert 'quantum_timeline' in result_dict
        assert 'economic_impact' in result_dict


class TestSimulationState:
    """Test SimulationState class."""
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        state = SimulationState(total_iterations=100)
        
        assert state.progress == 0.0
        
        state.completed_iterations = 50
        assert state.progress == 50.0
        
        state.completed_iterations = 100
        assert state.progress == 100.0
    
    def test_runtime_calculation(self):
        """Test runtime calculation."""
        state = SimulationState(total_iterations=100)
        time.sleep(0.1)  # Small delay
        
        assert state.runtime > 0
        assert state.runtime < 1  # Should be less than 1 second
        
        # Test with end time
        state.end_time = state.start_time + 10
        assert state.runtime == 10
    
    def test_iterations_per_second(self):
        """Test iteration rate calculation."""
        state = SimulationState(total_iterations=100)
        state.completed_iterations = 50
        state.end_time = state.start_time + 5  # 5 seconds
        
        assert state.iterations_per_second == 10.0


class TestMonteCarloSimulation:
    """Test MonteCarloSimulation class."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        config = get_test_config()
        sim = MonteCarloSimulation(config)
        
        assert sim.config == config
        assert isinstance(sim.random_engine, RandomEngine)
        assert isinstance(sim.results_collector, ResultsCollector)
        assert sim.state.total_iterations == config.n_iterations
    
    def test_validation(self):
        """Test configuration validation during initialization."""
        # Invalid configuration
        config = SimulationParameters(n_iterations=-100)
        
        with pytest.raises(ValueError):
            sim = MonteCarloSimulation(config)
    
    def test_single_iteration(self):
        """Test running a single iteration."""
        config = get_test_config()
        sim = MonteCarloSimulation(config)
        
        result = sim._run_single_iteration(0, 12345)
        
        assert isinstance(result, SimulationResult)
        assert result.iteration_id == 0
        assert 'crqc_year' in result.quantum_timeline
        assert 'total_loss_usd' in result.economic_impact
    
    def test_batch_creation(self):
        """Test batch creation for parallel execution."""
        config = SimulationParameters(n_iterations=100)
        sim = MonteCarloSimulation(config)
        
        batches = sim._create_batches(batch_size=25)
        
        assert len(batches) == 4
        assert batches[0] == (0, 25)
        assert batches[-1] == (75, 100)
    
    def test_sequential_run(self):
        """Test sequential execution."""
        config = SimulationParameters(n_iterations=10, n_cores=1)
        sim = MonteCarloSimulation(config)
        
        results = sim.run()
        
        assert 'metadata' in results
        assert results['metadata']['successful_iterations'] == 10
        assert results['metadata']['failed_iterations'] == 0
    
    def test_parallel_run(self):
        """Test parallel execution."""
        config = SimulationParameters(n_iterations=20, n_cores=2)
        sim = MonteCarloSimulation(config)
        
        results = sim.run()
        
        assert 'metadata' in results
        assert results['metadata']['successful_iterations'] == 20
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config1 = SimulationParameters(n_iterations=10, random_seed=42)
        sim1 = MonteCarloSimulation(config1)
        results1 = sim1.run()
        
        config2 = SimulationParameters(n_iterations=10, random_seed=42)
        sim2 = MonteCarloSimulation(config2)
        results2 = sim2.run()
        
        # Check that key metrics are identical
        if 'raw_results' in results1 and 'raw_results' in results2:
            for i in range(10):
                assert results1['raw_results'][i]['iteration_id'] == \
                       results2['raw_results'][i]['iteration_id']
    
    def test_model_validation(self):
        """Test model validation."""
        config = get_test_config()
        sim = MonteCarloSimulation(config)
        
        # No models provided
        assert sim.validate_models() is False
        
        # With models
        models = {
            'quantum_timeline': Mock(),
            'network_state': Mock(),
            'attack_model': Mock(),
            'economic_model': Mock()
        }
        sim = MonteCarloSimulation(config, models=models)
        assert sim.validate_models() is True


class TestRandomEngine:
    """Test RandomEngine class."""
    
    def test_initialization(self):
        """Test random engine initialization."""
        engine = RandomEngine(seed=42)
        
        assert engine.master_seed == 42
        assert 'quantum' in engine.component_seeds
        assert 'network' in engine.component_seeds
    
    def test_iteration_seed_generation(self):
        """Test iteration seed generation."""
        engine = RandomEngine(seed=42)
        
        seed1 = engine.get_iteration_seed(0)
        seed2 = engine.get_iteration_seed(1)
        
        assert seed1 != seed2  # Different iterations get different seeds
        
        # Same iteration should get same seed
        seed1_again = engine.get_iteration_seed(0)
        assert seed1 == seed1_again
    
    def test_component_rng(self):
        """Test component RNG creation."""
        engine = RandomEngine(seed=42)
        
        rng1 = engine.get_component_rng('quantum', 1000)
        rng2 = engine.get_component_rng('network', 1000)
        
        # Different components should produce different values
        val1 = rng1.random()
        val2 = rng2.random()
        assert val1 != val2
    
    def test_batch_seeds(self):
        """Test batch seed creation."""
        engine = RandomEngine(seed=42)
        
        seeds = engine.create_batch_seeds(batch_size=5, batch_id=0)
        
        assert len(seeds) == 5
        assert len(set(seeds)) == 5  # All unique
    
    def test_seed_splitting(self):
        """Test seed splitting for multiple streams."""
        engine = RandomEngine(seed=42)
        
        seeds = engine.split_seed(1000, n_streams=4)
        
        assert len(seeds) == 4
        assert len(set(seeds)) == 4  # All unique


class TestRandomStreamManager:
    """Test RandomStreamManager class."""
    
    def test_stream_creation(self):
        """Test creating random streams."""
        manager = RandomStreamManager(base_seed=42)
        
        stream1 = manager.create_stream('test1')
        stream2 = manager.create_stream('test2')
        
        assert isinstance(stream1, np.random.RandomState)
        assert isinstance(stream2, np.random.RandomState)
        
        # Different streams produce different values
        val1 = stream1.random()
        val2 = stream2.random()
        assert val1 != val2
    
    def test_stream_retrieval(self):
        """Test getting existing streams."""
        manager = RandomStreamManager(base_seed=42)
        
        stream1 = manager.create_stream('test')
        stream2 = manager.get_stream('test')
        
        assert stream1 is stream2  # Same object
        
        with pytest.raises(KeyError):
            manager.get_stream('nonexistent')
    
    def test_stream_reset(self):
        """Test resetting streams."""
        manager = RandomStreamManager(base_seed=42)
        
        stream = manager.create_stream('test')
        val1 = stream.random()
        
        manager.reset_stream('test')
        stream = manager.get_stream('test')
        val2 = stream.random()
        
        # After reset, should produce same value as initially
        assert val1 == val2


class TestResultsCollector:
    """Test ResultsCollector class."""
    
    def test_add_result(self):
        """Test adding results."""
        collector = ResultsCollector()
        
        result = SimulationResult(
            iteration_id=0,
            quantum_timeline={'crqc_year': 2035},
            network_state={},
            attack_results={'attacks_successful': 1},
            economic_impact={'total_loss_usd': 1e10},
            first_attack_year=2035
        )
        
        collector.add_result(result)
        
        assert len(collector.results) == 1
        assert len(collector.first_attack_years) == 1
        assert collector.first_attack_years[0] == 2035
    
    def test_summary_computation(self):
        """Test summary statistics computation."""
        collector = ResultsCollector()
        
        # Add multiple results
        for i in range(10):
            result = SimulationResult(
                iteration_id=i,
                quantum_timeline={'crqc_year': 2030 + i},
                network_state={},
                attack_results={'attacks_successful': 1 if i % 2 == 0 else 0},
                economic_impact={'total_loss_usd': (i + 1) * 1e9},
                first_attack_year=2030 + i
            )
            collector.add_result(result)
        
        summary = collector.get_summary()
        
        assert summary['n_iterations'] == 10
        assert 'first_attack_year' in summary['metrics']
        assert 'economic_loss_usd' in summary['metrics']
        assert summary['metrics']['attack_success_rate'] == 0.5
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        collector = ResultsCollector()
        
        # Add results with economic losses
        np.random.seed(42)
        for i in range(100):
            result = SimulationResult(
                iteration_id=i,
                quantum_timeline={},
                network_state={},
                attack_results={},
                economic_impact={'total_loss_usd': np.random.exponential(1e10)},
                first_attack_year=2030 + np.random.normal(0, 3)
            )
            collector.add_result(result)
        
        risk_metrics = collector.get_risk_metrics()
        
        assert 'var' in risk_metrics
        assert 'cvar' in risk_metrics
        assert 'attack_timing' in risk_metrics
        assert 95 in risk_metrics['var']
    
    def test_dataframe_conversion(self):
        """Test conversion to DataFrame."""
        collector = ResultsCollector()
        
        # Add some results
        for i in range(5):
            result = SimulationResult(
                iteration_id=i,
                quantum_timeline={'crqc_year': 2030 + i},
                network_state={},
                attack_results={'attacks_successful': 1},
                economic_impact={'total_loss_usd': 1e10},
                first_attack_year=2030 + i
            )
            collector.add_result(result)
        
        df = collector.get_dataframe()
        
        assert len(df) == 5
        assert 'iteration_id' in df.columns
        assert 'first_attack_year' in df.columns
        assert 'total_loss_usd' in df.columns


class TestResultsStatistics:
    """Test ResultsStatistics class."""
    
    def test_from_array(self):
        """Test creating statistics from array."""
        data = np.array([1, 2, 3, 4, 5])
        
        stats = ResultsStatistics.from_array(data)
        
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert 50 in stats.percentiles
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ResultsStatistics(
            mean=10.0,
            median=9.0,
            std=2.0,
            min=5.0,
            max=15.0,
            percentiles={25: 7.5, 75: 12.5}
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict['mean'] == 10.0
        assert stats_dict['percentiles'][25] == 7.5
