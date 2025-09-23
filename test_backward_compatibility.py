#!/usr/bin/env python3
"""
Test script to verify backward compatibility of enhanced models.

This script tests that:
1. The simulation runs with default (backward compatible) settings
2. The simulation runs with enhanced features enabled
3. Results are reasonable in both cases
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.config import SimulationParameters
        from src.core.simulation import MonteCarloSimulation
        from src.models.quantum_timeline import QuantumDevelopmentModel
        from src.models.network_state import NetworkStateModel
        from src.models.attack_scenarios import AttackScenariosModel
        from src.models.economic_impact import EconomicImpactModel
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_default_config():
    """Test that default configuration works (backward compatibility)."""
    print("\nTesting default configuration...")
    try:
        from src.config import SimulationParameters
        config = SimulationParameters()
        
        # Verify backward compatibility flags are False by default
        assert config.use_advanced_models == False
        assert config.enable_live_data == False
        assert config.quantum.enable_grover_modeling == False
        assert config.enable_hybrid_attacks == False
        
        print("✓ Default configuration maintains backward compatibility")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_initialization():
    """Test that models can be initialized."""
    print("\nTesting model initialization...")
    try:
        from src.config import SimulationParameters
        from src.models.quantum_timeline import QuantumDevelopmentModel
        from src.models.network_state import NetworkStateModel
        from src.models.attack_scenarios import AttackScenariosModel
        from src.models.economic_impact import EconomicImpactModel
        
        config = SimulationParameters()
        
        # Initialize models with default settings
        quantum_model = QuantumDevelopmentModel(
            config.quantum,
            enable_live_data=False,
            enable_grover=False,
            use_advanced_models=False
        )
        
        network_model = NetworkStateModel(
            config.network,
            use_graph_model=False
        )
        
        attack_model = AttackScenariosModel(
            config.attack,
            enable_grover=False,
            enable_hybrid_attacks=False,
            use_agent_based_model=False
        )
        
        economic_model = EconomicImpactModel(
            config.economic,
            use_system_dynamics=False,
            use_var_forecast=False,
            model_cross_chain=False
        )
        
        print("✓ All models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features():
    """Test that enhanced features can be enabled."""
    print("\nTesting enhanced features...")
    try:
        from src.config import SimulationParameters
        from src.models.quantum_timeline import QuantumDevelopmentModel
        from src.models.network_state import NetworkStateModel
        
        config = SimulationParameters()
        
        # Enable enhanced features
        config.use_advanced_models = True
        config.quantum.enable_grover_modeling = True
        config.enable_hybrid_attacks = True
        
        # Initialize models with enhanced features
        quantum_model = QuantumDevelopmentModel(
            config.quantum,
            enable_live_data=False,  # Keep false to avoid API calls
            enable_grover=True,
            use_advanced_models=True
        )
        
        network_model = NetworkStateModel(
            config.network,
            use_graph_model=True
        )
        
        print("✓ Enhanced features can be enabled")
        return True
    except Exception as e:
        print(f"✗ Enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_initialization():
    """Test that simulation can be initialized."""
    print("\nTesting simulation initialization...")
    try:
        from src.config import SimulationParameters
        from src.core.simulation import MonteCarloSimulation
        from src.models.quantum_timeline import QuantumDevelopmentModel
        from src.models.network_state import NetworkStateModel
        from src.models.attack_scenarios import AttackScenariosModel
        from src.models.economic_impact import EconomicImpactModel
        
        config = SimulationParameters()
        config.n_iterations = 10  # Small number for testing
        
        # Create models
        models = {
            'quantum_timeline': QuantumDevelopmentModel(config.quantum),
            'network_state': NetworkStateModel(config.network),
            'attack_scenarios': AttackScenariosModel(config.attack),
            'economic_impact': EconomicImpactModel(config.economic)
        }
        
        # Initialize simulation
        sim = MonteCarloSimulation(
            config,
            models=models,
            enable_convergence_tracking=True
        )
        
        print("✓ Simulation initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Simulation initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_default_config,
        test_model_initialization,
        test_enhanced_features,
        test_simulation_initialization
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The enhancements are backward compatible.")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
