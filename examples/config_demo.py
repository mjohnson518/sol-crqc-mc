#!/usr/bin/env python3
"""
Demonstration of the configuration system for Solana CRQC Monte Carlo Simulation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    SimulationParameters,
    QuantumParameters,
    NetworkParameters,
    EconomicParameters,
    get_default_config,
    get_test_config
)


def demonstrate_basic_usage():
    """Show basic configuration usage."""
    print("=" * 60)
    print("BASIC CONFIGURATION USAGE")
    print("=" * 60)
    
    # Create default configuration
    config = get_default_config()
    print(f"\nDefault configuration created:")
    print(f"  - Iterations: {config.n_iterations:,}")
    print(f"  - Random seed: {config.random_seed}")
    print(f"  - Time range: {config.start_year}-{config.end_year}")
    
    # Access nested parameters
    print(f"\nQuantum parameters:")
    print(f"  - Ed25519 qubits needed: {config.quantum.logical_qubits_for_ed25519:,}")
    print(f"  - Qubit growth rate: {config.quantum.qubit_growth_rate:.1%} per year")
    
    print(f"\nNetwork parameters:")
    print(f"  - Validators: {config.network.n_validators:,}")
    print(f"  - Total stake: {config.network.total_stake_sol:,.0f} SOL")
    
    print(f"\nEconomic parameters:")
    print(f"  - SOL price: ${config.economic.sol_price_usd:,.2f}")
    print(f"  - TVL: ${config.economic.total_value_locked_usd:,.0f}")


def demonstrate_custom_config():
    """Show how to create custom configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION")
    print("=" * 60)
    
    # Create custom configuration
    config = SimulationParameters(
        n_iterations=50_000,
        random_seed=123,
        n_cores=4,
        start_year=2026,
        end_year=2040,
        quantum=QuantumParameters(
            qubit_growth_rate=1.3,  # More conservative growth
            initial_qubits=500
        ),
        network=NetworkParameters(
            n_validators=5000,  # Projected future growth
            validator_growth_rate=1.2
        ),
        economic=EconomicParameters(
            sol_price_usd=250.0,  # Bullish scenario
            total_value_locked_usd=20_000_000_000  # $20B TVL
        )
    )
    
    print(f"\nCustom configuration created:")
    print(f"  - Iterations: {config.n_iterations:,}")
    print(f"  - Cores: {config.n_cores}")
    print(f"  - Time range: {config.start_year}-{config.end_year}")
    print(f"  - Qubit growth: {config.quantum.qubit_growth_rate:.1%} per year")
    print(f"  - Validators: {config.network.n_validators:,}")
    print(f"  - SOL price: ${config.economic.sol_price_usd:,.2f}")


def demonstrate_validation():
    """Show configuration validation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Valid configuration
    valid_config = get_default_config()
    try:
        valid_config.validate()
        print("\n✓ Default configuration is valid")
    except ValueError as e:
        print(f"\n✗ Validation failed: {e}")
    
    # Invalid configuration
    invalid_config = SimulationParameters(
        n_iterations=-100,  # Invalid: negative
        confidence_level=1.5,  # Invalid: > 1
        start_year=2030,
        end_year=2025  # Invalid: end before start
    )
    
    print("\nTesting invalid configuration:")
    try:
        invalid_config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Validation failed (expected):\n{e}")


def demonstrate_save_load():
    """Show saving and loading configuration."""
    print("\n" + "=" * 60)
    print("SAVE AND LOAD CONFIGURATION")
    print("=" * 60)
    
    # Create configuration
    config = SimulationParameters(
        n_iterations=25_000,
        random_seed=999
    )
    
    # Save configuration
    save_path = config.save()
    print(f"\nConfiguration saved to: {save_path}")
    
    # Load configuration
    loaded_config = SimulationParameters.load(save_path)
    print(f"\nConfiguration loaded successfully:")
    print(f"  - Iterations: {loaded_config.n_iterations:,}")
    print(f"  - Random seed: {loaded_config.random_seed}")
    
    # Clean up
    save_path.unlink()
    print(f"\nCleaned up temporary file")


def demonstrate_summary():
    """Show configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    config = get_default_config()
    print("\n" + config.summary())


def demonstrate_scenarios():
    """Show different simulation scenarios."""
    print("\n" + "=" * 60)
    print("SIMULATION SCENARIOS")
    print("=" * 60)
    
    scenarios = {
        "Conservative": SimulationParameters(
            quantum=QuantumParameters(
                qubit_growth_rate=1.2,  # Slower quantum progress
                breakthrough_probability_annual=0.02
            ),
            network=NetworkParameters(
                migration_adoption_rate=0.95  # Fast migration to quantum-safe
            )
        ),
        "Baseline": get_default_config(),
        "Aggressive": SimulationParameters(
            quantum=QuantumParameters(
                qubit_growth_rate=2.0,  # Rapid quantum progress
                breakthrough_probability_annual=0.10
            ),
            network=NetworkParameters(
                migration_adoption_rate=0.5  # Slow migration to quantum-safe
            )
        ),
        "Quick Test": get_test_config()
    }
    
    for name, config in scenarios.items():
        print(f"\n{name} Scenario:")
        print(f"  - Iterations: {config.n_iterations:,}")
        print(f"  - Qubit growth: {config.quantum.qubit_growth_rate:.1%}/year")
        print(f"  - Migration rate: {config.network.migration_adoption_rate:.0%}")


def main():
    """Run all demonstrations."""
    demonstrate_basic_usage()
    demonstrate_custom_config()
    demonstrate_validation()
    demonstrate_save_load()
    demonstrate_summary()
    demonstrate_scenarios()
    
    print("\n" + "=" * 60)
    print("Configuration demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
