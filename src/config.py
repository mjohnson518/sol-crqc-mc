"""
Configuration module 

This module provides centralized configuration management for all simulation
parameters, including quantum development, network state, and economic factors.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any
import logging
from datetime import datetime


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantumParameters:
    """Parameters related to quantum computing development and capabilities."""
    
    # Resource requirements for Ed25519 breaking (Roetteler et al. 2017)
    logical_qubits_for_ed25519: int = 2330
    physical_to_logical_ratio: int = 1000  # Error correction overhead
    gate_speed_hz: float = 1e6  # 1 MHz gate speed
    circuit_depth: float = 1.42e9  # Circuit depth for Ed25519
    error_correction_distance: int = 15  # Code distance for error correction
    success_probability: float = 0.9  # Shor's algorithm success rate
    
    # Quantum development timeline parameters
    initial_qubits: int = 1000  # Starting logical qubits in 2025
    qubit_growth_rate: float = 1.5  # Annual growth multiplier
    breakthrough_probability_annual: float = 0.05  # Annual breakthrough chance
    
    # Breakthrough probabilities by year (cumulative)
    breakthrough_timeline: Dict[int, float] = field(default_factory=lambda: {
        2026: 0.01,
        2027: 0.02,
        2028: 0.05,
        2029: 0.08,
        2030: 0.12,
        2031: 0.15,
        2032: 0.18,
        2033: 0.20,
        2034: 0.25,
        2035: 0.30,
    })


@dataclass
class NetworkParameters:
    """Parameters describing the Solana network state and characteristics."""
    
    # Validator network
    n_validators: int = 1950  # Current number of validators (Solana Beach Dec 2024)
    total_stake_sol: float = 380_000_000  # Total staked SOL
    stake_gini_coefficient: float = 0.82  # Stake concentration metric
    
    # Geographic distribution
    geographic_distribution: Dict[str, float] = field(default_factory=lambda: {
        'north_america': 0.40,
        'europe': 0.30,
        'asia': 0.20,
        'other': 0.10
    })
    
    # Consensus thresholds
    consensus_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'halt': 0.333,  # 33.3% to halt the network
        'control': 0.667  # 66.7% for network control
    })
    
    # Network growth parameters
    validator_growth_rate: float = 1.05  # 5% annual growth in validator count
    stake_growth_rate: float = 1.10  # 10% annual growth in total stake
    
    # Key rotation and migration
    key_rotation_frequency_days: int = 365  # How often keys are rotated
    migration_adoption_rate: float = 0.8  # Fraction adopting quantum-safe


@dataclass
class EconomicParameters:
    """Economic and financial parameters for impact calculation."""
    
    # Current market values (Updated Dec 2024)
    sol_price_usd: float = 235.0
    total_value_locked_usd: float = 8_500_000_000  # $8.5B TVL
    daily_volume_usd: float = 3_800_000_000  # $3.8B daily volume
    
    # DeFi ecosystem breakdown
    defi_protocols: Dict[str, float] = field(default_factory=lambda: {
        'lending': 3_200_000_000,  # $3.2B (Marginfi, Kamino, etc.)
        'dex': 3_500_000_000,  # $3.5B (Jupiter, Raydium, Orca)
        'liquid_staking': 1_800_000_000  # $1.8B (Marinade, Jito)
    })
    
    # Market impact parameters
    attack_market_impact_multiplier: float = 3.0  # Total impact = direct loss * multiplier
    confidence_loss_factor: float = 0.7  # Network value reduction after attack
    recovery_time_months: int = 12  # Time to recover from attack
    
    # Risk metrics
    var_confidence_level: float = 0.95  # Value at Risk confidence
    cvar_confidence_level: float = 0.95  # Conditional VaR confidence


@dataclass
class SimulationParameters:
    """Main simulation parameters and execution settings."""
    
    # Simulation settings
    n_iterations: int = 10_000  # Number of Monte Carlo iterations
    random_seed: Optional[int] = 42  # For reproducibility
    confidence_level: float = 0.95  # Statistical confidence level
    n_cores: int = field(default_factory=lambda: os.cpu_count() or 4)
    
    # Time parameters
    start_year: int = 2025
    end_year: int = 2045
    time_step_days: int = 30  # Simulation time step
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    save_raw_results: bool = True
    generate_reports: bool = True
    verbose_logging: bool = False
    
    # Component parameters
    quantum: QuantumParameters = field(default_factory=QuantumParameters)
    network: NetworkParameters = field(default_factory=NetworkParameters)
    economic: EconomicParameters = field(default_factory=EconomicParameters)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create output directories if they don't exist
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Setup logging level
        if self.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration. If None, uses timestamp.
            
        Returns:
            Path to saved configuration file.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"config_{timestamp}.json"
        
        filepath = Path(filepath)
        
        # Convert to dictionary and handle Path objects
        config_dict = self._to_serializable_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'SimulationParameters':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file.
            
        Returns:
            SimulationParameters instance.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert nested dictionaries to dataclass instances
        if 'quantum' in data and isinstance(data['quantum'], dict):
            data['quantum'] = QuantumParameters(**data['quantum'])
        if 'network' in data and isinstance(data['network'], dict):
            data['network'] = NetworkParameters(**data['network'])
        if 'economic' in data and isinstance(data['economic'], dict):
            data['economic'] = EconomicParameters(**data['economic'])
        
        # Convert output_dir to Path
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
        
        config = cls(**data)
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    @classmethod
    def from_env(cls) -> 'SimulationParameters':
        """
        Create configuration from environment variables.
        
        Environment variables should be prefixed with 'SQMC_'.
        Example: SQMC_N_ITERATIONS=50000
        
        Returns:
            SimulationParameters instance.
        """
        config_dict = {}
        
        # Map environment variables to configuration parameters
        env_mappings = {
            'SQMC_N_ITERATIONS': ('n_iterations', int),
            'SQMC_RANDOM_SEED': ('random_seed', int),
            'SQMC_N_CORES': ('n_cores', int),
            'SQMC_START_YEAR': ('start_year', int),
            'SQMC_END_YEAR': ('end_year', int),
            'SQMC_OUTPUT_DIR': ('output_dir', str),
            'SQMC_VERBOSE': ('verbose_logging', lambda x: x.lower() == 'true'),
            # Quantum parameters
            'SQMC_LOGICAL_QUBITS': ('logical_qubits_for_ed25519', int),
            'SQMC_QUBIT_GROWTH_RATE': ('qubit_growth_rate', float),
            # Network parameters
            'SQMC_N_VALIDATORS': ('n_validators', int),
            'SQMC_TOTAL_STAKE': ('total_stake_sol', float),
            # Economic parameters
            'SQMC_SOL_PRICE': ('sol_price_usd', float),
            'SQMC_TVL': ('total_value_locked_usd', float),
        }
        
        # Process environment variables
        for env_var, (param_path, converter) in env_mappings.items():
            if env_var in os.environ:
                value = converter(os.environ[env_var])
                
                # Handle nested parameters
                if '.' in param_path:
                    parts = param_path.split('.')
                    if parts[0] not in config_dict:
                        config_dict[parts[0]] = {}
                    config_dict[parts[0]][parts[1]] = value
                else:
                    config_dict[param_path] = value
        
        # Create nested dataclasses
        quantum_dict = config_dict.pop('quantum', {})
        network_dict = config_dict.pop('network', {})
        economic_dict = config_dict.pop('economic', {})
        
        if quantum_dict:
            config_dict['quantum'] = QuantumParameters(**quantum_dict)
        if network_dict:
            config_dict['network'] = NetworkParameters(**network_dict)
        if economic_dict:
            config_dict['economic'] = EconomicParameters(**economic_dict)
        
        return cls(**config_dict)
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        config_dict = asdict(self)
        
        # Convert Path objects to strings
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = str(config_dict['output_dir'])
        
        return config_dict
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid.
            
        Raises:
            ValueError: If configuration is invalid.
        """
        errors = []
        
        # Validate simulation parameters
        if self.n_iterations <= 0:
            errors.append("n_iterations must be positive")
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            errors.append("confidence_level must be between 0 and 1")
        if self.start_year >= self.end_year:
            errors.append("start_year must be before end_year")
        
        # Validate quantum parameters
        if self.quantum.logical_qubits_for_ed25519 <= 0:
            errors.append("logical_qubits_for_ed25519 must be positive")
        if self.quantum.qubit_growth_rate <= 0:
            errors.append("qubit_growth_rate must be positive")
        
        # Validate network parameters
        if self.network.n_validators <= 0:
            errors.append("n_validators must be positive")
        if self.network.total_stake_sol <= 0:
            errors.append("total_stake_sol must be positive")
        
        # Validate economic parameters
        if self.economic.sol_price_usd <= 0:
            errors.append("sol_price_usd must be positive")
        if self.economic.total_value_locked_usd <= 0:
            errors.append("total_value_locked_usd must be positive")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
        return True
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Returns:
            String summary of key parameters.
        """
        summary_lines = [
            "=" * 60,
            "Solana Quantum Monte Carlo Simulation Configuration",
            "=" * 60,
            f"Simulation: {self.n_iterations:,} iterations on {self.n_cores} cores",
            f"Time Range: {self.start_year} - {self.end_year}",
            f"Random Seed: {self.random_seed}",
            "",
            "Quantum Parameters:",
            f"  - Ed25519 Logical Qubits: {self.quantum.logical_qubits_for_ed25519:,}",
            f"  - Physical/Logical Ratio: {self.quantum.physical_to_logical_ratio:,}:1",
            f"  - Qubit Growth Rate: {self.quantum.qubit_growth_rate:.1%} annually",
            "",
            "Network Parameters:",
            f"  - Validators: {self.network.n_validators:,}",
            f"  - Total Stake: {self.network.total_stake_sol:,.0f} SOL",
            f"  - Stake Gini: {self.network.stake_gini_coefficient:.2f}",
            "",
            "Economic Parameters:",
            f"  - SOL Price: ${self.economic.sol_price_usd:,.2f}",
            f"  - Total Value Locked: ${self.economic.total_value_locked_usd:,.0f}",
            f"  - Daily Volume: ${self.economic.daily_volume_usd:,.0f}",
            "=" * 60
        ]
        
        return "\n".join(summary_lines)


# Convenience function for quick setup
def get_default_config() -> SimulationParameters:
    """Get default configuration with standard parameters."""
    return SimulationParameters()


def get_test_config() -> SimulationParameters:
    """Get test configuration with reduced iterations for quick testing."""
    return SimulationParameters(
        n_iterations=100,
        n_cores=2,
        verbose_logging=True
    )


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    
    # Validate configuration
    config.validate()
    
    # Print summary
    print(config.summary())
    
    # Save configuration
    saved_path = config.save()
    
    # Load configuration
    loaded_config = SimulationParameters.load(saved_path)
