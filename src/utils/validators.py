"""
Validation utilities for simulation parameters and data.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates simulation configuration parameters."""
    
    @staticmethod
    def validate_numeric_range(
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        param_name: str = "parameter"
    ) -> bool:
        """
        Validate that a numeric value is within specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            param_name: Name of parameter for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If value is out of range
        """
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        return True
    
    @staticmethod
    def validate_probability(value: float, param_name: str = "probability") -> bool:
        """
        Validate that a value is a valid probability [0, 1].
        
        Args:
            value: Value to validate
            param_name: Name of parameter for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If value is not a valid probability
        """
        return ConfigValidator.validate_numeric_range(
            value, 0.0, 1.0, param_name
        )
    
    @staticmethod
    def validate_distribution(
        distribution: Dict[Any, float],
        param_name: str = "distribution",
        should_sum_to_one: bool = True
    ) -> bool:
        """
        Validate a probability distribution.
        
        Args:
            distribution: Dictionary mapping keys to probabilities
            param_name: Name of parameter for error messages
            should_sum_to_one: Whether probabilities should sum to 1.0
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If distribution is invalid
        """
        if not distribution:
            raise ValueError(f"{param_name} cannot be empty")
        
        # Check all values are valid probabilities
        for key, prob in distribution.items():
            ConfigValidator.validate_probability(
                prob, f"{param_name}[{key}]"
            )
        
        # Check sum if required
        if should_sum_to_one:
            total = sum(distribution.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"{param_name} probabilities must sum to 1.0, got {total}"
                )
        
        return True
    
    @staticmethod
    def validate_year_range(
        start_year: int,
        end_year: int,
        min_year: int = 2020,
        max_year: int = 2100
    ) -> bool:
        """
        Validate a year range.
        
        Args:
            start_year: Start year
            end_year: End year
            min_year: Minimum allowed year
            max_year: Maximum allowed year
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If year range is invalid
        """
        ConfigValidator.validate_numeric_range(
            start_year, min_year, max_year, "start_year"
        )
        ConfigValidator.validate_numeric_range(
            end_year, min_year, max_year, "end_year"
        )
        
        if start_year >= end_year:
            raise ValueError(
                f"start_year ({start_year}) must be before end_year ({end_year})"
            )
        
        return True
    
    @staticmethod
    def validate_file_path(
        filepath: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = True
    ) -> bool:
        """
        Validate a file path.
        
        Args:
            filepath: Path to validate
            must_exist: Whether file must already exist
            must_be_file: Whether path must be a file (not directory)
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If path is invalid
        """
        path = Path(filepath)
        
        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if path.exists():
            if must_be_file and not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            if not must_be_file and not path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")
        
        return True


class DataValidator:
    """Validates simulation input and output data."""
    
    @staticmethod
    def validate_quantum_timeline(timeline_data: Dict[str, Any]) -> bool:
        """
        Validate quantum development timeline data.
        
        Args:
            timeline_data: Timeline data to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        required_fields = ['crqc_year', 'qubit_trajectory']
        
        for field in required_fields:
            if field not in timeline_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate CRQC year is reasonable
        ConfigValidator.validate_numeric_range(
            timeline_data['crqc_year'],
            2025, 2100,
            "crqc_year"
        )
        
        # Validate qubit trajectory is non-empty
        if not timeline_data['qubit_trajectory']:
            raise ValueError("qubit_trajectory cannot be empty")
        
        return True
    
    @staticmethod
    def validate_attack_result(attack_data: Dict[str, Any]) -> bool:
        """
        Validate attack simulation result data.
        
        Args:
            attack_data: Attack result data to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data is invalid
        """
        required_fields = ['feasible', 'success_probability']
        
        for field in required_fields:
            if field not in attack_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate success probability
        if attack_data['feasible']:
            ConfigValidator.validate_probability(
                attack_data['success_probability'],
                "attack success_probability"
            )
        
        return True
    
    @staticmethod
    def validate_simulation_results(results: List[Dict[str, Any]]) -> bool:
        """
        Validate complete simulation results.
        
        Args:
            results: List of simulation iteration results
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If results are invalid
        """
        if not results:
            raise ValueError("Simulation results cannot be empty")
        
        # Check first result has required structure
        first_result = results[0]
        required_fields = ['quantum_timeline', 'attack_results', 'economic_impact']
        
        for field in required_fields:
            if field not in first_result:
                raise ValueError(f"Missing required field in results: {field}")
        
        # Validate sample of results
        sample_size = min(10, len(results))
        for i in range(sample_size):
            try:
                DataValidator.validate_quantum_timeline(
                    results[i]['quantum_timeline']
                )
            except ValueError as e:
                raise ValueError(f"Invalid result at index {i}: {e}")
        
        logger.info(f"Validated {len(results)} simulation results")
        return True


class ParameterValidator:
    """Validates and sanitizes simulation parameters."""
    
    @staticmethod
    def sanitize_iterations(n_iterations: int) -> int:
        """
        Sanitize number of iterations to reasonable bounds.
        
        Args:
            n_iterations: Requested number of iterations
            
        Returns:
            Sanitized number of iterations
        """
        min_iterations = 100
        max_iterations = 10_000_000
        
        if n_iterations < min_iterations:
            logger.warning(
                f"n_iterations {n_iterations} below minimum, using {min_iterations}"
            )
            return min_iterations
        
        if n_iterations > max_iterations:
            logger.warning(
                f"n_iterations {n_iterations} above maximum, using {max_iterations}"
            )
            return max_iterations
        
        return n_iterations
    
    @staticmethod
    def sanitize_cores(n_cores: int, max_cores: Optional[int] = None) -> int:
        """
        Sanitize number of CPU cores to available resources.
        
        Args:
            n_cores: Requested number of cores
            max_cores: Maximum available cores (auto-detected if None)
            
        Returns:
            Sanitized number of cores
        """
        import os
        
        if max_cores is None:
            max_cores = os.cpu_count() or 1
        
        if n_cores < 1:
            logger.warning(f"n_cores {n_cores} invalid, using 1")
            return 1
        
        if n_cores > max_cores:
            logger.warning(
                f"n_cores {n_cores} exceeds available {max_cores}, "
                f"using {max_cores}"
            )
            return max_cores
        
        return n_cores
    
    @staticmethod
    def validate_and_sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize complete configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized configuration dictionary
        """
        # Sanitize simulation parameters
        if 'n_iterations' in config:
            config['n_iterations'] = ParameterValidator.sanitize_iterations(
                config['n_iterations']
            )
        
        if 'n_cores' in config:
            config['n_cores'] = ParameterValidator.sanitize_cores(
                config['n_cores']
            )
        
        # Validate probabilities
        prob_fields = [
            'confidence_level',
            'quantum.success_probability',
            'network.migration_adoption_rate'
        ]
        
        for field in prob_fields:
            if '.' in field:
                parts = field.split('.')
                if parts[0] in config and parts[1] in config[parts[0]]:
                    ConfigValidator.validate_probability(
                        config[parts[0]][parts[1]], field
                    )
            elif field in config:
                ConfigValidator.validate_probability(config[field], field)
        
        return config
