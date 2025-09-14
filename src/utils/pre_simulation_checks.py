"""
Pre-simulation validation and resource checks.

Ensures the simulation environment is properly configured and has
sufficient resources before starting a potentially long-running simulation.
"""

import logging
import psutil
import platform
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np

from src.config import SimulationParameters

logger = logging.getLogger(__name__)


class PreSimulationValidator:
    """
    Validates simulation parameters and system resources before execution.
    
    Performs checks for:
    - Parameter validity and ranges
    - System resource availability
    - Runtime estimation
    - Optimal configuration recommendations
    """
    
    def __init__(self, config: SimulationParameters):
        """
        Initialize validator with simulation configuration.
        
        Args:
            config: Simulation parameters to validate
        """
        self.config = config
        self.validation_results = {
            'parameter_checks': [],
            'resource_checks': [],
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
    
    def validate_all(self) -> Tuple[bool, Dict]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Starting pre-simulation validation...")
        
        # Run all checks
        self._check_parameters()
        self._check_system_resources()
        self._estimate_runtime()
        self._check_output_directory()
        self._generate_recommendations()
        
        # Determine overall validity
        is_valid = len(self.validation_results['errors']) == 0
        
        # Log results
        self._log_validation_results()
        
        return is_valid, self.validation_results
    
    def _check_parameters(self) -> None:
        """Validate simulation parameters are within reasonable ranges."""
        checks = []
        
        # Iteration count
        if self.config.n_iterations < 100:
            self.validation_results['warnings'].append(
                f"Low iteration count ({self.config.n_iterations}) may not provide "
                "statistically significant results"
            )
            checks.append(('Iteration Count', 'WARNING'))
        elif self.config.n_iterations > 1000000:
            self.validation_results['warnings'].append(
                f"Very high iteration count ({self.config.n_iterations:,}) will require "
                "significant computation time"
            )
            checks.append(('Iteration Count', 'WARNING'))
        else:
            checks.append(('Iteration Count', 'OK'))
        
        # Time horizon
        years = self.config.end_year - self.config.start_year
        if years < 5:
            self.validation_results['warnings'].append(
                f"Short time horizon ({years} years) may not capture long-term risks"
            )
            checks.append(('Time Horizon', 'WARNING'))
        elif years > 50:
            self.validation_results['warnings'].append(
                f"Very long time horizon ({years} years) increases uncertainty"
            )
            checks.append(('Time Horizon', 'WARNING'))
        else:
            checks.append(('Time Horizon', 'OK'))
        
        # Confidence level
        if self.config.confidence_level < 0.9:
            self.validation_results['warnings'].append(
                f"Low confidence level ({self.config.confidence_level:.0%}) may not be suitable "
                "for risk analysis"
            )
            checks.append(('Confidence Level', 'WARNING'))
        else:
            checks.append(('Confidence Level', 'OK'))
        
        # CPU cores
        available_cores = multiprocessing.cpu_count()
        if self.config.n_cores > available_cores:
            self.validation_results['errors'].append(
                f"Requested cores ({self.config.n_cores}) exceeds available cores ({available_cores})"
            )
            checks.append(('CPU Cores', 'ERROR'))
        elif self.config.n_cores > available_cores * 0.8:
            self.validation_results['warnings'].append(
                f"Using {self.config.n_cores}/{available_cores} cores may impact system responsiveness"
            )
            checks.append(('CPU Cores', 'WARNING'))
        else:
            checks.append(('CPU Cores', 'OK'))
        
        # Quantum parameters
        if hasattr(self.config, 'quantum_growth_rate'):
            if self.config.quantum_growth_rate < 0.05:
                self.validation_results['warnings'].append(
                    "Very low quantum growth rate may underestimate risk"
                )
                checks.append(('Quantum Growth Rate', 'WARNING'))
            elif self.config.quantum_growth_rate > 0.5:
                self.validation_results['warnings'].append(
                    "Very high quantum growth rate may overestimate near-term risk"
                )
                checks.append(('Quantum Growth Rate', 'WARNING'))
            else:
                checks.append(('Quantum Growth Rate', 'OK'))
        
        self.validation_results['parameter_checks'] = checks
    
    def _check_system_resources(self) -> None:
        """Check available system resources."""
        checks = []
        
        # Memory check
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Estimate memory requirements (rough estimate)
        bytes_per_iteration = 10000  # Rough estimate
        required_gb = (self.config.n_iterations * bytes_per_iteration) / (1024**3)
        
        if required_gb > available_gb:
            self.validation_results['errors'].append(
                f"Insufficient memory: {required_gb:.1f}GB required, {available_gb:.1f}GB available"
            )
            checks.append(('Memory', 'ERROR'))
        elif required_gb > available_gb * 0.8:
            self.validation_results['warnings'].append(
                f"High memory usage expected: {required_gb:.1f}GB of {available_gb:.1f}GB available"
            )
            checks.append(('Memory', 'WARNING'))
        else:
            checks.append(('Memory', f'OK ({available_gb:.1f}GB available)'))
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            self.validation_results['warnings'].append(
                f"High CPU usage detected ({cpu_percent:.0f}%). Simulation may run slowly."
            )
            checks.append(('CPU Load', 'WARNING'))
        else:
            checks.append(('CPU Load', f'OK ({cpu_percent:.0f}% usage)'))
        
        # Disk space check
        output_dir = Path(self.config.output_dir)
        if output_dir.exists():
            disk = psutil.disk_usage(str(output_dir))
            free_gb = disk.free / (1024**3)
            
            # Estimate disk requirements
            estimated_output_gb = self.config.n_iterations * 1000 / (1024**3)  # Rough estimate
            
            if estimated_output_gb > free_gb:
                self.validation_results['errors'].append(
                    f"Insufficient disk space: {estimated_output_gb:.1f}GB required, "
                    f"{free_gb:.1f}GB available"
                )
                checks.append(('Disk Space', 'ERROR'))
            elif estimated_output_gb > free_gb * 0.5:
                self.validation_results['warnings'].append(
                    f"Low disk space: {free_gb:.1f}GB available, {estimated_output_gb:.1f}GB needed"
                )
                checks.append(('Disk Space', 'WARNING'))
            else:
                checks.append(('Disk Space', f'OK ({free_gb:.1f}GB available)'))
        
        # System info
        checks.append(('Platform', platform.platform()))
        checks.append(('Python Version', platform.python_version()))
        checks.append(('CPU Count', str(multiprocessing.cpu_count())))
        
        self.validation_results['resource_checks'] = checks
    
    def _estimate_runtime(self) -> None:
        """Estimate simulation runtime based on parameters."""
        # Rough estimation based on empirical observations
        # Adjust these based on actual performance measurements
        seconds_per_iteration = 0.01  # Base time per iteration
        
        # Adjust for model complexity
        if hasattr(self.config, 'model_complexity'):
            complexity_factor = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
            seconds_per_iteration *= complexity_factor.get(self.config.model_complexity, 1.0)
        
        # Adjust for parallelization efficiency
        parallel_efficiency = 0.7  # Typical parallel efficiency
        if self.config.n_cores > 1:
            speedup = 1 + (self.config.n_cores - 1) * parallel_efficiency
        else:
            speedup = 1.0
        
        # Calculate total time
        total_seconds = (self.config.n_iterations * seconds_per_iteration) / speedup
        
        # Format runtime estimate
        if total_seconds < 60:
            runtime_str = f"{total_seconds:.0f} seconds"
        elif total_seconds < 3600:
            runtime_str = f"{total_seconds/60:.1f} minutes"
        elif total_seconds < 86400:
            runtime_str = f"{total_seconds/3600:.1f} hours"
        else:
            runtime_str = f"{total_seconds/86400:.1f} days"
        
        self.validation_results['runtime_estimate'] = runtime_str
        
        # Add warning for long runtimes
        if total_seconds > 3600:  # More than 1 hour
            self.validation_results['warnings'].append(
                f"Long runtime expected: approximately {runtime_str}"
            )
    
    def _check_output_directory(self) -> None:
        """Check output directory configuration."""
        output_dir = Path(self.config.output_dir)
        
        try:
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check write permissions
            test_file = output_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
            
            self.validation_results['parameter_checks'].append(
                ('Output Directory', 'OK')
            )
        except Exception as e:
            self.validation_results['errors'].append(
                f"Cannot write to output directory {output_dir}: {e}"
            )
            self.validation_results['parameter_checks'].append(
                ('Output Directory', 'ERROR')
            )
    
    def _generate_recommendations(self) -> None:
        """Generate recommendations for optimal simulation configuration."""
        recommendations = []
        
        # Iteration count recommendations
        if self.config.n_iterations < 1000:
            recommendations.append(
                "Consider increasing iterations to at least 1,000 for basic statistical significance"
            )
        elif self.config.n_iterations < 10000:
            recommendations.append(
                "For publication-quality results, consider 10,000+ iterations"
            )
        
        # Optimal core usage
        available_cores = multiprocessing.cpu_count()
        optimal_cores = max(1, available_cores - 1)  # Leave one core for system
        
        if self.config.n_cores == 1 and available_cores > 1:
            recommendations.append(
                f"Enable parallel processing with n_cores={optimal_cores} for faster execution"
            )
        elif self.config.n_cores == available_cores:
            recommendations.append(
                f"Consider using n_cores={optimal_cores} to maintain system responsiveness"
            )
        
        # Batch size optimization
        if self.config.n_cores > 1:
            optimal_batch = max(10, self.config.n_iterations // (self.config.n_cores * 10))
            recommendations.append(
                f"Optimal batch size for parallel execution: ~{optimal_batch} iterations per batch"
            )
        
        # Memory optimization
        memory = psutil.virtual_memory()
        if memory.available < 8 * (1024**3):  # Less than 8GB available
            recommendations.append(
                "Consider closing other applications to free up memory"
            )
            if self.config.save_raw_results:
                recommendations.append(
                    "Disable save_raw_results to reduce memory usage"
                )
        
        # Convergence recommendations
        recommendations.append(
            "Enable convergence tracking to monitor statistical reliability"
        )
        
        if self.config.n_iterations >= 10000:
            recommendations.append(
                "Consider using adaptive iteration stopping based on convergence"
            )
        
        self.validation_results['recommendations'] = recommendations
    
    def _log_validation_results(self) -> None:
        """Log validation results."""
        logger.info("=" * 60)
        logger.info("PRE-SIMULATION VALIDATION REPORT")
        logger.info("=" * 60)
        
        # Parameter checks
        logger.info("Parameter Checks:")
        for check, status in self.validation_results['parameter_checks']:
            symbol = "✓" if status == "OK" else "⚠" if "WARNING" in str(status) else "✗"
            logger.info(f"  {symbol} {check}: {status}")
        
        # Resource checks
        logger.info("\nResource Checks:")
        for check, status in self.validation_results['resource_checks']:
            logger.info(f"  • {check}: {status}")
        
        # Runtime estimate
        if 'runtime_estimate' in self.validation_results:
            logger.info(f"\nEstimated Runtime: {self.validation_results['runtime_estimate']}")
        
        # Warnings
        if self.validation_results['warnings']:
            logger.warning("\nWarnings:")
            for warning in self.validation_results['warnings']:
                logger.warning(f"  ⚠ {warning}")
        
        # Errors
        if self.validation_results['errors']:
            logger.error("\nErrors:")
            for error in self.validation_results['errors']:
                logger.error(f"  ✗ {error}")
        
        # Recommendations
        if self.validation_results['recommendations']:
            logger.info("\nRecommendations:")
            for rec in self.validation_results['recommendations']:
                logger.info(f"  → {rec}")
        
        logger.info("=" * 60)
    
    def suggest_optimal_iterations(self) -> int:
        """
        Suggest optimal iteration count based on available resources.
        
        Returns:
            Suggested number of iterations
        """
        # Get available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Estimate iterations that fit in memory (conservative)
        bytes_per_iteration = 10000
        max_iterations_memory = int((available_gb * 0.5 * 1024**3) / bytes_per_iteration)
        
        # Consider time constraints (target ~1 hour maximum)
        seconds_per_iteration = 0.01
        if self.config.n_cores > 1:
            speedup = 1 + (self.config.n_cores - 1) * 0.7
        else:
            speedup = 1.0
        
        max_iterations_time = int((3600 * speedup) / seconds_per_iteration)
        
        # Take minimum of constraints
        suggested = min(max_iterations_memory, max_iterations_time)
        
        # Round to nice number
        if suggested > 100000:
            suggested = (suggested // 10000) * 10000
        elif suggested > 10000:
            suggested = (suggested // 1000) * 1000
        elif suggested > 1000:
            suggested = (suggested // 100) * 100
        
        # Ensure minimum
        suggested = max(suggested, 1000)
        
        return suggested


def validate_before_simulation(config: SimulationParameters) -> bool:
    """
    Convenience function to run pre-simulation validation.
    
    Args:
        config: Simulation configuration
        
    Returns:
        True if validation passes, False otherwise
    """
    validator = PreSimulationValidator(config)
    is_valid, report = validator.validate_all()
    
    if not is_valid:
        logger.error("Pre-simulation validation failed. Please address errors before proceeding.")
        
        # Offer to continue anyway
        if len(report['errors']) > 0:
            response = input("Critical errors detected. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
    
    elif report['warnings']:
        logger.warning("Validation completed with warnings.")
        response = input("Continue with simulation? (Y/n): ")
        if response.lower() == 'n':
            return False
    
    else:
        logger.info("Pre-simulation validation passed successfully.")
    
    return True


def estimate_resource_requirements(
    n_iterations: int,
    n_cores: int = 1,
    model_complexity: str = 'medium'
) -> Dict[str, any]:
    """
    Estimate resource requirements for given parameters.
    
    Args:
        n_iterations: Number of Monte Carlo iterations
        n_cores: Number of CPU cores to use
        model_complexity: 'low', 'medium', or 'high'
        
    Returns:
        Dictionary with resource estimates
    """
    # Memory estimate
    bytes_per_iteration = {'low': 5000, 'medium': 10000, 'high': 20000}
    memory_gb = (n_iterations * bytes_per_iteration[model_complexity]) / (1024**3)
    
    # Time estimate
    seconds_per_iteration = {'low': 0.005, 'medium': 0.01, 'high': 0.02}
    parallel_efficiency = 0.7
    
    if n_cores > 1:
        speedup = 1 + (n_cores - 1) * parallel_efficiency
    else:
        speedup = 1.0
    
    runtime_seconds = (n_iterations * seconds_per_iteration[model_complexity]) / speedup
    
    # Disk space estimate
    disk_gb = n_iterations * 1000 / (1024**3)  # Rough estimate
    
    return {
        'memory_gb': round(memory_gb, 2),
        'runtime_seconds': round(runtime_seconds, 2),
        'runtime_formatted': _format_time(runtime_seconds),
        'disk_gb': round(disk_gb, 2),
        'recommended_cores': min(n_cores, multiprocessing.cpu_count() - 1)
    }


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"
