"""
Convergence analysis module for Monte Carlo simulations.

This module provides comprehensive tools for monitoring and analyzing the convergence
of Monte Carlo simulations, including real-time tracking, statistical diagnostics,
and recommendations for optimal iteration counts.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
from scipy import stats
from scipy.signal import periodogram

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics of a single variable."""
    
    variable_name: str
    iterations: int
    running_mean: float
    running_std: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    coefficient_of_variation: float
    effective_sample_size: float
    autocorrelation: float
    is_converged: bool
    convergence_iteration: Optional[int] = None
    gelman_rubin_statistic: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'variable_name': self.variable_name,
            'iterations': self.iterations,
            'running_mean': float(self.running_mean),
            'running_std': float(self.running_std),
            'standard_error': float(self.standard_error),
            'confidence_interval_95': [float(x) for x in self.confidence_interval_95],
            'coefficient_of_variation': float(self.coefficient_of_variation),
            'effective_sample_size': float(self.effective_sample_size),
            'autocorrelation': float(self.autocorrelation),
            'is_converged': self.is_converged,
            'convergence_iteration': self.convergence_iteration,
            'gelman_rubin_statistic': float(self.gelman_rubin_statistic) if self.gelman_rubin_statistic else None
        }


@dataclass
class ConvergenceReport:
    """Comprehensive convergence analysis report."""
    
    timestamp: str
    total_iterations: int
    converged_variables: List[str]
    non_converged_variables: List[str]
    metrics: Dict[str, ConvergenceMetrics]
    overall_convergence: bool
    recommended_iterations: int
    quality_score: str  # A-F grade
    warnings: List[str] = field(default_factory=list)
    
    def to_json(self, filepath: Path) -> None:
        """Save report to JSON file."""
        data = {
            'timestamp': self.timestamp,
            'total_iterations': self.total_iterations,
            'converged_variables': self.converged_variables,
            'non_converged_variables': self.non_converged_variables,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'overall_convergence': self.overall_convergence,
            'recommended_iterations': self.recommended_iterations,
            'quality_score': self.quality_score,
            'warnings': self.warnings
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Convergence report saved to {filepath}")


class ConvergenceAnalyzer:
    """
    Analyzes convergence of Monte Carlo simulations.
    
    This class provides comprehensive convergence monitoring including:
    - Running statistics (mean, standard deviation, standard error)
    - Confidence intervals
    - Coefficient of variation
    - Effective sample size accounting for autocorrelation
    - Gelman-Rubin convergence diagnostic
    - Automatic convergence detection
    """
    
    def __init__(
        self,
        convergence_threshold: float = 0.01,
        confidence_level: float = 0.95,
        min_iterations: int = 100,
        window_size: int = 100,
        check_interval: int = 100
    ):
        """
        Initialize convergence analyzer.
        
        Args:
            convergence_threshold: Coefficient of variation threshold for convergence
            confidence_level: Confidence level for intervals (default 95%)
            min_iterations: Minimum iterations before checking convergence
            window_size: Window size for moving statistics
            check_interval: How often to check convergence (every N iterations)
        """
        self.convergence_threshold = convergence_threshold
        self.confidence_level = confidence_level
        self.min_iterations = min_iterations
        self.window_size = window_size
        self.check_interval = check_interval
        
        # Storage for tracked variables
        self.data: Dict[str, List[float]] = {}
        self.convergence_history: Dict[str, List[ConvergenceMetrics]] = {}
        self.converged_at: Dict[str, Optional[int]] = {}
        
        # Z-score for confidence intervals
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
    def track(self, iteration: int, values: Dict[str, float]) -> None:
        """
        Track values for convergence analysis.
        
        Args:
            iteration: Current iteration number
            values: Dictionary of variable names to values
        """
        for var_name, value in values.items():
            if var_name not in self.data:
                self.data[var_name] = []
                self.convergence_history[var_name] = []
                self.converged_at[var_name] = None
            
            self.data[var_name].append(value)
            
            # Check convergence at specified intervals
            if iteration >= self.min_iterations and iteration % self.check_interval == 0:
                metrics = self._calculate_metrics(var_name, iteration)
                self.convergence_history[var_name].append(metrics)
                
                if metrics.is_converged and self.converged_at[var_name] is None:
                    self.converged_at[var_name] = iteration
                    logger.info(f"{var_name} converged at iteration {iteration}")
    
    def _calculate_metrics(self, var_name: str, iteration: int) -> ConvergenceMetrics:
        """
        Calculate convergence metrics for a variable.
        
        Args:
            var_name: Variable name
            iteration: Current iteration number
            
        Returns:
            ConvergenceMetrics object
        """
        data = np.array(self.data[var_name])
        n = len(data)
        
        # Basic statistics
        running_mean = np.mean(data)
        running_std = np.std(data, ddof=1)
        
        # Standard error
        standard_error = running_std / np.sqrt(n)
        
        # Confidence interval
        margin = self.z_score * standard_error
        ci_lower = running_mean - margin
        ci_upper = running_mean + margin
        
        # Coefficient of variation
        cv = running_std / abs(running_mean) if running_mean != 0 else float('inf')
        
        # Effective sample size (accounting for autocorrelation)
        ess = self._calculate_effective_sample_size(data)
        
        # Autocorrelation at lag 1
        if n > 1:
            autocorr = self._calculate_autocorrelation(data, lag=1)
        else:
            autocorr = 0.0
        
        # Check convergence
        is_converged = cv < self.convergence_threshold and n >= self.min_iterations
        
        # Gelman-Rubin statistic (if we can split chains)
        gr_stat = None
        if n >= 2 * self.min_iterations:
            gr_stat = self._calculate_gelman_rubin(data)
            # Update convergence check with G-R criterion
            is_converged = is_converged and (gr_stat < 1.1 if gr_stat else True)
        
        return ConvergenceMetrics(
            variable_name=var_name,
            iterations=n,
            running_mean=running_mean,
            running_std=running_std,
            standard_error=standard_error,
            confidence_interval_95=(ci_lower, ci_upper),
            coefficient_of_variation=cv,
            effective_sample_size=ess,
            autocorrelation=autocorr,
            is_converged=is_converged,
            convergence_iteration=self.converged_at.get(var_name),
            gelman_rubin_statistic=gr_stat
        )
    
    def _calculate_effective_sample_size(self, data: np.ndarray) -> float:
        """
        Calculate effective sample size accounting for autocorrelation.
        
        Args:
            data: Time series data
            
        Returns:
            Effective sample size
        """
        n = len(data)
        if n < 10:
            return float(n)
        
        # Calculate autocorrelation function
        acf = self._calculate_acf(data, max_lag=min(n//4, 100))
        
        # Find first negative autocorrelation or cutoff
        first_negative = n
        for i, ac in enumerate(acf[1:], 1):
            if ac < 0:
                first_negative = i
                break
        
        # Sum autocorrelations up to first negative
        sum_acf = 1 + 2 * np.sum(acf[1:first_negative])
        
        # Effective sample size
        ess = n / max(1, sum_acf)
        
        return min(ess, n)  # ESS shouldn't exceed actual sample size
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation at specified lag.
        
        Args:
            data: Time series data
            lag: Lag for autocorrelation
            
        Returns:
            Autocorrelation coefficient
        """
        n = len(data)
        if n <= lag:
            return 0.0
        
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / n
        c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
        
        if c0 == 0:
            return 0.0
        
        return c_lag / c0
    
    def _calculate_acf(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Calculate autocorrelation function up to max_lag.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to calculate
            
        Returns:
            Array of autocorrelation coefficients
        """
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            acf[lag] = self._calculate_autocorrelation(data, lag)
        return acf
    
    def _calculate_gelman_rubin(self, data: np.ndarray, n_chains: int = 4) -> Optional[float]:
        """
        Calculate Gelman-Rubin convergence diagnostic.
        
        Splits data into multiple chains and compares within-chain
        and between-chain variance.
        
        Args:
            data: Combined chain data
            n_chains: Number of chains to split into
            
        Returns:
            Gelman-Rubin statistic (should be < 1.1 for convergence)
        """
        n = len(data)
        if n < n_chains * 10:  # Need reasonable chain length
            return None
        
        # Split into chains
        chain_length = n // n_chains
        chains = [data[i*chain_length:(i+1)*chain_length] for i in range(n_chains)]
        
        # Calculate within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in chains])
        
        # Calculate between-chain variance
        chain_means = [np.mean(chain) for chain in chains]
        grand_mean = np.mean(chain_means)
        B = chain_length * np.var(chain_means, ddof=1)
        
        # Pooled variance estimate
        var_pooled = (1 - 1/chain_length) * W + (1/chain_length) * B
        
        # Gelman-Rubin statistic
        R_hat = np.sqrt(var_pooled / W) if W > 0 else 1.0
        
        return R_hat
    
    def get_current_metrics(self, var_name: str) -> Optional[ConvergenceMetrics]:
        """
        Get current convergence metrics for a variable.
        
        Args:
            var_name: Variable name
            
        Returns:
            Current ConvergenceMetrics or None if variable not tracked
        """
        if var_name not in self.data or not self.data[var_name]:
            return None
        
        return self._calculate_metrics(var_name, len(self.data[var_name]))
    
    def is_converged(self, var_name: Optional[str] = None) -> bool:
        """
        Check if variable(s) have converged.
        
        Args:
            var_name: Specific variable to check, or None for all variables
            
        Returns:
            True if converged
        """
        if var_name:
            metrics = self.get_current_metrics(var_name)
            return metrics.is_converged if metrics else False
        else:
            # Check all variables
            return all(
                self.get_current_metrics(var).is_converged
                for var in self.data.keys()
                if self.get_current_metrics(var)
            )
    
    def estimate_required_iterations(self, target_error: float = 0.01) -> int:
        """
        Estimate required iterations to achieve target standard error.
        
        Args:
            target_error: Target relative standard error
            
        Returns:
            Estimated number of iterations needed
        """
        estimates = []
        
        for var_name in self.data.keys():
            metrics = self.get_current_metrics(var_name)
            if not metrics or metrics.running_mean == 0:
                continue
            
            current_error = metrics.standard_error / abs(metrics.running_mean)
            if current_error <= target_error:
                estimates.append(metrics.iterations)
            else:
                # Estimate based on inverse square root relationship
                required = metrics.iterations * (current_error / target_error) ** 2
                # Account for autocorrelation
                required *= (metrics.iterations / metrics.effective_sample_size)
                estimates.append(int(required))
        
        if not estimates:
            return 10000  # Default recommendation
        
        # Return conservative estimate (75th percentile)
        return int(np.percentile(estimates, 75))
    
    def generate_report(self, output_path: Optional[Path] = None) -> ConvergenceReport:
        """
        Generate comprehensive convergence report.
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            ConvergenceReport object
        """
        from datetime import datetime
        
        # Collect metrics for all variables
        metrics = {}
        converged_vars = []
        non_converged_vars = []
        
        for var_name in self.data.keys():
            var_metrics = self.get_current_metrics(var_name)
            if var_metrics:
                metrics[var_name] = var_metrics
                if var_metrics.is_converged:
                    converged_vars.append(var_name)
                else:
                    non_converged_vars.append(var_name)
        
        # Overall convergence
        overall_convergence = len(converged_vars) == len(metrics) and len(metrics) > 0
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(metrics, overall_convergence)
        
        # Recommended iterations
        recommended = self.estimate_required_iterations()
        if overall_convergence:
            # If converged, recommend current + 20% buffer
            current_max = max(m.iterations for m in metrics.values())
            recommended = min(recommended, int(current_max * 1.2))
        
        # Generate warnings
        warnings = self._generate_warnings(metrics, overall_convergence)
        
        # Create report
        report = ConvergenceReport(
            timestamp=datetime.now().isoformat(),
            total_iterations=max((m.iterations for m in metrics.values()), default=0),
            converged_variables=converged_vars,
            non_converged_variables=non_converged_vars,
            metrics=metrics,
            overall_convergence=overall_convergence,
            recommended_iterations=recommended,
            quality_score=quality_score,
            warnings=warnings
        )
        
        # Save if path provided
        if output_path:
            report.to_json(output_path)
        
        return report
    
    def _calculate_quality_score(
        self,
        metrics: Dict[str, ConvergenceMetrics],
        overall_convergence: bool
    ) -> str:
        """
        Calculate quality score (A-F) based on convergence metrics.
        
        Args:
            metrics: Dictionary of convergence metrics
            overall_convergence: Whether all variables converged
            
        Returns:
            Letter grade A-F
        """
        if not metrics:
            return "F"
        
        score = 100.0
        
        # Convergence status (40 points)
        convergence_rate = sum(1 for m in metrics.values() if m.is_converged) / len(metrics)
        score -= (1 - convergence_rate) * 40
        
        # Coefficient of variation (20 points)
        avg_cv = np.mean([m.coefficient_of_variation for m in metrics.values()])
        if avg_cv > 0.05:
            score -= min(20, (avg_cv - 0.05) * 200)
        
        # Effective sample size ratio (20 points)
        avg_ess_ratio = np.mean([
            m.effective_sample_size / m.iterations 
            for m in metrics.values()
        ])
        score -= (1 - avg_ess_ratio) * 20
        
        # Gelman-Rubin statistic (20 points)
        gr_stats = [m.gelman_rubin_statistic for m in metrics.values() if m.gelman_rubin_statistic]
        if gr_stats:
            avg_gr = np.mean(gr_stats)
            if avg_gr > 1.1:
                score -= min(20, (avg_gr - 1.1) * 100)
        
        # Convert to letter grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_warnings(
        self,
        metrics: Dict[str, ConvergenceMetrics],
        overall_convergence: bool
    ) -> List[str]:
        """
        Generate warnings based on convergence analysis.
        
        Args:
            metrics: Dictionary of convergence metrics
            overall_convergence: Whether all variables converged
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not overall_convergence:
            warnings.append(
                f"Not all variables have converged. Consider increasing iterations."
            )
        
        # Check for high autocorrelation
        high_autocorr = [
            (name, m.autocorrelation) 
            for name, m in metrics.items() 
            if abs(m.autocorrelation) > 0.5
        ]
        if high_autocorr:
            warnings.append(
                f"High autocorrelation detected in: {', '.join(v[0] for v in high_autocorr)}. "
                "Consider thinning the samples."
            )
        
        # Check for poor Gelman-Rubin statistics
        poor_gr = [
            (name, m.gelman_rubin_statistic)
            for name, m in metrics.items()
            if m.gelman_rubin_statistic and m.gelman_rubin_statistic > 1.2
        ]
        if poor_gr:
            warnings.append(
                f"Poor mixing detected (G-R > 1.2) in: {', '.join(v[0] for v in poor_gr)}. "
                "Consider running longer chains."
            )
        
        # Check for low effective sample size
        low_ess = [
            (name, m.effective_sample_size / m.iterations)
            for name, m in metrics.items()
            if m.effective_sample_size / m.iterations < 0.1
        ]
        if low_ess:
            warnings.append(
                f"Low effective sample size in: {', '.join(v[0] for v in low_ess)}. "
                "Samples may be highly correlated."
            )
        
        # Check coefficient of variation
        high_cv = [
            (name, m.coefficient_of_variation)
            for name, m in metrics.items()
            if m.coefficient_of_variation > 0.1
        ]
        if high_cv:
            warnings.append(
                f"High variance in: {', '.join(v[0] for v in high_cv)}. "
                "Results may be unstable."
            )
        
        return warnings
    
    def get_convergence_summary(self) -> str:
        """
        Get a human-readable convergence summary.
        
        Returns:
            Summary string
        """
        lines = ["=" * 60, "CONVERGENCE SUMMARY", "=" * 60]
        
        for var_name in sorted(self.data.keys()):
            metrics = self.get_current_metrics(var_name)
            if not metrics:
                continue
            
            status = "✓ CONVERGED" if metrics.is_converged else "✗ NOT CONVERGED"
            lines.append(f"\n{var_name}: {status}")
            lines.append(f"  Iterations: {metrics.iterations:,}")
            lines.append(f"  Mean: {metrics.running_mean:.4f}")
            lines.append(f"  Std Error: {metrics.standard_error:.4f}")
            lines.append(f"  95% CI: [{metrics.confidence_interval_95[0]:.4f}, "
                        f"{metrics.confidence_interval_95[1]:.4f}]")
            lines.append(f"  CV: {metrics.coefficient_of_variation:.4f}")
            lines.append(f"  ESS: {metrics.effective_sample_size:.0f}")
            
            if metrics.gelman_rubin_statistic:
                lines.append(f"  G-R Statistic: {metrics.gelman_rubin_statistic:.3f}")
            
            if metrics.convergence_iteration:
                lines.append(f"  Converged at: iteration {metrics.convergence_iteration}")
        
        lines.append("\n" + "=" * 60)
        lines.append(f"Overall Convergence: {'YES' if self.is_converged() else 'NO'}")
        lines.append(f"Recommended Iterations: {self.estimate_required_iterations():,}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
