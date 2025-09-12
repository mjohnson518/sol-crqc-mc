"""
Results collection and aggregation for Monte Carlo simulation.

Handles the collection, aggregation, and basic analysis of simulation results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ResultsStatistics:
    """Container for statistical summaries of results."""
    
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    @classmethod
    def from_array(cls, data: np.ndarray, percentiles: List[int] = None) -> 'ResultsStatistics':
        """
        Create statistics from data array.
        
        Args:
            data: Data array
            percentiles: List of percentiles to calculate (default: [5, 25, 50, 75, 95])
            
        Returns:
            ResultsStatistics instance
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        
        percentile_values = {
            p: float(np.percentile(data, p))
            for p in percentiles
        }
        
        return cls(
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            percentiles=percentile_values
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'percentiles': self.percentiles
        }


class ResultsCollector:
    """
    Collects and aggregates Monte Carlo simulation results.
    
    Provides methods for:
    - Adding individual iteration results
    - Computing statistics across iterations
    - Generating summary reports
    - Tracking convergence
    """
    
    def __init__(self):
        """Initialize the results collector."""
        self.results = []
        self.first_attack_years = []
        self.economic_losses = []
        self.attack_success_rates = []
        
        # Convergence tracking
        self.convergence_history = defaultdict(list)
        self.convergence_checkpoints = [100, 500, 1000, 5000, 10000, 50000, 100000]
        
        # Summary statistics
        self._summary = None
        self._needs_update = True
    
    def add_result(self, result: Any):
        """
        Add a single simulation result.
        
        Args:
            result: SimulationResult instance
        """
        self.results.append(result)
        
        # Extract key metrics
        if hasattr(result, 'first_attack_year') and result.first_attack_year is not None:
            self.first_attack_years.append(result.first_attack_year)
        
        if hasattr(result, 'economic_impact'):
            self.economic_losses.append(result.economic_impact.get('total_loss_usd', 0))
        
        if hasattr(result, 'attack_results'):
            success = result.attack_results.get('attacks_successful', 0) > 0
            self.attack_success_rates.append(1 if success else 0)
        
        # Check convergence at checkpoints
        n = len(self.results)
        if n in self.convergence_checkpoints:
            self._update_convergence(n)
        
        self._needs_update = True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self._needs_update:
            self._compute_summary()
        return self._summary
    
    def _compute_summary(self):
        """Compute summary statistics from collected results."""
        n_results = len(self.results)
        
        if n_results == 0:
            self._summary = {'error': 'No results collected'}
            return
        
        summary = {
            'n_iterations': n_results,
            'metrics': {}
        }
        
        # First attack year statistics
        if self.first_attack_years:
            attack_years_array = np.array(self.first_attack_years)
            summary['metrics']['first_attack_year'] = ResultsStatistics.from_array(
                attack_years_array
            ).to_dict()
            
            # Cumulative probability by year
            summary['metrics']['cumulative_attack_probability'] = self._compute_cumulative_probability()
        else:
            summary['metrics']['first_attack_year'] = None
            summary['metrics']['cumulative_attack_probability'] = {}
        
        # Economic loss statistics
        if self.economic_losses:
            losses_array = np.array(self.economic_losses)
            summary['metrics']['economic_loss_usd'] = ResultsStatistics.from_array(
                losses_array
            ).to_dict()
            
            # Value at Risk (VaR) and Conditional VaR
            summary['metrics']['var_95'] = float(np.percentile(losses_array, 95))
            summary['metrics']['cvar_95'] = float(
                np.mean(losses_array[losses_array >= summary['metrics']['var_95']])
            )
        else:
            summary['metrics']['economic_loss_usd'] = None
        
        # Attack success rate
        if self.attack_success_rates:
            success_rate = np.mean(self.attack_success_rates)
            summary['metrics']['attack_success_rate'] = float(success_rate)
        else:
            summary['metrics']['attack_success_rate'] = 0.0
        
        # Convergence metrics
        summary['convergence'] = self._get_convergence_metrics()
        
        self._summary = summary
        self._needs_update = False
    
    def _compute_cumulative_probability(self) -> Dict[int, float]:
        """
        Compute cumulative probability of attack by year.
        
        Returns:
            Dictionary mapping year to cumulative probability
        """
        if not self.first_attack_years:
            return {}
        
        years = range(2025, 2046)
        cumulative_prob = {}
        n_total = len(self.results)
        
        for year in years:
            n_attacks = sum(1 for y in self.first_attack_years if y <= year)
            cumulative_prob[year] = n_attacks / n_total
        
        return cumulative_prob
    
    def _update_convergence(self, n: int):
        """
        Update convergence tracking.
        
        Args:
            n: Number of iterations completed
        """
        if self.first_attack_years:
            mean_attack_year = np.mean(self.first_attack_years)
            std_attack_year = np.std(self.first_attack_years)
            
            self.convergence_history['n_iterations'].append(n)
            self.convergence_history['mean_attack_year'].append(mean_attack_year)
            self.convergence_history['std_attack_year'].append(std_attack_year)
            
            if self.economic_losses:
                mean_loss = np.mean(self.economic_losses)
                self.convergence_history['mean_economic_loss'].append(mean_loss)
    
    def _get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Calculate convergence metrics.
        
        Returns:
            Dictionary containing convergence analysis
        """
        if len(self.convergence_history['n_iterations']) < 2:
            return {'converged': False, 'message': 'Insufficient data for convergence analysis'}
        
        # Check if key metrics have stabilized
        recent_means = self.convergence_history['mean_attack_year'][-3:]
        if len(recent_means) >= 3:
            variation = np.std(recent_means) / np.mean(recent_means)
            converged = variation < 0.01  # Less than 1% variation
        else:
            converged = False
        
        return {
            'converged': converged,
            'history': dict(self.convergence_history),
            'final_mean': self.convergence_history['mean_attack_year'][-1] if self.convergence_history['mean_attack_year'] else None,
            'coefficient_of_variation': variation if 'variation' in locals() else None
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Extract data for DataFrame
        data = []
        for result in self.results:
            row = {
                'iteration_id': result.iteration_id,
                'first_attack_year': result.first_attack_year,
                'runtime_seconds': result.runtime_seconds
            }
            
            # Add economic impact metrics
            if hasattr(result, 'economic_impact'):
                row['direct_loss_usd'] = result.economic_impact.get('direct_loss_usd', 0)
                row['total_loss_usd'] = result.economic_impact.get('total_loss_usd', 0)
            
            # Add attack metrics
            if hasattr(result, 'attack_results'):
                row['attacks_attempted'] = result.attack_results.get('attacks_attempted', 0)
                row['attacks_successful'] = result.attack_results.get('attacks_successful', 0)
            
            # Add quantum timeline metrics
            if hasattr(result, 'quantum_timeline'):
                row['crqc_year'] = result.quantum_timeline.get('crqc_year', None)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_percentiles(
        self,
        metric: str,
        percentiles: List[int] = None
    ) -> Dict[int, float]:
        """
        Calculate percentiles for a specific metric.
        
        Args:
            metric: Name of metric ('first_attack_year', 'economic_loss', etc.)
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary mapping percentile to value
        """
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        
        if metric == 'first_attack_year' and self.first_attack_years:
            data = np.array(self.first_attack_years)
        elif metric == 'economic_loss' and self.economic_losses:
            data = np.array(self.economic_losses)
        else:
            return {}
        
        return {
            p: float(np.percentile(data, p))
            for p in percentiles
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk-specific metrics.
        
        Returns:
            Dictionary containing risk metrics
        """
        risk_metrics = {}
        
        if self.economic_losses:
            losses = np.array(self.economic_losses)
            
            # Value at Risk (VaR) at different confidence levels
            risk_metrics['var'] = {
                90: float(np.percentile(losses, 90)),
                95: float(np.percentile(losses, 95)),
                99: float(np.percentile(losses, 99))
            }
            
            # Conditional Value at Risk (CVaR)
            risk_metrics['cvar'] = {}
            for conf in [90, 95, 99]:
                threshold = risk_metrics['var'][conf]
                tail_losses = losses[losses >= threshold]
                if len(tail_losses) > 0:
                    risk_metrics['cvar'][conf] = float(np.mean(tail_losses))
                else:
                    risk_metrics['cvar'][conf] = threshold
            
            # Maximum probable loss
            risk_metrics['max_probable_loss'] = risk_metrics['var'][99]
            
            # Expected shortfall
            risk_metrics['expected_shortfall'] = risk_metrics['cvar'][95]
        
        # Attack timing risk
        if self.first_attack_years:
            years = np.array(self.first_attack_years)
            
            risk_metrics['attack_timing'] = {
                'before_2030': float(np.mean(years < 2030)),
                'before_2035': float(np.mean(years < 2035)),
                'before_2040': float(np.mean(years < 2040)),
                'median_year': float(np.median(years))
            }
        
        return risk_metrics
    
    def clear(self):
        """Clear all collected results."""
        self.results.clear()
        self.first_attack_years.clear()
        self.economic_losses.clear()
        self.attack_success_rates.clear()
        self.convergence_history.clear()
        self._summary = None
        self._needs_update = True
        
        logger.info("Results collector cleared")


def test_results_collector():
    """Test the results collector functionality."""
    from src.core.simulation import SimulationResult
    
    collector = ResultsCollector()
    
    # Add some test results
    for i in range(100):
        result = SimulationResult(
            iteration_id=i,
            quantum_timeline={'crqc_year': 2030 + np.random.normal(0, 3)},
            network_state={},
            attack_results={'attacks_successful': 1 if np.random.random() > 0.3 else 0},
            economic_impact={'total_loss_usd': np.random.exponential(1e10)},
            first_attack_year=2030 + np.random.normal(0, 3),
            runtime_seconds=0.1
        )
        collector.add_result(result)
    
    # Get summary
    summary = collector.get_summary()
    
    print("Summary Statistics:")
    print(f"  Iterations: {summary['n_iterations']}")
    print(f"  Mean first attack: {summary['metrics']['first_attack_year']['mean']:.1f}")
    print(f"  Attack success rate: {summary['metrics']['attack_success_rate']:.1%}")
    print(f"  VaR(95%): ${summary['metrics']['var_95']/1e9:.1f}B")
    
    # Get risk metrics
    risk = collector.get_risk_metrics()
    print("\nRisk Metrics:")
    print(f"  P(attack < 2030): {risk['attack_timing']['before_2030']:.1%}")
    print(f"  Expected shortfall: ${risk['expected_shortfall']/1e9:.1f}B")
    
    print("\n✓ Results collector test passed")


if __name__ == "__main__":
    test_results_collector()
