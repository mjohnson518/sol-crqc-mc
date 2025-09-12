"""
Scenario comparison tools for analyzing different simulation configurations.

This module enables comparison of multiple simulation scenarios to identify
optimal strategies and understand the impact of different parameters.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from .statistical_analysis import StatisticalAnalyzer, StatisticalSummary


@dataclass
class ScenarioResult:
    """Container for scenario-specific results."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    metrics: Dict[str, StatisticalSummary]
    raw_data: Optional[Dict[str, List[float]]] = None
    
    def get_metric(self, metric_name: str, statistic: str = 'mean') -> float:
        """
        Get specific statistic for a metric.
        
        Args:
            metric_name: Name of the metric
            statistic: Statistic to retrieve ('mean', 'median', 'std', etc.)
            
        Returns:
            Requested statistic value
        """
        if metric_name not in self.metrics:
            raise KeyError(f"Metric '{metric_name}' not found in scenario '{self.name}'")
        
        metric = self.metrics[metric_name]
        
        if statistic in ['mean', 'median', 'std', 'var', 'min', 'max',
                        'skewness', 'kurtosis', 'sample_size']:
            return getattr(metric, statistic)
        elif statistic.startswith('p'):
            # Handle percentiles (e.g., 'p95' for 95th percentile)
            try:
                percentile = int(statistic[1:])
                return metric.percentiles.get(percentile, np.nan)
            except (ValueError, IndexError):
                raise ValueError(f"Invalid statistic: {statistic}")
        else:
            raise ValueError(f"Unknown statistic: {statistic}")


@dataclass
class ComparisonResult:
    """Results of scenario comparison."""
    
    scenarios: List[ScenarioResult]
    ranking: Dict[str, List[str]]  # Metric -> ordered scenario names
    best_scenario: Dict[str, str]  # Metric -> best scenario name
    statistical_tests: Dict[str, Dict[str, Any]]  # Test results
    dominance_matrix: Optional[pd.DataFrame] = None


class ScenarioComparator:
    """
    Compare and analyze multiple simulation scenarios.
    
    Provides tools for ranking scenarios, performing statistical tests,
    and identifying optimal configurations.
    """
    
    def __init__(self):
        """Initialize scenario comparator."""
        self.analyzer = StatisticalAnalyzer()
        self.scenarios: Dict[str, ScenarioResult] = {}
    
    def add_scenario(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        results: Dict[str, List[float]]
    ) -> None:
        """
        Add a scenario for comparison.
        
        Args:
            name: Unique scenario name
            description: Scenario description
            parameters: Simulation parameters used
            results: Raw simulation results (metric -> values)
        """
        # Analyze each metric
        metrics = {}
        for metric_name, values in results.items():
            if len(values) > 0:
                metrics[metric_name] = self.analyzer.analyze_variable(values, metric_name)
        
        # Store scenario
        self.scenarios[name] = ScenarioResult(
            name=name,
            description=description,
            parameters=parameters,
            metrics=metrics,
            raw_data=results
        )
    
    def compare_scenarios(
        self,
        metric_names: Optional[List[str]] = None,
        optimization_direction: Dict[str, str] = None
    ) -> ComparisonResult:
        """
        Compare all scenarios across specified metrics.
        
        Args:
            metric_names: Metrics to compare (None = all common metrics)
            optimization_direction: Dict mapping metric to 'min' or 'max'
                                  (default: 'min' for loss/risk, 'max' for others)
        
        Returns:
            ComparisonResult with rankings and comparisons
        """
        if len(self.scenarios) < 2:
            raise ValueError("Need at least 2 scenarios to compare")
        
        # Determine metrics to compare
        if metric_names is None:
            # Find common metrics across all scenarios
            metric_sets = [set(s.metrics.keys()) for s in self.scenarios.values()]
            metric_names = list(set.intersection(*metric_sets))
        
        # Default optimization directions
        if optimization_direction is None:
            optimization_direction = {}
            for metric in metric_names:
                if any(word in metric.lower() for word in 
                      ['loss', 'risk', 'cost', 'failure', 'attack']):
                    optimization_direction[metric] = 'min'
                else:
                    optimization_direction[metric] = 'max'
        
        # Rank scenarios for each metric
        ranking = {}
        best_scenario = {}
        
        for metric in metric_names:
            # Get values for ranking
            scenario_values = []
            for scenario_name, scenario in self.scenarios.items():
                if metric in scenario.metrics:
                    value = scenario.metrics[metric].mean
                    scenario_values.append((scenario_name, value))
            
            # Sort based on optimization direction
            reverse = (optimization_direction.get(metric, 'max') == 'max')
            scenario_values.sort(key=lambda x: x[1], reverse=reverse)
            
            # Store ranking
            ranking[metric] = [name for name, _ in scenario_values]
            best_scenario[metric] = ranking[metric][0] if ranking[metric] else None
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(metric_names)
        
        # Create dominance matrix
        dominance_matrix = self._calculate_dominance_matrix(
            metric_names,
            optimization_direction
        )
        
        return ComparisonResult(
            scenarios=list(self.scenarios.values()),
            ranking=ranking,
            best_scenario=best_scenario,
            statistical_tests=statistical_tests,
            dominance_matrix=dominance_matrix
        )
    
    def _perform_statistical_tests(
        self,
        metric_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical tests between scenarios.
        
        Args:
            metric_names: Metrics to test
            
        Returns:
            Test results for each metric
        """
        test_results = {}
        
        for metric in metric_names:
            # Collect raw data for each scenario
            scenario_data = {}
            for name, scenario in self.scenarios.items():
                if scenario.raw_data and metric in scenario.raw_data:
                    scenario_data[name] = scenario.raw_data[metric]
            
            if len(scenario_data) < 2:
                continue
            
            # Perform ANOVA if more than 2 groups
            if len(scenario_data) > 2:
                groups = list(scenario_data.values())
                f_stat, p_value = stats.f_oneway(*groups)
                
                test_results[metric] = {
                    'test': 'ANOVA',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_groups': len(groups)
                }
            
            # Perform pairwise t-tests
            pairwise_results = {}
            scenario_names = list(scenario_data.keys())
            
            for i in range(len(scenario_names)):
                for j in range(i + 1, len(scenario_names)):
                    name1, name2 = scenario_names[i], scenario_names[j]
                    data1, data2 = scenario_data[name1], scenario_data[name2]
                    
                    # Perform t-test
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.var(data1) + np.var(data2)) / 2
                    )
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std != 0 else 0
                    
                    pairwise_results[f"{name1}_vs_{name2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': cohens_d,
                        'significant': p_val < 0.05
                    }
            
            if pairwise_results:
                if metric not in test_results:
                    test_results[metric] = {}
                test_results[metric]['pairwise'] = pairwise_results
        
        return test_results
    
    def _calculate_dominance_matrix(
        self,
        metric_names: List[str],
        optimization_direction: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Calculate dominance matrix showing scenario superiority.
        
        Args:
            metric_names: Metrics to consider
            optimization_direction: Optimization direction for each metric
            
        Returns:
            DataFrame showing dominance relationships
        """
        scenario_names = list(self.scenarios.keys())
        n_scenarios = len(scenario_names)
        
        # Initialize dominance matrix
        dominance = np.zeros((n_scenarios, n_scenarios))
        
        for i, scenario1 in enumerate(scenario_names):
            for j, scenario2 in enumerate(scenario_names):
                if i == j:
                    continue
                
                # Count metrics where scenario1 dominates scenario2
                dominance_count = 0
                total_metrics = 0
                
                for metric in metric_names:
                    if (metric in self.scenarios[scenario1].metrics and
                        metric in self.scenarios[scenario2].metrics):
                        
                        val1 = self.scenarios[scenario1].metrics[metric].mean
                        val2 = self.scenarios[scenario2].metrics[metric].mean
                        
                        direction = optimization_direction.get(metric, 'max')
                        
                        if direction == 'max' and val1 > val2:
                            dominance_count += 1
                        elif direction == 'min' and val1 < val2:
                            dominance_count += 1
                        
                        total_metrics += 1
                
                # Calculate dominance percentage
                if total_metrics > 0:
                    dominance[i, j] = dominance_count / total_metrics
        
        return pd.DataFrame(
            dominance,
            index=scenario_names,
            columns=scenario_names
        )
    
    def identify_pareto_optimal(
        self,
        metrics: List[str],
        optimization_direction: Dict[str, str] = None
    ) -> List[str]:
        """
        Identify Pareto-optimal scenarios.
        
        A scenario is Pareto-optimal if no other scenario dominates it
        across all metrics.
        
        Args:
            metrics: Metrics to consider
            optimization_direction: Direction for each metric
            
        Returns:
            List of Pareto-optimal scenario names
        """
        if optimization_direction is None:
            optimization_direction = {}
            for metric in metrics:
                if any(word in metric.lower() for word in 
                      ['loss', 'risk', 'cost', 'failure']):
                    optimization_direction[metric] = 'min'
                else:
                    optimization_direction[metric] = 'max'
        
        pareto_optimal = []
        
        for scenario1_name, scenario1 in self.scenarios.items():
            is_dominated = False
            
            for scenario2_name, scenario2 in self.scenarios.items():
                if scenario1_name == scenario2_name:
                    continue
                
                # Check if scenario2 dominates scenario1
                dominates = True
                at_least_one_better = False
                
                for metric in metrics:
                    if (metric not in scenario1.metrics or
                        metric not in scenario2.metrics):
                        dominates = False
                        break
                    
                    val1 = scenario1.metrics[metric].mean
                    val2 = scenario2.metrics[metric].mean
                    direction = optimization_direction.get(metric, 'max')
                    
                    if direction == 'max':
                        if val2 < val1:
                            dominates = False
                            break
                        if val2 > val1:
                            at_least_one_better = True
                    else:  # min
                        if val2 > val1:
                            dominates = False
                            break
                        if val2 < val1:
                            at_least_one_better = True
                
                if dominates and at_least_one_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(scenario1_name)
        
        return pareto_optimal
    
    def calculate_scenario_scores(
        self,
        weights: Dict[str, float],
        normalization: str = 'minmax'
    ) -> Dict[str, float]:
        """
        Calculate weighted scores for scenarios.
        
        Args:
            weights: Weights for each metric (should sum to 1)
            normalization: Normalization method ('minmax', 'zscore')
            
        Returns:
            Dictionary mapping scenario names to scores
        """
        # Validate weights
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights should sum to 1.0")
        
        # Prepare data matrix
        metrics = list(weights.keys())
        scenario_names = list(self.scenarios.keys())
        
        data_matrix = []
        for scenario in scenario_names:
            row = []
            for metric in metrics:
                if metric in self.scenarios[scenario].metrics:
                    row.append(self.scenarios[scenario].metrics[metric].mean)
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalize data
        if normalization == 'minmax':
            # Min-max normalization
            mins = np.nanmin(data_matrix, axis=0)
            maxs = np.nanmax(data_matrix, axis=0)
            normalized = (data_matrix - mins) / (maxs - mins)
        elif normalization == 'zscore':
            # Z-score normalization
            means = np.nanmean(data_matrix, axis=0)
            stds = np.nanstd(data_matrix, axis=0)
            normalized = (data_matrix - means) / stds
        else:
            normalized = data_matrix
        
        # Calculate weighted scores
        weight_array = np.array([weights[m] for m in metrics])
        scores = np.nansum(normalized * weight_array, axis=1)
        
        return dict(zip(scenario_names, scores))
    
    def generate_comparison_report(self) -> str:
        """
        Generate a text report comparing all scenarios.
        
        Returns:
            Formatted comparison report
        """
        if not self.scenarios:
            return "No scenarios to compare"
        
        report = []
        report.append("=" * 80)
        report.append("SCENARIO COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Scenario descriptions
        report.append("SCENARIOS:")
        report.append("-" * 40)
        for name, scenario in self.scenarios.items():
            report.append(f"\n{name}:")
            report.append(f"  Description: {scenario.description}")
            report.append(f"  Parameters:")
            for param, value in scenario.parameters.items():
                report.append(f"    - {param}: {value}")
        
        # Metric summaries
        report.append("\n" + "=" * 80)
        report.append("METRIC SUMMARIES:")
        report.append("-" * 40)
        
        # Find common metrics
        metric_sets = [set(s.metrics.keys()) for s in self.scenarios.values()]
        common_metrics = list(set.intersection(*metric_sets))
        
        for metric in sorted(common_metrics):
            report.append(f"\n{metric}:")
            for name, scenario in self.scenarios.items():
                if metric in scenario.metrics:
                    stats = scenario.metrics[metric]
                    report.append(f"  {name}:")
                    report.append(f"    Mean: {stats.mean:.2f}")
                    report.append(f"    Median: {stats.median:.2f}")
                    report.append(f"    Std Dev: {stats.std:.2f}")
                    report.append(f"    95% CI: [{stats.confidence_interval[0]:.2f}, "
                                f"{stats.confidence_interval[1]:.2f}]")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
