"""
Unit tests for statistical analysis tools.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from analysis import (
    StatisticalAnalyzer,
    StatisticalSummary,
    ScenarioComparator,
    RiskAssessor,
    RiskLevel,
    ReportGenerator,
    ReportConfig
)


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        assert analyzer.confidence_level == 0.95
        assert abs(analyzer.alpha - 0.05) < 1e-10  # Handle floating point precision
    
    def test_analyze_variable(self):
        """Test variable analysis."""
        analyzer = StatisticalAnalyzer()
        
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        
        # Analyze
        summary = analyzer.analyze_variable(data, "test_variable")
        
        # Check results
        assert isinstance(summary, StatisticalSummary)
        assert 95 < summary.mean < 105
        assert 12 < summary.std < 18
        assert summary.sample_size == 1000
        assert len(summary.percentiles) == 9
        assert summary.percentiles[50] == summary.median
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        analyzer = StatisticalAnalyzer()
        
        with pytest.raises(ValueError, match="No valid data"):
            analyzer.analyze_variable([])
    
    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Generate loss data
        np.random.seed(42)
        losses = np.random.exponential(1000, 1000)
        
        # Calculate VaR
        var_95 = analyzer.calculate_var(losses, 0.95)
        var_99 = analyzer.calculate_var(losses, 0.99)
        
        assert var_95 > 0
        assert var_99 > var_95
        assert var_95 == np.percentile(losses, 95)
    
    def test_calculate_cvar(self):
        """Test Conditional Value at Risk calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Generate loss data
        np.random.seed(42)
        losses = np.random.exponential(1000, 1000)
        
        # Calculate CVaR
        cvar_95 = analyzer.calculate_cvar(losses, 0.95)
        var_95 = analyzer.calculate_var(losses, 0.95)
        
        assert cvar_95 > var_95  # CVaR should exceed VaR
        assert cvar_95 == np.mean(losses[losses >= var_95])
    
    def test_distribution_fitting(self):
        """Test distribution fitting."""
        analyzer = StatisticalAnalyzer()
        
        # Generate data from known distribution
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        
        # Fit distributions
        fits = analyzer.fit_distribution(data, ['norm', 'lognorm'])
        
        assert 'norm' in fits
        assert 'lognorm' in fits
        
        # Normal should fit better
        assert fits['norm'].ks_pvalue > 0.05  # Good fit
        assert fits['norm'].aic < fits['lognorm'].aic  # Better AIC
    
    def test_outlier_detection(self):
        """Test outlier detection methods."""
        analyzer = StatisticalAnalyzer()
        
        # Create data with outliers
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        data = np.append(data, [200, 250, -50])  # Add outliers
        
        # Test IQR method
        outliers_iqr = analyzer.identify_outliers(data, method='iqr')
        assert outliers_iqr['n_outliers'] >= 3
        
        # Test z-score method
        outliers_z = analyzer.identify_outliers(data, method='zscore', threshold=3)
        assert outliers_z['n_outliers'] >= 2
    
    def test_correlation_calculation(self):
        """Test correlation matrix calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Create correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)
        z = -x + np.random.normal(0, 0.5, 100)
        
        data = {'x': x, 'y': y, 'z': z}
        
        # Calculate correlations
        corr_matrix = analyzer.calculate_correlations(data)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.loc['x', 'y'] > 0.8  # Strong positive
        assert corr_matrix.loc['x', 'z'] < -0.8  # Strong negative
        assert abs(corr_matrix.loc['x', 'x'] - 1.0) < 0.001  # Self-correlation


class TestScenarioComparator:
    """Test scenario comparison functionality."""
    
    def test_scenario_addition(self):
        """Test adding scenarios for comparison."""
        comparator = ScenarioComparator()
        
        # Add scenarios
        comparator.add_scenario(
            name="Base",
            description="Base case",
            parameters={'param1': 1.0},
            results={'metric1': [1, 2, 3, 4, 5]}
        )
        
        comparator.add_scenario(
            name="Alternative",
            description="Alternative case",
            parameters={'param1': 2.0},
            results={'metric1': [2, 3, 4, 5, 6]}
        )
        
        assert len(comparator.scenarios) == 2
        assert 'Base' in comparator.scenarios
        assert 'Alternative' in comparator.scenarios
    
    def test_scenario_comparison(self):
        """Test comparing scenarios."""
        comparator = ScenarioComparator()
        
        # Add scenarios
        np.random.seed(42)
        comparator.add_scenario(
            "Scenario1",
            "First scenario",
            {'param': 1},
            {'loss': np.random.normal(100, 10, 100).tolist(),
             'gain': np.random.normal(50, 5, 100).tolist()}
        )
        
        comparator.add_scenario(
            "Scenario2",
            "Second scenario",
            {'param': 2},
            {'loss': np.random.normal(80, 8, 100).tolist(),
             'gain': np.random.normal(60, 6, 100).tolist()}
        )
        
        # Compare
        result = comparator.compare_scenarios()
        
        assert len(result.scenarios) == 2
        assert 'loss' in result.ranking
        assert 'gain' in result.ranking
        assert result.best_scenario['loss'] == 'Scenario2'  # Lower loss is better
        assert result.best_scenario['gain'] == 'Scenario2'  # Higher gain is better
    
    def test_pareto_optimality(self):
        """Test Pareto optimal identification."""
        comparator = ScenarioComparator()
        
        # Add scenarios with trade-offs
        comparator.add_scenario(
            "A", "Scenario A", {},
            {'cost': [10], 'benefit': [100]}
        )
        comparator.add_scenario(
            "B", "Scenario B", {},
            {'cost': [20], 'benefit': [150]}
        )
        comparator.add_scenario(
            "C", "Scenario C", {},  # Dominated by A
            {'cost': [15], 'benefit': [90]}
        )
        
        # Find Pareto optimal
        pareto = comparator.identify_pareto_optimal(
            ['cost', 'benefit'],
            {'cost': 'min', 'benefit': 'max'}
        )
        
        assert 'A' in pareto
        assert 'B' in pareto
        assert 'C' not in pareto  # Dominated
    
    def test_scenario_scoring(self):
        """Test weighted scenario scoring."""
        comparator = ScenarioComparator()
        
        # Add scenarios
        comparator.add_scenario(
            "S1", "Scenario 1", {},
            {'metric1': [100], 'metric2': [50]}
        )
        comparator.add_scenario(
            "S2", "Scenario 2", {},
            {'metric1': [80], 'metric2': [70]}
        )
        
        # Calculate scores
        weights = {'metric1': 0.7, 'metric2': 0.3}
        scores = comparator.calculate_scenario_scores(weights)
        
        assert 'S1' in scores
        assert 'S2' in scores
        assert isinstance(scores['S1'], float)


class TestRiskAssessor:
    """Test risk assessment functionality."""
    
    def test_risk_assessment(self):
        """Test basic risk assessment."""
        assessor = RiskAssessor()
        
        # Create mock simulation results
        results = {
            'metrics': {
                'first_attack_year': {
                    'mean': 2035,
                    'std': 3,
                    'min': 2030,
                    'max': 2045
                },
                'economic_loss_usd': {
                    'mean': 50_000_000_000,
                    'max': 100_000_000_000,
                    'percentile_95': 80_000_000_000
                }
            },
            'metadata': {
                'successful_iterations': 5000
            }
        }
        
        # Assess risk
        risk_metrics = assessor.assess_quantum_risk(results, current_year=2025)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert 0 <= risk_metrics.risk_score <= 100
        assert risk_metrics.risk_level in RiskLevel
        assert 0 <= risk_metrics.probability <= 1
        assert 0 <= risk_metrics.impact <= 1
        assert risk_metrics.time_horizon > 0
    
    def test_risk_categorization(self):
        """Test risk level categorization."""
        assessor = RiskAssessor()
        
        # Test different probability/impact combinations
        assert assessor.risk_matrix.categorize_risk(0.1, 0.1) == RiskLevel.MINIMAL
        assert assessor.risk_matrix.categorize_risk(0.9, 0.9) == RiskLevel.CRITICAL
        assert assessor.risk_matrix.categorize_risk(0.5, 0.5) == RiskLevel.MODERATE
    
    def test_risk_trajectory(self):
        """Test risk trajectory calculation."""
        assessor = RiskAssessor()
        
        # Mock results
        results = {
            'metrics': {
                'first_attack_year': {'mean': 2035, 'std': 3}
            }
        }
        
        # Calculate trajectory
        years = [2025, 2030, 2035, 2040]
        trajectory = assessor.calculate_risk_trajectory(results, years)
        
        assert len(trajectory) == 4
        for year in years:
            assert year in trajectory
            assert isinstance(trajectory[year], RiskMetrics)
    
    def test_risk_report_generation(self):
        """Test risk report generation."""
        assessor = RiskAssessor()
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            risk_score=75,
            risk_level=RiskLevel.HIGH,
            probability=0.7,
            impact=0.8,
            time_horizon=10,
            confidence=0.9
        )
        
        # Generate report
        report = assessor.generate_risk_report(risk_metrics)
        
        assert isinstance(report, str)
        assert "QUANTUM RISK ASSESSMENT REPORT" in report
        assert "Risk Level: High" in report
        assert "Risk Score: 75" in report


class TestReportGenerator:
    """Test report generation functionality."""
    
    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        config = ReportConfig(
            title="Test Report",
            output_format="markdown"
        )
        generator = ReportGenerator(config)
        
        # Mock results
        results = {
            'metrics': {
                'test_metric': {
                    'mean': 100,
                    'median': 99,
                    'std': 10,
                    'min': 80,
                    'max': 120
                }
            },
            'metadata': {
                'successful_iterations': 1000,
                'total_runtime': 10.5
            }
        }
        
        # Generate report
        report = generator.generate_report(results)
        
        assert isinstance(report, str)
        assert "# Test Report" in report
        assert "Executive Summary" in report
        assert "Key Findings" in report
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        config = ReportConfig(output_format="json")
        generator = ReportGenerator(config)
        
        # Mock results
        results = {'metrics': {'test': {'mean': 100}}}
        
        # Generate report
        report = generator.generate_report(results)
        
        # Validate JSON
        import json
        data = json.loads(report)
        assert 'metadata' in data
        assert 'results' in data
    
    def test_csv_report_generation(self):
        """Test CSV report generation."""
        config = ReportConfig(output_format="csv")
        generator = ReportGenerator(config)
        
        # Mock results
        results = {
            'metrics': {
                'metric1': {'mean': 100, 'median': 99, 'std': 10},
                'metric2': {'mean': 200, 'median': 199, 'std': 20}
            }
        }
        
        # Generate report
        report = generator.generate_report(results)
        
        assert isinstance(report, str)
        assert "metric,mean,median,std" in report
        assert "metric1" in report
        assert "metric2" in report
    
    def test_summary_table_generation(self):
        """Test summary table generation."""
        generator = ReportGenerator()
        
        # Mock results
        results = {
            'metrics': {
                'metric1': {
                    'mean': 100,
                    'median': 99,
                    'std': 10,
                    'min': 80,
                    'max': 120,
                    'percentile_5': 85,
                    'percentile_95': 115
                }
            }
        }
        
        # Generate table
        df = generator.generate_summary_table(results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Metric' in df.columns
        assert df.iloc[0]['Metric'] == 'metric1'
        assert df.iloc[0]['Mean'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
