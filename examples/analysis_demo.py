#!/usr/bin/env python3
"""
Demonstration of statistical analysis tools for Monte Carlo simulation.

This script shows how to use the analysis package to:
- Perform statistical analysis on simulation results
- Compare different scenarios
- Assess risk levels
- Generate comprehensive reports
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import (
    StatisticalAnalyzer,
    ScenarioComparator,
    RiskAssessor,
    RiskMetrics,
    RiskLevel,
    ReportGenerator,
    ReportConfig
)


def demonstrate_statistical_analysis():
    """Demonstrate statistical analysis capabilities."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Create analyzer
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Generate sample data (simulating CRQC emergence years)
    np.random.seed(42)
    crqc_years = np.random.normal(2035, 3, 1000)
    crqc_years = np.clip(crqc_years, 2028, 2050)  # Realistic bounds
    
    print("\n1. Analyzing CRQC Emergence Timeline")
    print("-" * 40)
    
    # Analyze the data
    summary = analyzer.analyze_variable(crqc_years, "CRQC Emergence Year")
    
    print(f"Mean: {summary.mean:.1f}")
    print(f"Median: {summary.median:.1f}")
    print(f"Std Dev: {summary.std:.1f}")
    print(f"95% CI: [{summary.confidence_interval[0]:.1f}, {summary.confidence_interval[1]:.1f}]")
    print(f"5th Percentile: {summary.percentiles[5]:.1f}")
    print(f"95th Percentile: {summary.percentiles[95]:.1f}")
    print(f"Skewness: {summary.skewness:.3f}")
    print(f"Kurtosis: {summary.kurtosis:.3f}")
    
    # Calculate risk metrics
    print("\n2. Risk Metrics")
    print("-" * 40)
    
    # Generate sample loss data
    losses = np.random.lognormal(24, 1, 1000)  # Log-normal losses in billions
    
    var_95 = analyzer.calculate_var(losses, 0.95)
    cvar_95 = analyzer.calculate_cvar(losses, 0.95)
    
    print(f"Value at Risk (95%): ${var_95:.1f}B")
    print(f"Conditional VaR (95%): ${cvar_95:.1f}B")
    print(f"Expected Shortfall Ratio: {cvar_95/var_95:.2f}")
    
    # Fit distributions
    print("\n3. Distribution Fitting")
    print("-" * 40)
    
    fits = analyzer.fit_distribution(crqc_years, ['norm', 'gamma'])
    
    for dist_name, fit in fits.items():
        print(f"\n{dist_name.capitalize()} Distribution:")
        print(f"  KS Test p-value: {fit.ks_pvalue:.4f}")
        print(f"  AIC: {fit.aic:.1f}")
        print(f"  BIC: {fit.bic:.1f}")
        print(f"  Good Fit: {'Yes' if fit.is_good_fit else 'No'}")
    
    # Identify outliers
    print("\n4. Outlier Detection")
    print("-" * 40)
    
    # Add some outliers
    data_with_outliers = np.append(crqc_years, [2025, 2026, 2055, 2060])
    
    outliers = analyzer.identify_outliers(data_with_outliers, method='iqr')
    print(f"Number of outliers: {outliers['n_outliers']}")
    print(f"Outlier percentage: {outliers['outlier_percentage']:.1f}%")
    print(f"Outlier values: {sorted(outliers['outlier_values'])[:5]}")  # Show first 5


def demonstrate_scenario_comparison():
    """Demonstrate scenario comparison capabilities."""
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON DEMONSTRATION")
    print("=" * 80)
    
    comparator = ScenarioComparator()
    
    # Create scenarios with different migration strategies
    np.random.seed(42)
    
    # Scenario 1: No Migration
    no_migration_losses = np.random.lognormal(25, 1.2, 1000)
    no_migration_years = np.random.normal(2033, 2, 1000)
    
    comparator.add_scenario(
        name="No Migration",
        description="No quantum-safe migration",
        parameters={'migration_rate': 0.0},
        results={
            'economic_loss': no_migration_losses.tolist(),
            'attack_year': no_migration_years.tolist()
        }
    )
    
    # Scenario 2: Gradual Migration
    gradual_losses = np.random.lognormal(24, 1, 1000)
    gradual_years = np.random.normal(2035, 3, 1000)
    
    comparator.add_scenario(
        name="Gradual Migration",
        description="50% migration by 2030",
        parameters={'migration_rate': 0.5},
        results={
            'economic_loss': gradual_losses.tolist(),
            'attack_year': gradual_years.tolist()
        }
    )
    
    # Scenario 3: Aggressive Migration
    aggressive_losses = np.random.lognormal(23, 0.8, 1000)
    aggressive_years = np.random.normal(2037, 4, 1000)
    
    comparator.add_scenario(
        name="Aggressive Migration",
        description="90% migration by 2030",
        parameters={'migration_rate': 0.9},
        results={
            'economic_loss': aggressive_losses.tolist(),
            'attack_year': aggressive_years.tolist()
        }
    )
    
    print("\n1. Scenario Comparison Results")
    print("-" * 40)
    
    # Compare scenarios
    comparison = comparator.compare_scenarios(
        optimization_direction={
            'economic_loss': 'min',
            'attack_year': 'max'
        }
    )
    
    # Print rankings
    for metric, ranking in comparison.ranking.items():
        print(f"\n{metric} Ranking:")
        for i, scenario in enumerate(ranking, 1):
            scenario_obj = comparator.scenarios[scenario]
            mean_value = scenario_obj.get_metric(metric, 'mean')
            print(f"  {i}. {scenario}: {mean_value:.1f}")
    
    # Best scenarios
    print("\n2. Best Scenarios by Metric")
    print("-" * 40)
    for metric, best in comparison.best_scenario.items():
        print(f"{metric}: {best}")
    
    # Pareto optimal scenarios
    print("\n3. Pareto Optimal Scenarios")
    print("-" * 40)
    pareto = comparator.identify_pareto_optimal(
        ['economic_loss', 'attack_year'],
        {'economic_loss': 'min', 'attack_year': 'max'}
    )
    print(f"Pareto optimal: {', '.join(pareto)}")
    
    # Weighted scoring
    print("\n4. Weighted Scenario Scores")
    print("-" * 40)
    weights = {'economic_loss': 0.6, 'attack_year': 0.4}
    scores = comparator.calculate_scenario_scores(weights, normalization='minmax')
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for scenario, score in sorted_scores:
        print(f"{scenario}: {score:.3f}")
    
    # Statistical tests
    print("\n5. Statistical Significance")
    print("-" * 40)
    if 'economic_loss' in comparison.statistical_tests:
        tests = comparison.statistical_tests['economic_loss']
        if 'test' in tests:
            print(f"ANOVA F-statistic: {tests['f_statistic']:.2f}")
            print(f"ANOVA p-value: {tests['p_value']:.4f}")
            print(f"Significant difference: {tests['significant']}")


def demonstrate_risk_assessment():
    """Demonstrate risk assessment capabilities."""
    print("\n" + "=" * 80)
    print("RISK ASSESSMENT DEMONSTRATION")
    print("=" * 80)
    
    assessor = RiskAssessor()
    
    # Create mock simulation results
    simulation_results = {
        'metrics': {
            'first_attack_year': {
                'mean': 2035,
                'std': 3,
                'min': 2030,
                'max': 2045,
                'percentile_5': 2031,
                'percentile_95': 2040
            },
            'economic_loss_usd': {
                'mean': 50_000_000_000,
                'median': 45_000_000_000,
                'std': 20_000_000_000,
                'max': 150_000_000_000,
                'percentile_95': 90_000_000_000,
                'percentile_99': 120_000_000_000
            },
            'attack_success_rate': 0.35
        },
        'metadata': {
            'successful_iterations': 10000,
            'start_year': 2025,
            'end_year': 2045
        }
    }
    
    print("\n1. Overall Risk Assessment")
    print("-" * 40)
    
    # Assess risk
    risk_metrics = assessor.assess_quantum_risk(simulation_results, current_year=2025)
    
    print(f"Risk Level: {risk_metrics.risk_level.value}")
    print(f"Risk Score: {risk_metrics.risk_score:.1f}/100")
    print(f"Probability: {risk_metrics.probability:.1%}")
    print(f"Impact: {risk_metrics.impact:.1%}")
    print(f"Time Horizon: {risk_metrics.time_horizon:.1f} years")
    print(f"Confidence: {risk_metrics.confidence:.1%}")
    print(f"Risk Index: {risk_metrics.risk_index:.1f}")
    
    # Assess threat categories
    print("\n2. Threat Category Assessment")
    print("-" * 40)
    
    threats = assessor.assess_threat_categories(simulation_results)
    
    # Sort by threat score
    sorted_threats = sorted(
        threats.values(),
        key=lambda x: x.threat_score,
        reverse=True
    )
    
    for threat in sorted_threats[:3]:  # Top 3 threats
        print(f"\n{threat.category.value}:")
        print(f"  Threat Score: {threat.threat_score:.2f}")
        print(f"  Likelihood: {threat.likelihood:.1%}")
        print(f"  Severity: {threat.severity:.1%}")
        print(f"  Detection Difficulty: {threat.detection_difficulty:.1%}")
        print(f"  Mitigation Effectiveness: {threat.mitigation_effectiveness:.1%}")
    
    # Risk trajectory
    print("\n3. Risk Trajectory Over Time")
    print("-" * 40)
    
    years = [2025, 2030, 2035, 2040]
    trajectory = assessor.calculate_risk_trajectory(simulation_results, years)
    
    print("Year | Risk Score | Risk Level")
    print("-----|------------|------------")
    for year, metrics in trajectory.items():
        print(f"{year} | {metrics.risk_score:10.1f} | {metrics.risk_level.value}")
    
    # Critical thresholds
    print("\n4. Critical Thresholds")
    print("-" * 40)
    
    thresholds = assessor.identify_critical_thresholds(simulation_results)
    for threshold_name, year in thresholds.items():
        print(f"{threshold_name}: {year:.0f}")


def demonstrate_report_generation():
    """Demonstrate report generation capabilities."""
    print("\n" + "=" * 80)
    print("REPORT GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Create mock results
    results = {
        'metrics': {
            'first_attack_year': {
                'mean': 2035.2,
                'median': 2035,
                'std': 3.1,
                'min': 2028,
                'max': 2048,
                'percentile_5': 2030,
                'percentile_95': 2041,
                'skewness': 0.15,
                'kurtosis': -0.3
            },
            'economic_loss_usd': {
                'mean': 52_300_000_000,
                'median': 48_000_000_000,
                'std': 18_500_000_000,
                'min': 5_000_000_000,
                'max': 142_000_000_000,
                'percentile_5': 22_000_000_000,
                'percentile_95': 89_000_000_000
            },
            'attack_success_rate': 0.42
        },
        'metadata': {
            'successful_iterations': 10000,
            'failed_iterations': 0,
            'total_runtime': 45.3,
            'iterations_per_second': 220.8,
            'start_year': 2025,
            'end_year': 2045
        },
        'parameters': {
            'n_iterations': 10000,
            'random_seed': 42,
            'quantum_growth_rate': 1.5,
            'migration_rate': 0.8
        }
    }
    
    # Create risk metrics
    risk_metrics = RiskMetrics(
        risk_score=68.5,
        risk_level=RiskLevel.HIGH,
        probability=0.65,
        impact=0.75,
        time_horizon=10.2,
        confidence=0.95
    )
    
    print("\n1. Markdown Report (Preview)")
    print("-" * 40)
    
    # Generate Markdown report
    config_md = ReportConfig(
        title="Quantum Risk Analysis - Demo",
        author="Analysis System",
        output_format="markdown"
    )
    generator_md = ReportGenerator(config_md)
    
    report_md = generator_md.generate_report(results, risk_metrics)
    
    # Show first 500 characters
    print(report_md[:500] + "...\n")
    
    print("2. Summary Table")
    print("-" * 40)
    
    # Generate summary table
    generator = ReportGenerator()
    df = generator.generate_summary_table(results)
    print(df.to_string())
    
    print("\n3. JSON Report Structure")
    print("-" * 40)
    
    # Generate JSON report
    config_json = ReportConfig(output_format="json")
    generator_json = ReportGenerator(config_json)
    
    import json
    report_json = generator_json.generate_report(results, risk_metrics)
    data = json.loads(report_json)
    
    print("JSON Report Keys:")
    for key in data.keys():
        print(f"  - {key}")
        if isinstance(data[key], dict):
            for subkey in list(data[key].keys())[:3]:
                print(f"    - {subkey}")
    
    print("\n4. Report File Paths")
    print("-" * 40)
    
    # Show where reports would be saved
    output_dir = Path("data/output/reports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Markdown: {output_dir}/report_{timestamp}.md")
    print(f"JSON: {output_dir}/report_{timestamp}.json")
    print(f"CSV: {output_dir}/report_{timestamp}.csv")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS TOOLS DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases the analysis capabilities for Monte Carlo simulations")
    
    # Run demonstrations
    demonstrate_statistical_analysis()
    demonstrate_scenario_comparison()
    demonstrate_risk_assessment()
    demonstrate_report_generation()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe analysis tools provide:")
    print("  ✓ Comprehensive statistical analysis")
    print("  ✓ Multi-scenario comparison")
    print("  ✓ Risk assessment and scoring")
    print("  ✓ Professional report generation")
    print("\nUse these tools to extract insights from your simulation results!")


if __name__ == "__main__":
    main()
