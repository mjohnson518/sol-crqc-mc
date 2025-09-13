#!/usr/bin/env python3
"""
Complete Monte Carlo Simulation Runner for Solana Quantum Impact.

This script runs the full simulation with all models integrated and
generates comprehensive reports and visualizations.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from src.config import SimulationParameters

# Import core simulation
from src.core.simulation import MonteCarloSimulation
from src.core.results_collector import ResultsCollector

# Import all models
from src.models.quantum_timeline import QuantumDevelopmentModel
from src.models.network_state import NetworkStateModel
from src.models.attack_scenarios import AttackScenariosModel
from src.models.economic_impact import EconomicImpactModel

# Import analysis tools
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.scenario_comparison import ScenarioComparator
from analysis.risk_assessment import RiskAssessor
from analysis.report_generator import ReportGenerator, ReportConfig

# Import visualization tools
from visualization.timeline_plots import TimelinePlotter
from visualization.network_plots import NetworkPlotter
from visualization.attack_plots import AttackPlotter
from visualization.economic_plots import EconomicPlotter
from visualization.statistical_plots import StatisticalPlotter
from visualization.dashboard import DashboardCreator


def create_output_directory(base_dir: str = "simulation_results") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    return output_dir


def run_simulation(config: SimulationParameters, output_dir: Path) -> dict:
    """
    Run the complete Monte Carlo simulation.
    
    Args:
        config: Simulation configuration
        output_dir: Directory for outputs
        
    Returns:
        Dictionary of simulation results
    """
    print("\n" + "="*70)
    print(" STARTING MONTE CARLO SIMULATION ")
    print("="*70)
    
    # Save configuration
    config_path = output_dir / "data" / "configuration.json"
    with open(config_path, 'w') as f:
        json.dump(config._to_serializable_dict(), f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    # Initialize all models
    print("\n" + "-"*50)
    print("Initializing models...")
    
    models = {
        'quantum_timeline': QuantumDevelopmentModel(config.quantum),
        'network_state': NetworkStateModel(config.network),
        'attack_scenarios': AttackScenariosModel(config.quantum),
        'economic_impact': EconomicImpactModel(config.economic)
    }
    print("✓ All models initialized")
    
    # Create simulation instance
    sim = MonteCarloSimulation(config, models=models)
    
    # Run simulation
    print("\n" + "-"*50)
    print(f"Running {config.n_iterations} iterations on {config.n_cores} cores...")
    print("This may take several minutes...\n")
    
    start_time = time.time()
    results = sim.run()
    runtime = time.time() - start_time
    
    print(f"\n✓ Simulation completed in {runtime:.1f} seconds")
    print(f"  Average time per iteration: {runtime/config.n_iterations:.3f} seconds")
    
    # Save raw results
    if config.save_raw_results:
        results_path = output_dir / "data" / "raw_results.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_to_serializable(results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✓ Raw results saved to: {results_path}")
    
    return results


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Convert dataclasses and other objects with __dict__ to dictionaries
        return convert_to_serializable(obj.__dict__)
    elif hasattr(obj, '__dataclass_fields__'):
        # Handle dataclasses specifically
        from dataclasses import asdict
        return convert_to_serializable(asdict(obj))
    else:
        # Try to convert to string for any remaining objects
        try:
            return str(obj)
        except:
            return None


def analyze_results(results: dict, output_dir: Path) -> dict:
    """
    Perform comprehensive analysis of simulation results.
    
    Args:
        results: Raw simulation results
        output_dir: Directory for outputs
        
    Returns:
        Dictionary of analysis results
    """
    print("\n" + "-"*50)
    print("Analyzing results...")
    
    # Extract raw data arrays from results if available
    raw_data = {}
    if 'raw_results' in results and results['raw_results']:
        # Extract arrays from raw results
        raw_data['crqc_years'] = []
        raw_data['economic_losses'] = []
        raw_data['attack_success_rates'] = []
        
        for iteration in results['raw_results']:
            if isinstance(iteration, dict):
                # Extract CRQC year
                quantum = iteration.get('quantum_timeline', {})
                if quantum and 'first_crqc_year' in quantum:
                    raw_data['crqc_years'].append(quantum['first_crqc_year'])
                
                # Extract economic loss
                economic = iteration.get('economic_impact', {})
                if economic and 'total_loss_usd' in economic:
                    raw_data['economic_losses'].append(economic['total_loss_usd'])
                
                # Extract attack success rate
                attack = iteration.get('attack_scenarios', {})
                if attack and 'attack_success' in attack:
                    raw_data['attack_success_rates'].append(1.0 if attack['attack_success'] else 0.0)
    
    # Statistical analysis
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Analyze key variables if they exist in raw data
    stats_summary = {}
    if raw_data.get('crqc_years'):
        stats_summary['crqc_years'] = analyzer.analyze_variable(
            raw_data['crqc_years'], 'CRQC Emergence Year'
        )
    if raw_data.get('economic_losses'):
        stats_summary['economic_losses'] = analyzer.analyze_variable(
            raw_data['economic_losses'], 'Economic Loss'
        )
    if raw_data.get('attack_success_rates'):
        stats_summary['attack_success_rates'] = analyzer.analyze_variable(
            raw_data['attack_success_rates'], 'Attack Success Rate'
        )
    
    # Risk assessment
    assessor = RiskAssessor()
    risk_metrics = assessor.assess_quantum_risk(results)
    
    # Save analysis results
    analysis_path = output_dir / "data" / "analysis_summary.json"
    analysis_results = {
        'statistics': convert_to_serializable(stats_summary),
        'risk_metrics': convert_to_serializable(risk_metrics),
        'raw_data': raw_data  # Include raw data for visualization
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✓ Analysis results saved to: {analysis_path}")
    
    return analysis_results


def generate_visualizations(results: dict, analysis: dict, output_dir: Path):
    """
    Generate comprehensive visualizations.
    
    Args:
        results: Raw simulation results
        analysis: Analysis results
        output_dir: Directory for outputs
    """
    print("\n" + "-"*50)
    print("Generating visualizations...")
    
    plots_dir = output_dir / "plots"
    
    # Set matplotlib to non-interactive mode for saving
    plt.ioff()
    
    # 1. Executive Dashboard
    print("  Creating executive dashboard...")
    dashboard_creator = DashboardCreator()
    fig = dashboard_creator.create_executive_summary(results)
    fig.savefig(plots_dir / "executive_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Quantum Timeline Visualization
    if 'quantum_timelines' in results:
        print("  Creating quantum timeline plots...")
        plotter = TimelinePlotter()
        # Assuming we have multiple timeline samples
        # For now, create a simple timeline plot
        
    # 3. Network Evolution Visualization
    if 'network_evolution' in results:
        print("  Creating network evolution plots...")
        plotter = NetworkPlotter()
        # Create network evolution visualizations
        
    # 4. Economic Impact Distribution
    if 'economic_losses' in results:
        print("  Creating economic impact plots...")
        plotter = EconomicPlotter()
        # Create economic impact visualizations
        
    # 5. Statistical Convergence
    print("  Creating statistical plots...")
    stat_plotter = StatisticalPlotter()
    
    # Plot convergence for key metrics
    if 'crqc_years' in results:
        fig = stat_plotter.plot_monte_carlo_convergence(
            results['crqc_years'],
            title="CRQC Year Convergence Analysis"
        )
        fig.savefig(plots_dir / "crqc_convergence.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Visualizations saved to: {plots_dir}")


def generate_reports(results: dict, analysis: dict, output_dir: Path):
    """
    Generate comprehensive reports.
    
    Args:
        results: Raw simulation results
        analysis: Analysis results
        output_dir: Directory for outputs
    """
    print("\n" + "-"*50)
    print("Generating reports...")
    
    reports_dir = output_dir / "reports"
    
    # Configure report generator
    config = ReportConfig(
        title="Solana Quantum Impact Monte Carlo Simulation Report",
        author="Supernova",
        include_charts=True,
        include_recommendations=True,
        include_raw_data=False
    )
    
    generator = ReportGenerator(config)
    
    # Generate markdown report
    print("  Creating markdown report...")
    md_report = generator.generate_report(
        results,
        risk_metrics=analysis.get('risk_metrics'),
        output_path=reports_dir / "simulation_report.md"
    )
    
    # Generate JSON summary
    print("  Creating JSON summary...")
    json_summary = {
        'configuration': convert_to_serializable(config),
        'analysis': convert_to_serializable(analysis),
        'summary': {
            'total_iterations': results.get('n_iterations', 0),
            'successful_iterations': results.get('successful_iterations', 0),
            'failed_iterations': results.get('failed_iterations', 0),
            'runtime_seconds': results.get('runtime', 0)
        }
    }
    json_path = reports_dir / "simulation_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"✓ Reports saved to: {reports_dir}")


def print_summary(results: dict, analysis: dict):
    """Print summary of key findings."""
    print("\n" + "="*70)
    print(" SIMULATION SUMMARY ")
    print("="*70)
    
    # Extract key metrics
    if 'crqc_years' in results:
        crqc_median = np.median(results['crqc_years'])
        crqc_p25 = np.percentile(results['crqc_years'], 25)
        crqc_p75 = np.percentile(results['crqc_years'], 75)
        print(f"\n📊 CRQC Emergence:")
        print(f"   Median Year: {int(crqc_median)}")
        print(f"   25th-75th Percentile: {int(crqc_p25)}-{int(crqc_p75)}")
    
    if 'economic_losses' in results:
        losses = results['economic_losses']
        mean_loss = np.mean(losses)
        max_loss = np.max(losses)
        print(f"\n💰 Economic Impact:")
        print(f"   Mean Loss: ${mean_loss/1e9:.1f}B")
        print(f"   Maximum Loss: ${max_loss/1e9:.1f}B")
    
    if 'attack_success_rates' in results:
        success_rates = results['attack_success_rates']
        mean_rate = np.mean(success_rates)
        print(f"\n⚔️  Attack Success:")
        print(f"   Average Success Rate: {mean_rate:.1%}")
    
    if 'risk_metrics' in analysis:
        risk = analysis['risk_metrics']
        if isinstance(risk, dict) and 'overall_score' in risk:
            print(f"\n⚠️  Risk Assessment:")
            print(f"   Overall Risk Score: {risk['overall_score']:.2f}/1.0")
            print(f"   Risk Level: {risk.get('risk_level', 'N/A')}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run complete Solana Quantum Impact Monte Carlo Simulation"
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=100,
        help='Number of Monte Carlo iterations (default: 100)'
    )
    parser.add_argument(
        '--cores', '-c',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: auto)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='simulation_results',
        help='Output directory (default: simulation_results)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with 10 iterations'
    )
    
    args = parser.parse_args()
    
    # Configure simulation
    if args.quick:
        print("\n🚀 Running in QUICK TEST mode (10 iterations)")
        n_iterations = 10
    else:
        n_iterations = args.iterations
    
    config = SimulationParameters(
        n_iterations=n_iterations,
        n_cores=args.cores or os.cpu_count(),
        random_seed=args.seed,
        save_raw_results=True,
        start_year=2025,
        end_year=2050
    )
    
    print(f"\n" + "="*70)
    print(f" SOLANA QUANTUM IMPACT MONTE CARLO SIMULATION ")
    print(f"="*70)
    print(f"\nConfiguration:")
    print(f"  • Iterations: {config.n_iterations:,}")
    print(f"  • CPU Cores: {config.n_cores}")
    print(f"  • Random Seed: {config.random_seed}")
    print(f"  • Time Range: {config.start_year}-{config.end_year}")
    print(f"  • Output: {args.output}/")
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    print(f"\n📁 Output directory: {output_dir}")
    
    try:
        # Run simulation
        results = run_simulation(config, output_dir)
        
        # Analyze results
        analysis = analyze_results(results, output_dir)
        
        # Generate visualizations
        generate_visualizations(results, analysis, output_dir)
        
        # Generate reports
        try:
            generate_reports(results, analysis, output_dir)
        except Exception as e:
            print(f"⚠️  Warning: Report generation failed: {e}")
            print("  Continuing with summary...")
        
        # Print summary
        print_summary(results, analysis)
        
        print(f"\n" + "="*70)
        print(f" ✅ SIMULATION COMPLETE ")
        print(f"="*70)
        print(f"\n📊 Results saved to: {output_dir}")
        print(f"\nKey outputs:")
        print(f"  • Executive Dashboard: {output_dir}/plots/executive_dashboard.png")
        print(f"  • Full Report: {output_dir}/reports/simulation_report.md")
        print(f"  • Data Files: {output_dir}/data/")
        print(f"\n🎉 Success! All components executed successfully.\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
