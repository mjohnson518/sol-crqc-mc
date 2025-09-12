#!/usr/bin/env python3
"""
Demonstration of visualization capabilities.

This script showcases the various visualization tools available
for analyzing Monte Carlo simulation results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import configuration and models
from src.config import SimulationParameters
from src.models.quantum_timeline import QuantumDevelopmentModel, QuantumTimeline, QuantumCapability, QuantumThreat
from src.models.network_state import NetworkStateModel, NetworkSnapshot, ValidatorState, ValidatorTier
from src.models.attack_scenarios import AttackScenariosModel, AttackScenario, AttackType, AttackSeverity
from src.models.economic_impact import EconomicImpactModel, EconomicLoss

# Import visualization modules
from visualization.timeline_plots import TimelinePlotter, plot_quantum_timeline
from visualization.network_plots import NetworkPlotter, plot_network_evolution
from visualization.attack_plots import AttackPlotter, plot_attack_windows
from visualization.economic_plots import EconomicPlotter, plot_economic_impact
from visualization.statistical_plots import StatisticalPlotter, plot_distributions
from visualization.dashboard import DashboardCreator


def demonstrate_timeline_visualization():
    """Demonstrate quantum timeline visualizations."""
    print("\n" + "="*60)
    print("QUANTUM TIMELINE VISUALIZATION")
    print("="*60)
    
    # Create sample quantum timelines
    config = SimulationParameters()
    model = QuantumDevelopmentModel(config.quantum)
    rng = np.random.RandomState(42)
    
    # Generate multiple timelines for ensemble plot
    timelines = []
    for i in range(50):
        rng_iter = np.random.RandomState(42 + i)
        timeline = model.project_timeline(rng_iter, 2025, 2050)
        timelines.append(timeline)
    
    # Create timeline plotter
    plotter = TimelinePlotter()
    
    # Plot ensemble
    print("\nCreating quantum timeline ensemble plot...")
    fig = plotter.plot_timeline_ensemble(
        timelines,
        title="Quantum Computing Development Scenarios (50 runs)"
    )
    plt.show()
    
    # Plot single timeline
    print("\nCreating single timeline plot...")
    fig = plot_quantum_timeline(
        timelines[0],
        title="Sample Quantum Development Timeline"
    )
    plt.show()
    
    print("Timeline visualization complete!")


def demonstrate_network_visualization():
    """Demonstrate network state visualizations."""
    print("\n" + "="*60)
    print("NETWORK STATE VISUALIZATION")
    print("="*60)
    
    # Create sample network evolution
    config = SimulationParameters()
    model = NetworkStateModel(config.network)
    rng = np.random.RandomState(42)
    
    # Generate network evolution
    evolution = model.simulate_evolution(rng, 2025, 2050)
    
    # Create network plotter
    plotter = NetworkPlotter()
    
    # Plot comprehensive evolution
    print("\nCreating comprehensive network evolution plot...")
    fig = plotter.plot_network_evolution_comprehensive(
        evolution,
        title="Solana Network Evolution Analysis"
    )
    plt.show()
    
    # Plot validator distribution for a specific year
    snapshot = evolution.get_snapshot_at_year(2035)
    print("\nCreating validator tier distribution plot...")
    fig = plotter.plot_validator_tier_distribution(
        snapshot,
        title="Validator Distribution Analysis"
    )
    plt.show()
    
    print("Network visualization complete!")


def demonstrate_attack_visualization():
    """Demonstrate attack scenario visualizations."""
    print("\n" + "="*60)
    print("ATTACK SCENARIO VISUALIZATION")
    print("="*60)
    
    # Create sample attack scenarios
    config = SimulationParameters()
    quantum_model = QuantumDevelopmentModel(config.quantum)
    attack_model = AttackScenariosModel(config.quantum)
    rng = np.random.RandomState(42)
    
    # Generate quantum timeline
    timeline = quantum_model.project_timeline(rng, 2025, 2050)
    
    # Generate attack plan
    attack_plan = attack_model.generate_attack_plan(rng, timeline)
    
    # Create attack plotter
    plotter = AttackPlotter()
    
    # Plot attack timeline
    print("\nCreating attack timeline analysis...")
    fig = plotter.plot_attack_timeline(
        [attack_plan],
        title="Quantum Attack Scenario Analysis"
    )
    plt.show()
    
    # Plot attack windows
    if attack_plan.attack_windows:
        print("\nCreating attack windows visualization...")
        fig = plot_attack_windows(
            attack_plan.attack_windows,
            title="Identified Attack Windows"
        )
        plt.show()
    
    print("Attack visualization complete!")


def demonstrate_economic_visualization():
    """Demonstrate economic impact visualizations."""
    print("\n" + "="*60)
    print("ECONOMIC IMPACT VISUALIZATION")
    print("="*60)
    
    # Create sample economic losses
    config = SimulationParameters()
    model = EconomicImpactModel(config.economic)
    rng = np.random.RandomState(42)
    
    # Generate multiple economic loss scenarios
    losses = []
    for i in range(100):
        rng_iter = np.random.RandomState(42 + i)
        
        # Create sample attack scenario
        from src.models.attack_scenarios import AttackScenario, AttackVector
        attack = AttackScenario(
            attack_type=np.random.choice(list(AttackType)),
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035 + np.random.randint(-5, 5),
            success_probability=0.5 + rng_iter.random() * 0.5,
            severity=np.random.choice(list(AttackSeverity)),
            validators_compromised=int(rng_iter.exponential(5)),
            stake_compromised=rng_iter.random() * 0.3,
            accounts_at_risk=int(rng_iter.exponential(100000)),
            time_to_execute=rng_iter.exponential(24),
            detection_probability=rng_iter.random(),
            mitigation_possible=rng_iter.random() > 0.3
        )
        
        # Create sample network snapshot
        from src.models.network_state import MigrationStatus
        network = NetworkSnapshot(
            year=attack.year,
            n_validators=1032,
            total_stake=400_000_000,
            validators=[],
            geographic_distribution={'north_america': 0.4, 'europe': 0.3, 
                                   'asia': 0.2, 'other': 0.1},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3 + rng_iter.random() * 0.4,
            superminority_count=22,
            gini_coefficient=0.8,
            network_resilience=0.5 + rng_iter.random() * 0.3
        )
        
        # Calculate economic loss
        loss = model.calculate_impact(rng_iter, attack, network)
        losses.append(loss)
    
    # Create economic plotter
    plotter = EconomicPlotter()
    
    # Plot comprehensive impact
    print("\nCreating comprehensive economic impact analysis...")
    fig = plotter.plot_economic_impact_comprehensive(
        losses,
        title="Economic Impact Analysis (100 scenarios)"
    )
    plt.show()
    
    # Plot single loss cascade
    print("\nCreating loss cascade analysis...")
    fig = plotter.plot_loss_cascade(
        losses[0],
        title="Sample Economic Loss Cascade"
    )
    plt.show()
    
    print("Economic visualization complete!")


def demonstrate_statistical_visualization():
    """Demonstrate statistical analysis visualizations."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS VISUALIZATION")
    print("="*60)
    
    # Generate sample data for statistical analysis
    np.random.seed(42)
    
    # Create sample Monte Carlo results
    n_iterations = 1000
    results = {
        'crqc_year': np.random.normal(2035, 5, n_iterations),
        'economic_loss': np.random.lognormal(22, 1.5, n_iterations),  # Log-normal losses
        'attack_success': np.random.beta(2, 5, n_iterations),  # Beta distributed probabilities
        'migration_progress': np.random.beta(3, 2, n_iterations)  # Beta distributed progress
    }
    
    # Create statistical plotter
    plotter = StatisticalPlotter()
    
    # Plot convergence
    print("\nCreating Monte Carlo convergence plot...")
    fig = plotter.plot_monte_carlo_convergence(
        results['economic_loss'],
        title="Economic Loss Convergence Analysis"
    )
    plt.show()
    
    # Plot distribution comparison
    print("\nCreating distribution comparison...")
    data_dict = {
        'CRQC Year': results['crqc_year'],
        'Economic Loss (log)': np.log10(results['economic_loss']),
        'Attack Success': results['attack_success'],
        'Migration Progress': results['migration_progress']
    }
    fig = plotter.plot_distribution_comparison(
        data_dict,
        title="Key Variable Distribution Comparison"
    )
    plt.show()
    
    # Plot sensitivity analysis
    print("\nCreating sensitivity analysis...")
    sensitivity_results = {
        'Quantum Dev Rate': 0.35,
        'Network Growth': -0.15,
        'Migration Speed': -0.42,
        'Attack Sophistication': 0.28,
        'Market Volatility': 0.22,
        'Defense Investment': -0.38,
        'Validator Count': -0.18,
        'Stake Distribution': 0.12
    }
    fig = plotter.plot_sensitivity_analysis(
        sensitivity_results,
        title="Parameter Sensitivity Analysis"
    )
    plt.show()
    
    print("Statistical visualization complete!")


def demonstrate_dashboard():
    """Demonstrate dashboard creation."""
    print("\n" + "="*60)
    print("DASHBOARD VISUALIZATION")
    print("="*60)
    
    # Create comprehensive sample results
    results = {
        'n_iterations': 1000,
        'random_seed': 42,
        'quantum_timeline': {
            'median_crqc_year': 2035,
            'crqc_probabilities': [0.01 * i**1.5 for i in range(26)],  # 2025-2050
            'years': list(range(2025, 2051))
        },
        'network_state': {
            'peak_validators': 1500,
            'final_migration': 0.75,
            'years': list(range(2025, 2051)),
            'migration_progress': [0.1 + 0.035 * i for i in range(26)]
        },
        'attack_scenarios': {
            'avg_success_rate': 0.42,
            'years': list(range(2025, 2051)),
            'success_rates': [0.1 + 0.02 * i for i in range(26)]
        },
        'economic_impact': {
            'mean_loss': 75e9,  # $75 billion
            'total_losses': np.random.lognormal(24, 1.2, 1000)
        }
    }
    
    # Create dashboard
    creator = DashboardCreator()
    
    print("\nCreating executive summary dashboard...")
    fig = creator.create_executive_summary(
        results,
        title="Solana Quantum Impact - Executive Summary"
    )
    plt.show()
    
    print("Dashboard visualization complete!")


def main():
    """Run all visualization demonstrations."""
    print("\n" + "="*70)
    print(" SOLANA QUANTUM IMPACT - VISUALIZATION DEMONSTRATION ")
    print("="*70)
    print("\nThis demo showcases the visualization capabilities")
    print("for Monte Carlo simulation results analysis.\n")
    
    # Set matplotlib backend for better display
    plt.ion()  # Interactive mode
    
    demonstrations = [
        ("Timeline Visualization", demonstrate_timeline_visualization),
        ("Network Visualization", demonstrate_network_visualization),
        ("Attack Visualization", demonstrate_attack_visualization),
        ("Economic Visualization", demonstrate_economic_visualization),
        ("Statistical Visualization", demonstrate_statistical_visualization),
        ("Dashboard Creation", demonstrate_dashboard)
    ]
    
    for i, (name, demo_func) in enumerate(demonstrations, 1):
        print(f"\n[{i}/{len(demonstrations)}] Running: {name}")
        try:
            demo_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(" ALL VISUALIZATIONS COMPLETE ")
    print("="*70)
    print("\nThe visualization engine provides comprehensive tools for:")
    print("• Quantum timeline analysis and projection")
    print("• Network evolution and migration tracking")
    print("• Attack scenario and risk assessment")
    print("• Economic impact and recovery analysis")
    print("• Statistical analysis and convergence")
    print("• Executive dashboards and reports\n")


if __name__ == "__main__":
    main()
