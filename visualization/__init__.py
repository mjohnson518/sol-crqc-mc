"""
Solana Quantum Impact Monte Carlo Simulation - Visualization Package

This package provides comprehensive visualization tools for displaying
simulation results, including quantum timelines, network evolution,
attack scenarios, economic impacts, and statistical analyses.

Key Components:
- timeline_plots: Quantum development timeline visualizations
- network_plots: Network state and evolution visualizations  
- attack_plots: Attack scenario and success probability visualizations
- economic_plots: Economic impact and recovery visualizations
- statistical_plots: Statistical distribution and analysis visualizations
- dashboard: Comprehensive dashboard views

Author: Marc Johnson
Date: 2025
"""

from .timeline_plots import (
    TimelinePlotter,
    plot_quantum_timeline,
    plot_threat_evolution,
    plot_capability_progression
)

from .network_plots import (
    NetworkPlotter,
    plot_network_evolution,
    plot_validator_distribution,
    plot_migration_progress
)

from .attack_plots import (
    AttackPlotter,
    plot_attack_windows,
    plot_success_probability,
    plot_attack_severity_matrix
)

from .economic_plots import (
    EconomicPlotter,
    plot_economic_impact,
    plot_recovery_timeline,
    plot_loss_distribution
)

from .statistical_plots import (
    StatisticalPlotter,
    plot_distributions,
    plot_correlation_matrix,
    plot_confidence_intervals
)

from .dashboard import (
    DashboardCreator,
    create_executive_dashboard,
    create_technical_dashboard,
    create_risk_dashboard
)

__all__ = [
    # Timeline plots
    'TimelinePlotter',
    'plot_quantum_timeline',
    'plot_threat_evolution',
    'plot_capability_progression',
    
    # Network plots
    'NetworkPlotter',
    'plot_network_evolution',
    'plot_validator_distribution',
    'plot_migration_progress',
    
    # Attack plots
    'AttackPlotter',
    'plot_attack_windows',
    'plot_success_probability',
    'plot_attack_severity_matrix',
    
    # Economic plots
    'EconomicPlotter',
    'plot_economic_impact',
    'plot_recovery_timeline',
    'plot_loss_distribution',
    
    # Statistical plots
    'StatisticalPlotter',
    'plot_distributions',
    'plot_correlation_matrix',
    'plot_confidence_intervals',
    
    # Dashboard
    'DashboardCreator',
    'create_executive_dashboard',
    'create_technical_dashboard',
    'create_risk_dashboard'
]
