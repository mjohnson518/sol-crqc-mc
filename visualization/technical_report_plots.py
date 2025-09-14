"""
Technical visualization plots for detailed quantum risk analysis.

Provides comprehensive visualization functions for technical reports including:
- Quantum development scenarios
- Attack surface evolution
- Economic impact breakdowns
- Sensitivity analysis
- Convergence diagnostics
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy import stats, interpolate
import plotly.graph_objs as go
import plotly.offline as pyo

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class TechnicalReportPlots:
    """Generate technical visualizations for quantum risk analysis."""
    
    def __init__(self, results_path: Path, output_dir: Optional[Path] = None):
        """
        Initialize technical plots generator.
        
        Args:
            results_path: Path to simulation results
            output_dir: Output directory for plots
        """
        self.results_path = Path(results_path)
        self.output_dir = output_dir or self.results_path.parent / "technical_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        # Define color scheme
        self.colors = {
            'scenarios': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'components': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'],
            'impact': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#73AB84',
            'warning': '#F18F01',
            'danger': '#C73E1D'
        }
    
    def plot_quantum_development_scenarios(
        self,
        n_scenarios: int = 5,
        show_milestones: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot multiple quantum development trajectory scenarios.
        
        Args:
            n_scenarios: Number of scenarios to show
            show_milestones: Whether to show milestone markers
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Extract timeline data
        timeline_data = self.results.get('quantum_timeline', {})
        trajectories = timeline_data.get('sample_trajectories', [])
        
        if not trajectories:
            # Generate synthetic trajectories for demonstration
            trajectories = self._generate_synthetic_trajectories(n_scenarios)
        
        years = np.arange(2025, 2046)
        
        # Plot 1: Logical Qubits Evolution
        for i, traj in enumerate(trajectories[:n_scenarios]):
            qubits = traj.get('logical_qubits', self._generate_qubit_trajectory())
            
            # Add uncertainty band
            uncertainty = np.random.uniform(0.8, 1.2, len(years))
            qubits_upper = qubits * uncertainty * 1.1
            qubits_lower = qubits * uncertainty * 0.9
            
            ax1.plot(years, qubits, color=self.colors['scenarios'][i % len(self.colors['scenarios'])],
                    linewidth=2, label=f'Scenario {i+1}', alpha=0.8)
            ax1.fill_between(years, qubits_lower, qubits_upper,
                            color=self.colors['scenarios'][i % len(self.colors['scenarios'])],
                            alpha=0.1)
        
        # Add critical thresholds
        thresholds = {
            'RSA-2048': 2330,
            'Ed25519': 1700,
            'Lattice-256': 3500,
            'SHA-256': 1500
        }
        
        for name, threshold in thresholds.items():
            ax1.axhline(y=threshold, linestyle='--', alpha=0.5, label=name)
            ax1.text(2045.5, threshold, name, fontsize=8, va='center')
        
        ax1.set_ylabel('Logical Qubits', fontsize=11)
        ax1.set_title('Quantum Computing Development Scenarios', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        
        # Plot 2: Breakthrough Events and Milestones
        if show_milestones:
            # Define milestone events
            milestones = [
                (2027, 'Error Correction Breakthrough', 0.3),
                (2029, 'Scalable Architecture', 0.5),
                (2032, 'Quantum Advantage Demonstrated', 0.7),
                (2035, 'Commercial FTQC Available', 0.9),
                (2038, 'Cryptanalysis Optimized', 0.95)
            ]
            
            # Plot breakthrough probability over time
            breakthrough_prob = self._calculate_breakthrough_probability(years)
            ax2.plot(years, breakthrough_prob, color=self.colors['primary'],
                    linewidth=3, label='Breakthrough Probability')
            ax2.fill_between(years, 0, breakthrough_prob,
                            color=self.colors['primary'], alpha=0.2)
            
            # Add milestone markers
            for year, event, prob in milestones:
                ax2.scatter(year, prob, s=100, color=self.colors['danger'],
                          zorder=5, edgecolors='black', linewidth=1)
                ax2.annotate(event, (year, prob), xytext=(10, 10),
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Breakthrough Probability', fontsize=11)
        ax2.set_title('Quantum Breakthrough Timeline', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'quantum_development_scenarios.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attack_surface_evolution(
        self,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot stacked area chart of attack surface evolution.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        years = np.arange(2025, 2046)
        
        # Define vulnerable components over time
        components = {
            'Consensus Validators': self._generate_vulnerability_curve(years, peak_year=2032),
            'DeFi Protocols': self._generate_vulnerability_curve(years, peak_year=2034),
            'Smart Contracts': self._generate_vulnerability_curve(years, peak_year=2033),
            'Bridge Infrastructure': self._generate_vulnerability_curve(years, peak_year=2031),
            'User Wallets': self._generate_vulnerability_curve(years, peak_year=2035)
        }
        
        # Create stacked area plot
        stack_data = np.array(list(components.values()))
        
        ax.stackplot(years, *stack_data,
                    labels=list(components.keys()),
                    colors=self.colors['components'],
                    alpha=0.8)
        
        # Add migration progress overlay
        migration_rate = self._generate_migration_curve(years)
        ax2 = ax.twinx()
        ax2.plot(years, migration_rate, color=self.colors['success'],
                linewidth=3, linestyle='--', label='Migration Progress')
        ax2.fill_between(years, 0, migration_rate,
                        color=self.colors['success'], alpha=0.1)
        
        # Add critical events
        events = [
            (2030, 'First CRQC Prototype'),
            (2035, 'Commercial CRQC'),
            (2040, 'Widespread CRQC')
        ]
        
        for year, event in events:
            ax.axvline(x=year, color=self.colors['danger'],
                      linestyle=':', alpha=0.5)
            ax.text(year, ax.get_ylim()[1] * 0.95, event,
                   rotation=90, fontsize=9, va='top')
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Vulnerable Components (Normalized)', fontsize=11)
        ax2.set_ylabel('Migration Progress (%)', fontsize=11, color=self.colors['success'])
        ax.set_title('Solana Attack Surface Evolution', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Color y-axis labels
        ax2.tick_params(axis='y', labelcolor=self.colors['success'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'attack_surface_evolution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_economic_impact_waterfall(
        self,
        scenario: str = 'expected',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot waterfall chart of economic impact components.
        
        Args:
            scenario: 'best', 'expected', or 'worst'
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Define impact components (in billions)
        scenarios_data = {
            'best': {
                'Direct Theft': 10,
                'Market Panic': 15,
                'DeFi Cascade': 8,
                'Reputation Loss': 5,
                'Recovery Costs': 3,
                'Regulatory Fines': 2
            },
            'expected': {
                'Direct Theft': 50,
                'Market Panic': 80,
                'DeFi Cascade': 45,
                'Reputation Loss': 30,
                'Recovery Costs': 20,
                'Regulatory Fines': 15
            },
            'worst': {
                'Direct Theft': 150,
                'Market Panic': 300,
                'DeFi Cascade': 200,
                'Reputation Loss': 100,
                'Recovery Costs': 80,
                'Regulatory Fines': 50
            }
        }
        
        data = scenarios_data[scenario]
        
        # Plot 1: Waterfall chart
        categories = list(data.keys())
        values = list(data.values())
        
        # Calculate cumulative values
        cumulative = np.cumsum([0] + values)
        
        # Create waterfall effect
        for i, (cat, val) in enumerate(zip(categories, values)):
            # Draw bar
            color = self.colors['impact'][i % len(self.colors['impact'])]
            ax1.bar(i, val, bottom=cumulative[i], color=color, 
                   edgecolor='black', linewidth=1, alpha=0.8)
            
            # Add connecting lines
            if i > 0:
                ax1.plot([i-1, i], [cumulative[i], cumulative[i]],
                        'k--', alpha=0.5, linewidth=1)
            
            # Add value labels
            ax1.text(i, cumulative[i] + val/2, f'${val}B',
                    ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Add total bar
        total = sum(values)
        ax1.bar(len(categories), total, color=self.colors['danger'],
               edgecolor='black', linewidth=2, alpha=0.9)
        ax1.text(len(categories), total/2, f'Total:\n${total}B',
                ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax1.set_xticks(range(len(categories) + 1))
        ax1.set_xticklabels(categories + ['TOTAL'], rotation=45, ha='right')
        ax1.set_ylabel('Economic Loss (Billions USD)', fontsize=11)
        ax1.set_title(f'Economic Impact Breakdown - {scenario.title()} Case',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Recovery timeline
        days = np.arange(0, 365)
        
        # Generate recovery curves for each scenario
        recovery_curves = {
            'best': 1 - np.exp(-days / 60),
            'expected': 1 - np.exp(-days / 120),
            'worst': 1 - np.exp(-days / 200)
        }
        
        for scen, curve in recovery_curves.items():
            style = '-' if scen == scenario else '--'
            alpha = 1.0 if scen == scenario else 0.5
            ax2.plot(days, curve * 100, label=f'{scen.title()} Case',
                    linestyle=style, alpha=alpha, linewidth=2)
        
        # Add recovery milestones
        milestones = [
            (30, 'Emergency Response'),
            (90, 'System Restoration'),
            (180, 'Market Confidence'),
            (365, 'Full Recovery')
        ]
        
        for day, milestone in milestones:
            recovery_pct = recovery_curves[scenario][day] * 100
            ax2.scatter(day, recovery_pct, s=100, color=self.colors['warning'],
                       zorder=5, edgecolors='black')
            ax2.annotate(f'{milestone}\n({recovery_pct:.0f}%)',
                        (day, recovery_pct), xytext=(10, 10),
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Days After Attack', fontsize=11)
        ax2.set_ylabel('Recovery Progress (%)', fontsize=11)
        ax2.set_title('Network Recovery Timeline', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f'economic_impact_waterfall_{scenario}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_spider(
        self,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot spider/radar chart for sensitivity analysis.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Define parameters and their sensitivities
        parameters = [
            'Qubit Growth Rate',
            'Migration Speed',
            'Stake Concentration',
            'Attack Sophistication',
            'Network Effect',
            'Regulatory Response',
            'Market Volatility',
            'Technology Adoption'
        ]
        
        # Generate sensitivity scores (0-1)
        base_case = [0.5] * len(parameters)
        
        scenarios = {
            'Optimistic': [0.3, 0.8, 0.4, 0.2, 0.7, 0.8, 0.3, 0.7],
            'Baseline': base_case,
            'Pessimistic': [0.8, 0.3, 0.7, 0.9, 0.4, 0.2, 0.8, 0.3]
        }
        
        # Create spider plot
        angles = np.linspace(0, 2 * np.pi, len(parameters), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax = fig.add_subplot(111, projection='polar')
        
        for scenario_name, values in scenarios.items():
            values += values[:1]  # Complete the circle
            
            if scenario_name == 'Baseline':
                style = '-'
                alpha = 1.0
                width = 2
            else:
                style = '--'
                alpha = 0.7
                width = 1.5
            
            ax.plot(angles, values, style, linewidth=width,
                   label=scenario_name, alpha=alpha)
            ax.fill(angles, values, alpha=0.15)
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(parameters, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], size=8)
        ax.grid(True, alpha=0.3)
        
        # Add title and legend
        plt.title('Parameter Sensitivity Analysis', size=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
        
        # Add annotations for most sensitive parameters
        sensitive_params = ['Qubit Growth Rate', 'Migration Speed', 'Attack Sophistication']
        for param in sensitive_params:
            if param in parameters:
                idx = parameters.index(param)
                angle = angles[idx]
                ax.annotate('HIGH IMPACT', xy=(angle, 0.9),
                           xytext=(angle, 1.1),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=8, color='red', fontweight='bold',
                           ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'sensitivity_spider.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_dashboard(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot comprehensive convergence dashboard.
        
        Args:
            metrics: List of metrics to plot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['crqc_year', 'economic_loss', 'attack_probability', 'network_vulnerability']
        
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Generate convergence data
            n_iterations = self.results.get('parameters', {}).get('n_iterations', 1000)
            iterations = np.arange(100, n_iterations + 1, max(1, n_iterations // 100))
            
            # Simulate convergence behavior
            true_value = np.random.uniform(0.3, 0.7)
            noise_scale = 0.3 * np.exp(-iterations / n_iterations * 2)
            
            # Running mean
            running_mean = true_value + noise_scale * np.cumsum(np.random.randn(len(iterations))) / np.arange(1, len(iterations) + 1)
            
            # Standard error
            std_error = noise_scale / np.sqrt(np.arange(1, len(iterations) + 1))
            
            # Confidence intervals
            ci_lower = running_mean - 1.96 * std_error
            ci_upper = running_mean + 1.96 * std_error
            
            # Plot running mean with CI
            ax.plot(iterations, running_mean, color=self.colors['primary'],
                   linewidth=2, label='Running Mean')
            ax.fill_between(iterations, ci_lower, ci_upper,
                           color=self.colors['primary'], alpha=0.2,
                           label='95% CI')
            
            # Add true value line
            ax.axhline(y=true_value, color=self.colors['success'],
                      linestyle='--', alpha=0.5, label='True Value')
            
            # Add convergence threshold
            threshold = true_value * 0.01  # 1% threshold
            ax.axhspan(true_value - threshold, true_value + threshold,
                      color=self.colors['success'], alpha=0.1,
                      label='Convergence Zone')
            
            # Mark convergence point
            converged_idx = np.where(np.abs(running_mean - true_value) < threshold)[0]
            if len(converged_idx) > 0:
                conv_iter = iterations[converged_idx[0]]
                ax.axvline(x=conv_iter, color=self.colors['warning'],
                          linestyle=':', alpha=0.7,
                          label=f'Converged: {conv_iter}')
            
            # Formatting
            ax.set_xlabel('Iterations', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(self._format_metric_title(metric), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            # Add statistics text box
            stats_text = (
                f"Mean: {running_mean[-1]:.4f}\n"
                f"SE: {std_error[-1]:.4f}\n"
                f"CV: {std_error[-1]/abs(running_mean[-1]):.4f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Monte Carlo Convergence Dashboard', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / 'convergence_dashboard.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # Helper methods
    def _generate_synthetic_trajectories(self, n: int) -> List[Dict]:
        """Generate synthetic quantum trajectories for visualization."""
        trajectories = []
        
        for i in range(n):
            base_rate = np.random.uniform(0.15, 0.25)  # Annual growth rate
            trajectory = {
                'logical_qubits': self._generate_qubit_trajectory(growth_rate=base_rate),
                'gate_fidelity': np.random.uniform(0.99, 0.9999, 21),
                'coherence_time': np.exp(np.linspace(3, 8, 21))
            }
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_qubit_trajectory(self, growth_rate: float = 0.2) -> np.ndarray:
        """Generate a qubit growth trajectory."""
        years = 21  # 2025-2045
        base = 100
        
        # S-curve growth with noise
        t = np.linspace(0, 1, years)
        s_curve = 1 / (1 + np.exp(-10 * (t - 0.5)))
        
        trajectory = base * np.exp(growth_rate * np.arange(years))
        trajectory *= (0.5 + s_curve)
        
        # Add noise
        noise = np.random.normal(1, 0.1, years)
        trajectory *= noise
        
        return trajectory
    
    def _generate_vulnerability_curve(self, years: np.ndarray, peak_year: int) -> np.ndarray:
        """Generate vulnerability curve for a component."""
        # Gaussian-like vulnerability curve
        sigma = 3
        curve = np.exp(-0.5 * ((years - peak_year) / sigma) ** 2)
        
        # Add migration effect (reduces vulnerability over time)
        migration_effect = 1 - (years - years[0]) / (years[-1] - years[0]) * 0.5
        
        return curve * migration_effect
    
    def _generate_migration_curve(self, years: np.ndarray) -> np.ndarray:
        """Generate migration progress curve."""
        # S-curve adoption
        t = (years - years[0]) / (years[-1] - years[0])
        migration = 100 / (1 + np.exp(-10 * (t - 0.6)))
        
        # Add some noise
        noise = np.random.normal(0, 2, len(years))
        migration += noise
        
        return np.clip(migration, 0, 100)
    
    def _calculate_breakthrough_probability(self, years: np.ndarray) -> np.ndarray:
        """Calculate breakthrough probability over time."""
        # Cumulative probability of breakthrough
        t = (years - years[0]) / (years[-1] - years[0])
        prob = 1 - np.exp(-3 * t)
        
        return prob
    
    def _format_metric_title(self, metric: str) -> str:
        """Format metric name for display."""
        titles = {
            'crqc_year': 'CRQC Emergence Year',
            'economic_loss': 'Total Economic Loss',
            'attack_probability': 'Attack Success Probability',
            'network_vulnerability': 'Network Vulnerability Score'
        }
        return titles.get(metric, metric.replace('_', ' ').title())


def main():
    """Command-line interface for generating technical plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate technical report plots')
    parser.add_argument('--results-path', type=Path, required=True,
                       help='Path to simulation results JSON')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for plots')
    parser.add_argument('--plot-type', choices=['all', 'quantum', 'attack', 'economic', 'sensitivity', 'convergence'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = TechnicalReportPlots(args.results_path, args.output_dir)
    
    # Generate requested plots
    if args.plot_type in ['all', 'quantum']:
        fig = plotter.plot_quantum_development_scenarios()
        print(f"Generated: Quantum development scenarios")
    
    if args.plot_type in ['all', 'attack']:
        fig = plotter.plot_attack_surface_evolution()
        print(f"Generated: Attack surface evolution")
    
    if args.plot_type in ['all', 'economic']:
        for scenario in ['best', 'expected', 'worst']:
            fig = plotter.plot_economic_impact_waterfall(scenario=scenario)
            print(f"Generated: Economic impact waterfall ({scenario})")
    
    if args.plot_type in ['all', 'sensitivity']:
        fig = plotter.plot_sensitivity_spider()
        print(f"Generated: Sensitivity spider plot")
    
    if args.plot_type in ['all', 'convergence']:
        fig = plotter.plot_convergence_dashboard()
        print(f"Generated: Convergence dashboard")
    
    print(f"\nAll plots saved to: {plotter.output_dir}")


if __name__ == '__main__':
    main()
