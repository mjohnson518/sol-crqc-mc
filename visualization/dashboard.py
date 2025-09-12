"""
Dashboard module for comprehensive simulation results visualization.

This module provides dashboard-style visualizations that combine
multiple aspects of the simulation results into comprehensive views.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from src.core.results_collector import ResultsCollector
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.risk_assessment import RiskAssessor, RiskLevel


class DashboardCreator:
    """Create comprehensive dashboard visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize dashboard creator.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.risk_colors = {
            RiskLevel.MINIMAL: '#2ecc71',    # Green
            RiskLevel.LOW: '#3498db',         # Blue  
            RiskLevel.MODERATE: '#f39c12',    # Yellow
            RiskLevel.HIGH: '#e67e22',        # Orange
            RiskLevel.CRITICAL: '#e74c3c'     # Red
        }
    
    def create_executive_summary(
        self,
        results: Dict[str, Any],
        title: str = "Executive Summary Dashboard",
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create executive-level summary dashboard.
        
        Args:
            results: Comprehensive simulation results
            title: Dashboard title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Key Metrics Panel (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_key_metrics(ax1, results)
        
        # 2. CRQC Timeline (top middle and right)
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_crqc_timeline(ax2, results)
        
        # 3. Risk Assessment (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_risk_gauge(ax3, results)
        
        # 4. Economic Impact Distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_impact_distribution(ax4, results)
        
        # 5. Attack Success Probability (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_attack_probability(ax5, results)
        
        # 6. Network Migration Progress (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_migration_timeline(ax6, results)
        
        # 7. Scenario Comparison (bottom middle and right)
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_scenario_comparison(ax7, results)
        
        # Add metadata
        self._add_metadata(fig, results)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_key_metrics(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot key metrics panel."""
        ax.axis('off')
        ax.set_title("Key Metrics", fontweight='bold', fontsize=12)
        
        # Extract key metrics
        metrics = self._extract_key_metrics(results)
        
        # Create text display
        y_pos = 0.9
        for metric, value in metrics.items():
            ax.text(0.05, y_pos, f"{metric}:", fontweight='bold', 
                   transform=ax.transAxes, fontsize=10)
            ax.text(0.95, y_pos, str(value), ha='right',
                   transform=ax.transAxes, fontsize=10)
            y_pos -= 0.15
    
    def _plot_crqc_timeline(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot CRQC emergence timeline."""
        ax.set_title("CRQC Emergence Probability", fontweight='bold', fontsize=12)
        
        # Extract timeline data
        years = range(2025, 2051)
        if 'quantum_timeline' in results:
            probabilities = results['quantum_timeline'].get('crqc_probabilities', [])
            if probabilities:
                ax.plot(years[:len(probabilities)], probabilities, 
                       linewidth=2, color='darkred')
                ax.fill_between(years[:len(probabilities)], 0, probabilities,
                              alpha=0.3, color='darkred')
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add key threshold
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
                  label='50% threshold')
        ax.legend()
    
    def _plot_risk_gauge(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot risk assessment gauge."""
        ax.set_title("Overall Risk Level", fontweight='bold', fontsize=12)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(results)
        risk_level = self._score_to_level(risk_score)
        
        # Create semi-circular gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 1
        
        # Draw colored segments
        segments = [
            (0, 0.2, self.risk_colors[RiskLevel.MINIMAL]),
            (0.2, 0.4, self.risk_colors[RiskLevel.LOW]),
            (0.4, 0.6, self.risk_colors[RiskLevel.MODERATE]),
            (0.6, 0.8, self.risk_colors[RiskLevel.HIGH]),
            (0.8, 1.0, self.risk_colors[RiskLevel.CRITICAL])
        ]
        
        for start, end, color in segments:
            theta_seg = theta[int(start*100):int(end*100)]
            ax.fill_between(theta_seg, 0, r, color=color, alpha=0.6)
        
        # Draw needle
        needle_angle = np.pi * (1 - risk_score)
        ax.plot([0, r * np.cos(needle_angle)], 
               [0, r * np.sin(needle_angle)],
               'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Labels
        ax.text(0, -0.3, f"Risk Score: {risk_score:.2f}", 
               ha='center', fontweight='bold', fontsize=11)
        ax.text(0, -0.5, f"Level: {risk_level.name}", 
               ha='center', color=self.risk_colors[risk_level], 
               fontweight='bold', fontsize=12)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-0.6, 1.2])
        ax.axis('off')
    
    def _plot_impact_distribution(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot economic impact distribution."""
        ax.set_title("Economic Impact", fontweight='bold', fontsize=12)
        
        if 'economic_impact' in results:
            losses = results['economic_impact'].get('total_losses', [])
            if losses:
                ax.hist(losses, bins=20, color='darkred', alpha=0.7, 
                       edgecolor='black')
                
                # Add statistics
                mean_loss = np.mean(losses)
                ax.axvline(mean_loss, color='blue', linestyle='--',
                          label=f'Mean: ${mean_loss/1e9:.1f}B')
                ax.legend()
        
        ax.set_xlabel("Total Loss (USD)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    def _plot_attack_probability(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot attack success probability over time."""
        ax.set_title("Attack Success Rate", fontweight='bold', fontsize=12)
        
        if 'attack_scenarios' in results:
            years = results['attack_scenarios'].get('years', [])
            probabilities = results['attack_scenarios'].get('success_rates', [])
            
            if years and probabilities:
                ax.plot(years, probabilities, linewidth=2, color='purple')
                ax.fill_between(years, 0, probabilities, alpha=0.3, color='purple')
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Success Probability")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    def _plot_migration_timeline(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot network migration progress."""
        ax.set_title("Migration Progress", fontweight='bold', fontsize=12)
        
        if 'network_state' in results:
            years = results['network_state'].get('years', [])
            progress = results['network_state'].get('migration_progress', [])
            
            if years and progress:
                ax.plot(years, progress, linewidth=2, color='green')
                ax.fill_between(years, 0, progress, alpha=0.3, color='green')
                
                # Add threshold lines
                ax.axhline(y=0.667, color='orange', linestyle='--', 
                          alpha=0.5, label='Safe threshold')
                ax.legend()
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Migration %")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    def _plot_scenario_comparison(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot scenario comparison."""
        ax.set_title("Scenario Impact Comparison", fontweight='bold', fontsize=12)
        
        # Create sample scenario data
        scenarios = ['Baseline', 'Accelerated QC', 'Enhanced Defense', 'Worst Case']
        impacts = [100, 250, 75, 500]
        probabilities = [0.3, 0.5, 0.2, 0.7]
        
        # Create bubble chart
        for i, (scenario, impact, prob) in enumerate(zip(scenarios, impacts, probabilities)):
            ax.scatter(prob, impact, s=impact*2, alpha=0.6,
                      label=scenario)
        
        ax.set_xlabel("Probability")
        ax.set_ylabel("Economic Impact ($B)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _add_metadata(self, fig: plt.Figure, results: Dict[str, Any]):
        """Add metadata to dashboard."""
        metadata_text = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Iterations: {results.get('n_iterations', 'N/A')}\n"
            f"Random Seed: {results.get('random_seed', 'N/A')}"
        )
        fig.text(0.99, 0.01, metadata_text, ha='right', va='bottom',
                fontsize=8, color='gray', style='italic')
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Extract key metrics from results."""
        metrics = {}
        
        # CRQC emergence year
        if 'quantum_timeline' in results:
            crqc_year = results['quantum_timeline'].get('median_crqc_year', 'N/A')
            metrics['CRQC Year (Median)'] = str(crqc_year)
        
        # Economic impact
        if 'economic_impact' in results:
            mean_loss = results['economic_impact'].get('mean_loss', 0)
            metrics['Avg Economic Loss'] = f"${mean_loss/1e9:.1f}B"
        
        # Network statistics
        if 'network_state' in results:
            validators = results['network_state'].get('peak_validators', 1032)
            metrics['Peak Validators'] = str(validators)
        
        # Attack statistics
        if 'attack_scenarios' in results:
            success_rate = results['attack_scenarios'].get('avg_success_rate', 0)
            metrics['Avg Attack Success'] = f"{success_rate:.1%}"
        
        return metrics
    
    def _calculate_risk_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall risk score from results."""
        score = 0.5  # Default moderate risk
        
        # Adjust based on CRQC timeline
        if 'quantum_timeline' in results:
            crqc_year = results['quantum_timeline'].get('median_crqc_year', 2040)
            if crqc_year < 2030:
                score += 0.3
            elif crqc_year < 2035:
                score += 0.2
            elif crqc_year > 2045:
                score -= 0.2
        
        # Adjust based on economic impact
        if 'economic_impact' in results:
            mean_loss = results['economic_impact'].get('mean_loss', 0)
            if mean_loss > 100e9:  # >$100B
                score += 0.2
            elif mean_loss > 50e9:  # >$50B
                score += 0.1
        
        # Adjust based on network migration
        if 'network_state' in results:
            migration = results['network_state'].get('final_migration', 0)
            if migration > 0.8:
                score -= 0.2
            elif migration < 0.3:
                score += 0.2
        
        return np.clip(score, 0, 1)
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if score < 0.2:
            return RiskLevel.MINIMAL
        elif score < 0.4:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MODERATE
        elif score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


def create_executive_dashboard(
    results: Dict[str, Any],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create executive-level dashboard.
    
    Args:
        results: Simulation results
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    creator = DashboardCreator()
    return creator.create_executive_summary(results, save_path=save_path)


def create_technical_dashboard(
    results: Dict[str, Any],
    title: str = "Technical Analysis Dashboard",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create technical analysis dashboard.
    
    Args:
        results: Simulation results
        title: Dashboard title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Create 4x3 grid
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add various technical plots
    # (Implementation would include detailed technical visualizations)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_risk_dashboard(
    results: Dict[str, Any],
    title: str = "Risk Assessment Dashboard",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create risk-focused dashboard.
    
    Args:
        results: Simulation results
        title: Dashboard title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Create custom grid for risk visualization
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add risk-specific visualizations
    # (Implementation would include risk matrices, heat maps, etc.)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
