"""
Economic impact visualization module.

This module provides visualization tools for economic impacts,
loss distributions, recovery timelines, and market reactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.models.economic_impact import (
    EconomicLoss, MarketReaction, ImpactComponent,
    ImpactType, RecoverySpeed
)


class EconomicPlotter:
    """Create economic impact visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize economic plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.impact_colors = {
            ImpactType.DIRECT_THEFT: '#e74c3c',        # Red
            ImpactType.MARKET_PANIC: '#e67e22',        # Orange
            ImpactType.DEFI_CASCADE: '#9b59b6',        # Purple
            ImpactType.REPUTATION_DAMAGE: '#f39c12',   # Yellow
            ImpactType.MIGRATION_COSTS: '#3498db',     # Blue
            ImpactType.OPPORTUNITY_COST: '#95a5a6'     # Gray
        }
        self.recovery_colors = {
            RecoverySpeed.RAPID: '#2ecc71',     # Green
            RecoverySpeed.MODERATE: '#f39c12',   # Yellow
            RecoverySpeed.SLOW: '#e67e22',       # Orange
            RecoverySpeed.STAGNANT: '#e74c3c'    # Red
        }
    
    def plot_economic_impact_comprehensive(
        self,
        economic_losses: List[EconomicLoss],
        title: str = "Comprehensive Economic Impact Analysis",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive economic impact visualization.
        
        Args:
            economic_losses: List of EconomicLoss instances
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Total Loss Distribution
        ax = axes[0, 0]
        total_losses = [loss.total_loss_usd for loss in economic_losses]
        
        ax.hist(total_losses, bins=30, edgecolor='black', alpha=0.7, color='darkred')
        ax.axvline(np.mean(total_losses), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean: ${np.mean(total_losses)/1e9:.1f}B')
        ax.axvline(np.median(total_losses), color='green', linestyle='--',
                  linewidth=2, label=f'Median: ${np.median(total_losses)/1e9:.1f}B')
        
        ax.set_title("Total Loss Distribution", fontweight='bold')
        ax.set_xlabel("Total Loss (USD)")
        ax.set_ylabel("Frequency")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss Components Breakdown
        ax = axes[0, 1]
        component_totals = {impact_type: [] for impact_type in ImpactType}
        
        for loss in economic_losses:
            for component in loss.components:
                component_totals[component.impact_type].append(component.amount_usd)
        
        # Calculate averages
        component_avgs = {
            impact_type: np.mean(amounts) if amounts else 0
            for impact_type, amounts in component_totals.items()
        }
        
        types = list(component_avgs.keys())
        values = list(component_avgs.values())
        colors = [self.impact_colors[t] for t in types]
        
        ax.pie(values, labels=[t.name for t in types], colors=colors,
              autopct='%1.1f%%', startangle=90)
        ax.set_title("Average Loss Components", fontweight='bold')
        
        # 3. Recovery Timeline Distribution
        ax = axes[1, 0]
        recovery_times = [loss.recovery_timeline_days for loss in economic_losses]
        
        ax.hist(recovery_times, bins=30, edgecolor='black', alpha=0.7, color='blue')
        ax.axvline(np.mean(recovery_times), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(recovery_times):.0f} days')
        
        ax.set_title("Recovery Timeline Distribution", fontweight='bold')
        ax.set_xlabel("Recovery Time (Days)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Market Impact Analysis
        ax = axes[1, 1]
        price_drops = [loss.market_reaction.price_drop_percent 
                      for loss in economic_losses]
        tvl_drops = [loss.market_reaction.tvl_drop_percent 
                    for loss in economic_losses]
        
        ax.scatter(price_drops, tvl_drops, alpha=0.5, s=50, c='purple')
        ax.set_title("Market Reaction Correlation", fontweight='bold')
        ax.set_xlabel("Price Drop (%)")
        ax.set_ylabel("TVL Drop (%)")
        ax.grid(True, alpha=0.3)
        
        # Add correlation line
        if len(price_drops) > 1:
            z = np.polyfit(price_drops, tvl_drops, 1)
            p = np.poly1d(z)
            ax.plot(sorted(price_drops), p(sorted(price_drops)), 
                   "r--", alpha=0.5, label=f'Correlation')
            ax.legend()
        
        # 5. Loss vs Attack Year
        ax = axes[2, 0]
        years = [loss.attack_year for loss in economic_losses]
        losses = [loss.total_loss_usd for loss in economic_losses]
        
        # Group by year and calculate statistics
        year_stats = {}
        for year, loss in zip(years, losses):
            if year not in year_stats:
                year_stats[year] = []
            year_stats[year].append(loss)
        
        sorted_years = sorted(year_stats.keys())
        means = [np.mean(year_stats[y]) for y in sorted_years]
        stds = [np.std(year_stats[y]) for y in sorted_years]
        
        ax.errorbar(sorted_years, means, yerr=stds, linewidth=2, 
                   capsize=5, capthick=2, color='darkgreen')
        ax.fill_between(sorted_years, 0, means, alpha=0.3, color='darkgreen')
        
        ax.set_title("Economic Loss by Attack Year", fontweight='bold')
        ax.set_xlabel("Attack Year")
        ax.set_ylabel("Average Total Loss (USD)")
        ax.grid(True, alpha=0.3)
        
        # 6. Recovery Speed Distribution
        ax = axes[2, 1]
        recovery_speeds = [loss.recovery_timeline_days / loss.total_loss_usd * 1e9
                          for loss in economic_losses if loss.total_loss_usd > 0]
        
        ax.hist(recovery_speeds, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax.set_title("Recovery Efficiency", fontweight='bold')
        ax.set_xlabel("Days per Billion USD Lost")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_loss_cascade(
        self,
        economic_loss: EconomicLoss,
        title: str = "Economic Loss Cascade Analysis",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize the cascade of economic impacts.
        
        Args:
            economic_loss: EconomicLoss instance
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f"{title} - Total Loss: ${economic_loss.total_loss_usd/1e9:.2f}B",
                    fontsize=14, fontweight='bold')
        
        # Top plot: Waterfall chart of loss components
        components = sorted(economic_loss.components, 
                          key=lambda c: c.amount_usd, reverse=True)
        
        cumulative = 0
        for i, component in enumerate(components):
            color = self.impact_colors[component.impact_type]
            ax1.bar(i, component.amount_usd, bottom=cumulative, 
                   color=color, alpha=0.8, edgecolor='black')
            
            # Add value label
            mid_point = cumulative + component.amount_usd / 2
            ax1.text(i, mid_point, f'${component.amount_usd/1e9:.1f}B',
                    ha='center', va='center', fontweight='bold', fontsize=9)
            
            cumulative += component.amount_usd
        
        ax1.set_xticks(range(len(components)))
        ax1.set_xticklabels([c.impact_type.name for c in components],
                           rotation=45, ha='right')
        ax1.set_title("Loss Component Cascade", fontweight='bold')
        ax1.set_ylabel("Loss Amount (USD)")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom plot: Recovery timeline
        recovery_days = economic_loss.recovery_timeline_days
        recovery_phases = [
            ("Immediate Response", 0, min(7, recovery_days), 'red'),
            ("Short-term Recovery", 7, min(30, recovery_days), 'orange'),
            ("Medium-term Recovery", 30, min(180, recovery_days), 'yellow'),
            ("Long-term Recovery", 180, recovery_days, 'green')
        ]
        
        for phase_name, start, end, color in recovery_phases:
            if end > start:
                ax2.barh(0, end - start, left=start, height=0.8,
                        color=color, alpha=0.6, edgecolor='black')
                mid = start + (end - start) / 2
                ax2.text(mid, 0, phase_name, ha='center', va='center',
                        fontweight='bold', fontsize=10)
        
        ax2.set_xlim([0, recovery_days * 1.1])
        ax2.set_ylim([-0.5, 0.5])
        ax2.set_xlabel("Days")
        ax2.set_title(f"Recovery Timeline ({recovery_days:.0f} days total)",
                     fontweight='bold')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_economic_impact(
    losses: List[float],
    title: str = "Economic Impact Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot economic impact distribution with statistics.
    
    Args:
        losses: List of loss amounts
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Left plot: Histogram with statistics
    ax1.hist(losses, bins=30, edgecolor='black', alpha=0.7, color='darkred')
    
    # Add statistical lines
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    p95_loss = np.percentile(losses, 95)
    
    ax1.axvline(mean_loss, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: ${mean_loss/1e9:.1f}B')
    ax1.axvline(median_loss, color='green', linestyle='--', linewidth=2,
               label=f'Median: ${median_loss/1e9:.1f}B')
    ax1.axvline(p95_loss, color='orange', linestyle='--', linewidth=2,
               label=f'95th %ile: ${p95_loss/1e9:.1f}B')
    
    ax1.set_xlabel("Economic Loss (USD)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Loss Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Cumulative distribution
    sorted_losses = np.sort(losses)
    cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    
    ax2.plot(sorted_losses, cumulative, linewidth=2, color='darkblue')
    ax2.fill_between(sorted_losses, 0, cumulative, alpha=0.3, color='darkblue')
    
    # Mark key percentiles
    for percentile, color, label in [(50, 'green', 'Median'),
                                     (90, 'orange', '90th %ile'),
                                     (99, 'red', '99th %ile')]:
        value = np.percentile(losses, percentile)
        ax2.axvline(value, color=color, linestyle=':', alpha=0.5)
        ax2.axhline(percentile/100, color=color, linestyle=':', alpha=0.5)
        ax2.text(value, percentile/100 + 0.02, 
                f'{label}: ${value/1e9:.1f}B',
                fontsize=9, color=color)
    
    ax2.set_xlabel("Economic Loss (USD)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_recovery_timeline(
    recovery_data: Dict[str, List[float]],
    title: str = "Economic Recovery Timeline",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot recovery timeline analysis.
    
    Args:
        recovery_data: Dictionary mapping scenarios to recovery times
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Left plot: Box plot comparison
    scenarios = list(recovery_data.keys())
    data = [recovery_data[s] for s in scenarios]
    
    bp = ax1.boxplot(data, labels=scenarios, patch_artist=True)
    
    # Color boxes by severity
    colors = ['green', 'yellow', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors[:len(scenarios)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Recovery Time (Days)")
    ax1.set_title("Recovery Time by Scenario")
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Right plot: Recovery curves
    max_days = max(max(times) for times in recovery_data.values())
    days = np.linspace(0, max_days, 100)
    
    for scenario, times in recovery_data.items():
        # Create synthetic recovery curve
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Logistic recovery function
        recovery = 1 / (1 + np.exp(-10 * (days - mean_time) / mean_time))
        ax2.plot(days, recovery, linewidth=2, label=scenario)
    
    ax2.set_xlabel("Days After Attack")
    ax2.set_ylabel("Recovery Progress")
    ax2.set_title("Recovery Trajectories")
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_distribution(
    losses: List[EconomicLoss],
    group_by: str = "attack_type",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot loss distribution grouped by various factors.
    
    Args:
        losses: List of EconomicLoss instances
        group_by: Grouping factor (attack_type, year, severity)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if group_by == "attack_type":
        # Group losses by attack type
        grouped = {}
        for loss in losses:
            attack_type = loss.attack_scenario.attack_type if hasattr(loss, 'attack_scenario') else 'Unknown'
            if attack_type not in grouped:
                grouped[attack_type] = []
            grouped[attack_type].append(loss.total_loss_usd)
        
        # Create violin plot
        data = list(grouped.values())
        labels = list(grouped.keys())
        
        parts = ax.violinplot(data, positions=range(len(labels)),
                             showmeans=True, showmedians=True)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel("Attack Type")
        default_title = "Loss Distribution by Attack Type"
        
    elif group_by == "year":
        # Group losses by year
        years = [loss.attack_year for loss in losses]
        total_losses = [loss.total_loss_usd for loss in losses]
        
        # Create scatter plot with trend line
        ax.scatter(years, total_losses, alpha=0.5, s=50, c='darkred')
        
        # Add trend line
        z = np.polyfit(years, total_losses, 1)
        p = np.poly1d(z)
        ax.plot(sorted(set(years)), p(sorted(set(years))), 
               "r--", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_xlabel("Attack Year")
        ax.legend()
        default_title = "Loss Distribution Over Time"
        
    elif group_by == "severity":
        # Group losses by severity level
        severity_losses = {}
        for loss in losses:
            # Categorize by loss magnitude
            if loss.total_loss_usd < 1e9:
                severity = "Low (<$1B)"
            elif loss.total_loss_usd < 10e9:
                severity = "Medium ($1-10B)"
            elif loss.total_loss_usd < 50e9:
                severity = "High ($10-50B)"
            else:
                severity = "Critical (>$50B)"
            
            if severity not in severity_losses:
                severity_losses[severity] = []
            severity_losses[severity].append(loss.total_loss_usd)
        
        # Create bar plot with error bars
        severities = ["Low (<$1B)", "Medium ($1-10B)", 
                     "High ($10-50B)", "Critical (>$50B)"]
        means = []
        stds = []
        
        for severity in severities:
            if severity in severity_losses:
                means.append(np.mean(severity_losses[severity]))
                stds.append(np.std(severity_losses[severity]))
            else:
                means.append(0)
                stds.append(0)
        
        colors = ['green', 'yellow', 'orange', 'red']
        ax.bar(range(len(severities)), means, yerr=stds,
              color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(range(len(severities)))
        ax.set_xticklabels(severities, rotation=45, ha='right')
        ax.set_xlabel("Loss Severity")
        default_title = "Average Loss by Severity"
    
    else:
        raise ValueError(f"Unknown grouping: {group_by}")
    
    ax.set_ylabel("Economic Loss (USD)")
    ax.set_title(title or default_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
