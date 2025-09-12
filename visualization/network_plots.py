"""
Network visualization module for Solana network evolution.

This module provides visualization tools for network state dynamics,
validator distribution, stake concentration, and migration progress.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.models.network_state import (
    NetworkEvolution, NetworkSnapshot, ValidatorState, 
    MigrationStatus, ValidatorTier
)


class NetworkPlotter:
    """Create network state and evolution visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize network plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.tier_colors = {
            ValidatorTier.SUPERMINORITY: '#e74c3c',  # Red - highest stake
            ValidatorTier.CRITICAL: '#e67e22',       # Orange  
            ValidatorTier.SIGNIFICANT: '#f39c12',    # Yellow
            ValidatorTier.STANDARD: '#3498db',       # Blue
            ValidatorTier.SMALL: '#95a5a6'          # Gray
        }
        self.migration_colors = {
            MigrationStatus.VULNERABLE: '#e74c3c',    # Red
            MigrationStatus.IN_PROGRESS: '#f39c12',   # Yellow
            MigrationStatus.PROTECTED: '#2ecc71'      # Green
        }
    
    def plot_network_evolution_comprehensive(
        self,
        evolution: NetworkEvolution,
        title: str = "Solana Network Evolution",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive network evolution visualization.
        
        Args:
            evolution: NetworkEvolution instance
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        years = sorted(evolution.snapshots.keys())
        snapshots = [evolution.snapshots[year] for year in years]
        
        # 1. Validator Count Evolution
        ax = axes[0, 0]
        validator_counts = [s.n_validators for s in snapshots]
        superminority_counts = [s.superminority_count for s in snapshots]
        
        ax.plot(years, validator_counts, 'b-', linewidth=2, label='Total Validators')
        ax.plot(years, superminority_counts, 'r--', linewidth=2, 
                label='Superminority')
        ax.fill_between(years, 0, superminority_counts, alpha=0.3, color='red')
        
        ax.set_title("Validator Network Growth", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Validators")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Stake Distribution (Gini Coefficient)
        ax = axes[0, 1]
        gini_coefficients = [s.gini_coefficient for s in snapshots]
        
        ax.plot(years, gini_coefficients, 'purple', linewidth=2)
        ax.fill_between(years, 0, gini_coefficients, alpha=0.3, color='purple')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, 
                  label='High Concentration')
        
        ax.set_title("Stake Concentration (Gini Coefficient)", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Gini Coefficient")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Geographic Distribution
        ax = axes[1, 0]
        regions = list(snapshots[0].geographic_distribution.keys())
        region_data = {region: [] for region in regions}
        
        for snapshot in snapshots:
            for region in regions:
                region_data[region].append(
                    snapshot.geographic_distribution.get(region, 0)
                )
        
        bottom = np.zeros(len(years))
        for region, values in region_data.items():
            ax.bar(years, values, bottom=bottom, label=region, alpha=0.8)
            bottom += values
        
        ax.set_title("Geographic Distribution", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Fraction of Validators")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Migration Progress
        ax = axes[1, 1]
        migration_progress = [s.migration_progress for s in snapshots]
        
        ax.plot(years, migration_progress, 'green', linewidth=2)
        ax.fill_between(years, 0, migration_progress, alpha=0.3, color='green')
        ax.axhline(y=0.667, color='orange', linestyle='--', alpha=0.5,
                  label='Consensus Threshold')
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5,
                  label='Target Coverage')
        
        ax.set_title("Quantum-Safe Migration Progress", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Migration Progress")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Network Resilience Score
        ax = axes[2, 0]
        resilience_scores = [s.network_resilience for s in snapshots]
        
        ax.plot(years, resilience_scores, 'darkblue', linewidth=2)
        ax.fill_between(years, 0, resilience_scores, alpha=0.3, color='darkblue')
        
        # Add color bands for resilience levels
        ax.axhspan(0, 0.3, alpha=0.2, color='red', label='Critical')
        ax.axhspan(0.3, 0.6, alpha=0.2, color='orange', label='Vulnerable')
        ax.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Moderate')
        ax.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Strong')
        
        ax.set_title("Network Resilience Score", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Resilience Score")
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 6. Attack Surface Analysis
        ax = axes[2, 1]
        vulnerable_validators = []
        protected_validators = []
        
        for snapshot in snapshots:
            vulnerable = sum(1 for v in snapshot.validators if not v.quantum_safe)
            protected = sum(1 for v in snapshot.validators if v.quantum_safe)
            vulnerable_validators.append(vulnerable)
            protected_validators.append(protected)
        
        ax.plot(years, vulnerable_validators, 'r-', linewidth=2, 
                label='Vulnerable')
        ax.plot(years, protected_validators, 'g-', linewidth=2, 
                label='Protected')
        ax.fill_between(years, 0, vulnerable_validators, alpha=0.3, color='red')
        ax.fill_between(years, 0, protected_validators, alpha=0.3, color='green')
        
        ax.set_title("Validator Protection Status", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Validators")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_validator_tier_distribution(
        self,
        snapshot: NetworkSnapshot,
        title: str = "Validator Tier Distribution",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot validator tier distribution with stake weights.
        
        Args:
            snapshot: NetworkSnapshot instance
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"{title} - Year {snapshot.year}", fontsize=14, fontweight='bold')
        
        # Count validators by tier
        tier_counts = {tier: 0 for tier in ValidatorTier}
        tier_stakes = {tier: 0.0 for tier in ValidatorTier}
        
        for validator in snapshot.validators:
            tier_counts[validator.tier] += 1
            tier_stakes[validator.tier] += validator.stake
        
        # Left plot: Validator count by tier
        tiers = list(ValidatorTier)
        counts = [tier_counts[tier] for tier in tiers]
        colors = [self.tier_colors[tier] for tier in tiers]
        
        ax1.bar(range(len(tiers)), counts, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(tiers)))
        ax1.set_xticklabels([tier.name for tier in tiers], rotation=45)
        ax1.set_title("Validators by Tier")
        ax1.set_ylabel("Number of Validators")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right plot: Stake distribution by tier (pie chart)
        stakes = [tier_stakes[tier] for tier in tiers]
        explode = [0.1 if tier == ValidatorTier.SUPERMINORITY else 0 for tier in tiers]
        
        ax2.pie(stakes, labels=[tier.name for tier in tiers], 
                colors=colors, autopct='%1.1f%%', explode=explode,
                startangle=90)
        ax2.set_title("Stake Distribution by Tier")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_network_evolution(
    evolution: NetworkEvolution,
    metric: str = "validators",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot specific network metric evolution over time.
    
    Args:
        evolution: NetworkEvolution instance
        metric: Metric to plot (validators, stake, resilience, migration)
        title: Plot title (auto-generated if None)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    years = sorted(evolution.snapshots.keys())
    snapshots = [evolution.snapshots[year] for year in years]
    
    if metric == "validators":
        values = [s.n_validators for s in snapshots]
        ylabel = "Number of Validators"
        default_title = "Validator Network Growth"
        color = 'blue'
    elif metric == "stake":
        values = [s.total_stake for s in snapshots]
        ylabel = "Total Stake (SOL)"
        default_title = "Total Network Stake Evolution"
        color = 'green'
    elif metric == "resilience":
        values = [s.network_resilience for s in snapshots]
        ylabel = "Resilience Score"
        default_title = "Network Resilience Over Time"
        color = 'purple'
    elif metric == "migration":
        values = [s.migration_progress for s in snapshots]
        ylabel = "Migration Progress"
        default_title = "Quantum-Safe Migration Progress"
        color = 'orange'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    ax.plot(years, values, linewidth=2, color=color)
    ax.fill_between(years, 0, values, alpha=0.3, color=color)
    
    ax.set_title(title or default_title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_validator_distribution(
    validators: List[ValidatorState],
    title: str = "Validator Stake Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot validator stake distribution analysis.
    
    Args:
        validators: List of ValidatorState instances
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Sort validators by stake
    validators_sorted = sorted(validators, key=lambda v: v.stake, reverse=True)
    stakes = [v.stake for v in validators_sorted]
    
    # Left plot: Stake distribution (log scale)
    ax1.bar(range(len(stakes[:100])), stakes[:100], color='blue', alpha=0.6)
    ax1.set_yscale('log')
    ax1.set_title("Top 100 Validators by Stake")
    ax1.set_xlabel("Validator Rank")
    ax1.set_ylabel("Stake (SOL, log scale)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Middle plot: Cumulative stake distribution
    cumulative_stake = np.cumsum(stakes) / sum(stakes)
    ax2.plot(range(len(cumulative_stake)), cumulative_stake, linewidth=2)
    ax2.axhline(y=0.333, color='orange', linestyle='--', alpha=0.5, 
                label='Halt Threshold (33.3%)')
    ax2.axhline(y=0.667, color='red', linestyle='--', alpha=0.5,
                label='Control Threshold (66.7%)')
    
    # Find superminority size
    superminority_idx = next(i for i, cs in enumerate(cumulative_stake) if cs > 0.333)
    ax2.axvline(x=superminority_idx, color='green', linestyle=':', alpha=0.5,
                label=f'Superminority: {superminority_idx+1}')
    
    ax2.set_title("Cumulative Stake Distribution")
    ax2.set_xlabel("Number of Validators")
    ax2.set_ylabel("Cumulative Stake Fraction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Right plot: Geographic distribution (if available)
    regions = {}
    for v in validators:
        region = v.geographic_region
        regions[region] = regions.get(region, 0) + 1
    
    if regions:
        ax3.pie(regions.values(), labels=regions.keys(), autopct='%1.1f%%',
                startangle=90)
        ax3.set_title("Geographic Distribution")
    else:
        ax3.text(0.5, 0.5, "No geographic data", ha='center', va='center')
        ax3.set_title("Geographic Distribution")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_migration_progress(
    migration_data: Dict[int, float],
    title: str = "Quantum-Safe Migration Progress",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot migration progress to quantum-safe cryptography.
    
    Args:
        migration_data: Dictionary mapping years to migration progress
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    years = sorted(migration_data.keys())
    progress = [migration_data[year] for year in years]
    
    # Create gradient fill based on migration status
    for i in range(len(years) - 1):
        y = progress[i]
        if y < 0.333:
            color = 'red'
            alpha = 0.3
        elif y < 0.667:
            color = 'orange'
            alpha = 0.4
        else:
            color = 'green'
            alpha = 0.5
        
        ax.fill_between([years[i], years[i+1]], 0, [y, progress[i+1]], 
                       color=color, alpha=alpha)
    
    ax.plot(years, progress, 'black', linewidth=2)
    
    # Add threshold lines
    ax.axhline(y=0.333, color='orange', linestyle='--', alpha=0.5,
              label='Minimum Safe (33.3%)')
    ax.axhline(y=0.667, color='blue', linestyle='--', alpha=0.5,
              label='Consensus Safe (66.7%)')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5,
              label='Target (95%)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Migration Progress")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
