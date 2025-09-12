"""
Attack scenario visualization module.

This module provides visualization tools for quantum attack scenarios,
success probabilities, attack windows, and severity assessments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.models.attack_scenarios import (
    AttackPlan, AttackScenario, AttackWindow,
    AttackType, AttackVector, AttackSeverity
)


class AttackPlotter:
    """Create attack scenario visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize attack plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.attack_colors = {
            AttackType.KEY_COMPROMISE: '#3498db',     # Blue
            AttackType.DOUBLE_SPEND: '#9b59b6',       # Purple
            AttackType.CONSENSUS_HALT: '#e67e22',     # Orange
            AttackType.CONSENSUS_CONTROL: '#e74c3c',  # Red
            AttackType.TARGETED_THEFT: '#f39c12',     # Yellow
            AttackType.SYSTEMIC_FAILURE: '#c0392b'    # Dark Red
        }
        self.severity_colors = {
            AttackSeverity.LOW: '#2ecc71',       # Green
            AttackSeverity.MEDIUM: '#f39c12',    # Yellow
            AttackSeverity.HIGH: '#e67e22',      # Orange
            AttackSeverity.CRITICAL: '#e74c3c'   # Red
        }
    
    def plot_attack_timeline(
        self,
        attack_plans: List[AttackPlan],
        title: str = "Quantum Attack Timeline Analysis",
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot comprehensive attack timeline analysis.
        
        Args:
            attack_plans: List of AttackPlan instances
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract attack windows and scenarios
        all_windows = []
        all_scenarios = []
        for plan in attack_plans:
            all_windows.extend(plan.attack_windows)
            all_scenarios.extend(plan.scenarios)
        
        # 1. Attack Windows Over Time
        ax = axes[0, 0]
        if all_windows:
            years = sorted(set(w.start_year for w in all_windows))
            window_counts = []
            for year in years:
                count = sum(1 for w in all_windows 
                          if w.start_year <= year <= w.end_year)
                window_counts.append(count)
            
            ax.plot(years, window_counts, linewidth=2, color='darkred')
            ax.fill_between(years, 0, window_counts, alpha=0.3, color='darkred')
            ax.set_title("Attack Windows Over Time", fontweight='bold')
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Open Windows")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No attack windows identified", 
                   ha='center', va='center')
            ax.set_title("Attack Windows Over Time", fontweight='bold')
        
        # 2. Attack Type Distribution
        ax = axes[0, 1]
        if all_scenarios:
            attack_types = {}
            for scenario in all_scenarios:
                attack_type = scenario.attack_type
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            
            types = list(attack_types.keys())
            counts = list(attack_types.values())
            colors = [self.attack_colors[t] for t in types]
            
            ax.bar(range(len(types)), counts, color=colors, alpha=0.8)
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels([t.name for t in types], rotation=45, ha='right')
            ax.set_title("Attack Type Distribution", fontweight='bold')
            ax.set_ylabel("Number of Scenarios")
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "No attack scenarios generated", 
                   ha='center', va='center')
            ax.set_title("Attack Type Distribution", fontweight='bold')
        
        # 3. Success Probability by Year
        ax = axes[1, 0]
        if all_scenarios:
            years = sorted(set(s.year for s in all_scenarios))
            avg_probabilities = []
            for year in years:
                year_scenarios = [s for s in all_scenarios if s.year == year]
                avg_prob = np.mean([s.success_probability for s in year_scenarios])
                avg_probabilities.append(avg_prob)
            
            ax.plot(years, avg_probabilities, linewidth=2, color='purple')
            ax.fill_between(years, 0, avg_probabilities, alpha=0.3, color='purple')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5,
                      label='50% threshold')
            ax.set_title("Average Attack Success Probability", fontweight='bold')
            ax.set_xlabel("Year")
            ax.set_ylabel("Success Probability")
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No success probabilities available", 
                   ha='center', va='center')
            ax.set_title("Average Attack Success Probability", fontweight='bold')
        
        # 4. Severity Distribution Over Time
        ax = axes[1, 1]
        if all_scenarios:
            years = sorted(set(s.year for s in all_scenarios))
            severity_counts = {sev: [] for sev in AttackSeverity}
            
            for year in years:
                year_scenarios = [s for s in all_scenarios if s.year == year]
                year_counts = {sev: 0 for sev in AttackSeverity}
                for scenario in year_scenarios:
                    year_counts[scenario.severity] += 1
                
                total = len(year_scenarios)
                for sev in AttackSeverity:
                    severity_counts[sev].append(
                        year_counts[sev] / total if total > 0 else 0
                    )
            
            # Stacked area plot
            bottom = np.zeros(len(years))
            for severity in AttackSeverity:
                values = severity_counts[severity]
                ax.fill_between(years, bottom, bottom + values,
                              color=self.severity_colors[severity],
                              label=severity.name, alpha=0.8)
                bottom += values
            
            ax.set_title("Attack Severity Distribution", fontweight='bold')
            ax.set_xlabel("Year")
            ax.set_ylabel("Fraction of Attacks")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No severity data available", 
                   ha='center', va='center')
            ax.set_title("Attack Severity Distribution", fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attack_matrix(
        self,
        scenarios: List[AttackScenario],
        title: str = "Attack Scenario Matrix",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot attack scenario matrix showing relationships.
        
        Args:
            scenarios: List of AttackScenario instances
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Create matrix data
        attack_types = list(AttackType)
        vectors = list(AttackVector)
        
        # Left plot: Attack Type vs Vector heatmap
        matrix = np.zeros((len(attack_types), len(vectors)))
        for scenario in scenarios:
            i = attack_types.index(scenario.attack_type)
            j = vectors.index(scenario.vector)
            matrix[i, j] += scenario.success_probability
        
        # Normalize by count
        count_matrix = np.zeros_like(matrix)
        for scenario in scenarios:
            i = attack_types.index(scenario.attack_type)
            j = vectors.index(scenario.vector)
            count_matrix[i, j] += 1
        
        matrix = np.divide(matrix, count_matrix, 
                          out=np.zeros_like(matrix), where=count_matrix!=0)
        
        im1 = ax1.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(vectors)))
        ax1.set_xticklabels([v.name for v in vectors], rotation=45, ha='right')
        ax1.set_yticks(range(len(attack_types)))
        ax1.set_yticklabels([t.name for t in attack_types])
        ax1.set_title("Success Probability by Type & Vector")
        ax1.set_xlabel("Attack Vector")
        ax1.set_ylabel("Attack Type")
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Avg Success Probability')
        
        # Right plot: Severity vs Success Probability scatter
        severities = [scenario.severity for scenario in scenarios]
        probabilities = [scenario.success_probability for scenario in scenarios]
        colors = [self.attack_colors[scenario.attack_type] for scenario in scenarios]
        
        ax2.scatter(probabilities, severities, c=colors, alpha=0.6, s=50)
        ax2.set_xlabel("Success Probability")
        ax2.set_ylabel("Severity")
        ax2.set_title("Attack Severity vs Success Probability")
        ax2.set_yticks(range(len(AttackSeverity)))
        ax2.set_yticklabels([s.name for s in AttackSeverity])
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        
        # Add legend for attack types
        legend_elements = [mpatches.Patch(color=color, label=attack_type.name)
                          for attack_type, color in self.attack_colors.items()]
        ax2.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_attack_windows(
    windows: List[AttackWindow],
    title: str = "Quantum Attack Windows",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot attack windows as a Gantt chart.
    
    Args:
        windows: List of AttackWindow instances
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if not windows:
        ax.text(0.5, 0.5, "No attack windows identified", 
               ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig
    
    # Sort windows by threat level and start year
    windows_sorted = sorted(windows, 
                           key=lambda w: (w.threat_level.value, w.start_year))
    
    # Create Gantt chart
    threat_colors = {
        'NONE': '#2ecc71',
        'EMERGING': '#f39c12',
        'MODERATE': '#e67e22',
        'HIGH': '#e74c3c',
        'CRITICAL': '#c0392b'
    }
    
    for i, window in enumerate(windows_sorted):
        duration = window.end_year - window.start_year
        color = threat_colors.get(window.threat_level.name, 'gray')
        
        ax.barh(i, duration, left=window.start_year, height=0.8,
               color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add vulnerability score text
        mid_year = window.start_year + duration / 2
        ax.text(mid_year, i, f'{window.vulnerability_score:.2f}',
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Set labels
    ax.set_yticks(range(len(windows_sorted)))
    ax.set_yticklabels([f"{w.threat_level.name[:3]}" for w in windows_sorted])
    ax.set_xlabel("Year")
    ax.set_ylabel("Threat Level")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [mpatches.Patch(color=color, label=threat, alpha=0.7)
                      for threat, color in threat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_success_probability(
    scenarios: List[AttackScenario],
    by: str = "year",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot attack success probability analysis.
    
    Args:
        scenarios: List of AttackScenario instances
        by: Group by (year, type, severity)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if by == "year":
        years = sorted(set(s.year for s in scenarios))
        probabilities = []
        error_bars = []
        
        for year in years:
            year_probs = [s.success_probability for s in scenarios 
                         if s.year == year]
            probabilities.append(np.mean(year_probs))
            error_bars.append(np.std(year_probs))
        
        ax.errorbar(years, probabilities, yerr=error_bars, 
                   linewidth=2, capsize=5, capthick=2)
        ax.fill_between(years, 0, probabilities, alpha=0.3)
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Success Probability")
        default_title = "Attack Success Probability Over Time"
        
    elif by == "type":
        attack_types = list(set(s.attack_type for s in scenarios))
        probabilities = []
        error_bars = []
        
        for attack_type in attack_types:
            type_probs = [s.success_probability for s in scenarios 
                         if s.attack_type == attack_type]
            probabilities.append(np.mean(type_probs))
            error_bars.append(np.std(type_probs))
        
        colors = [AttackPlotter().attack_colors[t] for t in attack_types]
        ax.bar(range(len(attack_types)), probabilities, yerr=error_bars,
              color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels([t.name for t in attack_types], rotation=45, ha='right')
        
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Success Probability")
        default_title = "Success Probability by Attack Type"
        
    elif by == "severity":
        severities = list(AttackSeverity)
        probabilities = []
        error_bars = []
        
        for severity in severities:
            sev_probs = [s.success_probability for s in scenarios 
                        if s.severity == severity]
            if sev_probs:
                probabilities.append(np.mean(sev_probs))
                error_bars.append(np.std(sev_probs))
            else:
                probabilities.append(0)
                error_bars.append(0)
        
        colors = [AttackPlotter().severity_colors[s] for s in severities]
        ax.bar(range(len(severities)), probabilities, yerr=error_bars,
              color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(range(len(severities)))
        ax.set_xticklabels([s.name for s in severities])
        
        ax.set_xlabel("Attack Severity")
        ax.set_ylabel("Success Probability")
        default_title = "Success Probability by Severity"
    
    else:
        raise ValueError(f"Unknown grouping: {by}")
    
    ax.set_title(title or default_title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add 50% threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
              label='50% threshold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attack_severity_matrix(
    scenarios: List[AttackScenario],
    title: str = "Attack Severity Impact Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot attack severity impact matrix.
    
    Args:
        scenarios: List of AttackScenario instances
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create severity impact matrix
    severities = list(AttackSeverity)
    attack_types = list(AttackType)
    
    matrix = np.zeros((len(severities), len(attack_types)))
    counts = np.zeros((len(severities), len(attack_types)))
    
    for scenario in scenarios:
        i = severities.index(scenario.severity)
        j = attack_types.index(scenario.attack_type)
        # Weight by validators compromised and success probability
        impact = (scenario.validators_compromised * 
                 scenario.success_probability)
        matrix[i, j] += impact
        counts[i, j] += 1
    
    # Normalize by count
    matrix = np.divide(matrix, counts, 
                      out=np.zeros_like(matrix), where=counts!=0)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(attack_types)))
    ax.set_xticklabels([t.name for t in attack_types], rotation=45, ha='right')
    ax.set_yticks(range(len(severities)))
    ax.set_yticklabels([s.name for s in severities])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Severity Level")
    
    # Add text annotations
    for i in range(len(severities)):
        for j in range(len(attack_types)):
            if counts[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha='center', va='center', color='white',
                             fontweight='bold', fontsize=9)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Weighted Impact Score')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
