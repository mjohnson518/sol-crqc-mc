"""
Timeline visualization module for quantum development.

This module provides visualization tools for quantum computing timelines,
including capability progression, threat evolution, and CRQC emergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from src.models.quantum_timeline import QuantumTimeline, QuantumThreat, QuantumCapability


class TimelinePlotter:
    """Create timeline visualizations for quantum development."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize timeline plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'NONE': '#2ecc71',      # Green - safe
            'EMERGING': '#f39c12',   # Orange - warning
            'MODERATE': '#e67e22',   # Dark orange
            'HIGH': '#e74c3c',       # Red - danger
            'CRITICAL': '#c0392b'    # Dark red - critical
        }
    
    def plot_timeline_ensemble(
        self,
        timelines: List[QuantumTimeline],
        title: str = "Quantum Computing Timeline Ensemble",
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ensemble of quantum timelines showing uncertainty.
        
        Args:
            timelines: List of quantum timeline instances
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data from timelines
        years = sorted(set(
            year for timeline in timelines
            for year in timeline.capabilities.keys()
        ))
        
        # 1. Logical Qubits Progression
        ax = axes[0, 0]
        logical_qubits = []
        for timeline in timelines:
            qubits = [
                timeline.capabilities.get(year, QuantumCapability()).logical_qubits
                for year in years
            ]
            logical_qubits.append(qubits)
        
        logical_qubits = np.array(logical_qubits)
        self._plot_uncertainty_band(ax, years, logical_qubits)
        ax.set_title("Logical Qubits Progression", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Logical Qubits")
        ax.axhline(y=2330, color='red', linestyle='--', alpha=0.5, label='Ed25519 threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gate Fidelity Evolution
        ax = axes[0, 1]
        gate_fidelities = []
        for timeline in timelines:
            fidelities = [
                timeline.capabilities.get(year, QuantumCapability()).gate_fidelity
                for year in years
            ]
            gate_fidelities.append(fidelities)
        
        gate_fidelities = np.array(gate_fidelities)
        self._plot_uncertainty_band(ax, years, gate_fidelities)
        ax.set_title("Gate Fidelity Evolution", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Gate Fidelity")
        ax.axhline(y=0.999, color='red', linestyle='--', alpha=0.5, label='FTQC threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Threat Level Distribution
        ax = axes[1, 0]
        threat_counts = {threat: [] for threat in QuantumThreat}
        for year in years:
            year_threats = [
                timeline.threat_assessments.get(year, QuantumThreat.NONE)
                for timeline in timelines
            ]
            for threat in QuantumThreat:
                count = sum(1 for t in year_threats if t == threat)
                threat_counts[threat].append(count / len(timelines))
        
        # Stack area plot for threat evolution
        bottom = np.zeros(len(years))
        for threat in QuantumThreat:
            ax.fill_between(
                years, bottom, bottom + threat_counts[threat],
                color=self.colors[threat.name],
                label=threat.name,
                alpha=0.8
            )
            bottom += threat_counts[threat]
        
        ax.set_title("Threat Level Distribution", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Probability")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. CRQC Emergence Probability
        ax = axes[1, 1]
        crqc_prob = []
        for year in years:
            crqc_count = sum(
                1 for timeline in timelines
                if timeline.crqc_year and timeline.crqc_year <= year
            )
            crqc_prob.append(crqc_count / len(timelines))
        
        ax.plot(years, crqc_prob, linewidth=2, color='darkred')
        ax.fill_between(years, 0, crqc_prob, alpha=0.3, color='darkred')
        ax.set_title("CRQC Emergence Probability", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Probability")
        ax.grid(True, alpha=0.3)
        
        # Add 50% line
        if any(p >= 0.5 for p in crqc_prob):
            year_50 = years[next(i for i, p in enumerate(crqc_prob) if p >= 0.5)]
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(x=year_50, color='orange', linestyle='--', alpha=0.5)
            ax.text(year_50, 0.52, f'{year_50}', fontsize=10, color='orange')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_uncertainty_band(
        self,
        ax: plt.Axes,
        x: List[int],
        y_data: np.ndarray,
        confidence: float = 0.9
    ):
        """
        Plot median with confidence bands.
        
        Args:
            ax: Matplotlib axes
            x: X-axis values (years)
            y_data: 2D array of y values (simulations x years)
            confidence: Confidence level for bands
        """
        median = np.median(y_data, axis=0)
        lower = np.percentile(y_data, (1 - confidence) * 50, axis=0)
        upper = np.percentile(y_data, 100 - (1 - confidence) * 50, axis=0)
        
        ax.plot(x, median, linewidth=2, color='blue', label='Median')
        ax.fill_between(x, lower, upper, alpha=0.3, color='blue', 
                        label=f'{int(confidence*100)}% CI')


def plot_quantum_timeline(
    timeline: QuantumTimeline,
    title: str = "Quantum Development Timeline",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a comprehensive quantum timeline visualization.
    
    Args:
        timeline: QuantumTimeline instance
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    years = sorted(timeline.capabilities.keys())
    
    # Top plot: Quantum capabilities
    capabilities = [timeline.capabilities[year] for year in years]
    logical_qubits = [c.logical_qubits for c in capabilities]
    physical_qubits = [c.physical_qubits for c in capabilities]
    
    ax1.plot(years, logical_qubits, 'b-', linewidth=2, label='Logical Qubits')
    ax1.plot(years, physical_qubits, 'g--', linewidth=1, alpha=0.7, label='Physical Qubits')
    ax1.axhline(y=2330, color='red', linestyle='--', alpha=0.5, label='Ed25519 Threshold')
    
    ax1.set_ylabel('Number of Qubits')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Quantum Computing Capabilities', fontsize=12)
    
    # Bottom plot: Threat assessment
    threats = [timeline.threat_assessments.get(year, QuantumThreat.NONE) for year in years]
    threat_values = [list(QuantumThreat).index(t) for t in threats]
    colors = [TimelinePlotter().colors[t.name] for t in threats]
    
    ax2.scatter(years, threat_values, c=colors, s=100, alpha=0.8)
    ax2.plot(years, threat_values, 'k-', alpha=0.3, linewidth=1)
    
    # Set threat level labels
    ax2.set_yticks(range(len(QuantumThreat)))
    ax2.set_yticklabels([t.name for t in QuantumThreat])
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Threat Level')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Quantum Threat Assessment', fontsize=12)
    
    # Mark CRQC year if available
    if timeline.crqc_year:
        for ax in [ax1, ax2]:
            ax.axvline(x=timeline.crqc_year, color='red', linestyle=':', 
                      alpha=0.5, label=f'CRQC: {timeline.crqc_year}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_threat_evolution(
    threat_data: Dict[int, Dict[str, float]],
    title: str = "Quantum Threat Evolution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot evolution of threat probabilities over time.
    
    Args:
        threat_data: Dictionary mapping years to threat probabilities
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    years = sorted(threat_data.keys())
    threat_types = list(next(iter(threat_data.values())).keys())
    
    # Create stacked area plot
    bottom = np.zeros(len(years))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(threat_types)))
    
    for i, threat in enumerate(threat_types):
        values = [threat_data[year].get(threat, 0) for year in years]
        ax.fill_between(years, bottom, bottom + values, 
                       color=colors[i], label=threat, alpha=0.8)
        bottom += values
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_capability_progression(
    capabilities: Dict[int, QuantumCapability],
    metrics: List[str] = ['logical_qubits', 'gate_fidelity', 'coherence_time_ms'],
    title: str = "Quantum Capability Progression",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple quantum capability metrics over time.
    
    Args:
        capabilities: Dictionary mapping years to QuantumCapability
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    years = sorted(capabilities.keys())
    
    metric_info = {
        'logical_qubits': ('Logical Qubits', 'log', 2330, 'Ed25519 threshold'),
        'physical_qubits': ('Physical Qubits', 'log', None, None),
        'gate_fidelity': ('Gate Fidelity', 'linear', 0.999, 'FTQC threshold'),
        'coherence_time_ms': ('Coherence Time (ms)', 'log', None, None),
        'error_rate': ('Error Rate', 'log', 1e-9, 'Target error rate')
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [getattr(capabilities[year], metric, 0) for year in years]
        
        ax.plot(years, values, linewidth=2, color='blue')
        ax.fill_between(years, 0, values, alpha=0.3, color='blue')
        
        # Add threshold line if applicable
        ylabel, yscale, threshold, threshold_label = metric_info.get(
            metric, (metric.replace('_', ' ').title(), 'linear', None, None)
        )
        
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      alpha=0.5, label=threshold_label)
            ax.legend()
        
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Year')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
