"""
Statistical visualization module.

This module provides visualization tools for statistical distributions,
correlations, confidence intervals, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path


class StatisticalPlotter:
    """Create statistical analysis visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize statistical plotter.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.default_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    def plot_monte_carlo_convergence(
        self,
        results: List[float],
        title: str = "Monte Carlo Convergence Analysis",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot Monte Carlo simulation convergence.
        
        Args:
            results: List of simulation results
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Calculate running statistics
        n_iterations = len(results)
        iterations = np.arange(1, n_iterations + 1)
        running_mean = np.cumsum(results) / iterations
        running_std = [np.std(results[:i+1]) for i in range(n_iterations)]
        
        # Left plot: Running mean with confidence bands
        ax1.plot(iterations, running_mean, linewidth=2, color='blue', label='Running Mean')
        ax1.fill_between(iterations,
                         running_mean - 1.96 * np.array(running_std) / np.sqrt(iterations),
                         running_mean + 1.96 * np.array(running_std) / np.sqrt(iterations),
                         alpha=0.3, color='blue', label='95% CI')
        
        # Add final value line
        final_mean = running_mean[-1]
        ax1.axhline(y=final_mean, color='red', linestyle='--', alpha=0.5,
                   label=f'Final: {final_mean:.2e}')
        
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Running Mean")
        ax1.set_title("Convergence of Mean")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Standard error convergence
        standard_errors = np.array(running_std) / np.sqrt(iterations)
        ax2.plot(iterations, standard_errors, linewidth=2, color='green')
        ax2.fill_between(iterations, 0, standard_errors, alpha=0.3, color='green')
        
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Standard Error")
        ax2.set_title("Standard Error Convergence")
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_comparison(
        self,
        data_dict: Dict[str, List[float]],
        title: str = "Distribution Comparison",
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Compare multiple distributions.
        
        Args:
            data_dict: Dictionary mapping names to data lists
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_distributions = len(data_dict)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Overlapping histograms
        ax = axes[0, 0]
        for i, (name, data) in enumerate(data_dict.items()):
            ax.hist(data, bins=30, alpha=0.5, label=name, 
                   color=self.default_colors[i % len(self.default_colors)])
        ax.set_title("Histogram Comparison")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plots
        ax = axes[0, 1]
        bp = ax.boxplot(list(data_dict.values()), labels=list(data_dict.keys()),
                       patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.default_colors[i % len(self.default_colors)])
            patch.set_alpha(0.7)
        ax.set_title("Box Plot Comparison")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Violin plots
        ax = axes[1, 0]
        parts = ax.violinplot(list(data_dict.values()), 
                             positions=range(len(data_dict)),
                             showmeans=True, showmedians=True)
        ax.set_xticks(range(len(data_dict)))
        ax.set_xticklabels(list(data_dict.keys()), rotation=45, ha='right')
        ax.set_title("Violin Plot Comparison")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q plots
        ax = axes[1, 1]
        if n_distributions == 2:
            # Q-Q plot for two distributions
            names = list(data_dict.keys())
            data1, data2 = list(data_dict.values())
            
            # Calculate quantiles
            quantiles = np.percentile(data1, np.linspace(0, 100, 100))
            quantiles2 = np.percentile(data2, np.linspace(0, 100, 100))
            
            ax.scatter(quantiles, quantiles2, alpha=0.5, s=20)
            ax.plot([min(quantiles), max(quantiles)],
                   [min(quantiles), max(quantiles)],
                   'r--', alpha=0.5, label='y=x')
            
            ax.set_xlabel(f"{names[0]} Quantiles")
            ax.set_ylabel(f"{names[1]} Quantiles")
            ax.set_title("Q-Q Plot")
            ax.legend()
        else:
            # Statistical summary table
            summary_data = []
            for name, data in data_dict.items():
                summary_data.append([
                    name[:10],
                    f"{np.mean(data):.2e}",
                    f"{np.std(data):.2e}",
                    f"{np.median(data):.2e}",
                    f"{stats.skew(data):.2f}"
                ])
            
            table = ax.table(cellText=summary_data,
                           colLabels=['Dist', 'Mean', 'Std', 'Median', 'Skew'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            ax.axis('off')
            ax.set_title("Statistical Summary")
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_analysis(
        self,
        sensitivity_results: Dict[str, Dict[str, float]],
        title: str = "Sensitivity Analysis",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_results: Dictionary of parameter impacts
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Extract parameters and their impacts
        parameters = list(sensitivity_results.keys())
        impacts = []
        for param in parameters:
            if isinstance(sensitivity_results[param], dict):
                # Calculate average impact
                impact = np.mean(list(sensitivity_results[param].values()))
            else:
                impact = sensitivity_results[param]
            impacts.append(impact)
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(impacts))[::-1]
        parameters = [parameters[i] for i in sorted_indices]
        impacts = [impacts[i] for i in sorted_indices]
        
        # Left plot: Tornado diagram
        colors = ['red' if x < 0 else 'green' for x in impacts]
        ax1.barh(range(len(parameters)), impacts, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(parameters)))
        ax1.set_yticklabels(parameters)
        ax1.set_xlabel("Impact on Output")
        ax1.set_title("Parameter Sensitivity (Tornado)")
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right plot: Spider/Radar chart for top parameters
        top_n = min(8, len(parameters))
        top_params = parameters[:top_n]
        top_impacts = np.abs(impacts[:top_n])
        
        # Normalize impacts to [0, 1]
        if max(top_impacts) > 0:
            top_impacts = top_impacts / max(top_impacts)
        
        angles = np.linspace(0, 2 * np.pi, len(top_params), endpoint=False)
        top_impacts = np.concatenate((top_impacts, [top_impacts[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, top_impacts, 'o-', linewidth=2, color='darkblue')
        ax2.fill(angles, top_impacts, alpha=0.25, color='darkblue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(top_params, size=8)
        ax2.set_ylim(0, 1)
        ax2.set_title("Top Parameter Impacts (Normalized)", pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_distributions(
    data: Union[List[float], Dict[str, List[float]]],
    title: str = "Distribution Analysis",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot comprehensive distribution analysis.
    
    Args:
        data: Single list or dictionary of data
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Convert to dictionary if single list
    if isinstance(data, list):
        data = {"Data": data}
    
    for i, (name, values) in enumerate(data.items()):
        color = plt.cm.Set1(i / max(1, len(data) - 1))
        
        # Left plot: Histogram with KDE
        ax1.hist(values, bins=30, alpha=0.5, density=True, 
                color=color, label=name)
        
        # Add KDE
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(min(values), max(values), 100)
        ax1.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        # Middle plot: CDF
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax2.plot(sorted_values, cdf, linewidth=2, color=color, label=name)
        
        # Right plot: Q-Q plot against normal
        stats.probplot(values, dist="norm", plot=ax3)
    
    ax1.set_title("Histogram with KDE")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Cumulative Distribution")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title("Q-Q Plot (Normal)")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame with variables as columns
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, square=True,
               linewidths=1, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confidence_intervals(
    estimates: Dict[str, Tuple[float, float, float]],
    title: str = "Confidence Interval Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confidence intervals for multiple estimates.
    
    Args:
        estimates: Dictionary mapping names to (mean, lower, upper) tuples
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(estimates.keys())
    means = [estimates[name][0] for name in names]
    lowers = [estimates[name][1] for name in names]
    uppers = [estimates[name][2] for name in names]
    
    # Calculate error bars
    yerr_lower = [m - l for m, l in zip(means, lowers)]
    yerr_upper = [u - m for u, m in zip(uppers, means)]
    
    # Create error bar plot
    x_pos = np.arange(len(names))
    ax.errorbar(x_pos, means, yerr=[yerr_lower, yerr_upper],
               fmt='o', markersize=8, capsize=5, capthick=2,
               linewidth=2, elinewidth=2)
    
    # Add value labels
    for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        ax.text(i, mean, f'{mean:.2e}', ha='center', va='bottom', fontsize=9)
        
        # Add CI range
        ci_range = upper - lower
        ax.text(i, lower - ci_range * 0.1, 
               f'[{lower:.2e}, {upper:.2e}]',
               ha='center', va='top', fontsize=8, color='gray')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Estimate Value")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add reference line at zero if applicable
    if min(lowers) < 0 < max(uppers):
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
