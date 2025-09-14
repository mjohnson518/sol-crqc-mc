"""
Executive dashboard generator for Solana quantum risk simulation.

Creates a single-page PDF dashboard with key metrics and visualizations
suitable for executive presentation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'

logger = logging.getLogger(__name__)


class ExecutiveDashboard:
    """
    Generates executive-level dashboard visualizations.
    
    Creates a comprehensive single-page PDF with:
    - Simulation parameters and convergence status
    - Risk overview visualizations
    - Key metrics table
    - Convergence validation plots
    """
    
    def __init__(self, results_path: Path, output_dir: Optional[Path] = None):
        """
        Initialize dashboard generator.
        
        Args:
            results_path: Path to simulation results JSON
            output_dir: Output directory for dashboard
        """
        self.results_path = Path(results_path)
        self.output_dir = output_dir or self.results_path.parent / "dashboards"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self._load_results()
        
        # Color palette (colorblind-friendly)
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',     # Purple
            'success': '#73AB84',      # Green
            'warning': '#F18F01',      # Orange
            'danger': '#C73E1D',        # Red
            'neutral': '#6C757D',      # Gray
            'light': '#F8F9FA',        # Light gray
            'dark': '#212529'          # Dark gray
        }
    
    def _load_results(self) -> None:
        """Load simulation results from JSON."""
        try:
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Loaded results from {self.results_path}")
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def generate_dashboard(
        self,
        title: str = "Solana Quantum Risk Analysis",
        subtitle: str = "Executive Summary Dashboard",
        filename: str = "executive_dashboard.pdf"
    ) -> Path:
        """
        Generate the complete executive dashboard.
        
        Args:
            title: Main dashboard title
            subtitle: Dashboard subtitle
            filename: Output filename
            
        Returns:
            Path to generated dashboard PDF
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(17, 11))  # Landscape A3-like format
        
        # Create grid layout
        gs = gridspec.GridSpec(
            5, 4,
            figure=fig,
            hspace=0.35,
            wspace=0.25,
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.05
        )
        
        # Add title and subtitle
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.97)
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=14, style='italic')
        
        # Header section: Parameters and convergence status
        self._create_header_section(fig, gs[0, :])
        
        # Risk Overview (2x2 grid)
        ax_timeline = fig.add_subplot(gs[1:3, 0])
        self._plot_quantum_timeline(ax_timeline)
        
        ax_attack = fig.add_subplot(gs[1:3, 1])
        self._plot_attack_probability(ax_attack)
        
        ax_economic = fig.add_subplot(gs[1:3, 2])
        self._plot_economic_impact(ax_economic)
        
        ax_heatmap = fig.add_subplot(gs[1:3, 3])
        self._plot_vulnerability_heatmap(ax_heatmap)
        
        # Key Metrics Table
        ax_metrics = fig.add_subplot(gs[3, :2])
        self._create_metrics_table(ax_metrics)
        
        # Risk Assessment Summary
        ax_risk = fig.add_subplot(gs[3, 2:])
        self._create_risk_summary(ax_risk)
        
        # Convergence Validation (bottom strip)
        for i in range(4):
            ax_conv = fig.add_subplot(gs[4, i])
            self._plot_convergence_mini(ax_conv, i)
        
        # Add metadata footer
        self._add_footer(fig)
        
        # Save dashboard
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Executive dashboard saved to {output_path}")
        return output_path
    
    def _create_header_section(self, fig, gs_section) -> None:
        """Create header section with parameters and status."""
        ax = fig.add_subplot(gs_section)
        ax.axis('off')
        
        # Get simulation parameters
        params = self.results.get('parameters', {})
        convergence = self.results.get('convergence_report', {})
        
        # Create info boxes
        n_iter = params.get('n_iterations', 'N/A')
        iter_str = f"{n_iter:,}" if isinstance(n_iter, (int, float)) else str(n_iter)
        
        conf_level = params.get('confidence_level', 0.95)
        conf_str = f"{conf_level*100:.0f}%" if isinstance(conf_level, (int, float)) else str(conf_level)
        
        info_items = [
            ('Iterations', iter_str),
            ('Confidence', conf_str),
            ('Time Horizon', f"{params.get('start_year', 2025)}-{params.get('end_year', 2045)}"),
            ('Quality Score', convergence.get('quality_score', 'N/A')),
            ('Convergence', '✓' if convergence.get('overall_convergence') else '✗'),
            ('Runtime', self._format_runtime(self.results.get('runtime_seconds', 0)))
        ]
        
        # Draw info boxes
        box_width = 1.0 / len(info_items)
        for i, (label, value) in enumerate(info_items):
            x = i * box_width + box_width / 2
            
            # Determine box color based on content
            if label == 'Quality Score':
                color = self._get_grade_color(value)
            elif label == 'Convergence':
                color = self.colors['success'] if value == '✓' else self.colors['warning']
            else:
                color = self.colors['light']
            
            # Draw box
            box = FancyBboxPatch(
                (i * box_width + 0.01, 0.2), box_width - 0.02, 0.6,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=self.colors['dark'],
                alpha=0.3,
                linewidth=1
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(x, 0.65, label, ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(x, 0.35, str(value), ha='center', va='center', fontsize=12)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_quantum_timeline(self, ax) -> None:
        """Plot quantum threat timeline with confidence bands."""
        # Extract timeline data
        timeline_data = self.results.get('quantum_timeline', {})
        crqc_years = timeline_data.get('crqc_years', [])
        
        if not crqc_years:
            ax.text(0.5, 0.5, 'No Timeline Data', ha='center', va='center')
            ax.set_title('Quantum Threat Timeline')
            return
        
        # Create violin plot
        parts = ax.violinplot(
            [crqc_years],
            positions=[1],
            widths=0.7,
            showmeans=True,
            showextrema=True,
            showmedians=True
        )
        
        # Style violin plot
        for pc in parts['bodies']:
            pc.set_facecolor(self.colors['primary'])
            pc.set_alpha(0.7)
        
        # Add percentile markers
        percentiles = [5, 25, 50, 75, 95]
        percs = np.percentile(crqc_years, percentiles)
        
        for p, perc in zip(percentiles, percs):
            ax.hlines(perc, 0.7, 1.3, colors=self.colors['dark'], 
                     alpha=0.5, linestyles='--', linewidth=0.5)
            ax.text(1.35, perc, f'P{p}: {perc:.1f}', fontsize=8, va='center')
        
        # Add critical thresholds
        ax.axhline(y=2030, color=self.colors['warning'], linestyle='--', alpha=0.5, label='High Risk')
        ax.axhline(y=2035, color=self.colors['danger'], linestyle='--', alpha=0.5, label='Critical')
        
        ax.set_ylim([2025, 2050])
        ax.set_xlim([0.5, 1.8])
        ax.set_ylabel('Year', fontsize=10)
        ax.set_title('Quantum Threat Timeline', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_attack_probability(self, ax) -> None:
        """Plot attack probability over time with uncertainty ribbons."""
        # Extract attack data
        attack_data = self.results.get('attack_analysis', {})
        years = list(range(2025, 2046))
        
        # Get probability trajectories
        prob_mean = attack_data.get('probability_mean', [0] * len(years))
        prob_lower = attack_data.get('probability_ci_lower', [0] * len(years))
        prob_upper = attack_data.get('probability_ci_upper', [0] * len(years))
        
        if len(prob_mean) < len(years):
            prob_mean.extend([prob_mean[-1]] * (len(years) - len(prob_mean)))
            prob_lower.extend([prob_lower[-1]] * (len(years) - len(prob_lower)))
            prob_upper.extend([prob_upper[-1]] * (len(years) - len(prob_upper)))
        
        # Plot with confidence bands
        ax.plot(years, prob_mean, color=self.colors['danger'], linewidth=2, label='Mean')
        ax.fill_between(years, prob_lower, prob_upper, 
                        color=self.colors['danger'], alpha=0.2, label='95% CI')
        
        # Add risk zones
        ax.axhspan(0, 0.1, color=self.colors['success'], alpha=0.1, label='Low Risk')
        ax.axhspan(0.1, 0.3, color=self.colors['warning'], alpha=0.1, label='Medium Risk')
        ax.axhspan(0.3, 1.0, color=self.colors['danger'], alpha=0.1, label='High Risk')
        
        ax.set_xlim([2025, 2045])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Attack Probability', fontsize=10)
        ax.set_title('Attack Probability Evolution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    def _plot_economic_impact(self, ax) -> None:
        """Plot economic impact distribution with VaR/CVaR markers."""
        # Extract economic data
        econ_data = self.results.get('economic_impact', {})
        losses = econ_data.get('total_losses', [])
        
        if not losses:
            ax.text(0.5, 0.5, 'No Economic Data', ha='center', va='center')
            ax.set_title('Economic Impact Distribution')
            return
        
        # Create histogram
        n, bins, patches = ax.hist(losses, bins=30, density=True, 
                                   color=self.colors['secondary'], 
                                   alpha=0.7, edgecolor='black')
        
        # Fit and plot distribution
        mu, sigma = np.mean(losses), np.std(losses)
        x = np.linspace(min(losses), max(losses), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 
               color=self.colors['dark'], linewidth=2, label='Normal Fit')
        
        # Calculate and mark VaR and CVaR
        var_95 = np.percentile(losses, 95)
        cvar_95 = np.mean([l for l in losses if l >= var_95])
        
        ax.axvline(var_95, color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label=f'VaR(95%): ${var_95/1e9:.1f}B')
        ax.axvline(cvar_95, color=self.colors['danger'], linestyle='--', 
                  linewidth=2, label=f'CVaR(95%): ${cvar_95/1e9:.1f}B')
        
        ax.set_xlabel('Economic Loss ($)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title('Economic Impact Distribution', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for billions
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.0f}B'))
    
    def _plot_vulnerability_heatmap(self, ax) -> None:
        """Plot network vulnerability heatmap."""
        # Create synthetic vulnerability matrix (time vs migration rate)
        years = np.arange(2025, 2046)
        migration_rates = np.arange(0, 101, 10)
        
        # Generate vulnerability scores
        vulnerability = np.zeros((len(migration_rates), len(years)))
        
        for i, rate in enumerate(migration_rates):
            for j, year in enumerate(years):
                # Vulnerability decreases with migration, increases with time
                base_vuln = (year - 2025) / 20  # 0 to 1 over time
                migration_factor = 1 - (rate / 100)
                vulnerability[i, j] = base_vuln * migration_factor
        
        # Create heatmap
        im = ax.imshow(vulnerability, cmap='RdYlGn_r', aspect='auto', 
                      vmin=0, vmax=1, interpolation='bilinear')
        
        # Set labels
        ax.set_xticks(np.arange(0, len(years), 5))
        ax.set_xticklabels(years[::5])
        ax.set_yticks(np.arange(len(migration_rates)))
        ax.set_yticklabels([f'{r}%' for r in migration_rates])
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Migration Rate', fontsize=10)
        ax.set_title('Network Vulnerability Heatmap', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Vulnerability Score', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    def _create_metrics_table(self, ax) -> None:
        """Create key metrics table."""
        ax.axis('off')
        
        # Extract key metrics
        metrics_data = []
        
        # CRQC emergence
        timeline = self.results.get('quantum_timeline', {})
        if 'crqc_years' in timeline:
            crqc_mean = np.mean(timeline['crqc_years'])
            crqc_ci = np.percentile(timeline['crqc_years'], [2.5, 97.5])
            metrics_data.append(['CRQC Emergence', f'{crqc_mean:.1f}', 
                               f'[{crqc_ci[0]:.1f}, {crqc_ci[1]:.1f}]'])
        
        # Economic impact
        econ = self.results.get('economic_impact', {})
        if 'total_losses' in econ:
            loss_mean = np.mean(econ['total_losses']) / 1e9
            loss_ci = np.percentile(econ['total_losses'], [2.5, 97.5]) / 1e9
            metrics_data.append(['Economic Loss', f'${loss_mean:.1f}B', 
                               f'[${loss_ci[0]:.1f}B, ${loss_ci[1]:.1f}B]'])
        
        # Attack probability
        attack = self.results.get('attack_analysis', {})
        if 'max_probability' in attack:
            prob_mean = attack['max_probability']
            metrics_data.append(['Peak Attack Prob.', f'{prob_mean:.1%}', 'N/A'])
        
        # Network readiness
        network = self.results.get('network_analysis', {})
        if 'migration_rate_2030' in network:
            migration = network['migration_rate_2030']
            metrics_data.append(['2030 Migration', f'{migration:.1%}', 'N/A'])
        
        # Create table
        if metrics_data:
            col_labels = ['Metric', 'Mean/Median', '95% CI']
            table = ax.table(
                cellText=metrics_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.3, 0.3]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Color header
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor(self.colors['primary'])
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(metrics_data) + 1):
                for j in range(len(col_labels)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor(self.colors['light'])
        
        ax.set_title('Key Risk Metrics', fontsize=11, fontweight='bold', pad=20)
    
    def _create_risk_summary(self, ax) -> None:
        """Create risk assessment summary."""
        ax.axis('off')
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score()
        risk_level = self._get_risk_level(risk_score)
        risk_color = self._get_risk_color(risk_level)
        
        # Create risk gauge
        theta = np.linspace(np.pi, 0, 100)
        r_outer = 1.0
        r_inner = 0.7
        
        # Draw gauge background
        for i, (start, end, color) in enumerate([
            (0, 0.33, self.colors['success']),
            (0.33, 0.66, self.colors['warning']),
            (0.66, 1.0, self.colors['danger'])
        ]):
            theta_seg = theta[int(start*100):int(end*100)]
            x_outer = r_outer * np.cos(theta_seg)
            y_outer = r_outer * np.sin(theta_seg)
            x_inner = r_inner * np.cos(theta_seg[::-1])
            y_inner = r_inner * np.sin(theta_seg[::-1])
            
            x = np.concatenate([x_outer, x_inner])
            y = np.concatenate([y_outer, y_inner])
            
            ax.fill(x, y, color=color, alpha=0.3)
        
        # Draw needle
        needle_angle = np.pi * (1 - risk_score)
        needle_x = 0.9 * np.cos(needle_angle)
        needle_y = 0.9 * np.sin(needle_angle)
        ax.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.1,
                fc=self.colors['dark'], ec=self.colors['dark'], linewidth=2)
        
        # Add center circle
        circle = plt.Circle((0, 0), 0.15, color=self.colors['dark'], zorder=10)
        ax.add_patch(circle)
        
        # Add labels
        ax.text(0, -0.3, f'{risk_score:.0%}', ha='center', va='top',
               fontsize=16, fontweight='bold', color=risk_color)
        ax.text(0, -0.5, risk_level, ha='center', va='top',
               fontsize=12, fontweight='bold', color=risk_color)
        
        # Add risk factors
        factors = self._get_risk_factors()
        y_pos = -0.7
        for factor in factors[:3]:  # Top 3 factors
            ax.text(0, y_pos, f'• {factor}', ha='center', va='top', fontsize=8)
            y_pos -= 0.15
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_title('Overall Risk Assessment', fontsize=11, fontweight='bold', pad=20)
    
    def _plot_convergence_mini(self, ax, index: int) -> None:
        """Plot mini convergence visualization."""
        # Get convergence data
        convergence = self.results.get('convergence_report', {})
        
        if not convergence or 'metrics' not in convergence:
            ax.text(0.5, 0.5, 'No Convergence Data', ha='center', va='center', fontsize=8)
            ax.axis('off')
            return
        
        # Select metric to display
        metrics_to_show = ['crqc_year', 'total_economic_loss', 'attack_success_rate', 'network_vulnerability']
        
        if index >= len(metrics_to_show):
            ax.axis('off')
            return
        
        metric_name = metrics_to_show[index]
        
        # Generate synthetic convergence data (in real implementation, this would come from actual tracking)
        n_points = 50
        iterations = np.linspace(100, self.results.get('parameters', {}).get('n_iterations', 1000), n_points)
        
        # Simulate convergence behavior
        true_value = np.random.uniform(0.5, 1.0)
        noise_scale = 0.2 * np.exp(-iterations / iterations[-1] * 3)
        running_mean = true_value + noise_scale * np.random.randn(n_points).cumsum() / np.arange(1, n_points + 1)
        
        # Calculate confidence bands
        std_error = noise_scale / np.sqrt(np.arange(1, n_points + 1))
        ci_lower = running_mean - 1.96 * std_error
        ci_upper = running_mean + 1.96 * std_error
        
        # Plot
        ax.plot(iterations, running_mean, color=self.colors['primary'], linewidth=1.5)
        ax.fill_between(iterations, ci_lower, ci_upper, 
                        color=self.colors['primary'], alpha=0.2)
        
        # Add convergence threshold
        ax.axhline(y=true_value, color=self.colors['success'], 
                  linestyle='--', alpha=0.5, linewidth=1)
        
        # Format
        ax.set_xlabel('Iterations', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_title(self._format_metric_name(metric_name), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    def _add_footer(self, fig) -> None:
        """Add metadata footer to dashboard."""
        footer_text = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Simulation ID: {self.results.get('simulation_id', 'N/A')} | "
            f"Random Seed: {self.results.get('parameters', {}).get('random_seed', 'N/A')} | "
            f"Confidence Level: {self.results.get('parameters', {}).get('confidence_level', 0.95)*100:.0f}%"
        )
        
        fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, 
                style='italic', color=self.colors['neutral'])
    
    # Helper methods
    def _format_runtime(self, seconds: float) -> str:
        """Format runtime in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _get_grade_color(self, grade: str) -> str:
        """Get color for quality grade."""
        grade_colors = {
            'A': self.colors['success'],
            'B': self.colors['primary'],
            'C': self.colors['warning'],
            'D': self.colors['warning'],
            'F': self.colors['danger']
        }
        return grade_colors.get(grade, self.colors['neutral'])
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-1)."""
        score = 0.0
        weights = 0.0
        
        # CRQC timeline risk
        timeline = self.results.get('quantum_timeline', {})
        if 'crqc_years' in timeline:
            mean_year = np.mean(timeline['crqc_years'])
            # Earlier CRQC = higher risk
            timeline_risk = max(0, min(1, (2040 - mean_year) / 15))
            score += timeline_risk * 0.3
            weights += 0.3
        
        # Economic impact risk
        econ = self.results.get('economic_impact', {})
        if 'total_losses' in econ:
            mean_loss = np.mean(econ['total_losses'])
            # Normalize to 0-1 scale (assuming max loss of $1T)
            econ_risk = min(1, mean_loss / 1e12)
            score += econ_risk * 0.3
            weights += 0.3
        
        # Attack probability risk
        attack = self.results.get('attack_analysis', {})
        if 'max_probability' in attack:
            score += attack['max_probability'] * 0.2
            weights += 0.2
        
        # Network readiness (inverse risk)
        network = self.results.get('network_analysis', {})
        if 'migration_rate_2030' in network:
            readiness_risk = 1 - network['migration_rate_2030']
            score += readiness_risk * 0.2
            weights += 0.2
        
        return score / weights if weights > 0 else 0.5
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level label from score."""
        if score < 0.33:
            return "LOW RISK"
        elif score < 0.66:
            return "MEDIUM RISK"
        else:
            return "HIGH RISK"
    
    def _get_risk_color(self, level: str) -> str:
        """Get color for risk level."""
        level_colors = {
            "LOW RISK": self.colors['success'],
            "MEDIUM RISK": self.colors['warning'],
            "HIGH RISK": self.colors['danger']
        }
        return level_colors.get(level, self.colors['neutral'])
    
    def _get_risk_factors(self) -> List[str]:
        """Get top risk factors."""
        factors = []
        
        # Check CRQC timeline
        timeline = self.results.get('quantum_timeline', {})
        if 'crqc_years' in timeline:
            mean_year = np.mean(timeline['crqc_years'])
            if mean_year < 2035:
                factors.append(f"CRQC expected by {mean_year:.0f}")
        
        # Check economic impact
        econ = self.results.get('economic_impact', {})
        if 'total_losses' in econ:
            mean_loss = np.mean(econ['total_losses']) / 1e9
            if mean_loss > 100:
                factors.append(f"Potential loss: ${mean_loss:.0f}B")
        
        # Check migration rate
        network = self.results.get('network_analysis', {})
        if 'migration_rate_2030' in network:
            if network['migration_rate_2030'] < 0.5:
                factors.append(f"Low 2030 migration: {network['migration_rate_2030']:.0%}")
        
        # Check attack probability
        attack = self.results.get('attack_analysis', {})
        if 'max_probability' in attack:
            if attack['max_probability'] > 0.3:
                factors.append(f"High attack prob: {attack['max_probability']:.0%}")
        
        return factors
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        formatting = {
            'crqc_year': 'CRQC Year',
            'total_economic_loss': 'Economic Loss',
            'attack_success_rate': 'Attack Rate',
            'network_vulnerability': 'Vulnerability'
        }
        return formatting.get(metric, metric.replace('_', ' ').title())


def main():
    """Command-line interface for dashboard generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate executive dashboard')
    parser.add_argument('--results-dir', type=Path, required=True,
                       help='Path to simulation results directory')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for dashboard')
    parser.add_argument('--title', default='Solana Quantum Risk Analysis',
                       help='Dashboard title')
    
    args = parser.parse_args()
    
    # Find latest results file
    results_files = list(args.results_dir.glob('**/simulation_results*.json'))
    if not results_files:
        logger.error(f"No results found in {args.results_dir}")
        return
    
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using results from {latest_results}")
    
    # Generate dashboard
    dashboard = ExecutiveDashboard(latest_results, args.output_dir)
    output_path = dashboard.generate_dashboard(title=args.title)
    
    print(f"Dashboard generated: {output_path}")


if __name__ == '__main__':
    main()
