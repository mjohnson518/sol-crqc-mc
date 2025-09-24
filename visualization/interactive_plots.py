"""
Interactive visualizations using Plotly for enhanced user experience.

This module provides interactive alternatives to static matplotlib plots,
allowing users to zoom, pan, hover for details, and export to HTML.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Solana-inspired color scheme
SOLANA_COLORS = {
    'primary': '#14F195',    # Solana green
    'secondary': '#9945FF',  # Solana purple  
    'dark': '#1A1B3A',       # Dark background
    'light': '#F5F5F5',      # Light text
    'warning': '#FF6B6B',    # Red for risks
    'success': '#4ECDC4',    # Teal for success
    'gradient_start': '#667EEA',
    'gradient_end': '#764BA2'
}


class InteractiveVisualizer:
    """Creates interactive Plotly visualizations for quantum risk analysis."""
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize interactive visualizer.
        
        Args:
            theme: Plotly template theme
        """
        self.theme = theme
        if PLOTLY_AVAILABLE:
            pio.templates.default = theme
    
    def create_risk_dashboard(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create an interactive risk dashboard with multiple panels.
        
        Args:
            results: Simulation results
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - cannot create interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'CRQC Emergence Timeline',
                'Attack Probability Over Time',
                'Economic Impact Distribution',
                'Network Migration Progress'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Extract data
        years = list(range(2025, 2046))
        
        # 1. CRQC Timeline (top-left)
        quantum_probs = results.get('quantum_timeline', {}).get('crqc_probability_by_year', {})
        crqc_probs = [quantum_probs.get(str(year), 0) for year in years]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=crqc_probs,
                mode='lines+markers',
                name='CRQC Probability',
                line=dict(color=SOLANA_COLORS['warning'], width=3),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Probability: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add 50% threshold line
        fig.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="50% Threshold",
            row=1, col=1
        )
        
        # 2. Attack Probability (top-right)
        attack_data = results.get('attack_scenarios', {})
        attack_probs = attack_data.get('attack_probability_timeline', {})
        
        # Create traces for different attack types
        for attack_type in ['PRIVATE_KEY', 'VALIDATOR', 'GROVER_POH']:
            probs = [attack_probs.get(str(year), {}).get(attack_type, 0) for year in years]
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=probs,
                    mode='lines',
                    name=attack_type.replace('_', ' ').title(),
                    stackgroup='one',  # Creates area chart
                    hovertemplate='%{y:.1%}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Economic Impact Histogram (bottom-left)
        economic_data = results.get('economic_impact', {})
        losses = economic_data.get('loss_samples', np.random.lognormal(20, 2, 1000))
        
        fig.add_trace(
            go.Histogram(
                x=np.log10(losses + 1),  # Log scale for better visualization
                nbinsx=30,
                name='Loss Distribution',
                marker_color=SOLANA_COLORS['secondary'],
                hovertemplate='Loss: $10^%{x:.1f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Network Migration (bottom-right)
        network_data = results.get('network_state', {})
        migration_timeline = network_data.get('migration_timeline', {})
        
        migrated = [migration_timeline.get(str(year), {}).get('migrated_percentage', 0) for year in years]
        at_risk = [100 - m for m in migrated]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=migrated,
                mode='lines',
                name='Migrated',
                fill='tonexty',
                line=dict(color=SOLANA_COLORS['success'], width=2),
                hovertemplate='Year: %{x}<br>Migrated: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=at_risk,
                mode='lines',
                name='At Risk',
                fill='tozeroy',
                line=dict(color=SOLANA_COLORS['warning'], width=2),
                hovertemplate='Year: %{x}<br>At Risk: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🔐 Solana Quantum Risk Dashboard',
                'font': {'size': 24, 'color': SOLANA_COLORS['light']}
            },
            showlegend=True,
            height=800,
            template=self.theme,
            hovermode='x unified',
            paper_bgcolor=SOLANA_COLORS['dark'],
            plot_bgcolor=SOLANA_COLORS['dark']
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Attack Probability", row=1, col=2)
        
        fig.update_xaxes(title_text="Log10(Loss + 1) USD", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Year", row=2, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=2, col=2)
        
        # Save if requested
        if output_path:
            fig.write_html(str(output_path))
            logger.info(f"Interactive dashboard saved to {output_path}")
        
        return fig
    
    def create_sensitivity_heatmap(
        self,
        sensitivity_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create an interactive sensitivity analysis heatmap.
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Extract Sobol indices
        params = list(sensitivity_results.get('first_order', {}).keys())
        first_order = list(sensitivity_results.get('first_order', {}).values())
        total_order = list(sensitivity_results.get('total_order', {}).values())
        
        # Create heatmap data
        z = [first_order, total_order]
        y = ['First Order', 'Total Order']
        x = params
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            text=[[f'{v:.3f}' for v in row] for row in z],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Parameter: %{x}<br>Index Type: %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Parameter Sensitivity Analysis (Sobol Indices)',
            xaxis_title='Parameters',
            yaxis_title='Sensitivity Index Type',
            height=400,
            template=self.theme
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_3d_risk_surface(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create a 3D surface plot of risk over time and attack scenarios.
        
        Args:
            results: Simulation results
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Create grid data
        years = np.arange(2025, 2046)
        scenarios = np.arange(0, 5)  # 5 attack scenarios
        
        # Generate risk surface (example - would use real data)
        X, Y = np.meshgrid(years, scenarios)
        Z = np.zeros_like(X, dtype=float)
        
        # Populate with risk values
        for i, scenario in enumerate(scenarios):
            for j, year in enumerate(years):
                # Risk increases over time and varies by scenario
                base_risk = (year - 2025) / 20  # 0 to 1 over 20 years
                scenario_mult = 1 + scenario * 0.3
                Z[i, j] = min(base_risk * scenario_mult * np.random.uniform(0.8, 1.2), 1.0)
        
        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=years,
            y=['Private Key', 'Validator', 'PoH', 'Hybrid', 'State Actor'],
            colorscale='RdYlGn_r',
            hovertemplate='Year: %{x}<br>Scenario: %{y}<br>Risk: %{z:.1%}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Risk Surface: Attack Scenarios Over Time',
            scene=dict(
                xaxis_title='Year',
                yaxis_title='Attack Scenario',
                zaxis_title='Risk Level',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            template=self.theme
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_network_graph(
        self,
        network_state: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create an interactive network graph of validator relationships.
        
        Args:
            network_state: Network topology data
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Generate example network data
        n_nodes = 50
        np.random.seed(42)
        
        # Create node positions (circular layout)
        theta = np.linspace(0, 2*np.pi, n_nodes)
        x = np.cos(theta) + np.random.normal(0, 0.1, n_nodes)
        y = np.sin(theta) + np.random.normal(0, 0.1, n_nodes)
        
        # Node sizes based on stake
        stakes = np.random.pareto(2, n_nodes) * 100
        
        # Create edges (preferential attachment)
        edge_x = []
        edge_y = []
        for i in range(n_nodes):
            for j in range(i+1, min(i+3, n_nodes)):
                edge_x.extend([x[i], x[j], None])
                edge_y.extend([y[i], y[j], None])
        
        # Edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node trace
        node_trace = go.Scatter(
            x=x, y=y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=stakes,
                color=stakes,
                colorbar=dict(
                    thickness=15,
                    title="Stake (SOL)",
                    xanchor="left",
                    titleside="right"
                ),
                line_width=2
            ),
            text=[f'V{i}' for i in range(n_nodes)],
            hovertext=[f'Validator {i}<br>Stake: {s:.0f} SOL' for i, s in enumerate(stakes)]
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Validator Network Topology',
                           titlefont_size=20,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           template=self.theme,
                           height=600
                       ))
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_comparison_chart(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        metric: str = 'total_risk',
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create comparison chart across multiple scenarios.
        
        Args:
            scenarios: Dictionary of scenario names to results
            metric: Metric to compare
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Extract data for comparison
        scenario_names = list(scenarios.keys())
        values = []
        errors = []
        
        for name, data in scenarios.items():
            val = data.get(metric, {}).get('mean', 0)
            err = data.get(metric, {}).get('std', 0)
            values.append(val)
            errors.append(err)
        
        # Create bar chart with error bars
        fig = go.Figure(data=[
            go.Bar(
                x=scenario_names,
                y=values,
                error_y=dict(
                    type='data',
                    array=errors,
                    visible=True
                ),
                marker_color=SOLANA_COLORS['primary'],
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
                hovertemplate='%{x}<br>Value: %{y:.2f} ± %{error_y.array:.2f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Scenario Comparison: {metric.replace("_", " ").title()}',
            xaxis_title='Scenario',
            yaxis_title=metric.replace('_', ' ').title(),
            template=self.theme,
            height=500
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def create_timeline_gantt(
        self,
        milestones: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> Optional[go.Figure]:
        """
        Create a Gantt chart for quantum development milestones.
        
        Args:
            milestones: List of milestone dictionaries
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE or not milestones:
            return None
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(milestones)
        
        if 'start' not in df.columns or 'end' not in df.columns:
            # Generate example data
            df = pd.DataFrame([
                {'task': 'Logical Qubits > 100', 'start': '2025-01-01', 'end': '2027-01-01', 'category': 'Hardware'},
                {'task': 'Error Rate < 0.1%', 'start': '2026-01-01', 'end': '2028-01-01', 'category': 'Hardware'},
                {'task': 'Shor\'s Algorithm Demo', 'start': '2027-01-01', 'end': '2029-01-01', 'category': 'Software'},
                {'task': 'CRQC Emergence', 'start': '2029-01-01', 'end': '2032-01-01', 'category': 'Threat'},
                {'task': 'Post-Quantum Migration', 'start': '2025-01-01', 'end': '2030-01-01', 'category': 'Defense'},
            ])
        
        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start='start',
            x_end='end',
            y='task',
            color='category',
            title='Quantum Development Timeline & Milestones',
            color_discrete_map={
                'Hardware': SOLANA_COLORS['primary'],
                'Software': SOLANA_COLORS['secondary'],
                'Threat': SOLANA_COLORS['warning'],
                'Defense': SOLANA_COLORS['success']
            }
        )
        
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(
            height=400,
            template=self.theme,
            xaxis_title='Timeline',
            yaxis_title='Milestones'
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig


def generate_interactive_report(
    results: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Generate a complete interactive HTML report.
    
    Args:
        results: Simulation results
        output_dir: Directory to save outputs
        
    Returns:
        Path to main HTML file
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not available - cannot generate interactive report")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    viz = InteractiveVisualizer()
    
    # Generate all visualizations
    dashboard = viz.create_risk_dashboard(results, output_dir / 'dashboard.html')
    surface = viz.create_3d_risk_surface(results, output_dir / '3d_risk.html')
    network = viz.create_network_graph(results.get('network_state', {}), output_dir / 'network.html')
    
    # Create main index.html
    index_path = output_dir / 'index.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Solana Quantum Risk - Interactive Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                color: #fff;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                text-align: center;
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .subtitle {{
                text-align: center;
                font-size: 1.2em;
                opacity: 0.9;
                margin-bottom: 40px;
            }}
            .nav {{
                display: flex;
                gap: 20px;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 40px;
            }}
            .nav-btn {{
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.5);
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.3s;
                backdrop-filter: blur(10px);
            }}
            .nav-btn:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .iframe-container {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            iframe {{
                width: 100%;
                height: 900px;
                border: none;
                border-radius: 8px;
            }}
            .timestamp {{
                text-align: center;
                opacity: 0.7;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 Solana Quantum Risk Analysis</h1>
            <p class="subtitle">Interactive Visualization Suite</p>
            
            <nav class="nav">
                <a href="dashboard.html" class="nav-btn" target="display">📊 Risk Dashboard</a>
                <a href="3d_risk.html" class="nav-btn" target="display">🎯 3D Risk Surface</a>
                <a href="network.html" class="nav-btn" target="display">🌐 Network Graph</a>
            </nav>
            
            <div class="iframe-container">
                <iframe name="display" src="dashboard.html"></iframe>
            </div>
            
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Interactive report generated at {index_path}")
    return index_path


if __name__ == "__main__":
    # Test with example data
    example_results = {
        'quantum_timeline': {
            'crqc_probability_by_year': {str(y): (y-2025)/20 for y in range(2025, 2046)}
        },
        'attack_scenarios': {
            'attack_probability_timeline': {
                str(y): {
                    'PRIVATE_KEY': 0.1 * (y-2025)/20,
                    'VALIDATOR': 0.2 * (y-2025)/20,
                    'GROVER_POH': 0.05 * (y-2025)/20
                } for y in range(2025, 2046)
            }
        },
        'economic_impact': {
            'loss_samples': np.random.lognormal(20, 2, 1000)
        },
        'network_state': {
            'migration_timeline': {
                str(y): {'migrated_percentage': min(100, 5 * (y-2025))}
                for y in range(2025, 2046)
            }
        }
    }
    
    output_path = Path('interactive_report_demo')
    report_path = generate_interactive_report(example_results, output_path)
    
    if report_path:
        print(f"✅ Interactive report generated: {report_path}")
        print(f"   Open {report_path} in a web browser to view")
