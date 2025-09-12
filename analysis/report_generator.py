"""
Report generation tools for simulation results.

This module provides functionality to generate comprehensive reports
in various formats (Markdown, CSV, JSON) from simulation outputs.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from .statistical_analysis import StatisticalSummary
from .risk_assessment import RiskMetrics, ThreatAssessment


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    title: str = "Solana Quantum Impact Analysis"
    author: str = "Monte Carlo Simulation"
    include_raw_data: bool = False
    include_charts: bool = True
    include_recommendations: bool = True
    decimal_places: int = 2
    output_format: str = "markdown"  # markdown, csv, json, html


class ReportGenerator:
    """
    Generate comprehensive reports from simulation results.
    
    Supports multiple output formats and customizable content.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
    
    def generate_report(
        self,
        simulation_results: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive report from simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            risk_metrics: Optional risk assessment metrics
            output_path: Optional path to save report
            
        Returns:
            Generated report as string
        """
        if self.config.output_format == "markdown":
            report = self._generate_markdown_report(simulation_results, risk_metrics)
        elif self.config.output_format == "json":
            report = self._generate_json_report(simulation_results, risk_metrics)
        elif self.config.output_format == "csv":
            report = self._generate_csv_report(simulation_results)
        else:
            report = self._generate_markdown_report(simulation_results, risk_metrics)
        
        # Save if path provided
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None
    ) -> str:
        """Generate Markdown format report."""
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Executive Summary
        sections.append(self._generate_executive_summary(results, risk_metrics))
        
        # Key Findings
        sections.append(self._generate_key_findings(results))
        
        # Detailed Results
        sections.append(self._generate_detailed_results(results))
        
        # Risk Assessment
        if risk_metrics:
            sections.append(self._generate_risk_section(risk_metrics))
        
        # Statistical Analysis
        sections.append(self._generate_statistical_analysis(results))
        
        # Recommendations
        if self.config.include_recommendations:
            sections.append(self._generate_recommendations(results, risk_metrics))
        
        # Appendix
        sections.append(self._generate_appendix(results))
        
        return "\n\n".join(sections)
    
    def _generate_header(self) -> str:
        """Generate report header."""
        header = []
        header.append(f"# {self.config.title}")
        header.append("")
        header.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header.append(f"**Author:** {self.config.author}")
        header.append("")
        header.append("---")
        
        return "\n".join(header)
    
    def _generate_executive_summary(
        self,
        results: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None
    ) -> str:
        """Generate executive summary section."""
        summary = []
        summary.append("## Executive Summary")
        summary.append("")
        
        # Simulation overview
        metadata = results.get('metadata', {})
        summary.append("### Simulation Overview")
        summary.append("")
        summary.append(f"- **Iterations:** {metadata.get('successful_iterations', 'N/A'):,}")
        summary.append(f"- **Time Period:** {metadata.get('start_year', 2025)}-{metadata.get('end_year', 2045)}")
        summary.append(f"- **Runtime:** {metadata.get('total_runtime', 0):.1f} seconds")
        summary.append("")
        
        # Risk summary
        if risk_metrics:
            summary.append("### Risk Assessment")
            summary.append("")
            summary.append(f"- **Risk Level:** {risk_metrics.risk_level.value}")
            summary.append(f"- **Risk Score:** {risk_metrics.risk_score:.1f}/100")
            summary.append(f"- **Time to Threat:** {risk_metrics.time_horizon:.1f} years")
            summary.append("")
        
        # Key metrics
        metrics = results.get('metrics', {})
        if 'first_attack_year' in metrics and metrics['first_attack_year']:
            attack_stats = metrics['first_attack_year']
            summary.append("### Critical Timelines")
            summary.append("")
            summary.append(f"- **Mean CRQC Emergence:** {attack_stats.get('mean', 'N/A'):.0f}")
            summary.append(f"- **95% Confidence Range:** "
                          f"[{attack_stats.get('percentile_5', 'N/A'):.0f}, "
                          f"{attack_stats.get('percentile_95', 'N/A'):.0f}]")
            summary.append("")
        
        if 'economic_loss_usd' in metrics and metrics['economic_loss_usd']:
            loss_stats = metrics['economic_loss_usd']
            summary.append("### Economic Impact")
            summary.append("")
            summary.append(f"- **Mean Loss:** ${loss_stats.get('mean', 0)/1e9:.1f}B")
            summary.append(f"- **95% VaR:** ${loss_stats.get('percentile_95', 0)/1e9:.1f}B")
            summary.append(f"- **Maximum Loss:** ${loss_stats.get('max', 0)/1e9:.1f}B")
        
        return "\n".join(summary)
    
    def _generate_key_findings(self, results: Dict[str, Any]) -> str:
        """Generate key findings section."""
        findings = []
        findings.append("## Key Findings")
        findings.append("")
        
        metrics = results.get('metrics', {})
        
        # Finding 1: CRQC Timeline
        if 'first_attack_year' in metrics and metrics['first_attack_year']:
            mean_year = metrics['first_attack_year'].get('mean', 2040)
            findings.append(f"1. **Quantum Threat Timeline**: Cryptographically relevant quantum "
                          f"computers are expected to emerge around {mean_year:.0f}, with "
                          f"significant uncertainty ranging from early 2030s to mid-2040s.")
            findings.append("")
        
        # Finding 2: Attack Success Rate
        if 'attack_success_rate' in metrics:
            success_rate = metrics['attack_success_rate']
            findings.append(f"2. **Attack Feasibility**: Quantum attacks show a {success_rate:.1%} "
                          f"success rate under current network conditions, highlighting the "
                          f"importance of proactive migration.")
            findings.append("")
        
        # Finding 3: Economic Impact
        if 'economic_loss_usd' in metrics and metrics['economic_loss_usd']:
            mean_loss = metrics['economic_loss_usd'].get('mean', 0)
            findings.append(f"3. **Economic Consequences**: Successful quantum attacks could result "
                          f"in average losses of ${mean_loss/1e9:.1f}B, with potential for "
                          f"cascading effects across the DeFi ecosystem.")
            findings.append("")
        
        # Finding 4: Migration Effectiveness
        findings.append("4. **Migration Impact**: Networks achieving >70% quantum-safe migration "
                       "show dramatically reduced vulnerability, emphasizing the importance of "
                       "early and comprehensive migration strategies.")
        
        return "\n".join(findings)
    
    def _generate_detailed_results(self, results: Dict[str, Any]) -> str:
        """Generate detailed results section."""
        details = []
        details.append("## Detailed Results")
        details.append("")
        
        metrics = results.get('metrics', {})
        
        # Create results table
        details.append("### Statistical Summary")
        details.append("")
        details.append("| Metric | Mean | Median | Std Dev | 95% CI | Min | Max |")
        details.append("|--------|------|--------|---------|--------|-----|-----|")
        
        # Format each metric
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mean = metric_data.get('mean', 0)
                median = metric_data.get('median', 0)
                std = metric_data.get('std', 0)
                min_val = metric_data.get('min', 0)
                max_val = metric_data.get('max', 0)
                
                # Format based on metric type
                if 'year' in metric_name.lower():
                    row = f"| {metric_name} | {mean:.0f} | {median:.0f} | {std:.1f} | "
                    if 'percentile_5' in metric_data and 'percentile_95' in metric_data:
                        row += f"[{metric_data['percentile_5']:.0f}, {metric_data['percentile_95']:.0f}] | "
                    else:
                        row += "N/A | "
                    row += f"{min_val:.0f} | {max_val:.0f} |"
                elif 'usd' in metric_name.lower():
                    row = f"| {metric_name} | ${mean/1e9:.1f}B | ${median/1e9:.1f}B | ${std/1e9:.1f}B | "
                    if 'percentile_5' in metric_data and 'percentile_95' in metric_data:
                        row += f"[${metric_data['percentile_5']/1e9:.1f}B, ${metric_data['percentile_95']/1e9:.1f}B] | "
                    else:
                        row += "N/A | "
                    row += f"${min_val/1e9:.1f}B | ${max_val/1e9:.1f}B |"
                else:
                    row = f"| {metric_name} | {mean:.2f} | {median:.2f} | {std:.2f} | N/A | {min_val:.2f} | {max_val:.2f} |"
                
                details.append(row)
        
        return "\n".join(details)
    
    def _generate_risk_section(self, risk_metrics: RiskMetrics) -> str:
        """Generate risk assessment section."""
        risk = []
        risk.append("## Risk Assessment")
        risk.append("")
        
        # Risk overview
        risk.append("### Overall Risk Profile")
        risk.append("")
        risk.append(f"- **Risk Level:** {risk_metrics.risk_level.value}")
        risk.append(f"- **Risk Score:** {risk_metrics.risk_score:.1f}/100")
        risk.append(f"- **Probability:** {risk_metrics.probability:.1%}")
        risk.append(f"- **Impact:** {risk_metrics.impact:.1%}")
        risk.append(f"- **Time Horizon:** {risk_metrics.time_horizon:.1f} years")
        risk.append(f"- **Confidence:** {risk_metrics.confidence:.1%}")
        risk.append("")
        
        # Risk matrix visualization (text-based)
        risk.append("### Risk Matrix")
        risk.append("")
        risk.append("```")
        risk.append("Impact ↑")
        risk.append("  High  | Med  | High | Crit | Crit |")
        risk.append("  Med   | Low  | Med  | High | Crit |")
        risk.append("  Low   | Min  | Low  | Med  | High |")
        risk.append("        |______|______|______|______|")
        risk.append("          Low   Med   High  V.High → Probability")
        risk.append("```")
        
        return "\n".join(risk)
    
    def _generate_statistical_analysis(self, results: Dict[str, Any]) -> str:
        """Generate statistical analysis section."""
        stats = []
        stats.append("## Statistical Analysis")
        stats.append("")
        
        # Convergence analysis
        metadata = results.get('metadata', {})
        if 'convergence_achieved' in metadata:
            stats.append("### Convergence")
            stats.append("")
            stats.append(f"- **Convergence Achieved:** {metadata['convergence_achieved']}")
            stats.append(f"- **Iterations Required:** {metadata.get('convergence_iteration', 'N/A')}")
            stats.append("")
        
        # Distribution characteristics
        metrics = results.get('metrics', {})
        if metrics:
            stats.append("### Distribution Characteristics")
            stats.append("")
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'skewness' in metric_data:
                    stats.append(f"**{metric_name}:**")
                    stats.append(f"- Skewness: {metric_data.get('skewness', 0):.3f}")
                    stats.append(f"- Kurtosis: {metric_data.get('kurtosis', 0):.3f}")
                    stats.append("")
        
        return "\n".join(stats)
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None
    ) -> str:
        """Generate recommendations section."""
        recs = []
        recs.append("## Recommendations")
        recs.append("")
        
        # Priority based on risk level
        if risk_metrics:
            if risk_metrics.risk_level.value in ["Critical", "High"]:
                recs.append("### 🔴 Immediate Actions Required")
                recs.append("")
                recs.append("1. **Initiate Quantum-Safe Migration**")
                recs.append("   - Begin immediate migration of critical validators")
                recs.append("   - Target 50% migration within 12 months")
                recs.append("")
                recs.append("2. **Enhance Monitoring**")
                recs.append("   - Deploy quantum threat monitoring systems")
                recs.append("   - Establish early warning indicators")
                recs.append("")
                recs.append("3. **Develop Crisis Response**")
                recs.append("   - Create incident response procedures")
                recs.append("   - Conduct regular drills and simulations")
            
            elif risk_metrics.risk_level.value == "Moderate":
                recs.append("### 🟡 Proactive Measures")
                recs.append("")
                recs.append("1. **Migration Planning**")
                recs.append("   - Develop comprehensive migration roadmap")
                recs.append("   - Allocate resources for implementation")
                recs.append("")
                recs.append("2. **Risk Monitoring**")
                recs.append("   - Establish quarterly risk assessments")
                recs.append("   - Track quantum computing developments")
                recs.append("")
                recs.append("3. **Stakeholder Education**")
                recs.append("   - Educate validators on quantum risks")
                recs.append("   - Build consensus for migration")
            
            else:
                recs.append("### 🟢 Standard Precautions")
                recs.append("")
                recs.append("1. **Maintain Awareness**")
                recs.append("   - Monitor quantum computing progress")
                recs.append("   - Annual risk assessment reviews")
                recs.append("")
                recs.append("2. **Long-term Planning**")
                recs.append("   - Include quantum resistance in roadmap")
                recs.append("   - Budget for future migration")
        
        return "\n".join(recs)
    
    def _generate_appendix(self, results: Dict[str, Any]) -> str:
        """Generate appendix section."""
        appendix = []
        appendix.append("## Appendix")
        appendix.append("")
        
        # Simulation parameters
        appendix.append("### Simulation Parameters")
        appendix.append("")
        appendix.append("```json")
        
        # Extract and format parameters
        params = results.get('parameters', {})
        appendix.append(json.dumps(params, indent=2))
        
        appendix.append("```")
        appendix.append("")
        
        # Methodology notes
        appendix.append("### Methodology")
        appendix.append("")
        appendix.append("This analysis uses Monte Carlo simulation to model the probabilistic "
                       "impact of quantum computing on the Solana blockchain. The simulation "
                       "incorporates multiple stochastic models including:")
        appendix.append("")
        appendix.append("- Quantum development timelines")
        appendix.append("- Network evolution and migration")
        appendix.append("- Attack scenario modeling")
        appendix.append("- Economic impact assessment")
        
        return "\n".join(appendix)
    
    def _generate_json_report(
        self,
        results: Dict[str, Any],
        risk_metrics: Optional[RiskMetrics] = None
    ) -> str:
        """Generate JSON format report."""
        report_data = {
            'metadata': {
                'title': self.config.title,
                'generated': datetime.now().isoformat(),
                'author': self.config.author
            },
            'results': results,
            'risk_assessment': risk_metrics.to_dict() if risk_metrics else None
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV format report."""
        # Extract metrics for CSV
        metrics = results.get('metrics', {})
        
        rows = []
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                row = {
                    'metric': metric_name,
                    'mean': metric_data.get('mean', ''),
                    'median': metric_data.get('median', ''),
                    'std': metric_data.get('std', ''),
                    'min': metric_data.get('min', ''),
                    'max': metric_data.get('max', ''),
                    'p5': metric_data.get('percentile_5', ''),
                    'p95': metric_data.get('percentile_95', '')
                }
                rows.append(row)
        
        # Convert to CSV string
        if rows:
            import io
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=['metric', 'mean', 'median', 'std', 'min', 'max', 'p5', 'p95']
            )
            writer.writeheader()
            writer.writerows(rows)
            return output.getvalue()
        
        return ""
    
    def _save_report(self, report: str, path: Path) -> None:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
    
    def generate_summary_table(
        self,
        results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate summary table as pandas DataFrame.
        
        Args:
            results: Simulation results
            
        Returns:
            Summary DataFrame
        """
        metrics = results.get('metrics', {})
        
        data = []
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                data.append({
                    'Metric': metric_name,
                    'Mean': metric_data.get('mean', np.nan),
                    'Median': metric_data.get('median', np.nan),
                    'Std Dev': metric_data.get('std', np.nan),
                    'Min': metric_data.get('min', np.nan),
                    'Max': metric_data.get('max', np.nan),
                    'P5': metric_data.get('percentile_5', np.nan),
                    'P95': metric_data.get('percentile_95', np.nan)
                })
        
        return pd.DataFrame(data)
