"""
Statistical analysis tools for Monte Carlo simulation results.

This package provides comprehensive analysis capabilities including:
- Statistical analysis and distribution fitting
- Scenario comparison and optimization
- Risk assessment and scoring
- Report generation in multiple formats
"""

from .statistical_analysis import (
    StatisticalAnalyzer,
    StatisticalSummary,
    DistributionFit
)

from .scenario_comparison import (
    ScenarioComparator,
    ScenarioResult,
    ComparisonResult
)

from .risk_assessment import (
    RiskAssessor,
    RiskMetrics,
    RiskLevel,
    ThreatCategory,
    ThreatAssessment,
    RiskMatrix
)

from .report_generator import (
    ReportGenerator,
    ReportConfig
)

__all__ = [
    # Statistical Analysis
    'StatisticalAnalyzer',
    'StatisticalSummary',
    'DistributionFit',
    
    # Scenario Comparison
    'ScenarioComparator',
    'ScenarioResult',
    'ComparisonResult',
    
    # Risk Assessment
    'RiskAssessor',
    'RiskMetrics',
    'RiskLevel',
    'ThreatCategory',
    'ThreatAssessment',
    'RiskMatrix',
    
    # Report Generation
    'ReportGenerator',
    'ReportConfig'
]

__version__ = '1.0.0'
