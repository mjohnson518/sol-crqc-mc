"""
Historical validation module for comparing simulation results with real events.

This module validates the Monte Carlo simulation against historical crypto crashes
and security incidents to ensure the model produces realistic outputs.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("Scipy not available - some validation tests will be skipped")

logger = logging.getLogger(__name__)


@dataclass
class HistoricalEvent:
    """Represents a historical crypto/blockchain event for comparison."""
    name: str
    date: str
    event_type: str  # 'crash', 'hack', 'collapse', 'attack'
    asset: str
    peak_value: float
    trough_value: float
    loss_percentage: float
    recovery_time_days: Optional[int]
    market_cap_loss_usd: float
    description: str
    relevant_metrics: Dict[str, Any]


class HistoricalValidator:
    """Validates simulation results against historical events."""
    
    def __init__(self):
        """Initialize with historical event database."""
        self.events = self._load_historical_events()
        self.validation_results = {}
    
    def _load_historical_events(self) -> List[HistoricalEvent]:
        """Load database of historical events for comparison."""
        events = [
            HistoricalEvent(
                name="Terra Luna Collapse",
                date="2022-05-07",
                event_type="collapse",
                asset="LUNA",
                peak_value=119.18,
                trough_value=0.00001,
                loss_percentage=99.99,
                recovery_time_days=None,  # Never recovered
                market_cap_loss_usd=60_000_000_000,
                description="Algorithmic stablecoin death spiral",
                relevant_metrics={
                    'time_to_collapse_days': 7,
                    'contagion_effect': True,
                    'defi_tvl_impact': 0.85,  # 85% TVL loss
                    'validator_impact': 0.95,  # 95% validators affected
                    'cascade_trigger': 'liquidity_crisis'
                }
            ),
            HistoricalEvent(
                name="FTX Bankruptcy",
                date="2022-11-11",
                event_type="collapse",
                asset="FTT",
                peak_value=84.18,
                trough_value=1.24,
                loss_percentage=98.5,
                recovery_time_days=None,
                market_cap_loss_usd=32_000_000_000,
                description="Exchange collapse due to fraud/mismanagement",
                relevant_metrics={
                    'time_to_collapse_days': 10,
                    'contagion_effect': True,
                    'market_impact': 0.15,  # 15% crypto market impact
                    'trust_erosion': 0.7,    # 70% trust loss
                    'regulatory_response': True
                }
            ),
            HistoricalEvent(
                name="Celsius Network Bankruptcy",
                date="2022-06-12",
                event_type="collapse",
                asset="CEL",
                peak_value=8.05,
                trough_value=0.09,
                loss_percentage=98.9,
                recovery_time_days=None,
                market_cap_loss_usd=8_000_000_000,
                description="Lending platform collapse",
                relevant_metrics={
                    'time_to_collapse_days': 30,
                    'withdrawal_freeze': True,
                    'leverage_ratio': 19,
                    'customer_funds_lost': 0.8  # 80% unrecoverable
                }
            ),
            HistoricalEvent(
                name="Ronin Bridge Hack",
                date="2022-03-23",
                event_type="hack",
                asset="ETH/USDC",
                peak_value=625_000_000,
                trough_value=0,
                loss_percentage=100,
                recovery_time_days=180,  # Partial recovery via refund
                market_cap_loss_usd=625_000_000,
                description="Validator key compromise",
                relevant_metrics={
                    'attack_vector': 'validator_compromise',
                    'validators_compromised': 5,
                    'detection_delay_days': 6,
                    'recovery_rate': 0.5  # 50% eventually recovered
                }
            ),
            HistoricalEvent(
                name="Wormhole Bridge Hack",
                date="2022-02-02",
                event_type="hack",
                asset="wETH",
                peak_value=326_000_000,
                trough_value=0,
                loss_percentage=100,
                recovery_time_days=1,  # Jump Crypto covered losses
                market_cap_loss_usd=326_000_000,
                description="Smart contract vulnerability",
                relevant_metrics={
                    'attack_vector': 'smart_contract',
                    'attack_sophistication': 'high',
                    'bailout': True,
                    'protocol_survival': True
                }
            ),
            HistoricalEvent(
                name="Mt. Gox Hack",
                date="2014-02-07",
                event_type="hack",
                asset="BTC",
                peak_value=850_000_000,
                trough_value=0,
                loss_percentage=100,
                recovery_time_days=3650,  # ~10 years for partial recovery
                market_cap_loss_usd=850_000_000,
                description="Exchange hack - 850,000 BTC stolen",
                relevant_metrics={
                    'btc_stolen': 850000,
                    'market_share_at_time': 0.7,  # 70% of BTC trading
                    'regulatory_impact': 'high',
                    'industry_maturation': True
                }
            ),
            HistoricalEvent(
                name="Black Thursday",
                date="2020-03-12",
                event_type="crash",
                asset="ETH",
                peak_value=194,
                trough_value=86,
                loss_percentage=55.7,
                recovery_time_days=45,
                market_cap_loss_usd=100_000_000_000,
                description="COVID-19 market crash",
                relevant_metrics={
                    'defi_liquidations': 140_000_000,
                    'gas_fees_spike': 10,  # 10x normal
                    'stablecoin_demand': 2.5,  # 2.5x normal
                    'correlation_with_tradfi': 0.85
                }
            )
        ]
        
        return events
    
    def validate_against_history(
        self,
        simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate simulation results against historical events.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            
        Returns:
            Validation report with statistical tests
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'tests_performed': [],
            'overall_validity': None,
            'recommendations': []
        }
        
        # 1. Loss magnitude validation
        loss_validation = self._validate_loss_magnitudes(simulation_results)
        validation_report['loss_magnitude_validation'] = loss_validation
        
        # 2. Time-to-event validation
        timing_validation = self._validate_timing(simulation_results)
        validation_report['timing_validation'] = timing_validation
        
        # 3. Cascade effect validation
        cascade_validation = self._validate_cascade_effects(simulation_results)
        validation_report['cascade_validation'] = cascade_validation
        
        # 4. Recovery pattern validation
        recovery_validation = self._validate_recovery_patterns(simulation_results)
        validation_report['recovery_validation'] = recovery_validation
        
        # 5. Statistical tests
        if SCIPY_AVAILABLE:
            stat_tests = self._perform_statistical_tests(simulation_results)
            validation_report['statistical_tests'] = stat_tests
        
        # Overall assessment
        validation_report['overall_validity'] = self._assess_overall_validity(validation_report)
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(validation_report)
        
        return validation_report
    
    def _validate_loss_magnitudes(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that simulated losses are comparable to historical events."""
        
        # Extract simulated losses
        economic_impact = results.get('economic_impact', {})
        simulated_losses = economic_impact.get('total_losses', {})
        
        # Get percentiles
        p50 = simulated_losses.get('p50', 0)
        p95 = simulated_losses.get('p95', 0)
        p99 = simulated_losses.get('p99', 0)
        
        # Historical loss range (in billions)
        historical_losses = [event.market_cap_loss_usd for event in self.events]
        hist_min = min(historical_losses) / 1e9
        hist_max = max(historical_losses) / 1e9
        hist_median = np.median(historical_losses) / 1e9 if SCIPY_AVAILABLE else sum(historical_losses) / len(historical_losses) / 1e9
        
        # Convert simulated losses to billions
        sim_p50_b = p50 / 1e9
        sim_p95_b = p95 / 1e9
        sim_p99_b = p99 / 1e9
        
        validation = {
            'historical_range_billions': [hist_min, hist_max],
            'historical_median_billions': hist_median,
            'simulated_p50_billions': sim_p50_b,
            'simulated_p95_billions': sim_p95_b,
            'simulated_p99_billions': sim_p99_b,
            'within_historical_range': hist_min <= sim_p95_b <= hist_max * 2,  # Allow 2x for quantum
            'assessment': 'PASS' if hist_min <= sim_p95_b <= hist_max * 2 else 'FAIL',
            'note': 'Quantum attacks could exceed historical losses'
        }
        
        return validation
    
    def _validate_timing(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate attack/collapse timing against historical events."""
        
        attack_scenarios = results.get('attack_scenarios', {})
        time_to_execute = attack_scenarios.get('time_to_execute_hours', {})
        
        # Historical timing (convert to hours)
        historical_timing_days = [
            event.relevant_metrics.get('time_to_collapse_days', 7)
            for event in self.events
            if event.event_type in ['collapse', 'crash']
        ]
        hist_timing_hours = [d * 24 for d in historical_timing_days if d]
        
        # Get simulated timing stats
        sim_mean = time_to_execute.get('mean', 168)  # Default 1 week
        sim_p95 = time_to_execute.get('p95', 336)    # Default 2 weeks
        
        hist_mean = np.mean(hist_timing_hours) if hist_timing_hours else 168
        
        validation = {
            'historical_mean_hours': hist_mean,
            'historical_range_hours': [min(hist_timing_hours), max(hist_timing_hours)] if hist_timing_hours else [24, 720],
            'simulated_mean_hours': sim_mean,
            'simulated_p95_hours': sim_p95,
            'timing_realistic': abs(sim_mean - hist_mean) / hist_mean < 0.5,  # Within 50%
            'assessment': 'PASS' if abs(sim_mean - hist_mean) / hist_mean < 0.5 else 'WARNING'
        }
        
        return validation
    
    def _validate_cascade_effects(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate DeFi cascade and contagion effects."""
        
        economic = results.get('economic_impact', {})
        cascade = economic.get('defi_cascade', {})
        
        # Historical cascade percentages
        hist_cascades = [
            event.relevant_metrics.get('defi_tvl_impact', 0)
            for event in self.events
            if 'defi_tvl_impact' in event.relevant_metrics
        ]
        
        hist_cascade_mean = np.mean(hist_cascades) if hist_cascades else 0.5
        sim_cascade = cascade.get('impact_percentage', 0.3)
        
        validation = {
            'historical_cascade_mean': hist_cascade_mean,
            'simulated_cascade': sim_cascade,
            'terra_luna_cascade': 0.85,  # Reference point
            'cascade_realistic': abs(sim_cascade - hist_cascade_mean) < 0.3,
            'assessment': 'PASS' if abs(sim_cascade - hist_cascade_mean) < 0.3 else 'WARNING'
        }
        
        return validation
    
    def _validate_recovery_patterns(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate recovery time and patterns."""
        
        economic = results.get('economic_impact', {})
        recovery = economic.get('recovery_timeline', {})
        
        # Historical recovery times (days)
        hist_recoveries = [
            event.recovery_time_days
            for event in self.events
            if event.recovery_time_days is not None
        ]
        
        if hist_recoveries:
            hist_recovery_mean = np.mean(hist_recoveries)
            hist_never_recovered = sum(1 for e in self.events if e.recovery_time_days is None)
        else:
            hist_recovery_mean = 180
            hist_never_recovered = 3
        
        sim_recovery_days = recovery.get('expected_days', 365)
        sim_no_recovery_prob = recovery.get('no_recovery_probability', 0.3)
        
        validation = {
            'historical_recovery_mean_days': hist_recovery_mean,
            'historical_never_recovered_count': hist_never_recovered,
            'simulated_recovery_days': sim_recovery_days,
            'simulated_no_recovery_prob': sim_no_recovery_prob,
            'recovery_realistic': sim_recovery_days > hist_recovery_mean,  # Quantum should be worse
            'assessment': 'PASS' if sim_recovery_days > hist_recovery_mean else 'WARNING',
            'note': 'Quantum attacks expected to have longer recovery than traditional'
        }
        
        return validation
    
    def _perform_statistical_tests(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical tests comparing simulated vs historical distributions."""
        
        if not SCIPY_AVAILABLE:
            return {'error': 'Scipy not available for statistical tests'}
        
        tests = {}
        
        # 1. Kolmogorov-Smirnov test for loss distributions
        economic = results.get('economic_impact', {})
        if 'loss_samples' in economic:
            sim_losses = np.array(economic['loss_samples']) / 1e9  # Convert to billions
            hist_losses = np.array([e.market_cap_loss_usd / 1e9 for e in self.events])
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(sim_losses[:len(hist_losses)*10], hist_losses)
            
            tests['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'reject_null': ks_pvalue < 0.05,
                'interpretation': 'Distributions significantly different' if ks_pvalue < 0.05 else 'Distributions similar'
            }
        
        # 2. Anderson-Darling test for normality
        if 'loss_samples' in economic:
            try:
                log_losses = np.log(sim_losses[sim_losses > 0])
                ad_result = stats.anderson(log_losses, dist='norm')
                
                tests['anderson_darling'] = {
                    'statistic': ad_result.statistic,
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': ad_result.significance_level.tolist(),
                    'log_normal': ad_result.statistic < ad_result.critical_values[2],  # 5% level
                    'interpretation': 'Log-normal distribution' if ad_result.statistic < ad_result.critical_values[2] else 'Not log-normal'
                }
            except:
                tests['anderson_darling'] = {'error': 'Test failed'}
        
        # 3. Mann-Whitney U test for severity comparison
        if 'severity_scores' in results:
            sim_severity = results['severity_scores']
            hist_severity = [e.loss_percentage for e in self.events]
            
            if len(sim_severity) >= len(hist_severity):
                u_stat, u_pvalue = stats.mannwhitneyu(
                    sim_severity[:len(hist_severity)*2],
                    hist_severity,
                    alternative='greater'  # Test if sim is more severe
                )
                
                tests['mann_whitney_u'] = {
                    'statistic': u_stat,
                    'p_value': u_pvalue,
                    'quantum_more_severe': u_pvalue < 0.05,
                    'interpretation': 'Quantum attacks more severe' if u_pvalue < 0.05 else 'Similar severity'
                }
        
        return tests
    
    def _assess_overall_validity(
        self,
        report: Dict[str, Any]
    ) -> str:
        """Assess overall validity of simulation based on all tests."""
        
        assessments = []
        
        # Check each validation component
        if 'loss_magnitude_validation' in report:
            assessments.append(report['loss_magnitude_validation'].get('assessment', 'UNKNOWN'))
        
        if 'timing_validation' in report:
            assessments.append(report['timing_validation'].get('assessment', 'UNKNOWN'))
        
        if 'cascade_validation' in report:
            assessments.append(report['cascade_validation'].get('assessment', 'UNKNOWN'))
        
        if 'recovery_validation' in report:
            assessments.append(report['recovery_validation'].get('assessment', 'UNKNOWN'))
        
        # Count results
        pass_count = assessments.count('PASS')
        warning_count = assessments.count('WARNING')
        fail_count = assessments.count('FAIL')
        
        # Overall assessment
        if fail_count > 0:
            return 'NEEDS_CALIBRATION'
        elif warning_count > 2:
            return 'ACCEPTABLE_WITH_CAVEATS'
        elif pass_count >= 3:
            return 'VALIDATED'
        else:
            return 'INCONCLUSIVE'
    
    def _generate_recommendations(
        self,
        report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        # Check loss magnitudes
        if 'loss_magnitude_validation' in report:
            loss_val = report['loss_magnitude_validation']
            if loss_val.get('assessment') == 'FAIL':
                recommendations.append(
                    "Consider adjusting loss parameters - current estimates may be unrealistic"
                )
        
        # Check timing
        if 'timing_validation' in report:
            timing_val = report['timing_validation']
            if not timing_val.get('timing_realistic'):
                recommendations.append(
                    "Review attack timing parameters - may be too fast/slow compared to historical events"
                )
        
        # Check cascades
        if 'cascade_validation' in report:
            cascade_val = report['cascade_validation']
            if not cascade_val.get('cascade_realistic'):
                recommendations.append(
                    "Cascade effects may be under/overestimated - review DeFi interconnection parameters"
                )
        
        # Check recovery
        if 'recovery_validation' in report:
            recovery_val = report['recovery_validation']
            if recovery_val.get('assessment') == 'WARNING':
                recommendations.append(
                    "Recovery timeline may be optimistic - quantum attacks likely harder to recover from"
                )
        
        # Statistical tests
        if 'statistical_tests' in report:
            stats_tests = report['statistical_tests']
            if 'kolmogorov_smirnov' in stats_tests:
                if stats_tests['kolmogorov_smirnov'].get('reject_null'):
                    recommendations.append(
                        "Loss distribution significantly different from historical - verify if intentional"
                    )
        
        if not recommendations:
            recommendations.append("Simulation parameters appear well-calibrated to historical events")
        
        return recommendations


def validate_simulation(
    results_path: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Validate simulation results against historical events.
    
    Args:
        results_path: Path to simulation results JSON
        output_path: Optional path to save validation report
        
    Returns:
        Validation report
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create validator
    validator = HistoricalValidator()
    
    # Run validation
    report = validator.validate_against_history(results)
    
    # Save report if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Test with example data
    example_results = {
        'economic_impact': {
            'total_losses': {
                'p50': 10_000_000_000,  # $10B median
                'p95': 60_000_000_000,  # $60B p95
                'p99': 100_000_000_000  # $100B p99
            },
            'loss_samples': np.random.lognormal(23, 1.5, 1000).tolist(),
            'defi_cascade': {
                'impact_percentage': 0.65
            },
            'recovery_timeline': {
                'expected_days': 500,
                'no_recovery_probability': 0.35
            }
        },
        'attack_scenarios': {
            'time_to_execute_hours': {
                'mean': 240,  # 10 days
                'p95': 720    # 30 days
            }
        },
        'severity_scores': np.random.beta(7, 2, 100).tolist()
    }
    
    validator = HistoricalValidator()
    report = validator.validate_against_history(example_results)
    
    print("\n" + "="*60)
    print("HISTORICAL VALIDATION REPORT")
    print("="*60)
    
    print(f"\nOverall Validity: {report['overall_validity']}")
    
    print("\n📊 Loss Magnitude Validation:")
    loss_val = report['loss_magnitude_validation']
    print(f"   Historical range: ${loss_val['historical_range_billions'][0]:.1f}B - ${loss_val['historical_range_billions'][1]:.1f}B")
    print(f"   Simulated P95: ${loss_val['simulated_p95_billions']:.1f}B")
    print(f"   Assessment: {loss_val['assessment']}")
    
    print("\n⏱️  Timing Validation:")
    timing = report['timing_validation']
    print(f"   Historical mean: {timing['historical_mean_hours']:.0f} hours")
    print(f"   Simulated mean: {timing['simulated_mean_hours']:.0f} hours")
    print(f"   Assessment: {timing['assessment']}")
    
    print("\n📉 Cascade Effects:")
    cascade = report['cascade_validation']
    print(f"   Terra Luna cascade: {cascade['terra_luna_cascade']:.0%}")
    print(f"   Simulated cascade: {cascade['simulated_cascade']:.0%}")
    print(f"   Assessment: {cascade['assessment']}")
    
    print("\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
