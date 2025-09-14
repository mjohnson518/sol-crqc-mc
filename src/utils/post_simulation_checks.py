"""
Post-simulation validation and quality checks.

Validates simulation results for quality, numerical stability,
and statistical significance after completion.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class PostSimulationValidator:
    """
    Validates simulation results after completion.
    
    Performs checks for:
    - Convergence achievement
    - Numerical stability
    - Statistical significance
    - Sanity checks on results
    - Quality scoring
    """
    
    def __init__(self, results: Dict[str, Any], convergence_report: Optional[Dict] = None):
        """
        Initialize validator with simulation results.
        
        Args:
            results: Simulation results dictionary
            convergence_report: Optional convergence analysis report
        """
        self.results = results
        self.convergence_report = convergence_report or results.get('convergence_report', {})
        
        self.validation_results = {
            'convergence_checks': [],
            'stability_checks': [],
            'sanity_checks': [],
            'statistical_checks': [],
            'warnings': [],
            'errors': [],
            'quality_score': None,
            'quality_grade': None
        }
    
    def validate_all(self) -> Tuple[str, Dict]:
        """
        Run all post-simulation validation checks.
        
        Returns:
            Tuple of (quality_grade, validation_report)
        """
        logger.info("Starting post-simulation validation...")
        
        # Run all checks
        self._check_convergence()
        self._check_numerical_stability()
        self._check_sanity()
        self._check_statistical_significance()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        quality_grade = self._score_to_grade(quality_score)
        
        self.validation_results['quality_score'] = quality_score
        self.validation_results['quality_grade'] = quality_grade
        
        # Log results
        self._log_validation_results()
        
        return quality_grade, self.validation_results
    
    def _check_convergence(self) -> None:
        """Check if key metrics have converged."""
        checks = []
        
        if not self.convergence_report:
            self.validation_results['warnings'].append(
                "No convergence report available. Cannot verify convergence."
            )
            checks.append(('Convergence Report', 'MISSING'))
            return
        
        # Overall convergence
        if self.convergence_report.get('overall_convergence'):
            checks.append(('Overall Convergence', 'ACHIEVED'))
        else:
            self.validation_results['warnings'].append(
                "Simulation did not achieve full convergence. Results may be unstable."
            )
            checks.append(('Overall Convergence', 'NOT ACHIEVED'))
        
        # Check individual metrics
        converged = self.convergence_report.get('converged_variables', [])
        non_converged = self.convergence_report.get('non_converged_variables', [])
        
        critical_metrics = ['crqc_year', 'total_economic_loss', 'attack_success_rate']
        
        for metric in critical_metrics:
            if metric in converged:
                checks.append((f'{metric} Convergence', 'OK'))
            elif metric in non_converged:
                self.validation_results['warnings'].append(
                    f"Critical metric '{metric}' did not converge"
                )
                checks.append((f'{metric} Convergence', 'FAILED'))
            else:
                checks.append((f'{metric} Convergence', 'NOT TRACKED'))
        
        # Check recommended iterations
        recommended = self.convergence_report.get('recommended_iterations', 0)
        actual = self.results.get('parameters', {}).get('n_iterations', 0)
        
        if recommended > actual:
            self.validation_results['warnings'].append(
                f"Simulation used {actual:,} iterations but {recommended:,} recommended for convergence"
            )
            checks.append(('Iteration Sufficiency', 'INSUFFICIENT'))
        else:
            checks.append(('Iteration Sufficiency', 'OK'))
        
        self.validation_results['convergence_checks'] = checks
    
    def _check_numerical_stability(self) -> None:
        """Check for numerical stability issues."""
        checks = []
        
        # Check for NaN or Inf values
        has_nan = False
        has_inf = False
        
        def check_for_invalid(obj, path=""):
            nonlocal has_nan, has_inf
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_for_invalid(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_for_invalid(item, f"{path}[{i}]")
            elif isinstance(obj, (int, float)):
                if np.isnan(obj):
                    has_nan = True
                    self.validation_results['errors'].append(f"NaN value found at {path}")
                elif np.isinf(obj):
                    has_inf = True
                    self.validation_results['errors'].append(f"Inf value found at {path}")
        
        check_for_invalid(self.results)
        
        if has_nan:
            checks.append(('NaN Values', 'FOUND'))
        else:
            checks.append(('NaN Values', 'NONE'))
        
        if has_inf:
            checks.append(('Inf Values', 'FOUND'))
        else:
            checks.append(('Inf Values', 'NONE'))
        
        # Check for extreme outliers
        if 'quantum_timeline' in self.results:
            timeline = self.results['quantum_timeline']
            if 'crqc_years' in timeline:
                years = timeline['crqc_years']
                if years:
                    mean = np.mean(years)
                    std = np.std(years)
                    
                    # Check for extreme outliers (> 5 sigma)
                    outliers = [y for y in years if abs(y - mean) > 5 * std]
                    
                    if outliers:
                        pct = len(outliers) / len(years) * 100
                        if pct > 1:
                            self.validation_results['warnings'].append(
                                f"{pct:.1f}% of CRQC years are extreme outliers (>5σ)"
                            )
                            checks.append(('Extreme Outliers', f'{pct:.1f}%'))
                        else:
                            checks.append(('Extreme Outliers', f'{pct:.2f}%'))
                    else:
                        checks.append(('Extreme Outliers', 'NONE'))
        
        # Check coefficient of variation for stability
        key_metrics = {
            'CRQC Year': self.results.get('quantum_timeline', {}).get('crqc_years', []),
            'Economic Loss': self.results.get('economic_impact', {}).get('total_losses', []),
            'Attack Probability': self.results.get('attack_analysis', {}).get('probabilities', [])
        }
        
        for metric_name, values in key_metrics.items():
            if values and len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                
                if mean != 0:
                    cv = std / abs(mean)
                    if cv > 2:
                        self.validation_results['warnings'].append(
                            f"{metric_name} has very high variance (CV={cv:.2f})"
                        )
                        checks.append((f'{metric_name} Stability', 'UNSTABLE'))
                    elif cv > 1:
                        checks.append((f'{metric_name} Stability', 'MARGINAL'))
                    else:
                        checks.append((f'{metric_name} Stability', 'STABLE'))
        
        self.validation_results['stability_checks'] = checks
    
    def _check_sanity(self) -> None:
        """Perform sanity checks on results."""
        checks = []
        
        # Check CRQC timeline sanity
        if 'quantum_timeline' in self.results:
            timeline = self.results['quantum_timeline']
            
            if 'crqc_years' in timeline and timeline['crqc_years']:
                years = timeline['crqc_years']
                mean_year = np.mean(years)
                
                # Check if CRQC year is reasonable (2025-2070)
                if mean_year < 2025:
                    self.validation_results['errors'].append(
                        f"CRQC year in the past: {mean_year:.1f}"
                    )
                    checks.append(('CRQC Year Range', 'INVALID'))
                elif mean_year > 2070:
                    self.validation_results['warnings'].append(
                        f"CRQC year very far in future: {mean_year:.1f}"
                    )
                    checks.append(('CRQC Year Range', 'QUESTIONABLE'))
                else:
                    checks.append(('CRQC Year Range', 'OK'))
                
                # Check distribution shape
                skewness = stats.skew(years)
                if abs(skewness) > 2:
                    self.validation_results['warnings'].append(
                        f"CRQC year distribution highly skewed: {skewness:.2f}"
                    )
                    checks.append(('CRQC Distribution', 'SKEWED'))
                else:
                    checks.append(('CRQC Distribution', 'NORMAL'))
        
        # Check economic losses sanity
        if 'economic_impact' in self.results:
            econ = self.results['economic_impact']
            
            if 'total_losses' in econ and econ['total_losses']:
                losses = econ['total_losses']
                mean_loss = np.mean(losses)
                max_loss = np.max(losses)
                
                # Check if losses are reasonable (not exceeding total Solana market cap * 10)
                # Assuming max reasonable loss is $10 trillion
                if max_loss > 1e13:  # $10 trillion
                    self.validation_results['warnings'].append(
                        f"Maximum loss exceeds $10T: ${max_loss/1e12:.1f}T"
                    )
                    checks.append(('Economic Loss Range', 'EXTREME'))
                elif mean_loss < 0:
                    self.validation_results['errors'].append(
                        f"Negative economic losses detected: ${mean_loss/1e9:.1f}B"
                    )
                    checks.append(('Economic Loss Range', 'INVALID'))
                else:
                    checks.append(('Economic Loss Range', 'OK'))
                
                # Check for reasonable variance
                if losses:
                    min_loss = np.min(losses)
                    if max_loss > 0 and min_loss > 0:
                        ratio = max_loss / min_loss
                        if ratio > 10000:
                            self.validation_results['warnings'].append(
                                f"Extreme variance in losses: {ratio:.0f}x range"
                            )
                            checks.append(('Loss Variance', 'EXTREME'))
                        else:
                            checks.append(('Loss Variance', 'OK'))
        
        # Check attack probabilities
        if 'attack_analysis' in self.results:
            attack = self.results['attack_analysis']
            
            if 'max_probability' in attack:
                max_prob = attack['max_probability']
                
                if max_prob < 0 or max_prob > 1:
                    self.validation_results['errors'].append(
                        f"Invalid probability value: {max_prob:.2f}"
                    )
                    checks.append(('Probability Values', 'INVALID'))
                else:
                    checks.append(('Probability Values', 'OK'))
        
        # Check iteration success rate
        params = self.results.get('parameters', {})
        total_iter = params.get('n_iterations', 0)
        successful_iter = params.get('successful_iterations', 0)
        
        if total_iter > 0:
            success_rate = successful_iter / total_iter
            
            if success_rate < 0.5:
                self.validation_results['errors'].append(
                    f"Low success rate: {success_rate:.1%} of iterations succeeded"
                )
                checks.append(('Success Rate', 'LOW'))
            elif success_rate < 0.8:
                self.validation_results['warnings'].append(
                    f"Moderate success rate: {success_rate:.1%} of iterations succeeded"
                )
                checks.append(('Success Rate', 'MODERATE'))
            else:
                checks.append(('Success Rate', f'OK ({success_rate:.1%})'))
        
        self.validation_results['sanity_checks'] = checks
    
    def _check_statistical_significance(self) -> None:
        """Check statistical significance of results."""
        checks = []
        
        # Check sample size
        params = self.results.get('parameters', {})
        n_iterations = params.get('successful_iterations', 0)
        
        if n_iterations < 30:
            self.validation_results['errors'].append(
                f"Sample size too small for statistical significance: {n_iterations}"
            )
            checks.append(('Sample Size', 'TOO SMALL'))
        elif n_iterations < 100:
            self.validation_results['warnings'].append(
                f"Small sample size may limit statistical power: {n_iterations}"
            )
            checks.append(('Sample Size', 'SMALL'))
        elif n_iterations < 1000:
            checks.append(('Sample Size', 'ADEQUATE'))
        else:
            checks.append(('Sample Size', 'GOOD'))
        
        # Check confidence intervals
        key_metrics = ['crqc_year', 'total_economic_loss', 'attack_success_rate']
        
        for metric in key_metrics:
            if metric in self.convergence_report.get('metrics', {}):
                metric_data = self.convergence_report['metrics'][metric]
                
                if 'confidence_interval_95' in metric_data:
                    ci = metric_data['confidence_interval_95']
                    mean = metric_data.get('running_mean', 0)
                    
                    if mean != 0:
                        ci_width = (ci[1] - ci[0]) / abs(mean)
                        
                        if ci_width > 1:
                            self.validation_results['warnings'].append(
                                f"{metric} has very wide confidence interval: ±{ci_width*50:.1f}%"
                            )
                            checks.append((f'{metric} CI', 'WIDE'))
                        elif ci_width > 0.5:
                            checks.append((f'{metric} CI', 'MODERATE'))
                        else:
                            checks.append((f'{metric} CI', 'NARROW'))
        
        # Check for statistical tests if available
        if 'statistical_tests' in self.results:
            tests = self.results['statistical_tests']
            
            for test_name, test_result in tests.items():
                p_value = test_result.get('p_value', 1.0)
                
                if p_value < 0.05:
                    checks.append((test_name, f'SIGNIFICANT (p={p_value:.3f})'))
                else:
                    checks.append((test_name, f'NOT SIGNIFICANT (p={p_value:.3f})'))
        
        self.validation_results['statistical_checks'] = checks
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100).
        
        Returns:
            Quality score from 0 to 100
        """
        score = 100.0
        
        # Convergence (30 points)
        if self.convergence_report.get('overall_convergence'):
            convergence_penalty = 0
        else:
            # Penalize based on number of non-converged metrics
            non_converged = len(self.convergence_report.get('non_converged_variables', []))
            total_metrics = len(self.convergence_report.get('metrics', {}))
            
            if total_metrics > 0:
                convergence_penalty = (non_converged / total_metrics) * 30
            else:
                convergence_penalty = 30
        
        score -= convergence_penalty
        
        # Numerical stability (20 points)
        stability_issues = sum(1 for check, status in self.validation_results['stability_checks']
                             if status in ['FOUND', 'UNSTABLE', 'EXTREME'])
        stability_penalty = min(20, stability_issues * 5)
        score -= stability_penalty
        
        # Sanity checks (20 points)
        sanity_issues = sum(1 for check, status in self.validation_results['sanity_checks']
                          if status in ['INVALID', 'EXTREME', 'LOW'])
        sanity_penalty = min(20, sanity_issues * 5)
        score -= sanity_penalty
        
        # Statistical significance (20 points)
        stat_issues = sum(1 for check, status in self.validation_results['statistical_checks']
                        if status in ['TOO SMALL', 'SMALL', 'WIDE'])
        stat_penalty = min(20, stat_issues * 5)
        score -= stat_penalty
        
        # Error count (10 points)
        error_penalty = min(10, len(self.validation_results['errors']) * 2)
        score -= error_penalty
        
        # Warning count (bonus/penalty)
        warning_adjustment = min(5, len(self.validation_results['warnings']) * 0.5)
        score -= warning_adjustment
        
        return max(0, min(100, score))
    
    def _score_to_grade(self, score: float) -> str:
        """
        Convert numerical score to letter grade.
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Letter grade A-F
        """
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _log_validation_results(self) -> None:
        """Log validation results."""
        logger.info("=" * 60)
        logger.info("POST-SIMULATION VALIDATION REPORT")
        logger.info("=" * 60)
        
        # Quality grade
        grade = self.validation_results['quality_grade']
        score = self.validation_results['quality_score']
        
        grade_emoji = {'A': '🌟', 'B': '✅', 'C': '📊', 'D': '⚠️', 'F': '❌'}
        logger.info(f"QUALITY GRADE: {grade} ({score:.1f}/100) {grade_emoji.get(grade, '')}")
        logger.info("")
        
        # Convergence checks
        if self.validation_results['convergence_checks']:
            logger.info("Convergence Checks:")
            for check, status in self.validation_results['convergence_checks']:
                symbol = "✓" if "OK" in status or "ACHIEVED" in status else "✗"
                logger.info(f"  {symbol} {check}: {status}")
        
        # Stability checks
        if self.validation_results['stability_checks']:
            logger.info("\nNumerical Stability:")
            for check, status in self.validation_results['stability_checks']:
                symbol = "✓" if status in ["NONE", "STABLE", "OK"] else "⚠"
                logger.info(f"  {symbol} {check}: {status}")
        
        # Sanity checks
        if self.validation_results['sanity_checks']:
            logger.info("\nSanity Checks:")
            for check, status in self.validation_results['sanity_checks']:
                symbol = "✓" if "OK" in status else "⚠" if "MODERATE" in status else "✗"
                logger.info(f"  {symbol} {check}: {status}")
        
        # Statistical checks
        if self.validation_results['statistical_checks']:
            logger.info("\nStatistical Significance:")
            for check, status in self.validation_results['statistical_checks']:
                logger.info(f"  • {check}: {status}")
        
        # Warnings
        if self.validation_results['warnings']:
            logger.warning("\nWarnings:")
            for warning in self.validation_results['warnings']:
                logger.warning(f"  ⚠ {warning}")
        
        # Errors
        if self.validation_results['errors']:
            logger.error("\nErrors:")
            for error in self.validation_results['errors']:
                logger.error(f"  ✗ {error}")
        
        # Interpretation
        logger.info("\n" + "=" * 60)
        logger.info("INTERPRETATION:")
        
        if grade == 'A':
            logger.info("✨ Excellent quality! Results are suitable for publication and executive presentation.")
        elif grade == 'B':
            logger.info("✅ Good quality. Results are reliable for internal reports and decision support.")
        elif grade == 'C':
            logger.info("📊 Acceptable quality. Results can be used for preliminary analysis with caveats.")
        elif grade == 'D':
            logger.info("⚠️ Poor quality. Results should be used with caution. Consider re-running with more iterations.")
        else:
            logger.info("❌ Unacceptable quality. Results are not reliable. Please address errors and re-run.")
        
        logger.info("=" * 60)
    
    def generate_quality_report(self, output_path: Optional[Path] = None) -> Dict:
        """
        Generate detailed quality report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Quality report dictionary
        """
        report = {
            'timestamp': str(Path.ctime(Path.cwd())),
            'quality_grade': self.validation_results['quality_grade'],
            'quality_score': self.validation_results['quality_score'],
            'checks': {
                'convergence': self.validation_results['convergence_checks'],
                'stability': self.validation_results['stability_checks'],
                'sanity': self.validation_results['sanity_checks'],
                'statistical': self.validation_results['statistical_checks']
            },
            'issues': {
                'errors': self.validation_results['errors'],
                'warnings': self.validation_results['warnings']
            },
            'suitable_for': self._get_suitability()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Quality report saved to {output_path}")
        
        return report
    
    def _get_suitability(self) -> List[str]:
        """Determine what the results are suitable for based on quality."""
        grade = self.validation_results['quality_grade']
        
        suitability_map = {
            'A': ['Publication', 'Executive Presentation', 'Strategic Decision', 'External Audit'],
            'B': ['Internal Report', 'Decision Support', 'Risk Assessment', 'Planning'],
            'C': ['Preliminary Analysis', 'Development', 'Testing', 'Exploration'],
            'D': ['Debugging', 'Method Validation'],
            'F': ['Not Suitable for Use']
        }
        
        return suitability_map.get(grade, ['Unknown'])
