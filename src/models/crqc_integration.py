"""
Integration layer for CRQC models with real-time calibration.

This module connects the real-time calibration system with the Bayesian,
competing risks, and quantum research models to provide continuously updated
CRQC predictions.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import math

from .quantum_research import QuantumResearchDatabase
from .bayesian_crqc import HierarchicalBayesianCRQC
from .competing_risks_crqc import CompetingRisksCRQC
from .realtime_calibration import RealTimeCalibrator, DataPoint
from .geopolitical_model import GeopoliticalModel
from .breakthrough_detector import BreakthroughDetector
from src.analysis.uncertainty_quantification import UncertaintyAnalysis, UncertaintyReport

logger = logging.getLogger(__name__)


@dataclass
class CRQCPrediction:
    """Unified CRQC prediction combining all models."""
    
    timestamp: datetime
    median_year: float
    confidence_interval_90: Tuple[float, float]
    confidence_interval_95: Tuple[float, float]
    probability_by_year: Dict[int, float]
    dominant_pathway: str
    risk_factors: Dict[str, float]
    confidence_score: float
    data_sources: List[str]
    anomaly_warnings: List[str]
    uncertainty_report: Optional[UncertaintyReport] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'median_year': self.median_year,
            'confidence_interval_90': self.confidence_interval_90,
            'confidence_interval_95': self.confidence_interval_95,
            'probability_by_year': self.probability_by_year,
            'dominant_pathway': self.dominant_pathway,
            'risk_factors': self.risk_factors,
            'confidence_score': self.confidence_score,
            'data_sources': self.data_sources,
            'anomaly_warnings': self.anomaly_warnings,
            'uncertainty_report': self.uncertainty_report.to_dict() if self.uncertainty_report else None,
        }


class CRQCIntegratedModel:
    """
    Integrated CRQC prediction model combining multiple approaches with
    real-time calibration.
    """
    
    def __init__(self, 
                 enable_realtime: bool = True,
                 cache_dir: Path = Path("data/cache/crqc")):
        """
        Initialize the integrated CRQC model.
        
        Args:
            enable_realtime: Whether to enable real-time updates
            cache_dir: Directory for caching model states
        """
        self.enable_realtime = enable_realtime
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component models
        self.quantum_research = QuantumResearchDatabase()
        self.bayesian_model = HierarchicalBayesianCRQC()
        self.competing_risks = CompetingRisksCRQC()
        self.geopolitical_model = GeopoliticalModel()
        self.breakthrough_detector = BreakthroughDetector()
        
        # Initialize real-time calibrator if enabled
        if self.enable_realtime:
            self.calibrator = RealTimeCalibrator()
        else:
            self.calibrator = None
        
        # Uncertainty analysis helper
        self.uncertainty_analyzer = UncertaintyAnalysis()
        self._latest_model_predictions: Dict[str, Dict[str, Any]] = {}
        
        # Model weights for ensemble
        self.model_weights = {
            'bayesian': 0.35,
            'competing_risks': 0.30,
            'quantum_research': 0.20,
            'geopolitical': 0.15,
        }
        
        # Calibration state
        self.last_calibration = None
        self.calibration_adjustments = {}
        
        # Load cached state if available
        self._load_state()
    
    def _load_state(self):
        """Load cached model state."""
        
        state_file = self.cache_dir / "integrated_model_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore model weights
                self.model_weights = state.get('model_weights', self.model_weights)
                
                # Restore calibration adjustments
                self.calibration_adjustments = state.get('calibration_adjustments', {})
                
                # Restore last calibration time
                if 'last_calibration' in state:
                    self.last_calibration = datetime.fromisoformat(state['last_calibration'])
                
                logger.info("Loaded integrated model state from cache")
                
            except Exception as e:
                logger.error(f"Error loading model state: {e}")
    
    def _save_state(self):
        """Save current model state."""
        
        state = {
            'model_weights': self.model_weights,
            'calibration_adjustments': self.calibration_adjustments,
            'last_calibration': self.last_calibration.isoformat() if self.last_calibration else None,
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = self.cache_dir / "integrated_model_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
    
    async def update_with_realtime_data(self):
        """Update models with real-time calibration data."""
        
        if not self.calibrator:
            logger.warning("Real-time calibration not enabled")
            return
        
        logger.info("Updating models with real-time data...")
        
        # Run calibration cycle
        update = await self.calibrator.run_calibration_cycle()
        
        # Get calibration adjustments
        self.calibration_adjustments = self.calibrator.get_current_adjustments()
        
        # Update Bayesian model with new data
        self._update_bayesian_model(self.calibrator.recent_data)
        
        # Update competing risks parameters
        self._update_competing_risks(update)
        
        # Update breakthrough detector using latest signals
        self._update_breakthrough_detector(self.calibrator.recent_data)
        
        # Adjust model weights based on recent performance
        self._update_model_weights(update)
        
        self.last_calibration = datetime.now()
        self._save_state()
        
        logger.info(f"Real-time update complete. Timeline adjustment: "
                   f"{self.calibration_adjustments.get('timeline_adjustment', 0):+.2f} years")
    
    def _update_bayesian_model(self, data_points: List[DataPoint]):
        """Update Bayesian model with new observations."""
        
        # Convert data points to observations
        for dp in data_points:
            if dp.data_type == "announcement" and "qubits" in dp.content.get("metrics", {}):
                # Extract company and technology info
                company = dp.content.get("company", "unknown")
                qubits = dp.content["metrics"]["qubits"]
                
                # Infer technology type from content
                technology = self._infer_technology(dp.content)
                
                # Create observation
                observation = {
                    'company': company,
                    'technology': technology,
                    'metric': 'logical_qubits',
                    'value': qubits,
                    'date': dp.timestamp
                }
                
                # Update Bayesian model
                self.bayesian_model.update_with_observation(observation)
        
        logger.info(f"Updated Bayesian model with {len(data_points)} data points")
    
    def _infer_technology(self, content: Dict[str, Any]) -> str:
        """Infer quantum technology type from content."""
        
        text = str(content).lower()
        
        if any(term in text for term in ["superconducting", "transmon", "josephson"]):
            return "superconducting"
        elif any(term in text for term in ["trapped ion", "ion trap"]):
            return "trapped_ion"
        elif any(term in text for term in ["neutral atom", "rydberg"]):
            return "neutral_atom"
        elif any(term in text for term in ["photonic", "optical"]):
            return "photonic"
        elif any(term in text for term in ["topological", "majorana"]):
            return "topological"
        else:
            return "superconducting"  # Default
    
    def _update_competing_risks(self, update):
        """Update competing risks model parameters."""
        
        # Adjust pathway hazards based on recent developments
        adjustments = self.calibration_adjustments
        
        # If timeline is accelerating, increase hazards
        timeline_factor = 1.0 - adjustments.get('timeline_adjustment', 0) / 10.0
        
        # Update breakthrough probability
        breakthrough_mult = adjustments.get('breakthrough_probability_multiplier', 1.0)
        
        # Apply adjustments to competing risks model
        # This would require adding update methods to CompetingRisksCRQC
        # For now, we'll store adjustments for use during prediction
        
        logger.info(f"Updated competing risks with timeline factor: {timeline_factor:.2f}")
    
    def _update_model_weights(self, update):
        """Dynamically adjust model weights based on performance."""
        
        # Simple adaptive weighting based on anomaly detection
        if update.anomalies_detected:
            # If anomalies detected, increase weight on Bayesian (more adaptive)
            self.model_weights['bayesian'] = min(0.5, self.model_weights['bayesian'] + 0.05)
            self.model_weights['competing_risks'] = max(0.3, self.model_weights['competing_risks'] - 0.025)
            self.model_weights['quantum_research'] = max(0.2, self.model_weights['quantum_research'] - 0.025)
        
        # Normalize weights
        total = sum(self.model_weights.values())
        for key in self.model_weights:
            self.model_weights[key] /= total
    
    def generate_prediction(self, 
                          current_year: int = 2025,
                          horizon_years: int = 25) -> CRQCPrediction:
        """
        Generate unified CRQC prediction combining all models.
        
        Args:
            current_year: Current year for prediction
            horizon_years: How many years ahead to predict
            
        Returns:
            CRQCPrediction object with ensemble results
        """
        
        logger.info("Generating integrated CRQC prediction...")
        
        # Get predictions from each model
        predictions = {}
        
        # 1. Quantum Research baseline
        qr_prediction = self._get_quantum_research_prediction(current_year, horizon_years)
        predictions['quantum_research'] = qr_prediction
        self._latest_model_predictions['quantum_research'] = qr_prediction
        
        # 2. Bayesian prediction
        bayesian_prediction = self._get_bayesian_prediction(current_year, horizon_years)
        predictions['bayesian'] = bayesian_prediction
        self._latest_model_predictions['bayesian'] = bayesian_prediction
        
        # 3. Competing risks prediction
        cr_prediction = self._get_competing_risks_prediction(current_year, horizon_years)
        predictions['competing_risks'] = cr_prediction
        self._latest_model_predictions['competing_risks'] = cr_prediction
        
        # 4. Geopolitical prediction
        geopolitical_prediction = self._get_geopolitical_prediction(current_year, horizon_years)
        predictions['geopolitical'] = geopolitical_prediction
        self._latest_model_predictions['geopolitical'] = predictions['geopolitical']

        # Breakthrough signal prediction
        predictions['breakthrough_detection'] = self._get_breakthrough_prediction(current_year, horizon_years)
        self._latest_model_predictions['breakthrough_detection'] = predictions['breakthrough_detection']
        
        # Apply calibration adjustments
        adjustments = self._apply_calibration_adjustments(predictions)
        
        # Combine predictions using weighted ensemble
        ensemble_prediction = self._ensemble_predictions(adjustments, current_year, horizon_years)
        
        # Attach uncertainty quantification
        ensemble_prediction.uncertainty_report = self.uncertainty_analyzer.analyze(
            model_predictions=self._latest_model_predictions,
            ensemble_prediction=ensemble_prediction,
        )
        
        # Add warnings based on recent calibration
        if self.calibrator and self.calibrator.anomaly_buffer:
            ensemble_prediction.anomaly_warnings = list(self.calibrator.anomaly_buffer)[-5:]
        breakthrough_assessment = self.breakthrough_detector.latest_assessment()
        if breakthrough_assessment:
            ensemble_prediction.anomaly_warnings.extend(breakthrough_assessment.alerts[:3])
        
        return ensemble_prediction
    
    def _get_quantum_research_prediction(self, 
                                       current_year: int,
                                       horizon_years: int) -> Dict[str, Any]:
        """Get prediction from quantum research model."""
        
        # Calculate requirements
        requirements = self.quantum_research.calculate_revised_requirements()
        
        # Project timeline based on current capabilities
        current_qubits = 433  # IBM Condor as of late 2024
        required_qubits = requirements['ed25519_logical']
        
        # Use exponential growth projection
        growth_rate = 1.5  # 50% annual growth
        years_to_crqc = np.log(required_qubits / current_qubits) / np.log(growth_rate)
        
        median_year = current_year + years_to_crqc
        
        # Generate probability distribution
        prob_by_year = {}
        for year in range(current_year, current_year + horizon_years):
            # Logistic curve
            t = year - median_year
            prob_by_year[year] = 1 / (1 + np.exp(-0.5 * t))
        
        return {
            'median_year': median_year,
            'probability_by_year': prob_by_year,
            'dominant_pathway': 'traditional_shors',
            'required_qubits': required_qubits
        }
    
    def _get_bayesian_prediction(self,
                                current_year: int,
                                horizon_years: int) -> Dict[str, Any]:
        """Get prediction from Bayesian model."""
        
        # Get prediction from Bayesian model
        prediction = self.bayesian_model.predict(include_breakthroughs=True)
        
        # Extract median year
        median_year = prediction['median']
        
        # Use probability_by_year if available, otherwise generate
        if 'probability_by_year' in prediction:
            prob_by_year = prediction['probability_by_year']
        else:
            # Fallback: generate from median and std
            prob_by_year = {}
            std = prediction.get('std', 3.0)
            for year in range(current_year, current_year + horizon_years):
                # Logistic curve around median
                z = (year - median_year) / std
                prob_by_year[year] = 1 / (1 + np.exp(-z))
        
        # Determine dominant technology from contributions
        tech_contributions = prediction.get('technology_contributions', {})
        if tech_contributions:
            # Find technology with highest positive effect (earliest CRQC)
            dominant_tech = min(tech_contributions.items(), 
                              key=lambda x: x[1]['mean_effect'])[0]
        else:
            dominant_tech = 'superconducting'
        
        return {
            'median_year': median_year,
            'probability_by_year': prob_by_year,
            'dominant_pathway': 'optimized_shors',
            'technologies': list(tech_contributions.keys()) if tech_contributions else ['superconducting']
        }
    
    def _get_competing_risks_prediction(self,
                                      current_year: int,
                                      horizon_years: int) -> Dict[str, Any]:
        """Get prediction from competing risks model."""
        
        # Generate timeline
        years = list(range(1, horizon_years + 1))
        
        # Get cumulative incidence for each pathway
        pathways = self.competing_risks.pathways
        cumulative_incidence = {}
        
        for pathway_name, pathway in pathways.items():
            # Calculate cumulative incidence at all time points
            ci_array = self.competing_risks.calculate_cumulative_incidence(
                pathway=pathway,
                time_points=np.array(years)
            )
            cumulative_incidence[pathway_name] = ci_array.tolist()
        
        # Find dominant pathway
        final_incidences = {p: ci[-1] for p, ci in cumulative_incidence.items()}
        dominant_pathway = max(final_incidences, key=final_incidences.get)
        
        # Calculate overall CRQC probability by year
        prob_by_year = {}
        for i, year in enumerate(range(current_year, current_year + horizon_years)):
            # Sum of all pathway probabilities (accounting for competing events)
            total_prob = sum(ci[i] for ci in cumulative_incidence.values())
            prob_by_year[year] = min(1.0, total_prob)
        
        # Find median year (50% probability)
        median_year = current_year + horizon_years  # Default
        for year, prob in prob_by_year.items():
            if prob >= 0.5:
                median_year = float(year)
                break
        
        return {
            'median_year': median_year,
            'probability_by_year': prob_by_year,
            'dominant_pathway': dominant_pathway,
            'pathway_probabilities': final_incidences
        }
    
    def _get_geopolitical_prediction(self,
                                     current_year: int,
                                     horizon_years: int) -> Dict[str, Any]:
        metrics = self.geopolitical_model.compute_geopolitical_adjustment(years=horizon_years)
        median_adjustment = metrics['timeline_adjustment']
        base_median = current_year + horizon_years / 2
        median_year = base_median + median_adjustment

        prob_by_year = {}
        for year in range(current_year, current_year + horizon_years):
            delta = year - median_year
            prob = 1 / (1 + math.exp(-delta / 2))
            prob_by_year[year] = prob

        return {
            'median_year': median_year,
            'probability_by_year': prob_by_year,
            'dominant_pathway': 'geopolitical_acceleration',
            'metrics': metrics,
        }
    
    def _get_breakthrough_prediction(self,
                                     current_year: int,
                                     horizon_years: int) -> Dict[str, Any]:
        assessment = self.breakthrough_detector.latest_assessment()
        if assessment is None:
            base_prob = 0.02
            composite_score = 0.0
        else:
            composite_score = assessment.composite_score
            base_prob = min(0.5, 0.02 * math.exp(composite_score / 2))

        prob_by_year = {}
        cumulative = 0.0
        for idx, year in enumerate(range(current_year, current_year + horizon_years)):
            annual_prob = min(0.9, base_prob * (1 + 0.1 * idx))
            cumulative = 1 - (1 - cumulative) * (1 - annual_prob)
            prob_by_year[year] = cumulative

        median_year = current_year + horizon_years
        for year, cumulative_prob in prob_by_year.items():
            if cumulative_prob >= 0.5:
                median_year = year
                break

        return {
            'median_year': float(median_year),
            'probability_by_year': prob_by_year,
            'dominant_pathway': 'breakthrough_detection',
            'composite_score': composite_score,
        }
    
    def _apply_calibration_adjustments(self, 
                                     predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply real-time calibration adjustments to predictions."""
        
        if not self.calibration_adjustments:
            return predictions
        
        timeline_adj = self.calibration_adjustments.get('timeline_adjustment', 0)
        confidence_mult = self.calibration_adjustments.get('confidence_multiplier', 1.0)
        
        # Apply adjustments to each model's predictions
        for model_name, pred in predictions.items():
            # Adjust median year
            pred['median_year'] += timeline_adj
            
            # Adjust probability distribution
            if 'probability_by_year' in pred:
                # Shift probabilities by timeline adjustment
                adjusted_probs = {}
                for year, prob in pred['probability_by_year'].items():
                    # Apply confidence multiplier and timeline shift
                    adj_year = year + int(timeline_adj)
                    if adj_year in pred['probability_by_year']:
                        adjusted_probs[year] = prob * confidence_mult
                    else:
                        adjusted_probs[year] = prob
                
                pred['probability_by_year'] = adjusted_probs
        
        return predictions
    
    def _ensemble_predictions(self,
                            predictions: Dict[str, Dict[str, Any]],
                            current_year: int,
                            horizon_years: int) -> CRQCPrediction:
        """Combine predictions using weighted ensemble."""
        
        # Calculate weighted median year
        median_years = []
        weights = []
        
        for model_name, pred in predictions.items():
            median_years.append(pred['median_year'])
            weights.append(self.model_weights.get(model_name, 0.33))
        
        ensemble_median = np.average(median_years, weights=weights)
        
        # Calculate ensemble probability distribution
        ensemble_probs = {}
        
        for year in range(current_year, current_year + horizon_years):
            year_probs = []
            year_weights = []
            
            for model_name, pred in predictions.items():
                if year in pred.get('probability_by_year', {}):
                    year_probs.append(pred['probability_by_year'][year])
                    year_weights.append(self.model_weights.get(model_name, 0.33))
            
            if year_probs:
                ensemble_probs[year] = np.average(year_probs, weights=year_weights)
            else:
                ensemble_probs[year] = 0.0
        
        # Calculate confidence intervals using bootstrap
        ci_90, ci_95 = self._calculate_confidence_intervals(
            median_years, weights, ensemble_median
        )
        
        # Determine dominant pathway (from model with highest weight)
        dominant_model = max(self.model_weights, key=self.model_weights.get)
        dominant_pathway = predictions[dominant_model].get('dominant_pathway', 'unknown')
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(predictions)
        geopolitical_metrics = self.geopolitical_model.get_cached_metrics()
        if geopolitical_metrics:
            risk_factors['geopolitical_investment_growth'] = geopolitical_metrics['investment_growth']
            risk_factors['geopolitical_timeline_shift'] = geopolitical_metrics['timeline_adjustment']

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(predictions)
        
        # Data sources
        data_sources = list(predictions.keys())
        if self.calibrator:
            data_sources.extend(['arxiv', 'quantum_news', 'github'])
        
        return CRQCPrediction(
            timestamp=datetime.now(),
            median_year=ensemble_median,
            confidence_interval_90=ci_90,
            confidence_interval_95=ci_95,
            probability_by_year=ensemble_probs,
            dominant_pathway=dominant_pathway,
            risk_factors=risk_factors,
            confidence_score=confidence_score,
            data_sources=data_sources,
            anomaly_warnings=[]
        )
    
    def _calculate_confidence_intervals(self,
                                      values: List[float],
                                      weights: List[float],
                                      center: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate confidence intervals for ensemble prediction."""
        
        # Simple weighted standard deviation
        weights = np.array(weights) / np.sum(weights)
        variance = np.sum(weights * (np.array(values) - center) ** 2)
        std = np.sqrt(variance)
        
        # Approximate confidence intervals
        ci_90 = (center - 1.645 * std, center + 1.645 * std)
        ci_95 = (center - 1.96 * std, center + 1.96 * std)
        
        return ci_90, ci_95
    
    def _calculate_risk_factors(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate key risk factors from predictions."""
        
        risk_factors = {}
        
        # Technology readiness (from Bayesian)
        if 'bayesian' in predictions:
            techs = predictions['bayesian'].get('technologies', [])
            risk_factors['technology_diversity'] = len(techs) / 5.0  # Normalize by max techs
        
        # Pathway concentration (from competing risks)
        if 'competing_risks' in predictions:
            pathway_probs = predictions['competing_risks'].get('pathway_probabilities', {})
            if pathway_probs:
                max_prob = max(pathway_probs.values())
                risk_factors['pathway_concentration'] = max_prob
        
        # Timeline uncertainty
        median_years = [p['median_year'] for p in predictions.values()]
        if len(median_years) > 1:
            risk_factors['prediction_variance'] = np.std(median_years) / np.mean(median_years)
        
        # Calibration impact
        if self.calibration_adjustments:
            risk_factors['calibration_impact'] = abs(
                self.calibration_adjustments.get('timeline_adjustment', 0)
            ) / 5.0  # Normalize by 5 years

        # Geopolitical metrics
        if 'geopolitical_investment_growth' not in risk_factors and 'geopolitical' in predictions:
            geo_metrics = predictions['geopolitical'].get('metrics', {})
            if geo_metrics:
                risk_factors['geopolitical_investment_growth'] = geo_metrics.get('investment_growth', 0.0)
                risk_factors['geopolitical_timeline_shift'] = geo_metrics.get('timeline_adjustment', 0.0)

        # Breakthrough risk score
        if 'breakthrough_detection' in predictions:
            risk_factors['breakthrough_risk_score'] = predictions['breakthrough_detection'].get('composite_score', 0.0)
        
        return risk_factors
    
    def _calculate_confidence_score(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence score for ensemble prediction."""
        
        confidence = 0.8  # Base confidence
        
        # Reduce confidence if predictions disagree
        median_years = [p['median_year'] for p in predictions.values()]
        if len(median_years) > 1:
            cv = np.std(median_years) / np.mean(median_years)
            confidence *= (1 - min(0.5, cv))  # Reduce by up to 50%
        
        # Boost confidence if we have recent calibration
        if self.last_calibration:
            days_since = (datetime.now() - self.last_calibration).days
            if days_since < 7:
                confidence *= 1.1
            elif days_since > 30:
                confidence *= 0.9
        
        # Apply calibration confidence multiplier
        if self.calibration_adjustments:
            confidence *= self.calibration_adjustments.get('confidence_multiplier', 1.0)
        
        return min(1.0, max(0.1, confidence))
    
    def generate_report(self, prediction: CRQCPrediction) -> str:
        """Generate human-readable report from prediction."""
        
        report = []
        report.append("=" * 60)
        report.append("INTEGRATED CRQC PREDICTION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Median CRQC Emergence: {prediction.median_year:.0f}")
        report.append(f"90% Confidence Interval: {prediction.confidence_interval_90[0]:.0f} - "
                     f"{prediction.confidence_interval_90[1]:.0f}")
        report.append(f"95% Confidence Interval: {prediction.confidence_interval_95[0]:.0f} - "
                     f"{prediction.confidence_interval_95[1]:.0f}")
        report.append(f"Dominant Pathway: {prediction.dominant_pathway.replace('_', ' ').title()}")
        report.append(f"Overall Confidence: {prediction.confidence_score:.1%}")
        report.append("")
        
        # Key Milestones
        report.append("KEY PROBABILITY MILESTONES")
        report.append("-" * 20)
        
        milestones = [0.05, 0.25, 0.50, 0.75, 0.95]
        for milestone in milestones:
            year = None
            for y, p in sorted(prediction.probability_by_year.items()):
                if p >= milestone:
                    year = y
                    break
            
            if year:
                report.append(f"{int(milestone*100)}% probability by: {year}")
        
        report.append("")
        
        # Risk Factors
        report.append("RISK FACTORS")
        report.append("-" * 20)
        
        for factor, value in sorted(prediction.risk_factors.items(), 
                                  key=lambda x: x[1], reverse=True):
            risk_level = "High" if value > 0.7 else "Medium" if value > 0.3 else "Low"
            report.append(f"{factor.replace('_', ' ').title()}: {value:.2f} ({risk_level})")
        
        report.append("")
        
        # Warnings
        if prediction.anomaly_warnings:
            report.append("⚠️  ANOMALY WARNINGS")
            report.append("-" * 20)
            for warning in prediction.anomaly_warnings:
                report.append(f"• {warning}")
            report.append("")
        
        # Data Sources
        report.append("DATA SOURCES")
        report.append("-" * 20)
        report.append(f"Models: {', '.join(prediction.data_sources[:3])}")
        if len(prediction.data_sources) > 3:
            report.append(f"Real-time: {', '.join(prediction.data_sources[3:])}")
        
        if hasattr(prediction, "uncertainty_report"):
            report.append("")
            report.append("UNCERTAINTY QUANTIFICATION")
            report.append("-" * 20)
            for interval in prediction.uncertainty_report.intervals:
                report.append(
                    f"{int(interval.level*100)}% interval: {interval.lower:.0f} – {interval.upper:.0f}"
                )
            report.append("Component contributions:")
            for comp in prediction.uncertainty_report.components:
                report.append(
                    f"• {comp.name.title()}: {comp.value:.2f} (contribution {comp.contribution:.0%})"
                )
        
        if self.last_calibration:
            days_ago = (datetime.now() - self.last_calibration).days
            report.append(f"Last calibration: {days_ago} days ago")
        
        return "\n".join(report)


def test_integrated_model():
    """Test the integrated CRQC model."""
    
    print("=" * 60)
    print("INTEGRATED CRQC MODEL TEST")
    print("=" * 60)
    
    # Initialize model
    model = CRQCIntegratedModel(enable_realtime=False)  # Disable for quick test
    
    # Generate prediction
    print("\nGenerating integrated prediction...")
    prediction = model.generate_prediction()
    
    # Print report
    report = model.generate_report(prediction)
    print("\n" + report)
    
    # Test with real-time updates (if available)
    if model.enable_realtime:
        print("\n" + "=" * 60)
        print("TESTING REAL-TIME UPDATES")
        print("=" * 60)
        
        # Run async update
        import asyncio
        
        async def run_update():
            await model.update_with_realtime_data()
            
            # Generate new prediction
            updated_prediction = model.generate_prediction()
            
            print("\nUpdated Prediction:")
            print(f"Original median: {prediction.median_year:.0f}")
            print(f"Updated median: {updated_prediction.median_year:.0f}")
            print(f"Adjustment: {updated_prediction.median_year - prediction.median_year:+.1f} years")
            
            if updated_prediction.anomaly_warnings:
                print("\nNew Warnings:")
                for warning in updated_prediction.anomaly_warnings:
                    print(f"• {warning}")
        
        asyncio.run(run_update())


if __name__ == "__main__":
    test_integrated_model()
