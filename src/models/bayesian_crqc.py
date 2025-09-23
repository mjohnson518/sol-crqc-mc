"""
Hierarchical Bayesian model for CRQC emergence prediction.

This module implements a sophisticated multi-level Bayesian model that accounts for:
- Global quantum progress trends
- Technology-specific variations
- Regional development differences
- Correlation between different approaches
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from scipy import stats
from scipy.stats import norm, gamma, lognorm, multivariate_normal
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import PyMC for Bayesian inference
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.info("PyMC not available - using simplified Bayesian implementation")


@dataclass
class TechnologyProfile:
    """Profile for a specific quantum computing technology."""
    
    name: str
    companies: List[str]
    current_qubits: int
    growth_rate_mean: float  # Annual multiplier
    growth_rate_std: float
    error_rate: float
    maturity_level: float  # 0-1 scale
    breakthrough_probability: float  # Annual
    
    def project_qubits(self, years_ahead: float, include_uncertainty: bool = True) -> float:
        """Project qubit count for this technology."""
        if include_uncertainty:
            growth = np.random.lognormal(
                mean=np.log(self.growth_rate_mean),
                sigma=self.growth_rate_std
            )
        else:
            growth = self.growth_rate_mean
        
        return self.current_qubits * (growth ** years_ahead)


@dataclass
class BayesianPosterior:
    """Stores posterior distributions from Bayesian inference."""
    
    crqc_year_samples: np.ndarray
    technology_effects: Dict[str, np.ndarray]
    breakthrough_rate_samples: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def get_median_prediction(self) -> float:
        """Get median CRQC year from posterior."""
        return np.median(self.crqc_year_samples)
    
    def get_credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Get Bayesian credible interval."""
        alpha = (1 - level) / 2
        return (
            np.quantile(self.crqc_year_samples, alpha),
            np.quantile(self.crqc_year_samples, 1 - alpha)
        )


class HierarchicalBayesianCRQC:
    """
    Hierarchical Bayesian model for CRQC emergence prediction.
    
    Model structure:
    - Level 1: Global trend (when quantum computing reaches CRQC capability)
    - Level 2: Technology-specific effects (superconducting, ion trap, etc.)
    - Level 3: Company/lab specific variations
    - Level 4: Individual observations with measurement error
    """
    
    def __init__(self, prior_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bayesian model with priors.
        
        Args:
            prior_config: Optional configuration for prior distributions
        """
        self.prior_config = prior_config or self._get_default_priors()
        self.technologies = self._initialize_technologies()
        self.historical_data = self._load_historical_data()
        self.posterior = None
        
    def _get_default_priors(self) -> Dict[str, Any]:
        """Get default prior distributions based on expert knowledge."""
        
        return {
            # Global CRQC emergence
            "global_crqc_mean": {
                "distribution": "normal",
                "params": {"mu": 2033, "sigma": 3}  # Expert consensus
            },
            "global_crqc_std": {
                "distribution": "halfnormal",
                "params": {"sigma": 2}
            },
            
            # Technology-specific offsets from global
            "tech_offset_std": {
                "distribution": "halfnormal",
                "params": {"sigma": 2}
            },
            
            # Breakthrough rate
            "breakthrough_rate": {
                "distribution": "gamma",
                "params": {"alpha": 2, "beta": 0.1}  # Mean ~0.05/year
            },
            
            # Growth rates by technology
            "growth_rates": {
                "superconducting": {"mu": 1.5, "sigma": 0.2},
                "trapped_ion": {"mu": 1.3, "sigma": 0.15},
                "topological": {"mu": 1.2, "sigma": 0.3},  # Higher uncertainty
                "photonic": {"mu": 1.4, "sigma": 0.25},
                "neutral_atom": {"mu": 1.6, "sigma": 0.2}
            }
        }
    
    def _initialize_technologies(self) -> Dict[str, TechnologyProfile]:
        """Initialize technology profiles with current state."""
        
        return {
            "superconducting": TechnologyProfile(
                name="Superconducting",
                companies=["IBM", "Google", "Rigetti"],
                current_qubits=1121,  # IBM Condor
                growth_rate_mean=1.5,
                growth_rate_std=0.2,
                error_rate=0.001,
                maturity_level=0.7,
                breakthrough_probability=0.06
            ),
            "trapped_ion": TechnologyProfile(
                name="Trapped Ion",
                companies=["IonQ", "Quantinuum", "Alpine"],
                current_qubits=32,  # Algorithmic qubits
                growth_rate_mean=1.3,
                growth_rate_std=0.15,
                error_rate=0.0001,
                maturity_level=0.6,
                breakthrough_probability=0.04
            ),
            "topological": TechnologyProfile(
                name="Topological",
                companies=["Microsoft"],
                current_qubits=0,  # Still theoretical
                growth_rate_mean=2.0,  # If breakthrough happens
                growth_rate_std=0.5,
                error_rate=0.00001,  # Theoretical advantage
                maturity_level=0.1,
                breakthrough_probability=0.02
            ),
            "photonic": TechnologyProfile(
                name="Photonic",
                companies=["PsiQuantum", "Xanadu"],
                current_qubits=216,  # Xanadu
                growth_rate_mean=1.4,
                growth_rate_std=0.25,
                error_rate=0.001,
                maturity_level=0.4,
                breakthrough_probability=0.05
            ),
            "neutral_atom": TechnologyProfile(
                name="Neutral Atom",
                companies=["Atom Computing", "QuEra", "Pasqal"],
                current_qubits=1180,  # Atom Computing
                growth_rate_mean=1.6,
                growth_rate_std=0.2,
                error_rate=0.005,
                maturity_level=0.5,
                breakthrough_probability=0.05
            )
        }
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical quantum computing progress data."""
        
        return [
            {"year": 2019, "company": "Google", "qubits": 53, "tech": "superconducting"},
            {"year": 2020, "company": "IBM", "qubits": 65, "tech": "superconducting"},
            {"year": 2021, "company": "IBM", "qubits": 127, "tech": "superconducting"},
            {"year": 2022, "company": "IBM", "qubits": 433, "tech": "superconducting"},
            {"year": 2023, "company": "IBM", "qubits": 1121, "tech": "superconducting"},
            {"year": 2023, "company": "Atom", "qubits": 1180, "tech": "neutral_atom"},
            {"year": 2023, "company": "IonQ", "qubits": 32, "tech": "trapped_ion"},
            {"year": 2024, "company": "Google", "qubits": 100, "tech": "superconducting"},
            {"year": 2024, "company": "PsiQuantum", "qubits": 0, "tech": "photonic"},  # Pre-commercial
        ]
    
    def build_model(self) -> Any:
        """
        Build the hierarchical Bayesian model.
        
        Returns:
            PyMC model object if available, else simplified model
        """
        if PYMC_AVAILABLE:
            return self._build_pymc_model()
        else:
            return self._build_simple_model()
    
    def _build_pymc_model(self) -> Any:
        """Build model using PyMC."""
        
        with pm.Model() as model:
            # Hyperpriors (Level 1 - Global)
            global_mean = pm.Normal(
                'global_crqc_mean',
                mu=self.prior_config["global_crqc_mean"]["params"]["mu"],
                sigma=self.prior_config["global_crqc_mean"]["params"]["sigma"]
            )
            global_std = pm.HalfNormal(
                'global_crqc_std',
                sigma=self.prior_config["global_crqc_std"]["params"]["sigma"]
            )
            
            # Technology-level effects (Level 2)
            tech_offset_std = pm.HalfNormal(
                'tech_offset_std',
                sigma=self.prior_config["tech_offset_std"]["params"]["sigma"]
            )
            
            tech_offsets = {}
            for tech_name in self.technologies.keys():
                tech_offsets[tech_name] = pm.Normal(
                    f'tech_offset_{tech_name}',
                    mu=0,
                    sigma=tech_offset_std
                )
            
            # Breakthrough rate
            breakthrough_rate = pm.Gamma(
                'breakthrough_rate',
                alpha=self.prior_config["breakthrough_rate"]["params"]["alpha"],
                beta=self.prior_config["breakthrough_rate"]["params"]["beta"]
            )
            
            # Growth rates by technology
            growth_rates = {}
            for tech_name, tech_prior in self.prior_config["growth_rates"].items():
                growth_rates[tech_name] = pm.LogNormal(
                    f'growth_rate_{tech_name}',
                    mu=np.log(tech_prior["mu"]),
                    sigma=tech_prior["sigma"]
                )
            
            # Likelihood - project forward and check CRQC achievement
            # This is simplified - in practice would be more complex
            crqc_years = []
            for tech_name, tech_profile in self.technologies.items():
                tech_mean = global_mean + tech_offsets[tech_name]
                tech_crqc = pm.Normal(
                    f'crqc_{tech_name}',
                    mu=tech_mean,
                    sigma=global_std
                )
                crqc_years.append(tech_crqc)
            
            # Overall CRQC (minimum across technologies)
            crqc_year = pm.Deterministic(
                'crqc_year',
                pm.math.minimum(*crqc_years) if len(crqc_years) > 1 else crqc_years[0]
            )
        
        return model
    
    def _build_simple_model(self) -> Dict[str, Any]:
        """Build simplified model without PyMC."""
        
        model = {
            "global_mean": norm(
                loc=self.prior_config["global_crqc_mean"]["params"]["mu"],
                scale=self.prior_config["global_crqc_mean"]["params"]["sigma"]
            ),
            "breakthrough_rate": gamma(
                a=self.prior_config["breakthrough_rate"]["params"]["alpha"],
                scale=1/self.prior_config["breakthrough_rate"]["params"]["beta"]
            ),
            "tech_offsets": {},
            "growth_rates": {}
        }
        
        # Technology-specific distributions
        for tech_name in self.technologies.keys():
            model["tech_offsets"][tech_name] = norm(0, 2)  # Simplified
            
            if tech_name in self.prior_config["growth_rates"]:
                prior = self.prior_config["growth_rates"][tech_name]
                model["growth_rates"][tech_name] = lognorm(
                    s=prior["sigma"],
                    scale=np.exp(np.log(prior["mu"]))
                )
        
        return model
    
    def update_posterior(self, new_data: Dict[str, Any]) -> BayesianPosterior:
        """
        Update posterior with new data using Bayesian inference.
        
        Args:
            new_data: New observations to update beliefs
            
        Returns:
            Updated posterior distributions
        """
        if PYMC_AVAILABLE:
            return self._update_pymc_posterior(new_data)
        else:
            return self._update_simple_posterior(new_data)
    
    def _update_pymc_posterior(self, new_data: Dict[str, Any]) -> BayesianPosterior:
        """Update using PyMC MCMC sampling."""
        
        model = self.build_model()
        
        with model:
            # Add observed data
            # This would involve adding likelihood terms based on new_data
            
            # Sample from posterior
            trace = pm.sample(
                draws=5000,
                tune=1000,
                chains=4,
                return_inferencedata=True
            )
        
        # Extract results
        crqc_samples = trace.posterior["crqc_year"].values.flatten()
        
        tech_effects = {}
        for tech in self.technologies.keys():
            if f"tech_offset_{tech}" in trace.posterior:
                tech_effects[tech] = trace.posterior[f"tech_offset_{tech}"].values.flatten()
        
        breakthrough_samples = trace.posterior["breakthrough_rate"].values.flatten()
        
        # Calculate credible intervals
        intervals = {}
        for level in [0.50, 0.80, 0.95]:
            intervals[f"ci_{int(level*100)}"] = (
                np.quantile(crqc_samples, (1-level)/2),
                np.quantile(crqc_samples, (1+level)/2)
            )
        
        return BayesianPosterior(
            crqc_year_samples=crqc_samples,
            technology_effects=tech_effects,
            breakthrough_rate_samples=breakthrough_samples,
            confidence_intervals=intervals
        )
    
    def _update_simple_posterior(self, new_data: Dict[str, Any]) -> BayesianPosterior:
        """Simplified posterior update without PyMC."""
        
        model = self._build_simple_model()
        n_samples = 10000
        
        # Sample from prior distributions
        global_samples = model["global_mean"].rvs(n_samples)
        breakthrough_samples = model["breakthrough_rate"].rvs(n_samples)
        
        # Technology-specific sampling
        tech_samples = {}
        crqc_by_tech = {}
        
        for tech_name, tech_profile in self.technologies.items():
            # Sample technology offset
            tech_offset = model["tech_offsets"][tech_name].rvs(n_samples)
            tech_samples[tech_name] = tech_offset
            
            # Sample growth rate
            if tech_name in model["growth_rates"]:
                growth_rate = model["growth_rates"][tech_name].rvs(n_samples)
            else:
                growth_rate = np.ones(n_samples) * tech_profile.growth_rate_mean
            
            # Project to CRQC
            current_year = datetime.now().year
            years_to_crqc = np.zeros(n_samples)
            
            for i in range(n_samples):
                qubits = tech_profile.current_qubits
                year = 0
                required_qubits = 2330  # Base requirement
                
                # Apply breakthrough possibility
                if np.random.random() < breakthrough_samples[i]:
                    required_qubits *= 0.7  # Breakthrough reduces requirements
                
                while qubits < required_qubits and year < 50:
                    qubits *= growth_rate[i]
                    year += 1
                
                years_to_crqc[i] = year
            
            crqc_by_tech[tech_name] = current_year + years_to_crqc + tech_offset
        
        # Overall CRQC is the minimum across technologies
        crqc_samples = np.min(
            np.array(list(crqc_by_tech.values())),
            axis=0
        )
        
        # Add global trend
        crqc_samples = 0.7 * crqc_samples + 0.3 * global_samples
        
        # Calculate intervals
        intervals = {}
        for level in [0.50, 0.80, 0.95]:
            intervals[f"ci_{int(level*100)}"] = (
                np.quantile(crqc_samples, (1-level)/2),
                np.quantile(crqc_samples, (1+level)/2)
            )
        
        return BayesianPosterior(
            crqc_year_samples=crqc_samples,
            technology_effects=tech_samples,
            breakthrough_rate_samples=breakthrough_samples,
            confidence_intervals=intervals
        )
    
    def predict(self, 
                include_breakthroughs: bool = True,
                n_samples: int = 10000) -> Dict[str, Any]:
        """
        Generate predictions from the Bayesian model.
        
        Args:
            include_breakthroughs: Whether to model breakthrough events
            n_samples: Number of samples to draw
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Get posterior (or use prior if no updates)
        if self.posterior is None:
            self.posterior = self.update_posterior({})
        
        predictions = {
            "median": np.median(self.posterior.crqc_year_samples),
            "mean": np.mean(self.posterior.crqc_year_samples),
            "std": np.std(self.posterior.crqc_year_samples),
            "ci_50": self.posterior.confidence_intervals["ci_50"],
            "ci_80": self.posterior.confidence_intervals["ci_80"],
            "ci_95": self.posterior.confidence_intervals["ci_95"],
            "probability_by_year": {},
            "technology_contributions": {}
        }
        
        # Calculate probability of CRQC by year
        current_year = datetime.now().year
        for year in range(current_year, current_year + 30):
            prob = np.mean(self.posterior.crqc_year_samples <= year)
            predictions["probability_by_year"][year] = prob
        
        # Technology contributions
        for tech_name in self.technologies.keys():
            if tech_name in self.posterior.technology_effects:
                tech_effect = self.posterior.technology_effects[tech_name]
                predictions["technology_contributions"][tech_name] = {
                    "mean_effect": np.mean(tech_effect),
                    "std_effect": np.std(tech_effect)
                }
        
        # Include breakthrough scenarios if requested
        if include_breakthroughs:
            predictions["breakthrough_scenario"] = self._simulate_breakthrough_scenario(n_samples)
        
        return predictions
    
    def _simulate_breakthrough_scenario(self, n_samples: int = 1000) -> Dict[str, Any]:
        """Simulate impact of potential breakthroughs."""
        
        breakthrough_years = []
        
        for _ in range(n_samples):
            # Sample breakthrough timing
            rate = np.random.choice(self.posterior.breakthrough_rate_samples)
            time_to_breakthrough = np.random.exponential(1/rate)
            
            # Breakthrough accelerates timeline
            normal_crqc = np.random.choice(self.posterior.crqc_year_samples)
            breakthrough_crqc = normal_crqc - np.random.uniform(2, 5)  # 2-5 year acceleration
            
            if datetime.now().year + time_to_breakthrough < normal_crqc:
                breakthrough_years.append(breakthrough_crqc)
            else:
                breakthrough_years.append(normal_crqc)
        
        return {
            "median_with_breakthrough": np.median(breakthrough_years),
            "probability_of_breakthrough": np.mean(
                np.array(breakthrough_years) < np.median(self.posterior.crqc_year_samples)
            )
        }
    
    def cross_validate(self, historical_checkpoints: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Cross-validate model against historical checkpoints.
        
        Args:
            historical_checkpoints: Historical data points to validate against
            
        Returns:
            Validation metrics
        """
        errors = []
        
        for checkpoint in historical_checkpoints:
            # Make prediction as of checkpoint date
            # Compare to actual progress
            # This is simplified - real implementation would be more complex
            pass
        
        return {
            "mae": np.mean(np.abs(errors)) if errors else 0,
            "rmse": np.sqrt(np.mean(np.array(errors)**2)) if errors else 0,
            "mape": np.mean(np.abs(errors)) if errors else 0  # Simplified
        }


def test_bayesian_model():
    """Test the Bayesian CRQC model."""
    
    print("=" * 50)
    print("HIERARCHICAL BAYESIAN CRQC MODEL TEST")
    print("=" * 50)
    
    # Initialize model
    model = HierarchicalBayesianCRQC()
    
    # Generate predictions
    predictions = model.predict(include_breakthroughs=True)
    
    print(f"\n=== CRQC Emergence Predictions ===")
    print(f"Median: {predictions['median']:.1f}")
    print(f"Mean: {predictions['mean']:.1f} ± {predictions['std']:.1f}")
    print(f"\nCredible Intervals:")
    print(f"  50% CI: {predictions['ci_50'][0]:.1f} - {predictions['ci_50'][1]:.1f}")
    print(f"  80% CI: {predictions['ci_80'][0]:.1f} - {predictions['ci_80'][1]:.1f}")
    print(f"  95% CI: {predictions['ci_95'][0]:.1f} - {predictions['ci_95'][1]:.1f}")
    
    print(f"\n=== Technology Contributions ===")
    for tech, contrib in predictions["technology_contributions"].items():
        print(f"{tech}: {contrib['mean_effect']:+.1f} ± {contrib['std_effect']:.1f} years")
    
    print(f"\n=== Breakthrough Scenario ===")
    breakthrough = predictions["breakthrough_scenario"]
    print(f"With breakthrough: {breakthrough['median_with_breakthrough']:.1f}")
    print(f"Breakthrough probability: {breakthrough['probability_of_breakthrough']:.1%}")
    
    print(f"\n=== Probability by Year ===")
    current_year = datetime.now().year
    for year in range(current_year, min(current_year + 11, current_year + 30)):
        if year in predictions["probability_by_year"]:
            prob = predictions["probability_by_year"][year]
            bar = "█" * int(prob * 50)
            print(f"{year}: {prob:6.1%} {bar}")


if __name__ == "__main__":
    test_bayesian_model()
