"""
Probability distributions for Monte Carlo simulation.

Provides various probability distributions and sampling methods
used throughout the simulation.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class DistributionSampler:
    """Utility class for sampling from various probability distributions."""
    
    @staticmethod
    def sample_lognormal(
        rng: np.random.RandomState,
        mean: float,
        std: float,
        size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Sample from log-normal distribution.
        
        Args:
            rng: Random number generator
            mean: Mean of underlying normal distribution
            std: Standard deviation of underlying normal
            size: Number of samples (None for single value)
            
        Returns:
            Sample(s) from log-normal distribution
        """
        return rng.lognormal(mean=np.log(mean), sigma=std, size=size)
    
    @staticmethod
    def sample_beta(
        rng: np.random.RandomState,
        alpha: float,
        beta: float,
        scale: float = 1.0,
        shift: float = 0.0,
        size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Sample from scaled and shifted beta distribution.
        
        Args:
            rng: Random number generator
            alpha: Alpha parameter
            beta: Beta parameter
            scale: Scale factor
            shift: Shift amount
            size: Number of samples
            
        Returns:
            Sample(s) from beta distribution
        """
        samples = rng.beta(alpha, beta, size=size)
        return shift + scale * samples
    
    @staticmethod
    def sample_truncated_normal(
        rng: np.random.RandomState,
        mean: float,
        std: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Sample from truncated normal distribution.
        
        Args:
            rng: Random number generator
            mean: Mean of normal distribution
            std: Standard deviation
            lower: Lower bound (None for no bound)
            upper: Upper bound (None for no bound)
            size: Number of samples
            
        Returns:
            Sample(s) from truncated normal
        """
        if lower is None and upper is None:
            return rng.normal(mean, std, size)
        
        # Convert bounds to standard normal scale
        a = (lower - mean) / std if lower is not None else -np.inf
        b = (upper - mean) / std if upper is not None else np.inf
        
        # Use scipy's truncated normal
        samples = stats.truncnorm.rvs(
            a, b, loc=mean, scale=std,
            size=size, random_state=rng.randint(2**31)
        )
        
        return samples
    
    @staticmethod
    def sample_mixture(
        rng: np.random.RandomState,
        distributions: List[Tuple[str, dict]],
        weights: Optional[List[float]] = None,
        size: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Sample from mixture of distributions.
        
        Args:
            rng: Random number generator
            distributions: List of (distribution_name, params) tuples
            weights: Weights for each distribution (equal if None)
            size: Number of samples
            
        Returns:
            Sample(s) from mixture distribution
        """
        n_dists = len(distributions)
        
        if weights is None:
            weights = [1.0 / n_dists] * n_dists
        else:
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)
        
        if size is None:
            # Single sample
            chosen = rng.choice(n_dists, p=weights)
            dist_name, params = distributions[chosen]
            return DistributionSampler._sample_single(rng, dist_name, params)
        else:
            # Multiple samples
            samples = np.zeros(size)
            choices = rng.choice(n_dists, size=size, p=weights)
            
            for i in range(size):
                dist_name, params = distributions[choices[i]]
                samples[i] = DistributionSampler._sample_single(rng, dist_name, params)
            
            return samples
    
    @staticmethod
    def _sample_single(
        rng: np.random.RandomState,
        dist_name: str,
        params: dict
    ) -> float:
        """Sample single value from named distribution."""
        if dist_name == 'normal':
            return rng.normal(**params)
        elif dist_name == 'lognormal':
            return DistributionSampler.sample_lognormal(rng, **params)
        elif dist_name == 'beta':
            return DistributionSampler.sample_beta(rng, **params)
        elif dist_name == 'exponential':
            return rng.exponential(**params)
        elif dist_name == 'uniform':
            return rng.uniform(**params)
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")


class TimeDistributions:
    """Probability distributions for time-related variables."""
    
    @staticmethod
    def sample_development_time(
        rng: np.random.RandomState,
        base_years: float,
        uncertainty: float = 0.3
    ) -> float:
        """
        Sample development time with uncertainty.
        
        Args:
            rng: Random number generator
            base_years: Base development time in years
            uncertainty: Relative uncertainty (0.3 = 30%)
            
        Returns:
            Sampled development time
        """
        # Log-normal to ensure positive values with right skew
        return rng.lognormal(
            mean=np.log(base_years),
            sigma=uncertainty
        )
    
    @staticmethod
    def sample_breakthrough_timing(
        rng: np.random.RandomState,
        earliest: float,
        most_likely: float,
        latest: float
    ) -> float:
        """
        Sample breakthrough timing using PERT distribution.
        
        Args:
            rng: Random number generator
            earliest: Earliest possible time
            most_likely: Most likely time
            latest: Latest possible time
            
        Returns:
            Sampled breakthrough time
        """
        # PERT distribution (modified beta)
        alpha = 1 + 4 * (most_likely - earliest) / (latest - earliest)
        beta = 1 + 4 * (latest - most_likely) / (latest - earliest)
        
        sample = rng.beta(alpha, beta)
        return earliest + sample * (latest - earliest)
    
    @staticmethod
    def sample_adoption_curve(
        rng: np.random.RandomState,
        t: float,
        midpoint: float = 0.5,
        steepness: float = 10
    ) -> float:
        """
        Sample from S-curve adoption model.
        
        Args:
            rng: Random number generator
            t: Time parameter (0 to 1)
            midpoint: Midpoint of adoption curve
            steepness: Steepness of curve
            
        Returns:
            Adoption level (0 to 1)
        """
        # Logistic function with noise
        base = 1 / (1 + np.exp(-steepness * (t - midpoint)))
        noise = rng.normal(0, 0.05)
        return np.clip(base + noise, 0, 1)


class EconomicDistributions:
    """Probability distributions for economic variables."""
    
    @staticmethod
    def sample_loss_severity(
        rng: np.random.RandomState,
        mean_loss: float,
        tail_index: float = 2.0
    ) -> float:
        """
        Sample loss severity from heavy-tailed distribution.
        
        Args:
            rng: Random number generator
            mean_loss: Mean loss amount
            tail_index: Tail heaviness (lower = heavier tail)
            
        Returns:
            Sampled loss amount
        """
        # Pareto distribution for heavy tails
        if tail_index <= 1:
            raise ValueError("Tail index must be > 1 for finite mean")
        
        # Set scale to achieve desired mean
        scale = mean_loss * (tail_index - 1) / tail_index
        
        return scale * (rng.pareto(tail_index) + 1)
    
    @staticmethod
    def sample_market_impact(
        rng: np.random.RandomState,
        direct_loss: float,
        multiplier_range: Tuple[float, float] = (2, 5)
    ) -> float:
        """
        Sample total market impact from direct loss.
        
        Args:
            rng: Random number generator
            direct_loss: Direct loss amount
            multiplier_range: Range for impact multiplier
            
        Returns:
            Total market impact
        """
        # Log-normal multiplier for realistic market impacts
        min_mult, max_mult = multiplier_range
        mean_mult = (min_mult + max_mult) / 2
        std_mult = (max_mult - min_mult) / 4
        
        multiplier = DistributionSampler.sample_truncated_normal(
            rng, mean_mult, std_mult, min_mult, max_mult
        )
        
        return direct_loss * multiplier
    
    @staticmethod
    def sample_recovery_time(
        rng: np.random.RandomState,
        severity: float,
        base_months: float = 12
    ) -> float:
        """
        Sample recovery time based on incident severity.
        
        Args:
            rng: Random number generator
            severity: Severity of incident (0 to 1)
            base_months: Base recovery time
            
        Returns:
            Recovery time in months
        """
        # Exponential relationship with severity
        mean_time = base_months * np.exp(severity)
        
        # Gamma distribution for realistic recovery times
        shape = 2.0  # Shape parameter
        scale = mean_time / shape
        
        return rng.gamma(shape, scale)


class NetworkDistributions:
    """Probability distributions for network-related variables."""
    
    @staticmethod
    def sample_validator_distribution(
        rng: np.random.RandomState,
        n_validators: int,
        gini_coefficient: float = 0.84
    ) -> np.ndarray:
        """
        Sample validator stake distribution.
        
        Args:
            rng: Random number generator
            n_validators: Number of validators
            gini_coefficient: Stake concentration (0=equal, 1=concentrated)
            
        Returns:
            Array of stake proportions
        """
        # Use Pareto distribution to achieve desired Gini coefficient
        # Relationship: Gini = 1 / (2 * alpha - 1) for Pareto
        alpha = (1 + gini_coefficient) / (2 * gini_coefficient)
        
        # Generate Pareto samples
        stakes = rng.pareto(alpha, size=n_validators) + 1
        
        # Normalize to proportions
        stakes = stakes / np.sum(stakes)
        
        # Sort in descending order
        return np.sort(stakes)[::-1]
    
    @staticmethod
    def sample_migration_progress(
        rng: np.random.RandomState,
        time_elapsed: float,
        adoption_rate: float,
        heterogeneity: float = 0.2
    ) -> float:
        """
        Sample migration progress to quantum-safe cryptography.
        
        Args:
            rng: Random number generator
            time_elapsed: Time since migration started (years)
            adoption_rate: Base adoption rate
            heterogeneity: Variation in adoption
            
        Returns:
            Fraction migrated (0 to 1)
        """
        # S-curve adoption with heterogeneity
        t = time_elapsed / 10  # Normalize to 10-year horizon
        
        # Base S-curve
        base_progress = 1 / (1 + np.exp(-10 * (t - 0.5)))
        
        # Adjust for adoption rate
        adjusted = base_progress * adoption_rate
        
        # Add heterogeneity
        noise = rng.normal(0, heterogeneity)
        
        return np.clip(adjusted + noise, 0, 1)


def test_distributions():
    """Test distribution sampling functions."""
    rng = np.random.RandomState(42)
    
    print("Testing Distribution Samplers:")
    
    # Test time distributions
    dev_time = TimeDistributions.sample_development_time(rng, 5.0)
    print(f"  Development time: {dev_time:.2f} years")
    
    breakthrough = TimeDistributions.sample_breakthrough_timing(rng, 2025, 2030, 2040)
    print(f"  Breakthrough year: {breakthrough:.1f}")
    
    # Test economic distributions
    loss = EconomicDistributions.sample_loss_severity(rng, 1e9)
    print(f"  Loss severity: ${loss/1e9:.2f}B")
    
    impact = EconomicDistributions.sample_market_impact(rng, 1e9)
    print(f"  Market impact: ${impact/1e9:.2f}B")
    
    # Test network distributions
    stakes = NetworkDistributions.sample_validator_distribution(rng, 100)
    print(f"  Top validator stake: {stakes[0]:.3%}")
    print(f"  Top 10 control: {np.sum(stakes[:10]):.1%}")
    
    # Test mixture distribution
    mixture = DistributionSampler.sample_mixture(
        rng,
        [
            ('normal', {'loc': 2030, 'scale': 2}),
            ('exponential', {'scale': 5})
        ],
        weights=[0.7, 0.3],
        size=1000
    )
    print(f"  Mixture mean: {np.mean(mixture):.1f}")
    
    print("\n✓ Distribution tests passed")


if __name__ == "__main__":
    test_distributions()
