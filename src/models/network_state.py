"""
Network state model for Solana blockchain.

This module models the evolution of the Solana network state over time,
including validator dynamics, stake distribution, and migration to
quantum-safe cryptography.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.config import NetworkParameters
from src.distributions.probability_dists import NetworkDistributions

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Status of quantum-safe migration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    COMPLETE = "complete"


class ValidatorTier(Enum):
    """Validator tiers based on stake."""
    SUPERMINORITY = "superminority"  # Top validators controlling 33%
    LARGE = "large"                   # Large but not superminority
    MEDIUM = "medium"                 # Medium stake
    SMALL = "small"                   # Small stake


@dataclass
class ValidatorState:
    """State of a single validator."""
    
    validator_id: int
    stake_amount: float
    stake_percentage: float
    tier: ValidatorTier
    location: str
    is_migrated: bool
    migration_year: Optional[float] = None
    uptime: float = 0.99
    
    @property
    def is_critical(self) -> bool:
        """Check if validator is critical for consensus."""
        return self.tier in [ValidatorTier.SUPERMINORITY, ValidatorTier.LARGE]


@dataclass
class NetworkSnapshot:
    """Snapshot of network state at a point in time."""
    
    year: float
    n_validators: int
    total_stake: float
    validators: List[ValidatorState]
    geographic_distribution: Dict[str, float]
    migration_status: MigrationStatus
    migration_progress: float  # 0 to 1
    superminority_count: int
    gini_coefficient: float
    network_resilience: float  # 0 to 1
    compromised_validators: int = 0  # Number of compromised validators
    attack_occurred: bool = False  # Whether an attack has occurred
    
    @property
    def migrated_stake_percentage(self) -> float:
        """Calculate percentage of stake that has migrated."""
        if not self.validators:
            return 0.0
        
        migrated_stake = sum(
            v.stake_amount for v in self.validators if v.is_migrated
        )
        return migrated_stake / self.total_stake if self.total_stake > 0 else 0.0
    
    @property
    def vulnerable_stake_percentage(self) -> float:
        """Calculate percentage of stake vulnerable to quantum attack."""
        return 1.0 - self.migrated_stake_percentage
    
    def get_attack_surface(self) -> Dict[str, Any]:
        """Calculate network attack surface metrics."""
        vulnerable_validators = [v for v in self.validators if not v.is_migrated]
        
        # Count vulnerable validators by tier
        vulnerable_by_tier = {}
        for tier in ValidatorTier:
            vulnerable_by_tier[tier.value] = sum(
                1 for v in vulnerable_validators if v.tier == tier
            )
        
        # Calculate stake concentration of vulnerable validators
        vulnerable_stake = sum(v.stake_amount for v in vulnerable_validators)
        
        return {
            'vulnerable_validators': len(vulnerable_validators),
            'vulnerable_stake': vulnerable_stake,
            'vulnerable_stake_percentage': self.vulnerable_stake_percentage,
            'vulnerable_by_tier': vulnerable_by_tier,
            'superminority_vulnerable': vulnerable_by_tier.get(
                ValidatorTier.SUPERMINORITY.value, 0
            ) > 0
        }


@dataclass
class NetworkEvolution:
    """Complete network evolution over time."""
    
    snapshots: List[NetworkSnapshot]
    migration_start_year: float
    migration_completion_year: Optional[float]
    peak_validators: int
    minimum_gini: float
    
    def get_snapshot_at_year(self, year: float) -> NetworkSnapshot:
        """Get network snapshot at specific year."""
        for snapshot in self.snapshots:
            if snapshot.year >= year:
                return snapshot
        return self.snapshots[-1]
    
    def get_migration_timeline(self) -> Dict[str, float]:
        """Get key migration milestones."""
        milestones = {
            'start': self.migration_start_year,
            'completion': self.migration_completion_year
        }
        
        # Find when different thresholds are reached
        for threshold in [0.25, 0.50, 0.75, 0.90]:
            for snapshot in self.snapshots:
                if snapshot.migration_progress >= threshold:
                    milestones[f'{int(threshold*100)}%'] = snapshot.year
                    break
        
        return milestones


class NetworkStateModel:
    """
    Models the evolution of Solana network state over time.
    
    Simulates:
    - Validator count growth
    - Stake distribution dynamics
    - Geographic distribution shifts
    - Migration to quantum-safe cryptography
    - Network resilience factors
    """
    
    def __init__(self, params: Optional[NetworkParameters] = None):
        """
        Initialize network state model.
        
        Args:
            params: Network parameters configuration
        """
        self.params = params or NetworkParameters()
        
        # Migration parameters
        self.migration_profiles = {
            'proactive': {
                'start_delay': -2,  # Start 2 years before CRQC
                'adoption_rate': 0.9,
                'speed': 2.0  # Fast adoption
            },
            'reactive': {
                'start_delay': 0,  # Start when CRQC emerges
                'adoption_rate': 0.7,
                'speed': 1.0  # Normal adoption
            },
            'laggard': {
                'start_delay': 2,  # Start 2 years after CRQC
                'adoption_rate': 0.5,
                'speed': 0.5  # Slow adoption
            }
        }
        
    def sample(
        self,
        rng: np.random.RandomState,
        quantum_timeline: Dict[str, Any]
    ) -> NetworkEvolution:
        """
        Sample a complete network evolution timeline.
        
        Args:
            rng: Random number generator
            quantum_timeline: Quantum development timeline
            
        Returns:
            NetworkEvolution instance
        """
        crqc_year = quantum_timeline.get('crqc_year', 2035)
        
        # Select migration profile
        profile = self._select_migration_profile(rng)
        migration_start = crqc_year + profile['start_delay']
        
        # Generate network snapshots for each year
        snapshots = []
        peak_validators = self.params.n_validators
        minimum_gini = self.params.stake_gini_coefficient
        
        for year in range(2025, 2046):
            snapshot = self._generate_snapshot(
                rng, year, migration_start, profile, crqc_year
            )
            snapshots.append(snapshot)
            
            # Track statistics
            peak_validators = max(peak_validators, snapshot.n_validators)
            minimum_gini = min(minimum_gini, snapshot.gini_coefficient)
        
        # Determine migration completion
        migration_completion = None
        for snapshot in snapshots:
            if snapshot.migration_progress >= 0.95:
                migration_completion = snapshot.year
                break
        
        return NetworkEvolution(
            snapshots=snapshots,
            migration_start_year=migration_start,
            migration_completion_year=migration_completion,
            peak_validators=peak_validators,
            minimum_gini=minimum_gini
        )
    
    def _select_migration_profile(self, rng: np.random.RandomState) -> Dict[str, Any]:
        """Select migration behavior profile."""
        # Weighted selection
        profiles = list(self.migration_profiles.keys())
        weights = [0.2, 0.6, 0.2]  # Most are reactive
        
        selected = rng.choice(profiles, p=weights)
        profile = self.migration_profiles[selected].copy()
        
        # Add some randomness
        profile['adoption_rate'] *= rng.uniform(0.8, 1.2)
        profile['speed'] *= rng.uniform(0.7, 1.3)
        profile['start_delay'] += rng.normal(0, 0.5)
        
        return profile
    
    def _generate_snapshot(
        self,
        rng: np.random.RandomState,
        year: float,
        migration_start: float,
        migration_profile: Dict[str, Any],
        crqc_year: float
    ) -> NetworkSnapshot:
        """Generate network snapshot for a specific year."""
        
        # Calculate validator count (growth over time)
        years_elapsed = year - 2025
        growth_factor = (self.params.validator_growth_rate ** years_elapsed)
        n_validators = int(
            self.params.n_validators * growth_factor * rng.uniform(0.9, 1.1)
        )
        
        # Calculate total stake (growth over time)
        stake_growth = (self.params.stake_growth_rate ** years_elapsed)
        total_stake = self.params.total_stake_sol * stake_growth
        
        # Generate validator distribution
        validators = self._generate_validators(rng, n_validators, total_stake)
        
        # Calculate migration progress
        migration_progress = self._calculate_migration_progress(
            year, migration_start, migration_profile
        )
        
        # Determine which validators have migrated
        self._apply_migration(rng, validators, migration_progress, year)
        
        # Calculate migration status
        if migration_progress == 0:
            status = MigrationStatus.NOT_STARTED
        elif migration_progress < 0.3:
            status = MigrationStatus.IN_PROGRESS
        elif migration_progress < 0.95:
            status = MigrationStatus.PARTIAL
        else:
            status = MigrationStatus.COMPLETE
        
        # Calculate geographic distribution (shifts over time)
        geographic = self._calculate_geographic_distribution(rng, year)
        
        # Calculate network metrics
        gini = self._calculate_gini_coefficient(validators)
        superminority_count = self._count_superminority(validators)
        resilience = self._calculate_resilience(
            validators, migration_progress, year, crqc_year
        )
        
        # Count compromised validators (for tracking purposes)
        compromised_count = sum(1 for v in validators if hasattr(v, 'is_compromised') and v.is_compromised)
        
        return NetworkSnapshot(
            year=year,
            n_validators=n_validators,
            total_stake=total_stake,
            validators=validators,
            geographic_distribution=geographic,
            migration_status=status,
            migration_progress=migration_progress,
            superminority_count=superminority_count,
            gini_coefficient=gini,
            network_resilience=resilience,
            compromised_validators=compromised_count,
            attack_occurred=compromised_count > 0
        )
    
    def _generate_validators(
        self,
        rng: np.random.RandomState,
        n_validators: int,
        total_stake: float
    ) -> List[ValidatorState]:
        """Generate validator distribution."""
        
        # Use Pareto distribution for realistic stake concentration
        stakes = NetworkDistributions.sample_validator_distribution(
            rng, n_validators, self.params.stake_gini_coefficient
        )
        
        # Convert to actual stake amounts
        stake_amounts = stakes * total_stake
        
        # Assign geographic locations
        locations = self._assign_locations(rng, n_validators)
        
        # Create validator states
        validators = []
        cumulative_stake = 0
        superminority_threshold = 0.33
        
        for i, (stake_amount, location) in enumerate(zip(stake_amounts, locations)):
            stake_percentage = stake_amount / total_stake
            cumulative_stake += stake_percentage
            
            # Determine tier
            if cumulative_stake <= superminority_threshold:
                tier = ValidatorTier.SUPERMINORITY
            elif stake_percentage > 0.01:  # > 1% of total
                tier = ValidatorTier.LARGE
            elif stake_percentage > 0.001:  # > 0.1% of total
                tier = ValidatorTier.MEDIUM
            else:
                tier = ValidatorTier.SMALL
            
            validator = ValidatorState(
                validator_id=i,
                stake_amount=stake_amount,
                stake_percentage=stake_percentage,
                tier=tier,
                location=location,
                is_migrated=False,
                uptime=rng.beta(20, 1)  # Most have high uptime
            )
            
            validators.append(validator)
        
        return validators
    
    def _assign_locations(
        self,
        rng: np.random.RandomState,
        n_validators: int
    ) -> List[str]:
        """Assign geographic locations to validators."""
        locations = []
        regions = list(self.params.geographic_distribution.keys())
        weights = list(self.params.geographic_distribution.values())
        
        for _ in range(n_validators):
            location = rng.choice(regions, p=weights)
            locations.append(location)
        
        return locations
    
    def _calculate_migration_progress(
        self,
        year: float,
        migration_start: float,
        profile: Dict[str, Any]
    ) -> float:
        """Calculate migration progress based on S-curve adoption."""
        if year < migration_start:
            return 0.0
        
        # Time since migration started
        t = year - migration_start
        
        # S-curve parameters
        midpoint = 3.0 / profile['speed']  # Years to 50% adoption
        steepness = 2.0 * profile['speed']
        
        # Logistic function
        progress = profile['adoption_rate'] / (1 + np.exp(-steepness * (t - midpoint)))
        
        return min(progress, profile['adoption_rate'])
    
    def _apply_migration(
        self,
        rng: np.random.RandomState,
        validators: List[ValidatorState],
        migration_progress: float,
        year: float
    ):
        """Apply migration status to validators."""
        if migration_progress == 0:
            return
        
        # Larger validators migrate first (more resources/awareness)
        # Sort by stake (descending)
        sorted_validators = sorted(
            validators,
            key=lambda v: v.stake_amount,
            reverse=True
        )
        
        # Determine how many should be migrated
        n_migrated = int(len(validators) * migration_progress)
        
        # Apply migration with some randomness
        for i, validator in enumerate(sorted_validators):
            # Base probability based on position
            base_prob = 1.0 if i < n_migrated else 0.0
            
            # Add randomness (±10%)
            actual_prob = base_prob + rng.uniform(-0.1, 0.1)
            
            if rng.random() < actual_prob:
                validator.is_migrated = True
                validator.migration_year = year - rng.uniform(0, 1)
    
    def _calculate_geographic_distribution(
        self,
        rng: np.random.RandomState,
        year: float
    ) -> Dict[str, float]:
        """Calculate geographic distribution with shifts over time."""
        base_dist = self.params.geographic_distribution.copy()
        
        # Simulate gradual shifts (e.g., growth in Asia)
        years_elapsed = year - 2025
        shift_factor = min(years_elapsed / 20, 1.0)  # Max shift over 20 years
        
        # Example: Asia grows, North America shrinks slightly
        if 'asia' in base_dist and 'north_america' in base_dist:
            shift = 0.1 * shift_factor * rng.uniform(0.5, 1.5)
            base_dist['asia'] = min(base_dist['asia'] + shift, 0.4)
            base_dist['north_america'] = max(base_dist['north_america'] - shift, 0.3)
        
        # Normalize to sum to 1
        total = sum(base_dist.values())
        return {k: v/total for k, v in base_dist.items()}
    
    def _calculate_gini_coefficient(self, validators: List[ValidatorState]) -> float:
        """Calculate Gini coefficient of stake distribution."""
        if not validators:
            return 0.0
        
        # Sort by stake
        sorted_stakes = sorted([v.stake_amount for v in validators])
        n = len(sorted_stakes)
        
        # Calculate Gini
        cumsum = np.cumsum(sorted_stakes)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_stakes)) / (n * cumsum[-1]) - (n + 1) / n
        
        return min(max(gini, 0), 1)  # Clamp to [0, 1]
    
    def _count_superminority(self, validators: List[ValidatorState]) -> int:
        """Count validators in superminority (top 33% of stake)."""
        return sum(1 for v in validators if v.tier == ValidatorTier.SUPERMINORITY)
    
    def _calculate_resilience(
        self,
        validators: List[ValidatorState],
        migration_progress: float,
        year: float,
        crqc_year: float
    ) -> float:
        """
        Calculate network resilience score (0 to 1).
        
        Factors:
        - Migration progress
        - Stake decentralization
        - Geographic distribution
        - Time until CRQC
        """
        # Base resilience from migration
        migration_score = migration_progress
        
        # Decentralization score (inverse of concentration)
        superminority_count = self._count_superminority(validators)
        total_validators = len(validators)
        decentralization_score = 1.0 - (superminority_count / total_validators) if total_validators > 0 else 0
        
        # Time buffer score
        years_until_crqc = max(0, crqc_year - year)
        time_score = min(years_until_crqc / 10, 1.0)  # Max score if 10+ years
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]  # Migration most important
        scores = [migration_score, decentralization_score, time_score]
        
        resilience = sum(w * s for w, s in zip(weights, scores))
        
        return min(max(resilience, 0), 1)


def test_network_model():
    """Test the network state model."""
    model = NetworkStateModel()
    rng = np.random.RandomState(42)
    
    # Create a sample quantum timeline
    quantum_timeline = {
        'crqc_year': 2035,
        'capabilities': []
    }
    
    # Generate network evolution
    evolution = model.sample(rng, quantum_timeline)
    
    print("Network Evolution Sample:")
    print(f"  Migration Start: {evolution.migration_start_year:.1f}")
    print(f"  Migration Complete: {evolution.migration_completion_year:.1f}" 
          if evolution.migration_completion_year else "  Migration Incomplete by 2045")
    print(f"  Peak Validators: {evolution.peak_validators}")
    print(f"  Minimum Gini: {evolution.minimum_gini:.3f}")
    
    # Show snapshot at specific years
    for year in [2025, 2030, 2035, 2040]:
        snapshot = evolution.get_snapshot_at_year(year)
        print(f"\n{year} Snapshot:")
        print(f"  Validators: {snapshot.n_validators}")
        print(f"  Migration: {snapshot.migration_progress:.1%}")
        print(f"  Vulnerable Stake: {snapshot.vulnerable_stake_percentage:.1%}")
        print(f"  Resilience: {snapshot.network_resilience:.2f}")
    
    # Show attack surface
    snapshot_2035 = evolution.get_snapshot_at_year(2035)
    attack_surface = snapshot_2035.get_attack_surface()
    print(f"\n2035 Attack Surface:")
    print(f"  Vulnerable Validators: {attack_surface['vulnerable_validators']}")
    print(f"  Vulnerable Stake: {attack_surface['vulnerable_stake_percentage']:.1%}")
    print(f"  Superminority Vulnerable: {attack_surface['superminority_vulnerable']}")
    
    print("\n✓ Network model test passed")


if __name__ == "__main__":
    test_network_model()
