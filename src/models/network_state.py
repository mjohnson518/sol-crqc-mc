"""
Network state model for Solana blockchain.

This module models the evolution of the Solana network state over time,
including validator dynamics, stake distribution, and migration to
quantum-safe cryptography.

Enhanced features (controlled by config flags):
- Graph-based validator topology using NetworkX
- Stake redistribution dynamics
- Live data fetching from Solana APIs
- What-if scenarios for stake centralization
"""

import numpy as np
from scipy import stats
from scipy.special import expit  # For logistic function
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Import NetworkX if available
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.info("NetworkX not available - graph features will use fallback implementation")

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
class ValidatorNode:
    """Extended validator representation for graph model."""
    
    id: str
    stake: float
    location: str
    performance_score: float  # 0-1 score
    migration_status: MigrationStatus
    connections: List[str]  # Connected validator IDs
    centrality_score: float  # Network centrality
    is_superminority: bool
    hardware_specs: Dict[str, Any]
    
    @property
    def influence_score(self) -> float:
        """Calculate validator's network influence."""
        return self.stake * self.centrality_score * self.performance_score


@dataclass
class NetworkTopology:
    """Graph-based network topology."""
    
    graph: Optional[Any] = None  # nx.Graph if NetworkX available
    validators: Dict[str, ValidatorNode] = field(default_factory=dict)
    stake_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    geographic_clusters: Dict[str, List[str]] = field(default_factory=dict)
    superminority_set: List[str] = field(default_factory=list)
    
    def get_centrality_metrics(self) -> Dict[str, float]:
        """Calculate various centrality metrics."""
        if NETWORKX_AVAILABLE and self.graph:
            return {
                'degree_centrality': nx.degree_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph),
                'closeness_centrality': nx.closeness_centrality(self.graph),
                'eigenvector_centrality': nx.eigenvector_centrality(self.graph, max_iter=1000)
            }
        return {}
    
    def get_clustering_coefficient(self) -> float:
        """Get network clustering coefficient."""
        if NETWORKX_AVAILABLE and self.graph:
            return nx.average_clustering(self.graph)
        return 0.0
    
    def get_connected_components(self) -> List[List[str]]:
        """Get connected components (validator clusters)."""
        if NETWORKX_AVAILABLE and self.graph:
            return [list(c) for c in nx.connected_components(self.graph)]
        return []


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
    graph_metrics: Optional[Dict[str, Any]] = None  # Graph-based metrics
    topology: Optional[NetworkTopology] = None  # Network topology if graph-based
    
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
    graph_topology: Optional[NetworkTopology] = None  # Final topology if graph-based
    final_resilience: Optional[float] = None  # Final network resilience
    
    def get_snapshot_at_year(self, year: float) -> NetworkSnapshot:
        """Get network snapshot at specific year."""
        # Handle None year - return first snapshot
        if year is None:
            if self.snapshots:
                return self.snapshots[0]
            # Create a default snapshot if none exist
            return NetworkSnapshot(
                year=2025,
                n_validators=self.snapshots[0].n_validators if self.snapshots else 1000,
                total_stake_sol=self.snapshots[0].total_stake_sol if self.snapshots else 400_000_000,
                validator_concentration_gini=0.85,
                quantum_resistant_validators=0,
                migration_progress=0.0,
                network_health_score=1.0,
                compromised_validators=0,
                attack_occurred=False
            )
            
        for snapshot in self.snapshots:
            if snapshot.year is not None and snapshot.year >= year:
                return snapshot
        return self.snapshots[-1] if self.snapshots else self.get_snapshot_at_year(None)
    
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
    
    def __init__(self, params: Optional[NetworkParameters] = None,
                 enable_live_data: bool = False,
                 use_graph_model: bool = False):
        """
        Initialize network state model.
        
        Args:
            params: Network parameters configuration
            enable_live_data: Whether to fetch live Solana data
            use_graph_model: Whether to use graph-based topology
        """
        self.params = params or NetworkParameters()
        self.enable_live_data = enable_live_data
        self.use_graph_model = use_graph_model and NETWORKX_AVAILABLE
        
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
        
        # Initialize topology if using graph model
        self.topology = None
        
        # Initialize data cache for live data
        if self.enable_live_data:
            self.data_cache = {}
            self.cache_duration = timedelta(hours=1)
        
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
        # Use graph-based model if enabled
        if self.use_graph_model:
            return self.sample_with_graph(rng, quantum_timeline)
        
        # Otherwise use standard model
        crqc_year = quantum_timeline.get('crqc_year', 2035)
        
        # Update with live data if enabled
        if self.enable_live_data:
            self._update_with_live_data()
        
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


    def sample_with_graph(
        self,
        rng: np.random.RandomState,
        quantum_timeline: Optional[Dict[str, Any]] = None
    ) -> NetworkEvolution:
        """
        Sample network evolution with graph-based model.
        
        Args:
            rng: Random number generator
            quantum_timeline: Quantum development timeline
            
        Returns:
            NetworkEvolution with graph topology
        """
        snapshots = []
        
        # Create initial topology
        self.topology = self._create_network_topology(self.params.n_validators, rng)
        
        # Get CRQC year if provided
        crqc_year = 2035
        if quantum_timeline and 'crqc_year' in quantum_timeline:
            crqc_year = quantum_timeline['crqc_year']
        
        # Select migration profile
        profile = self._select_migration_profile(rng)
        migration_start = crqc_year + profile['start_delay']
        
        # Track metrics
        peak_validators = self.params.n_validators
        minimum_gini = self.params.stake_gini_coefficient
        
        # Simulate evolution
        for year in range(2025, 2046):
            # Update topology with stake redistribution
            self.topology = self._simulate_stake_redistribution(self.topology, year, rng)
            
            # Sample migration progress
            migration_progress = self._calculate_migration_progress(
                year, migration_start, profile
            )
            
            # Update validator migration status
            self._update_migration_status(self.topology, migration_progress, year, rng)
            
            # Calculate network metrics
            snapshot = self._create_graph_snapshot(
                self.topology, year, migration_progress, crqc_year, rng
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
            minimum_gini=minimum_gini,
            graph_topology=self.topology,
            final_resilience=snapshots[-1].network_resilience if snapshots else 0.0
        )
    
    def _create_network_topology(
        self,
        n_validators: int,
        rng: np.random.RandomState
    ) -> NetworkTopology:
        """
        Create graph-based network topology.
        
        Args:
            n_validators: Number of validators
            rng: Random number generator
            
        Returns:
            NetworkTopology instance
        """
        topology = NetworkTopology()
        
        # Create scale-free network if NetworkX available
        if NETWORKX_AVAILABLE:
            m = 5  # Number of edges to attach from new node
            topology.graph = nx.barabasi_albert_graph(n_validators, m, seed=rng.randint(0, 2**32))
        
        # Create validator nodes
        validators = {}
        stake_distribution = self._sample_stake_distribution(n_validators, rng)
        
        # Geographic distribution
        locations = ['north_america', 'europe', 'asia', 'other']
        location_probs = list(self.params.geographic_distribution.values())
        
        for i in range(n_validators):
            validator_id = f"validator_{i}"
            
            # Assign properties
            stake = stake_distribution[i]
            location = rng.choice(locations, p=location_probs)
            performance = rng.beta(8, 2)  # Skewed toward high performance
            
            # Migration status (probabilistic based on stake)
            migration_prob = 0.3 + 0.5 * (1 - stake / stake_distribution.max())
            is_migrated = rng.random() < migration_prob
            
            validators[validator_id] = ValidatorNode(
                id=validator_id,
                stake=stake,
                location=location,
                performance_score=performance,
                migration_status=MigrationStatus.COMPLETED if is_migrated else MigrationStatus.NOT_STARTED,
                connections=[f"validator_{j}" for j in (topology.graph.neighbors(i) if NETWORKX_AVAILABLE and topology.graph else [])],
                centrality_score=0.0,
                is_superminority=False,
                hardware_specs={
                    'cpu_cores': rng.choice([32, 64, 128]),
                    'ram_gb': rng.choice([128, 256, 512]),
                    'bandwidth_gbps': rng.choice([1, 10, 40])
                }
            )
        
        # Calculate centrality scores if NetworkX available
        if NETWORKX_AVAILABLE and topology.graph:
            centrality = nx.eigenvector_centrality(topology.graph, max_iter=1000)
            for i, score in centrality.items():
                validators[f"validator_{i}"].centrality_score = score
        
        # Identify superminority
        sorted_validators = sorted(validators.values(), key=lambda v: v.stake, reverse=True)
        cumulative_stake = 0
        superminority_threshold = sum(stake_distribution) * 0.333
        superminority_set = []
        
        for v in sorted_validators:
            cumulative_stake += v.stake
            v.is_superminority = True
            superminority_set.append(v.id)
            if cumulative_stake >= superminority_threshold:
                break
        
        # Create geographic clusters
        geographic_clusters = {}
        for location in locations:
            geographic_clusters[location] = [
                v.id for v in validators.values() if v.location == location
            ]
        
        topology.validators = validators
        topology.stake_distribution = stake_distribution
        topology.geographic_clusters = geographic_clusters
        topology.superminority_set = superminority_set
        
        return topology
    
    def _sample_stake_distribution(
        self,
        n_validators: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Sample realistic stake distribution (Pareto-like).
        
        Args:
            n_validators: Number of validators
            rng: Random number generator
            
        Returns:
            Array of stake amounts
        """
        # Use Pareto distribution for realistic concentration
        shape = 1.5  # Shape parameter (lower = more concentrated)
        raw_stakes = rng.pareto(shape, n_validators) + 1
        
        # Normalize to total stake
        total_stake = self.params.total_stake_sol
        stakes = raw_stakes / raw_stakes.sum() * total_stake
        
        # Add minimum stake requirement
        min_stake = 5000  # Minimum 5000 SOL
        stakes = np.maximum(stakes, min_stake)
        
        # Renormalize
        stakes = stakes / stakes.sum() * total_stake
        
        return stakes
    
    def _simulate_stake_redistribution(
        self,
        topology: NetworkTopology,
        year: float,
        rng: np.random.RandomState
    ) -> NetworkTopology:
        """
        Simulate stake redistribution dynamics.
        
        Args:
            topology: Current network topology
            year: Current year
            rng: Random number generator
            
        Returns:
            Updated topology with redistributed stake
        """
        validators = topology.validators
        current_stakes = np.array([v.stake for v in validators.values()])
        
        # Migration curve (logistic)
        t = year - 2025
        t0 = 8  # Midpoint year (2033)
        k = 0.5  # Steepness
        migration_pressure = expit(k * (t - t0))
        
        # Stake flows based on migration status
        new_stakes = current_stakes.copy()
        
        for i, (vid, validator) in enumerate(validators.items()):
            # Validators lose stake if not migrated
            if validator.migration_status != MigrationStatus.COMPLETED:
                stake_loss = validator.stake * migration_pressure * 0.1 * rng.random()
                new_stakes[i] -= stake_loss
                
                # Redistribute lost stake to migrated validators
                migrated_indices = [
                    j for j, (_, v) in enumerate(validators.items())
                    if v.migration_status == MigrationStatus.COMPLETED
                ]
                
                if migrated_indices:
                    redistribution = stake_loss / len(migrated_indices)
                    for j in migrated_indices:
                        new_stakes[j] += redistribution
            
            # Performance-based stake changes
            if validator.performance_score < 0.5:
                # Poor performers lose stake
                performance_loss = validator.stake * 0.05 * rng.random()
                new_stakes[i] -= performance_loss
            elif validator.performance_score > 0.9:
                # Top performers gain stake
                performance_gain = validator.stake * 0.05 * rng.random()
                new_stakes[i] += performance_gain
        
        # Ensure non-negative stakes
        new_stakes = np.maximum(new_stakes, 1000)  # Minimum 1000 SOL
        
        # Update validator stakes
        for i, (vid, validator) in enumerate(validators.items()):
            validator.stake = new_stakes[i]
        
        # Recalculate superminority
        sorted_validators = sorted(validators.values(), key=lambda v: v.stake, reverse=True)
        cumulative_stake = 0
        superminority_threshold = new_stakes.sum() * 0.333
        new_superminority = []
        
        for v in sorted_validators:
            v.is_superminority = False
        
        for v in sorted_validators:
            cumulative_stake += v.stake
            v.is_superminority = True
            new_superminority.append(v.id)
            if cumulative_stake >= superminority_threshold:
                break
        
        topology.superminority_set = new_superminority
        topology.stake_distribution = new_stakes
        
        return topology
    
    def _update_migration_status(
        self,
        topology: NetworkTopology,
        migration_progress: float,
        year: float,
        rng: np.random.RandomState
    ):
        """Update validator migration status based on progress."""
        for validator in topology.validators.values():
            if validator.migration_status != MigrationStatus.COMPLETED:
                if rng.random() < migration_progress * 0.1:  # 10% chance per year
                    validator.migration_status = MigrationStatus.IN_PROGRESS
                    if rng.random() < 0.3:  # 30% complete immediately
                        validator.migration_status = MigrationStatus.COMPLETED
    
    def _create_graph_snapshot(
        self,
        topology: NetworkTopology,
        year: float,
        migration_progress: float,
        crqc_year: float,
        rng: np.random.RandomState
    ) -> NetworkSnapshot:
        """Create network snapshot from graph topology."""
        # Calculate metrics
        migrated_validators = sum(
            1 for v in topology.validators.values()
            if v.migration_status == MigrationStatus.COMPLETED
        )
        
        vulnerable_stake = sum(
            v.stake for v in topology.validators.values()
            if v.migration_status != MigrationStatus.COMPLETED
        )
        
        total_stake = sum(v.stake for v in topology.validators.values())
        vulnerable_stake_pct = vulnerable_stake / total_stake if total_stake > 0 else 0
        
        # Calculate network resilience based on graph metrics
        clustering = topology.get_clustering_coefficient()
        components = topology.get_connected_components()
        largest_component_size = max(len(c) for c in components) if components else 0
        
        resilience = (
            0.3 * (1 - vulnerable_stake_pct) +  # Migration progress
            0.2 * clustering +  # Network cohesion
            0.2 * (largest_component_size / len(topology.validators)) if len(topology.validators) > 0 else 0 +
            0.3 * (1 - self._calculate_gini_coefficient(list(topology.validators.values())))  # Decentralization
        )
        
        # Determine migration status
        if migration_progress == 0:
            status = MigrationStatus.NOT_STARTED
        elif migration_progress < 0.3:
            status = MigrationStatus.IN_PROGRESS
        elif migration_progress < 0.95:
            status = MigrationStatus.PARTIAL
        else:
            status = MigrationStatus.COMPLETE
        
        # Create simplified validator states for snapshot
        validator_states = []
        for i, (vid, v) in enumerate(list(topology.validators.items())[:100]):  # Sample for efficiency
            tier = ValidatorTier.SUPERMINORITY if v.is_superminority else ValidatorTier.MEDIUM
            validator_states.append(ValidatorState(
                validator_id=i,
                stake_amount=v.stake,
                stake_percentage=v.stake / total_stake if total_stake > 0 else 0,
                tier=tier,
                location=v.location,
                is_migrated=v.migration_status == MigrationStatus.COMPLETED,
                migration_year=year if v.migration_status == MigrationStatus.COMPLETED else None
            ))
        
        # Create snapshot
        snapshot = NetworkSnapshot(
            year=year,
            n_validators=len(topology.validators),
            total_stake=total_stake,
            validators=validator_states,
            geographic_distribution=self._calculate_geographic_distribution(rng, year),
            migration_status=status,
            migration_progress=migration_progress,
            superminority_count=len(topology.superminority_set),
            gini_coefficient=self._calculate_gini_coefficient(list(topology.validators.values())),
            network_resilience=resilience,
            compromised_validators=0,
            vulnerable_stake_percentage=vulnerable_stake_pct,
            attack_occurred=False,
            topology=topology
        )
        
        # Store graph metrics
        if NETWORKX_AVAILABLE and topology.graph:
            snapshot.graph_metrics = {
                'clustering_coefficient': clustering,
                'n_components': len(components),
                'largest_component_size': largest_component_size,
                'edge_count': topology.graph.number_of_edges(),
                'avg_degree': 2 * topology.graph.number_of_edges() / len(topology.validators)
            }
        
        return snapshot
    
    def simulate_centralization_scenario(
        self,
        topology: NetworkTopology,
        top_validator_stake_pct: float,
        rng: np.random.RandomState
    ) -> NetworkTopology:
        """
        Simulate stake centralization scenario.
        
        Args:
            topology: Current topology
            top_validator_stake_pct: Target stake % for top validators
            rng: Random number generator
            
        Returns:
            Modified topology with centralized stake
        """
        validators = list(topology.validators.values())
        total_stake = sum(v.stake for v in validators)
        
        # Sort by current stake
        validators.sort(key=lambda v: v.stake, reverse=True)
        
        # Number of top validators (e.g., top 10)
        n_top = 10
        target_top_stake = total_stake * top_validator_stake_pct
        current_top_stake = sum(v.stake for v in validators[:n_top])
        
        if current_top_stake < target_top_stake:
            # Need to concentrate more stake
            stake_to_move = target_top_stake - current_top_stake
            
            # Take stake from smaller validators
            for v in validators[n_top:]:
                if stake_to_move <= 0:
                    break
                
                # Move up to 50% of validator's stake
                move_amount = min(v.stake * 0.5, stake_to_move)
                v.stake -= move_amount
                stake_to_move -= move_amount
                
                # Distribute to top validators
                for top_v in validators[:n_top]:
                    top_v.stake += move_amount / n_top
        
        # Update topology
        topology.stake_distribution = np.array([v.stake for v in topology.validators.values()])
        
        return topology
    
    def simulate_attack_propagation(
        self,
        topology: NetworkTopology,
        initial_compromised: List[str],
        propagation_probability: float = 0.3
    ) -> Dict[str, Any]:
        """
        Simulate attack propagation through network.
        
        Args:
            topology: Network topology
            initial_compromised: Initially compromised validator IDs
            propagation_probability: Probability of spreading to neighbor
            
        Returns:
            Dictionary with propagation results
        """
        if not NETWORKX_AVAILABLE or not topology.graph:
            return {
                'rounds': [],
                'total_compromised': len(initial_compromised),
                'compromised_validators': initial_compromised,
                'compromised_stake': 0,
                'compromised_stake_pct': 0,
                'propagation_rounds': 0,
                'network_halted': False
            }
        
        G = topology.graph
        compromised = set(initial_compromised)
        rounds = []
        
        # Map validator IDs to graph nodes
        id_to_node = {f"validator_{i}": i for i in range(len(topology.validators))}
        
        # Simulate propagation rounds
        max_rounds = 10
        for round_num in range(max_rounds):
            new_compromised = set()
            
            for validator_id in compromised:
                if validator_id in id_to_node:
                    node = id_to_node[validator_id]
                    
                    # Check neighbors
                    for neighbor in G.neighbors(node):
                        neighbor_id = f"validator_{neighbor}"
                        if neighbor_id not in compromised:
                            # Propagation based on probability
                            if np.random.random() < propagation_probability:
                                new_compromised.add(neighbor_id)
            
            if not new_compromised:
                break  # No more propagation
            
            compromised.update(new_compromised)
            rounds.append({
                'round': round_num + 1,
                'new_compromised': len(new_compromised),
                'total_compromised': len(compromised)
            })
        
        # Calculate impact
        total_stake = sum(v.stake for v in topology.validators.values())
        compromised_stake = sum(
            topology.validators[vid].stake 
            for vid in compromised 
            if vid in topology.validators
        )
        
        return {
            'rounds': rounds,
            'total_compromised': len(compromised),
            'compromised_validators': list(compromised),
            'compromised_stake': compromised_stake,
            'compromised_stake_pct': compromised_stake / total_stake if total_stake > 0 else 0,
            'propagation_rounds': len(rounds),
            'network_halted': compromised_stake / total_stake > 0.333 if total_stake > 0 else False
        }
    
    def _update_with_live_data(self):
        """Update model parameters with live Solana data."""
        # This is a placeholder for live data integration
        # In production, this would fetch from Solana APIs
        pass


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
