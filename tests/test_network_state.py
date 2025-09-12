"""
Tests for network state model.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.models.network_state import (
    NetworkStateModel,
    NetworkEvolution,
    NetworkSnapshot,
    ValidatorState,
    ValidatorTier,
    MigrationStatus
)
from src.config import NetworkParameters


class TestValidatorState:
    """Test ValidatorState class."""
    
    def test_initialization(self):
        """Test validator state initialization."""
        validator = ValidatorState(
            validator_id=1,
            stake_amount=1000000,
            stake_percentage=0.01,
            tier=ValidatorTier.LARGE,
            location='north_america',
            is_migrated=False,
            uptime=0.99
        )
        
        assert validator.validator_id == 1
        assert validator.stake_amount == 1000000
        assert validator.tier == ValidatorTier.LARGE
        assert not validator.is_migrated
    
    def test_is_critical(self):
        """Test critical validator identification."""
        # Superminority validator is critical
        validator1 = ValidatorState(
            validator_id=1,
            stake_amount=10000000,
            stake_percentage=0.1,
            tier=ValidatorTier.SUPERMINORITY,
            location='europe',
            is_migrated=False
        )
        assert validator1.is_critical
        
        # Large validator is critical
        validator2 = ValidatorState(
            validator_id=2,
            stake_amount=5000000,
            stake_percentage=0.05,
            tier=ValidatorTier.LARGE,
            location='asia',
            is_migrated=False
        )
        assert validator2.is_critical
        
        # Small validator is not critical
        validator3 = ValidatorState(
            validator_id=3,
            stake_amount=10000,
            stake_percentage=0.0001,
            tier=ValidatorTier.SMALL,
            location='other',
            is_migrated=False
        )
        assert not validator3.is_critical


class TestNetworkSnapshot:
    """Test NetworkSnapshot class."""
    
    def test_initialization(self):
        """Test network snapshot initialization."""
        validators = [
            ValidatorState(
                validator_id=i,
                stake_amount=1000000 * (10 - i),
                stake_percentage=0.1 * (10 - i) / 55,
                tier=ValidatorTier.LARGE if i < 5 else ValidatorTier.SMALL,
                location='north_america',
                is_migrated=(i < 3)
            )
            for i in range(10)
        ]
        
        snapshot = NetworkSnapshot(
            year=2030,
            n_validators=10,
            total_stake=55000000,
            validators=validators,
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=3,
            gini_coefficient=0.8,
            network_resilience=0.5
        )
        
        assert snapshot.year == 2030
        assert snapshot.n_validators == 10
        assert len(snapshot.validators) == 10
    
    def test_migrated_stake_percentage(self):
        """Test calculation of migrated stake percentage."""
        validators = [
            ValidatorState(
                validator_id=0,
                stake_amount=6000000,
                stake_percentage=0.6,
                tier=ValidatorTier.SUPERMINORITY,
                location='north_america',
                is_migrated=True
            ),
            ValidatorState(
                validator_id=1,
                stake_amount=4000000,
                stake_percentage=0.4,
                tier=ValidatorTier.LARGE,
                location='europe',
                is_migrated=False
            )
        ]
        
        snapshot = NetworkSnapshot(
            year=2035,
            n_validators=2,
            total_stake=10000000,
            validators=validators,
            geographic_distribution={'north_america': 0.5, 'europe': 0.5},
            migration_status=MigrationStatus.PARTIAL,
            migration_progress=0.6,
            superminority_count=1,
            gini_coefficient=0.7,
            network_resilience=0.6
        )
        
        assert snapshot.migrated_stake_percentage == 0.6
        assert snapshot.vulnerable_stake_percentage == 0.4
    
    def test_attack_surface(self):
        """Test attack surface calculation."""
        validators = [
            ValidatorState(
                validator_id=0,
                stake_amount=5000000,
                stake_percentage=0.5,
                tier=ValidatorTier.SUPERMINORITY,
                location='north_america',
                is_migrated=False  # Vulnerable
            ),
            ValidatorState(
                validator_id=1,
                stake_amount=3000000,
                stake_percentage=0.3,
                tier=ValidatorTier.LARGE,
                location='europe',
                is_migrated=True  # Safe
            ),
            ValidatorState(
                validator_id=2,
                stake_amount=2000000,
                stake_percentage=0.2,
                tier=ValidatorTier.MEDIUM,
                location='asia',
                is_migrated=False  # Vulnerable
            )
        ]
        
        snapshot = NetworkSnapshot(
            year=2035,
            n_validators=3,
            total_stake=10000000,
            validators=validators,
            geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
            migration_status=MigrationStatus.PARTIAL,
            migration_progress=0.3,
            superminority_count=1,
            gini_coefficient=0.75,
            network_resilience=0.4
        )
        
        attack_surface = snapshot.get_attack_surface()
        
        assert attack_surface['vulnerable_validators'] == 2
        assert attack_surface['vulnerable_stake'] == 7000000
        assert attack_surface['vulnerable_stake_percentage'] == 0.7
        assert attack_surface['superminority_vulnerable'] is True


class TestNetworkEvolution:
    """Test NetworkEvolution class."""
    
    def test_initialization(self):
        """Test network evolution initialization."""
        snapshots = [
            NetworkSnapshot(
                year=2025 + i,
                n_validators=3000 + i * 100,
                total_stake=400000000 + i * 10000000,
                validators=[],
                geographic_distribution={},
                migration_status=MigrationStatus.NOT_STARTED,
                migration_progress=0,
                superminority_count=30,
                gini_coefficient=0.84,
                network_resilience=0.3
            )
            for i in range(5)
        ]
        
        evolution = NetworkEvolution(
            snapshots=snapshots,
            migration_start_year=2033,
            migration_completion_year=2038,
            peak_validators=3500,
            minimum_gini=0.80
        )
        
        assert len(evolution.snapshots) == 5
        assert evolution.migration_start_year == 2033
        assert evolution.peak_validators == 3500
    
    def test_get_snapshot_at_year(self):
        """Test getting snapshot at specific year."""
        snapshots = [
            NetworkSnapshot(
                year=year,
                n_validators=3000,
                total_stake=400000000,
                validators=[],
                geographic_distribution={},
                migration_status=MigrationStatus.NOT_STARTED,
                migration_progress=0,
                superminority_count=30,
                gini_coefficient=0.84,
                network_resilience=0.3
            )
            for year in [2025, 2030, 2035, 2040]
        ]
        
        evolution = NetworkEvolution(
            snapshots=snapshots,
            migration_start_year=2033,
            migration_completion_year=None,
            peak_validators=3000,
            minimum_gini=0.84
        )
        
        # Get exact year
        snapshot_2030 = evolution.get_snapshot_at_year(2030)
        assert snapshot_2030.year == 2030
        
        # Get interpolated year
        snapshot_2032 = evolution.get_snapshot_at_year(2032)
        assert snapshot_2032.year == 2035  # Returns next available
        
        # Beyond range
        snapshot_2045 = evolution.get_snapshot_at_year(2045)
        assert snapshot_2045.year == 2040  # Returns last available


class TestNetworkStateModel:
    """Test NetworkStateModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = NetworkStateModel()
        
        assert model.params is not None
        assert 'proactive' in model.migration_profiles
        assert 'reactive' in model.migration_profiles
        assert 'laggard' in model.migration_profiles
    
    def test_sample_generation(self):
        """Test network evolution generation."""
        model = NetworkStateModel()
        rng = np.random.RandomState(42)
        
        quantum_timeline = {
            'crqc_year': 2035,
            'capabilities': []
        }
        
        evolution = model.sample(rng, quantum_timeline)
        
        assert isinstance(evolution, NetworkEvolution)
        assert len(evolution.snapshots) == 21  # 2025-2045
        assert evolution.migration_start_year is not None
        
        # Check progression over time
        early_snapshot = evolution.get_snapshot_at_year(2025)
        late_snapshot = evolution.get_snapshot_at_year(2040)
        
        # Validators should grow
        assert late_snapshot.n_validators >= early_snapshot.n_validators
        
        # Migration should progress
        assert late_snapshot.migration_progress >= early_snapshot.migration_progress
    
    def test_migration_profiles(self):
        """Test different migration profiles."""
        model = NetworkStateModel()
        rng = np.random.RandomState(42)
        
        profile = model._select_migration_profile(rng)
        
        assert 'adoption_rate' in profile
        assert 'speed' in profile
        assert 'start_delay' in profile
        
        # Values should be reasonable
        assert 0 < profile['adoption_rate'] <= 1.2
        assert 0 < profile['speed'] <= 3
        assert -3 <= profile['start_delay'] <= 3
    
    def test_migration_progress_calculation(self):
        """Test migration progress calculation."""
        model = NetworkStateModel()
        
        profile = {
            'adoption_rate': 0.8,
            'speed': 1.0
        }
        
        # Before migration starts
        progress = model._calculate_migration_progress(2030, 2035, profile)
        assert progress == 0.0
        
        # During migration
        progress = model._calculate_migration_progress(2038, 2035, profile)
        assert 0 < progress < profile['adoption_rate']
        
        # Well after migration starts
        progress = model._calculate_migration_progress(2045, 2035, profile)
        assert progress == pytest.approx(profile['adoption_rate'], rel=0.01)
    
    def test_validator_generation(self):
        """Test validator distribution generation."""
        model = NetworkStateModel()
        rng = np.random.RandomState(42)
        
        validators = model._generate_validators(rng, 100, 1000000)
        
        assert len(validators) == 100
        
        # Check stake distribution
        total_stake = sum(v.stake_amount for v in validators)
        assert total_stake == pytest.approx(1000000, rel=0.01)
        
        # Check tier assignment
        tiers = [v.tier for v in validators]
        assert ValidatorTier.SUPERMINORITY in tiers
        assert ValidatorTier.SMALL in tiers
        
        # Superminority should control ~33% of stake
        superminority_stake = sum(
            v.stake_amount for v in validators 
            if v.tier == ValidatorTier.SUPERMINORITY
        )
        assert 0.25 < superminority_stake / total_stake < 0.40
    
    def test_gini_calculation(self):
        """Test Gini coefficient calculation."""
        model = NetworkStateModel()
        
        # Perfect equality
        equal_validators = [
            ValidatorState(
                validator_id=i,
                stake_amount=100,
                stake_percentage=0.1,
                tier=ValidatorTier.MEDIUM,
                location='north_america',
                is_migrated=False
            )
            for i in range(10)
        ]
        
        gini_equal = model._calculate_gini_coefficient(equal_validators)
        assert gini_equal < 0.1  # Should be close to 0
        
        # High inequality
        unequal_validators = [
            ValidatorState(
                validator_id=0,
                stake_amount=900,
                stake_percentage=0.9,
                tier=ValidatorTier.SUPERMINORITY,
                location='north_america',
                is_migrated=False
            )
        ] + [
            ValidatorState(
                validator_id=i,
                stake_amount=100/9,
                stake_percentage=0.1/9,
                tier=ValidatorTier.SMALL,
                location='north_america',
                is_migrated=False
            )
            for i in range(1, 10)
        ]
        
        gini_unequal = model._calculate_gini_coefficient(unequal_validators)
        assert gini_unequal > 0.7  # Should be high
    
    def test_resilience_calculation(self):
        """Test network resilience calculation."""
        model = NetworkStateModel()
        
        # High resilience scenario
        migrated_validators = [
            ValidatorState(
                validator_id=i,
                stake_amount=1000,
                stake_percentage=0.01,
                tier=ValidatorTier.MEDIUM,
                location='north_america',
                is_migrated=True
            )
            for i in range(100)
        ]
        
        resilience_high = model._calculate_resilience(
            migrated_validators,
            migration_progress=0.9,
            year=2030,
            crqc_year=2040
        )
        assert resilience_high > 0.7
        
        # Low resilience scenario
        vulnerable_validators = [
            ValidatorState(
                validator_id=i,
                stake_amount=1000,
                stake_percentage=0.01,
                tier=ValidatorTier.SUPERMINORITY if i < 10 else ValidatorTier.SMALL,
                location='north_america',
                is_migrated=False
            )
            for i in range(100)
        ]
        
        resilience_low = model._calculate_resilience(
            vulnerable_validators,
            migration_progress=0.1,
            year=2034,
            crqc_year=2035
        )
        assert resilience_low < 0.3
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        model = NetworkStateModel()
        quantum_timeline = {'crqc_year': 2035}
        
        rng1 = np.random.RandomState(123)
        evolution1 = model.sample(rng1, quantum_timeline)
        
        rng2 = np.random.RandomState(123)
        evolution2 = model.sample(rng2, quantum_timeline)
        
        # Check key metrics are identical
        assert evolution1.migration_start_year == evolution2.migration_start_year
        assert evolution1.peak_validators == evolution2.peak_validators
        
        # Check snapshots match
        for i in range(len(evolution1.snapshots)):
            snap1 = evolution1.snapshots[i]
            snap2 = evolution2.snapshots[i]
            assert snap1.n_validators == snap2.n_validators
            assert snap1.migration_progress == snap2.migration_progress
