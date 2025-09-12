"""
Tests for economic impact model.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.models.economic_impact import (
    EconomicImpactModel,
    EconomicLoss,
    EconomicRecovery,
    ImpactComponent,
    MarketReaction,
    ImpactType,
    RecoverySpeed
)
from src.models.attack_scenarios import (
    AttackScenario,
    AttackType,
    AttackVector,
    AttackSeverity
)
from src.models.network_state import (
    NetworkSnapshot,
    ValidatorState,
    ValidatorTier,
    MigrationStatus
)
from src.config import EconomicParameters


class TestImpactComponent:
    """Test ImpactComponent class."""
    
    def test_initialization(self):
        """Test impact component initialization."""
        component = ImpactComponent(
            impact_type=ImpactType.DIRECT_LOSS,
            amount_usd=1_000_000_000,
            percentage_of_tvl=0.082,
            time_to_realize_days=0.5,
            confidence_interval=(500_000_000, 1_500_000_000)
        )
        
        assert component.impact_type == ImpactType.DIRECT_LOSS
        assert component.amount_usd == 1_000_000_000
        assert component.percentage_of_tvl == 0.082
    
    def test_is_immediate(self):
        """Test immediate impact detection."""
        # Immediate impact (< 1 day)
        immediate = ImpactComponent(
            impact_type=ImpactType.DIRECT_LOSS,
            amount_usd=1_000_000_000,
            percentage_of_tvl=0.082,
            time_to_realize_days=0.5,
            confidence_interval=(500_000_000, 1_500_000_000)
        )
        assert immediate.is_immediate
        
        # Not immediate (> 1 day)
        delayed = ImpactComponent(
            impact_type=ImpactType.REPUTATION,
            amount_usd=500_000_000,
            percentage_of_tvl=0.041,
            time_to_realize_days=30,
            confidence_interval=(250_000_000, 750_000_000)
        )
        assert not delayed.is_immediate


class TestMarketReaction:
    """Test MarketReaction class."""
    
    def test_initialization(self):
        """Test market reaction initialization."""
        reaction = MarketReaction(
            sol_price_drop_percent=30,
            tvl_drop_percent=45,
            daily_volume_drop_percent=60,
            panic_duration_days=7,
            recovery_time_days=30
        )
        
        assert reaction.sol_price_drop_percent == 30
        assert reaction.tvl_drop_percent == 45
        assert reaction.panic_duration_days == 7
    
    def test_total_market_impact(self):
        """Test total market impact calculation."""
        # Mild reaction
        mild = MarketReaction(
            sol_price_drop_percent=10,
            tvl_drop_percent=15,
            daily_volume_drop_percent=20,
            panic_duration_days=2,
            recovery_time_days=7
        )
        
        # Severe reaction
        severe = MarketReaction(
            sol_price_drop_percent=50,
            tvl_drop_percent=70,
            daily_volume_drop_percent=80,
            panic_duration_days=30,
            recovery_time_days=180
        )
        
        assert mild.total_market_impact < severe.total_market_impact
        assert 0 <= mild.total_market_impact <= 1
        assert 0 <= severe.total_market_impact <= 1


class TestEconomicLoss:
    """Test EconomicLoss class."""
    
    def test_initialization(self):
        """Test economic loss initialization."""
        attack = AttackScenario(
            attack_type=AttackType.CONSENSUS_HALT,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=AttackSeverity.HIGH,
            validators_compromised=100,
            stake_compromised=0.35,
            accounts_at_risk=1000000,
            time_to_execute=24.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        components = [
            ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=1_000_000_000,
                percentage_of_tvl=0.082,
                time_to_realize_days=0.1,
                confidence_interval=(500_000_000, 1_500_000_000)
            ),
            ImpactComponent(
                impact_type=ImpactType.MARKET_CRASH,
                amount_usd=5_000_000_000,
                percentage_of_tvl=0.41,
                time_to_realize_days=1.0,
                confidence_interval=(2_500_000_000, 7_500_000_000)
            )
        ]
        
        market_reaction = MarketReaction(
            sol_price_drop_percent=30,
            tvl_drop_percent=45,
            daily_volume_drop_percent=60,
            panic_duration_days=7,
            recovery_time_days=30
        )
        
        loss = EconomicLoss(
            attack_scenario=attack,
            components=components,
            market_reaction=market_reaction,
            total_loss_usd=6_000_000_000,
            immediate_loss_usd=1_000_000_000,
            long_term_loss_usd=5_000_000_000,
            recovery_speed=RecoverySpeed.SLOW,
            recovery_timeline_days=90,
            confidence_level=0.95
        )
        
        assert loss.total_loss_usd == 6_000_000_000
        assert loss.immediate_loss_usd == 1_000_000_000
        assert len(loss.components) == 2
    
    def test_get_loss_by_type(self):
        """Test getting loss by impact type."""
        components = [
            ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=1_000_000_000,
                percentage_of_tvl=0.082,
                time_to_realize_days=0.1,
                confidence_interval=(500_000_000, 1_500_000_000)
            ),
            ImpactComponent(
                impact_type=ImpactType.MARKET_CRASH,
                amount_usd=5_000_000_000,
                percentage_of_tvl=0.41,
                time_to_realize_days=1.0,
                confidence_interval=(2_500_000_000, 7_500_000_000)
            ),
            ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=500_000_000,
                percentage_of_tvl=0.041,
                time_to_realize_days=0.2,
                confidence_interval=(250_000_000, 750_000_000)
            )
        ]
        
        loss = EconomicLoss(
            attack_scenario=Mock(),
            components=components,
            market_reaction=Mock(),
            total_loss_usd=6_500_000_000,
            immediate_loss_usd=1_500_000_000,
            long_term_loss_usd=5_000_000_000,
            recovery_speed=RecoverySpeed.MODERATE,
            recovery_timeline_days=30,
            confidence_level=0.95
        )
        
        # Should sum all DIRECT_LOSS components
        direct_loss = loss.get_loss_by_type(ImpactType.DIRECT_LOSS)
        assert direct_loss == 1_500_000_000
        
        # Should get MARKET_CRASH component
        market_loss = loss.get_loss_by_type(ImpactType.MARKET_CRASH)
        assert market_loss == 5_000_000_000
        
        # Should return 0 for non-existent type
        defi_loss = loss.get_loss_by_type(ImpactType.DEFI_CASCADE)
        assert defi_loss == 0
    
    def test_cumulative_loss_at_day(self):
        """Test cumulative loss calculation at specific days."""
        components = [
            ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=1_000_000_000,
                percentage_of_tvl=0.082,
                time_to_realize_days=0.5,
                confidence_interval=(500_000_000, 1_500_000_000)
            ),
            ImpactComponent(
                impact_type=ImpactType.MARKET_CRASH,
                amount_usd=5_000_000_000,
                percentage_of_tvl=0.41,
                time_to_realize_days=2.0,
                confidence_interval=(2_500_000_000, 7_500_000_000)
            ),
            ImpactComponent(
                impact_type=ImpactType.REPUTATION,
                amount_usd=2_000_000_000,
                percentage_of_tvl=0.164,
                time_to_realize_days=30.0,
                confidence_interval=(1_000_000_000, 3_000_000_000)
            )
        ]
        
        loss = EconomicLoss(
            attack_scenario=Mock(),
            components=components,
            market_reaction=Mock(),
            total_loss_usd=8_000_000_000,
            immediate_loss_usd=1_000_000_000,
            long_term_loss_usd=7_000_000_000,
            recovery_speed=RecoverySpeed.SLOW,
            recovery_timeline_days=90,
            confidence_level=0.95
        )
        
        # Day 0: Nothing realized yet
        assert loss.get_cumulative_loss_at_day(0) == 0
        
        # Day 1: First component realized
        assert loss.get_cumulative_loss_at_day(1) == 1_000_000_000
        
        # Day 3: First two components realized
        assert loss.get_cumulative_loss_at_day(3) == 6_000_000_000
        
        # Day 31: All components realized
        assert loss.get_cumulative_loss_at_day(31) == 8_000_000_000


class TestEconomicRecovery:
    """Test EconomicRecovery class."""
    
    def test_initialization(self):
        """Test economic recovery initialization."""
        phases = [
            {'start_day': 0, 'end_day': 7, 'start_tvl': 0.5, 'end_tvl': 0.6},
            {'start_day': 7, 'end_day': 30, 'start_tvl': 0.6, 'end_tvl': 0.8},
            {'start_day': 30, 'end_day': 90, 'start_tvl': 0.8, 'end_tvl': 0.9}
        ]
        
        milestones = {
            '50% recovery': 0,
            '75% recovery': 20,
            '90% recovery': 90
        }
        
        recovery = EconomicRecovery(
            recovery_phases=phases,
            milestones=milestones,
            final_tvl_percent=0.9,
            permanent_damage_percent=10
        )
        
        assert len(recovery.recovery_phases) == 3
        assert recovery.final_tvl_percent == 0.9
        assert recovery.permanent_damage_percent == 10
    
    def test_get_tvl_at_day(self):
        """Test TVL calculation during recovery."""
        phases = [
            {'start_day': 0, 'end_day': 10, 'start_tvl': 0.5, 'end_tvl': 0.7},
            {'start_day': 10, 'end_day': 30, 'start_tvl': 0.7, 'end_tvl': 0.9}
        ]
        
        recovery = EconomicRecovery(
            recovery_phases=phases,
            milestones={},
            final_tvl_percent=0.9,
            permanent_damage_percent=10
        )
        
        initial_tvl = 10_000_000_000
        
        # Before attack (day -1)
        assert recovery.get_tvl_at_day(-1, initial_tvl) == initial_tvl
        
        # Day 0 (start of first phase)
        assert recovery.get_tvl_at_day(0, initial_tvl) == initial_tvl * 0.5
        
        # Day 5 (middle of first phase)
        tvl_day5 = recovery.get_tvl_at_day(5, initial_tvl)
        assert initial_tvl * 0.5 < tvl_day5 < initial_tvl * 0.7
        
        # Day 10 (start of second phase)
        assert recovery.get_tvl_at_day(10, initial_tvl) == initial_tvl * 0.7
        
        # Day 30 (end of second phase)
        assert recovery.get_tvl_at_day(30, initial_tvl) == initial_tvl * 0.9
        
        # Day 100 (after all phases)
        assert recovery.get_tvl_at_day(100, initial_tvl) == initial_tvl * 0.9


class TestEconomicImpactModel:
    """Test EconomicImpactModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = EconomicImpactModel()
        
        assert model.params is not None
        assert AttackType.KEY_COMPROMISE in model.attack_impact_multipliers
        assert AttackSeverity.LOW in model.market_reaction_factors
        assert AttackSeverity.CRITICAL in model.recovery_speeds
    
    def test_calculate_impact(self):
        """Test economic impact calculation."""
        model = EconomicImpactModel()
        rng = np.random.RandomState(42)
        
        # Create test attack
        attack = AttackScenario(
            attack_type=AttackType.CONSENSUS_HALT,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=AttackSeverity.HIGH,
            validators_compromised=100,
            stake_compromised=0.35,
            accounts_at_risk=1000000,
            time_to_execute=24.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        # Create test network
        network = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.4
        )
        
        # Calculate impact
        impact = model.calculate_impact(rng, attack, network)
        
        assert isinstance(impact, EconomicLoss)
        assert impact.total_loss_usd > 0
        assert len(impact.components) > 0
        assert impact.recovery_speed == RecoverySpeed.SLOW
        
        # Check component types are present
        component_types = {c.impact_type for c in impact.components}
        assert ImpactType.DIRECT_LOSS in component_types
        assert ImpactType.MARKET_CRASH in component_types
    
    def test_impact_varies_by_severity(self):
        """Test that impact varies with attack severity."""
        model = EconomicImpactModel()
        rng = np.random.RandomState(42)
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.4
        )
        
        # Low severity attack
        low_attack = AttackScenario(
            attack_type=AttackType.KEY_COMPROMISE,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.9,
            severity=AttackSeverity.LOW,
            validators_compromised=1,
            stake_compromised=0.01,
            accounts_at_risk=10000,
            time_to_execute=1.0,
            detection_probability=0.3,
            mitigation_possible=True
        )
        
        # High severity attack
        high_attack = AttackScenario(
            attack_type=AttackType.CONSENSUS_CONTROL,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.5,
            severity=AttackSeverity.CRITICAL,
            validators_compromised=500,
            stake_compromised=0.7,
            accounts_at_risk=10000000,
            time_to_execute=100.0,
            detection_probability=0.95,
            mitigation_possible=False
        )
        
        low_impact = model.calculate_impact(rng, low_attack, network)
        high_impact = model.calculate_impact(rng, high_attack, network)
        
        assert low_impact.total_loss_usd < high_impact.total_loss_usd
        assert low_impact.recovery_timeline_days < high_impact.recovery_timeline_days
        assert low_impact.recovery_speed == RecoverySpeed.FAST
        assert high_impact.recovery_speed == RecoverySpeed.VERY_SLOW
    
    def test_simulate_recovery(self):
        """Test economic recovery simulation."""
        model = EconomicImpactModel()
        rng = np.random.RandomState(42)
        
        # Create a loss scenario
        attack = AttackScenario(
            attack_type=AttackType.CONSENSUS_HALT,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=AttackSeverity.HIGH,
            validators_compromised=100,
            stake_compromised=0.35,
            accounts_at_risk=1000000,
            time_to_execute=24.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.4
        )
        
        economic_loss = model.calculate_impact(rng, attack, network)
        initial_tvl = model.params.total_value_locked_usd
        
        # Simulate recovery
        recovery = model.simulate_recovery(rng, economic_loss, initial_tvl)
        
        assert isinstance(recovery, EconomicRecovery)
        assert len(recovery.recovery_phases) > 0
        assert 0 < recovery.final_tvl_percent <= 1
        assert recovery.permanent_damage_percent >= 0
        
        # Check TVL recovery over time
        tvl_day0 = recovery.get_tvl_at_day(0, initial_tvl)
        tvl_day30 = recovery.get_tvl_at_day(30, initial_tvl)
        tvl_day180 = recovery.get_tvl_at_day(180, initial_tvl)
        
        # TVL should generally increase over time
        assert tvl_day0 < initial_tvl
        assert tvl_day30 >= tvl_day0
        assert tvl_day180 >= tvl_day30
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        model = EconomicImpactModel()
        
        attack = AttackScenario(
            attack_type=AttackType.CONSENSUS_HALT,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=AttackSeverity.HIGH,
            validators_compromised=100,
            stake_compromised=0.35,
            accounts_at_risk=1000000,
            time_to_execute=24.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.4
        )
        
        # Run twice with same seed
        rng1 = np.random.RandomState(123)
        impact1 = model.calculate_impact(rng1, attack, network)
        
        rng2 = np.random.RandomState(123)
        impact2 = model.calculate_impact(rng2, attack, network)
        
        # Results should be identical
        assert impact1.total_loss_usd == impact2.total_loss_usd
        assert impact1.immediate_loss_usd == impact2.immediate_loss_usd
        assert impact1.recovery_timeline_days == impact2.recovery_timeline_days
