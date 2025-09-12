"""
Tests for attack scenarios model.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.models.attack_scenarios import (
    AttackScenariosModel,
    AttackPlan,
    AttackScenario,
    AttackWindow,
    AttackType,
    AttackVector,
    AttackSeverity
)
from src.models.quantum_timeline import QuantumCapability, QuantumThreat
from src.models.network_state import NetworkSnapshot, ValidatorState, ValidatorTier, MigrationStatus
from src.config import QuantumParameters


class TestAttackScenario:
    """Test AttackScenario class."""
    
    def test_initialization(self):
        """Test attack scenario initialization."""
        scenario = AttackScenario(
            attack_type=AttackType.KEY_COMPROMISE,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.8,
            severity=AttackSeverity.MEDIUM,
            validators_compromised=5,
            stake_compromised=0.1,
            accounts_at_risk=100000,
            time_to_execute=10.0,
            detection_probability=0.3,
            mitigation_possible=True
        )
        
        assert scenario.attack_type == AttackType.KEY_COMPROMISE
        assert scenario.success_probability == 0.8
        assert scenario.validators_compromised == 5
    
    def test_impact_score(self):
        """Test impact score calculation."""
        # Low severity attack
        low_attack = AttackScenario(
            attack_type=AttackType.KEY_COMPROMISE,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.5,
            severity=AttackSeverity.LOW,
            validators_compromised=1,
            stake_compromised=0.01,
            accounts_at_risk=1000,
            time_to_execute=1.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        # Critical severity attack
        critical_attack = AttackScenario(
            attack_type=AttackType.SYSTEMIC_FAILURE,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.9,
            severity=AttackSeverity.CRITICAL,
            validators_compromised=1000,
            stake_compromised=0.9,
            accounts_at_risk=10000000,
            time_to_execute=100.0,
            detection_probability=0.1,
            mitigation_possible=False
        )
        
        assert low_attack.impact_score < critical_attack.impact_score
        assert 0 <= low_attack.impact_score <= 1
        assert 0 <= critical_attack.impact_score <= 1


class TestAttackWindow:
    """Test AttackWindow class."""
    
    def test_initialization(self):
        """Test attack window initialization."""
        window = AttackWindow(
            start_year=2035,
            end_year=2038,
            peak_year=2036,
            opportunity_score=0.7
        )
        
        assert window.start_year == 2035
        assert window.end_year == 2038
        assert window.duration == 3.0
    
    def test_is_active(self):
        """Test window activity check."""
        window = AttackWindow(
            start_year=2035,
            end_year=2038,
            peak_year=2036,
            opportunity_score=0.7
        )
        
        assert not window.is_active(2034)
        assert window.is_active(2035)
        assert window.is_active(2036)
        assert window.is_active(2038)
        assert not window.is_active(2039)


class TestAttackPlan:
    """Test AttackPlan class."""
    
    def test_initialization(self):
        """Test attack plan initialization."""
        scenarios = [
            AttackScenario(
                attack_type=AttackType.KEY_COMPROMISE,
                vector=AttackVector.VALIDATOR_KEYS,
                year=2035,
                success_probability=0.8,
                severity=AttackSeverity.MEDIUM,
                validators_compromised=5,
                stake_compromised=0.1,
                accounts_at_risk=100000,
                time_to_execute=10.0,
                detection_probability=0.3,
                mitigation_possible=True
            )
        ]
        
        windows = [
            AttackWindow(
                start_year=2035,
                end_year=2038,
                peak_year=2036,
                opportunity_score=0.7
            )
        ]
        
        plan = AttackPlan(
            scenarios=scenarios,
            windows=windows,
            primary_target="validator",
            estimated_success_rate=0.8,
            total_stake_at_risk=0.1,
            recommended_year=2035,
            contingency_scenarios=[]
        )
        
        assert len(plan.scenarios) == 1
        assert len(plan.windows) == 1
        assert plan.primary_target == "validator"
    
    def test_get_scenario_at_year(self):
        """Test getting scenario at specific year."""
        early_scenario = AttackScenario(
            attack_type=AttackType.KEY_COMPROMISE,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.5,
            severity=AttackSeverity.LOW,
            validators_compromised=1,
            stake_compromised=0.01,
            accounts_at_risk=1000,
            time_to_execute=1.0,
            detection_probability=0.9,
            mitigation_possible=True
        )
        
        late_scenario = AttackScenario(
            attack_type=AttackType.CONSENSUS_HALT,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2037,
            success_probability=0.8,
            severity=AttackSeverity.HIGH,
            validators_compromised=100,
            stake_compromised=0.4,
            accounts_at_risk=1000000,
            time_to_execute=50.0,
            detection_probability=0.5,
            mitigation_possible=False
        )
        
        plan = AttackPlan(
            scenarios=[early_scenario, late_scenario],
            windows=[],
            primary_target="validator",
            estimated_success_rate=0.65,
            total_stake_at_risk=0.4,
            recommended_year=2035,
            contingency_scenarios=[]
        )
        
        # Before any scenarios
        assert plan.get_scenario_at_year(2034) is None
        
        # After first scenario
        scenario_2036 = plan.get_scenario_at_year(2036)
        assert scenario_2036 == early_scenario
        
        # After both scenarios (should return higher impact)
        scenario_2038 = plan.get_scenario_at_year(2038)
        assert scenario_2038 == late_scenario  # Higher impact


class TestAttackScenariosModel:
    """Test AttackScenariosModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = AttackScenariosModel()
        
        assert model.params is not None
        assert AttackType.KEY_COMPROMISE in model.attack_requirements
        assert AttackType.SYSTEMIC_FAILURE in model.attack_requirements
        assert all(0 < rate <= 1 for rate in model.base_success_rates.values())
    
    def test_feasible_attacks_identification(self):
        """Test identification of feasible attacks."""
        model = AttackScenariosModel()
        
        # Low capability - only simple attacks
        low_capability = QuantumCapability(
            year=2030,
            logical_qubits=2500,  # Just enough for key compromise
            physical_qubits=2500000,
            gate_fidelity=0.99,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.EMERGING
        )
        
        # High migration network - limits some attacks
        network_high_migration = NetworkSnapshot(
            year=2030,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={},
            migration_status=MigrationStatus.PARTIAL,
            migration_progress=0.8,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.7
        )
        network_high_migration.vulnerable_stake_percentage = 0.2  # Only 20% vulnerable
        
        feasible = model._identify_feasible_attacks(low_capability, network_high_migration)
        
        # Should allow key compromise but not consensus attacks
        assert AttackType.KEY_COMPROMISE in feasible
        assert AttackType.CONSENSUS_HALT not in feasible  # Not enough vulnerable stake
        assert AttackType.CONSENSUS_CONTROL not in feasible  # Not enough vulnerable stake
    
    def test_success_probability_calculation(self):
        """Test attack success probability calculation."""
        model = AttackScenariosModel()
        
        capability = QuantumCapability(
            year=2035,
            logical_qubits=5000,
            physical_qubits=5000000,
            gate_fidelity=0.9999,
            coherence_time_ms=10.0,
            threat_level=QuantumThreat.HIGH
        )
        
        # Low migration network
        network_low_migration = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.1,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.3
        )
        
        # High migration network
        network_high_migration = NetworkSnapshot(
            year=2035,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={},
            migration_status=MigrationStatus.PARTIAL,
            migration_progress=0.9,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.8
        )
        
        # Success should be higher with low migration
        prob_low_migration = model._calculate_success_probability(
            AttackType.KEY_COMPROMISE,
            capability,
            network_low_migration
        )
        
        prob_high_migration = model._calculate_success_probability(
            AttackType.KEY_COMPROMISE,
            capability,
            network_high_migration
        )
        
        assert prob_low_migration > prob_high_migration
        assert 0 <= prob_low_migration <= 1
        assert 0 <= prob_high_migration <= 1
    
    def test_sample_generation(self):
        """Test attack plan generation."""
        model = AttackScenariosModel()
        rng = np.random.RandomState(42)
        
        # Create test capability
        capability = QuantumCapability(
            year=2035,
            logical_qubits=10000,  # Enough for various attacks
            physical_qubits=10000000,
            gate_fidelity=0.9999,
            coherence_time_ms=10.0,
            threat_level=QuantumThreat.HIGH
        )
        
        # Create test network with validators
        validators = []
        for i in range(100):
            validators.append(ValidatorState(
                validator_id=i,
                stake_amount=1000000 * (100 - i),  # Decreasing stake
                stake_percentage=0.01 * (100 - i) / 50.5,
                tier=ValidatorTier.SUPERMINORITY if i < 10 else ValidatorTier.SMALL,
                location='north_america',
                is_migrated=(i < 20)  # 20% migrated
            ))
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=100,
            total_stake=50500000,
            validators=validators,
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.2,
            superminority_count=10,
            gini_coefficient=0.8,
            network_resilience=0.4
        )
        
        # Generate attack plan
        plan = model.sample(rng, capability, network)
        
        assert isinstance(plan, AttackPlan)
        assert len(plan.scenarios) > 0  # Should have feasible attacks
        assert plan.estimated_success_rate > 0
        assert plan.primary_target in ["validator", "user", "protocol", "none"]
    
    def test_no_attacks_possible(self):
        """Test when no attacks are possible."""
        model = AttackScenariosModel()
        rng = np.random.RandomState(42)
        
        # Very low capability
        capability = QuantumCapability(
            year=2025,
            logical_qubits=100,  # Too few for any attack
            physical_qubits=100000,
            gate_fidelity=0.99,
            coherence_time_ms=0.1,
            threat_level=QuantumThreat.NONE
        )
        
        network = NetworkSnapshot(
            year=2025,
            n_validators=3000,
            total_stake=400000000,
            validators=[],
            geographic_distribution={},
            migration_status=MigrationStatus.NOT_STARTED,
            migration_progress=0.0,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.5
        )
        
        plan = model.sample(rng, capability, network)
        
        assert len(plan.scenarios) == 0
        assert plan.estimated_success_rate == 0.0
        assert plan.primary_target == "none"
    
    def test_execution_time_calculation(self):
        """Test attack execution time calculation."""
        model = AttackScenariosModel()
        
        fast_capability = QuantumCapability(
            year=2040,
            logical_qubits=100000,
            physical_qubits=100000000,
            gate_fidelity=0.999999,
            coherence_time_ms=100.0,
            threat_level=QuantumThreat.CRITICAL
        )
        
        slow_capability = QuantumCapability(
            year=2030,
            logical_qubits=3000,
            physical_qubits=3000000,
            gate_fidelity=0.999,
            coherence_time_ms=1.0,
            threat_level=QuantumThreat.EMERGING
        )
        
        # Fast capability should have shorter execution time
        fast_time = model._calculate_execution_time(
            AttackType.KEY_COMPROMISE,
            fast_capability,
            validators_compromised=1
        )
        
        slow_time = model._calculate_execution_time(
            AttackType.KEY_COMPROMISE,
            slow_capability,
            validators_compromised=1
        )
        
        assert fast_time < slow_time
        assert fast_time > 0  # Should have minimum time
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        model = AttackScenariosModel()
        
        capability = QuantumCapability(
            year=2035,
            logical_qubits=5000,
            physical_qubits=5000000,
            gate_fidelity=0.9999,
            coherence_time_ms=10.0,
            threat_level=QuantumThreat.HIGH
        )
        
        validators = [
            ValidatorState(
                validator_id=i,
                stake_amount=1000000,
                stake_percentage=0.01,
                tier=ValidatorTier.SMALL,
                location='north_america',
                is_migrated=False
            )
            for i in range(100)
        ]
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=100,
            total_stake=100000000,
            validators=validators,
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.3,
            superminority_count=10,
            gini_coefficient=0.8,
            network_resilience=0.5
        )
        
        # Generate two plans with same seed
        rng1 = np.random.RandomState(123)
        plan1 = model.sample(rng1, capability, network)
        
        rng2 = np.random.RandomState(123)
        plan2 = model.sample(rng2, capability, network)
        
        # Should produce identical results
        assert plan1.estimated_success_rate == plan2.estimated_success_rate
        assert len(plan1.scenarios) == len(plan2.scenarios)
        if plan1.scenarios and plan2.scenarios:
            assert plan1.scenarios[0].success_probability == plan2.scenarios[0].success_probability
