"""
Attack scenarios model for quantum threats against Solana.

This module models different quantum attack scenarios, their success
probabilities, and potential impacts on the Solana network.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import math

from src.config import QuantumParameters
from src.models.quantum_timeline import QuantumCapability, QuantumThreat
from src.models.network_state import NetworkSnapshot, ValidatorState, ValidatorTier

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of quantum attacks."""
    KEY_COMPROMISE = "key_compromise"      # Break validator private keys
    DOUBLE_SPEND = "double_spend"          # Fork chain for double spending
    CONSENSUS_HALT = "consensus_halt"      # Halt consensus by controlling 33%+
    CONSENSUS_CONTROL = "consensus_control"  # Control consensus with 67%+
    TARGETED_THEFT = "targeted_theft"      # Target specific high-value accounts
    SYSTEMIC_FAILURE = "systemic_failure"  # Cascade failure from multiple attacks


class AttackVector(Enum):
    """Attack vectors for quantum attacks."""
    VALIDATOR_KEYS = "validator_keys"      # Attack validator Ed25519 keys
    USER_ACCOUNTS = "user_accounts"        # Attack user account keys
    SMART_CONTRACTS = "smart_contracts"    # Attack contract signatures
    NETWORK_PROTOCOL = "network_protocol"  # Attack protocol-level crypto


class AttackSeverity(Enum):
    """Severity levels of successful attacks."""
    LOW = "low"          # Minor disruption, quick recovery
    MEDIUM = "medium"    # Significant disruption, recoverable
    HIGH = "high"        # Major disruption, difficult recovery
    CRITICAL = "critical"  # Systemic failure, may be unrecoverable


class AttackerProfile(Enum):
    """Types of quantum attackers with different motivations."""
    PROFIT_DRIVEN = "profit"           # Criminal: maximize financial gain
    NATION_STATE = "nation_state"      # Strategic: destabilize economy
    CHAOS_AGENT = "chaos"              # Anarchist: maximum disruption
    DEMONSTRATOR = "demonstrator"      # Researcher: prove vulnerability
    COMPETITOR = "competitor"          # Rival chain: damage Solana


@dataclass
class AttackScenario:
    """Represents a specific attack scenario."""
    
    attack_type: AttackType
    vector: AttackVector
    year: float
    success_probability: float
    severity: AttackSeverity
    validators_compromised: int
    stake_compromised: float  # Percentage of total stake
    accounts_at_risk: int
    time_to_execute: float  # Hours
    detection_probability: float
    mitigation_possible: bool
    attacker_profile: AttackerProfile = AttackerProfile.PROFIT_DRIVEN
    attribution_difficulty: float = 0.5  # 0-1, how hard to attribute
    strategic_value: float = 1.0  # Multiplier for nation-state attacks
    
    @property
    def impact_score(self) -> float:
        """Calculate overall impact score (0-1)."""
        severity_weights = {
            AttackSeverity.LOW: 0.25,
            AttackSeverity.MEDIUM: 0.5,
            AttackSeverity.HIGH: 0.75,
            AttackSeverity.CRITICAL: 1.0
        }
        
        base_score = severity_weights[self.severity]
        
        # Adjust for stake compromised
        stake_factor = min(self.stake_compromised, 1.0)
        
        # Adjust for detection probability (harder to detect = higher impact)
        detection_factor = 1.0 - (self.detection_probability * 0.3)
        
        return base_score * (1 + stake_factor) * detection_factor / 2


@dataclass
class AttackWindow:
    """Time window where attack is feasible."""
    
    start_year: float
    end_year: float
    peak_year: float
    opportunity_score: float  # 0-1, how good the window is
    
    @property
    def duration(self) -> float:
        """Duration of attack window in years."""
        return self.end_year - self.start_year if self.end_year else float('inf')
    
    def is_active(self, year: float) -> bool:
        """Check if window is active at given year."""
        return self.start_year <= year <= (self.end_year or float('inf'))


@dataclass
class AttackPlan:
    """Complete attack plan with scenarios and timeline."""
    
    scenarios: List[AttackScenario]
    windows: List[AttackWindow]
    primary_target: str  # validator, user, protocol
    estimated_success_rate: float
    total_stake_at_risk: float
    recommended_year: float
    contingency_scenarios: List[AttackScenario]
    
    def get_scenario_at_year(self, year: float) -> Optional[AttackScenario]:
        """Get best attack scenario for a given year."""
        valid_scenarios = [s for s in self.scenarios if s.year <= year]
        if not valid_scenarios:
            return None
        
        # Return scenario with highest impact
        return max(valid_scenarios, key=lambda s: s.impact_score)
    
    def get_active_window(self, year: float) -> Optional[AttackWindow]:
        """Get active attack window at given year."""
        for window in self.windows:
            if window.is_active(year):
                return window
        return None


class AttackScenariosModel:
    """
    Models quantum attack scenarios against Solana.
    
    Simulates:
    - Attack type selection based on capabilities
    - Success probability calculations
    - Attack timing and windows
    - Impact assessment
    - Detection and mitigation factors
    """
    
    def __init__(self, params: Optional[QuantumParameters] = None):
        """
        Initialize attack scenarios model.
        
        Args:
            params: Quantum parameters configuration
        """
        self.params = params or QuantumParameters()
        
        # Attack requirements (logical qubits needed)
        self.attack_requirements = {
            AttackType.KEY_COMPROMISE: self.params.logical_qubits_for_ed25519,
            AttackType.DOUBLE_SPEND: self.params.logical_qubits_for_ed25519 * 10,  # Need many keys
            AttackType.CONSENSUS_HALT: self.params.logical_qubits_for_ed25519 * 33,  # 33% stake
            AttackType.CONSENSUS_CONTROL: self.params.logical_qubits_for_ed25519 * 67,  # 67% stake
            AttackType.TARGETED_THEFT: self.params.logical_qubits_for_ed25519,
            AttackType.SYSTEMIC_FAILURE: self.params.logical_qubits_for_ed25519 * 100
        }
        
        # Base success rates when requirements are met
        self.base_success_rates = {
            AttackType.KEY_COMPROMISE: 0.95,
            AttackType.DOUBLE_SPEND: 0.70,
            AttackType.CONSENSUS_HALT: 0.60,
            AttackType.CONSENSUS_CONTROL: 0.40,
            AttackType.TARGETED_THEFT: 0.85,
            AttackType.SYSTEMIC_FAILURE: 0.30
        }
        
        # Detection probabilities
        self.detection_rates = {
            AttackType.KEY_COMPROMISE: 0.3,      # Hard to detect single key compromise
            AttackType.DOUBLE_SPEND: 0.8,        # Double spends are obvious
            AttackType.CONSENSUS_HALT: 0.95,     # Network halt is immediately visible
            AttackType.CONSENSUS_CONTROL: 0.9,   # Consensus takeover is visible
            AttackType.TARGETED_THEFT: 0.5,      # Depends on monitoring
            AttackType.SYSTEMIC_FAILURE: 1.0     # Systemic failure is obvious
        }
    
    def sample(
        self,
        rng: np.random.RandomState,
        quantum_capability: QuantumCapability,
        network_snapshot: NetworkSnapshot
    ) -> AttackPlan:
        """
        Sample an attack plan based on quantum capabilities and network state.
        
        Args:
            rng: Random number generator
            quantum_capability: Current quantum computing capability
            network_snapshot: Current network state
            
        Returns:
            AttackPlan instance
        """
        # Identify feasible attack types
        feasible_attacks = self._identify_feasible_attacks(
            quantum_capability,
            network_snapshot
        )
        
        if not feasible_attacks:
            # No attacks possible
            return AttackPlan(
                scenarios=[],
                windows=[],
                primary_target="none",
                estimated_success_rate=0.0,
                total_stake_at_risk=0.0,
                recommended_year=float('inf'),
                contingency_scenarios=[]
            )
        
        # Generate attack scenarios
        scenarios = []
        for attack_type in feasible_attacks:
            scenario = self._generate_scenario(
                rng,
                attack_type,
                quantum_capability,
                network_snapshot
            )
            scenarios.append(scenario)
        
        # Identify attack windows
        windows = self._identify_attack_windows(
            quantum_capability,
            network_snapshot,
            scenarios
        )
        
        # Select primary target
        primary_target = self._select_primary_target(
            network_snapshot,
            scenarios
        )
        
        # Calculate overall success rate
        if scenarios:
            success_rate = np.mean([s.success_probability for s in scenarios])
            total_stake = max([s.stake_compromised for s in scenarios])
            recommended_year = min([s.year for s in scenarios])
        else:
            success_rate = 0.0
            total_stake = 0.0
            recommended_year = float('inf')
        
        # Generate contingency scenarios
        contingency = self._generate_contingency_scenarios(
            rng,
            scenarios,
            quantum_capability,
            network_snapshot
        )
        
        return AttackPlan(
            scenarios=scenarios,
            windows=windows,
            primary_target=primary_target,
            estimated_success_rate=success_rate,
            total_stake_at_risk=total_stake,
            recommended_year=recommended_year,
            contingency_scenarios=contingency
        )
    
    def _identify_feasible_attacks(
        self,
        capability: QuantumCapability,
        network: NetworkSnapshot
    ) -> List[AttackType]:
        """Identify which attacks are feasible with current capabilities."""
        feasible = []
        
        for attack_type, required_qubits in self.attack_requirements.items():
            # Check if we have enough qubits
            if capability.logical_qubits < required_qubits:
                continue
            
            # Additional checks based on network state
            if attack_type == AttackType.CONSENSUS_HALT:
                # Need to compromise 33% of stake
                if network.vulnerable_stake_percentage < 0.33:
                    continue
            
            elif attack_type == AttackType.CONSENSUS_CONTROL:
                # Need to compromise 67% of stake
                if network.vulnerable_stake_percentage < 0.67:
                    continue
            
            feasible.append(attack_type)
        
        return feasible
    
    def _generate_scenario(
        self,
        rng: np.random.RandomState,
        attack_type: AttackType,
        capability: QuantumCapability,
        network: NetworkSnapshot
    ) -> AttackScenario:
        """Generate a specific attack scenario."""
        
        # Determine attacker profile (30% nation-state, 60% profit, 10% other)
        profile_prob = rng.random()
        if profile_prob < 0.3:
            attacker_profile = AttackerProfile.NATION_STATE
        elif profile_prob < 0.9:
            attacker_profile = AttackerProfile.PROFIT_DRIVEN
        elif profile_prob < 0.95:
            attacker_profile = AttackerProfile.CHAOS_AGENT
        else:
            attacker_profile = AttackerProfile.DEMONSTRATOR
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(
            attack_type,
            capability,
            network
        )
        
        # Adjust for attacker profile
        if attacker_profile == AttackerProfile.NATION_STATE:
            success_prob *= 1.2  # Better resources
            strategic_value = 3.0  # High strategic value
            attribution_difficulty = 0.8  # Hard to attribute
        elif attacker_profile == AttackerProfile.CHAOS_AGENT:
            strategic_value = 2.0  # Chaos has value
            attribution_difficulty = 0.6
        else:
            strategic_value = 1.0
            attribution_difficulty = 0.3  # Easier to track criminals
        
        success_prob = min(0.99, success_prob)
        
        # Determine validators compromised
        validators_compromised = self._calculate_validators_compromised(
            rng,
            attack_type,
            network
        )
        
        # Nation-states might target more validators for systemic impact
        if attacker_profile == AttackerProfile.NATION_STATE:
            validators_compromised = min(
                int(validators_compromised * 1.5),
                network.n_validators
            )
        
        # Calculate stake compromised
        stake_compromised = self._calculate_stake_compromised(
            attack_type,
            network,
            validators_compromised
        )
        
        # Determine severity
        severity = self._determine_severity(
            attack_type,
            stake_compromised,
            network
        )
        
        # Calculate execution time (hours)
        execution_time = self._calculate_execution_time(
            attack_type,
            capability,
            validators_compromised
        )
        
        # Detection probability
        detection_prob = self.detection_rates[attack_type]
        
        # Nation-states better at avoiding detection
        if attacker_profile == AttackerProfile.NATION_STATE:
            detection_prob *= 0.5
        
        # Adjust for network monitoring improvements over time
        years_elapsed = network.year - 2025
        detection_prob = min(0.99, detection_prob * (1 + years_elapsed * 0.02))
        
        # Mitigation possibility
        mitigation = self._can_mitigate(attack_type, network, execution_time)
        
        # Accounts at risk
        accounts_at_risk = self._estimate_accounts_at_risk(
            attack_type,
            network,
            stake_compromised
        )
        
        return AttackScenario(
            attack_type=attack_type,
            vector=self._get_attack_vector(attack_type),
            year=network.year,
            success_probability=success_prob,
            severity=severity,
            validators_compromised=validators_compromised,
            stake_compromised=stake_compromised,
            accounts_at_risk=accounts_at_risk,
            time_to_execute=execution_time,
            detection_probability=detection_prob,
            mitigation_possible=mitigation,
            attacker_profile=attacker_profile,
            attribution_difficulty=attribution_difficulty,
            strategic_value=strategic_value
        )
    
    def _calculate_success_probability(
        self,
        attack_type: AttackType,
        capability: QuantumCapability,
        network: NetworkSnapshot
    ) -> float:
        """Calculate attack success probability."""
        base_rate = self.base_success_rates[attack_type]
        
        # Adjust for quantum capability excess
        required = self.attack_requirements[attack_type]
        excess_ratio = capability.logical_qubits / required if required > 0 else 1
        capability_factor = min(1.5, 1 + (excess_ratio - 1) * 0.2)
        
        # Adjust for network migration
        migration_penalty = network.migration_progress * 0.8
        
        # Adjust for threat level
        threat_multipliers = {
            QuantumThreat.NONE: 0.5,
            QuantumThreat.EMERGING: 0.7,
            QuantumThreat.MODERATE: 0.9,
            QuantumThreat.HIGH: 1.0,
            QuantumThreat.CRITICAL: 1.2
        }
        threat_factor = threat_multipliers.get(capability.threat_level, 1.0)
        
        # Calculate final probability
        success_prob = base_rate * capability_factor * (1 - migration_penalty) * threat_factor
        
        return min(0.99, max(0.01, success_prob))
    
    def _calculate_validators_compromised(
        self,
        rng: np.random.RandomState,
        attack_type: AttackType,
        network: NetworkSnapshot
    ) -> int:
        """Calculate number of validators compromised."""
        vulnerable_validators = [v for v in network.validators if not v.is_migrated]
        
        if attack_type == AttackType.KEY_COMPROMISE:
            # Single validator
            return 1
        
        elif attack_type == AttackType.TARGETED_THEFT:
            # A few high-value validators
            return min(5, len(vulnerable_validators))
        
        elif attack_type == AttackType.CONSENSUS_HALT:
            # Need 33% of validators by stake
            target_stake = 0.33
            compromised = 0
            cumulative_stake = 0
            
            # Sort by stake (attack largest first)
            sorted_validators = sorted(
                vulnerable_validators,
                key=lambda v: v.stake_percentage,
                reverse=True
            )
            
            for validator in sorted_validators:
                if cumulative_stake >= target_stake:
                    break
                compromised += 1
                cumulative_stake += validator.stake_percentage
            
            return compromised
        
        elif attack_type == AttackType.CONSENSUS_CONTROL:
            # Need 67% of validators by stake
            target_stake = 0.67
            compromised = 0
            cumulative_stake = 0
            
            sorted_validators = sorted(
                vulnerable_validators,
                key=lambda v: v.stake_percentage,
                reverse=True
            )
            
            for validator in sorted_validators:
                if cumulative_stake >= target_stake:
                    break
                compromised += 1
                cumulative_stake += validator.stake_percentage
            
            return compromised
        
        elif attack_type == AttackType.DOUBLE_SPEND:
            # Need enough validators to create a fork
            return min(10, len(vulnerable_validators))
        
        else:  # SYSTEMIC_FAILURE
            # Compromise most vulnerable validators
            return int(len(vulnerable_validators) * rng.uniform(0.5, 0.8))
    
    def _calculate_stake_compromised(
        self,
        attack_type: AttackType,
        network: NetworkSnapshot,
        validators_compromised: int
    ) -> float:
        """Calculate percentage of stake compromised."""
        if validators_compromised == 0:
            return 0.0
        
        vulnerable_validators = [v for v in network.validators if not v.is_migrated]
        
        # Sort by stake to simulate attacking largest first
        sorted_validators = sorted(
            vulnerable_validators,
            key=lambda v: v.stake_percentage,
            reverse=True
        )
        
        # Calculate stake of compromised validators
        compromised_stake = sum(
            v.stake_percentage 
            for v in sorted_validators[:validators_compromised]
        )
        
        return min(1.0, compromised_stake)
    
    def _determine_severity(
        self,
        attack_type: AttackType,
        stake_compromised: float,
        network: NetworkSnapshot
    ) -> AttackSeverity:
        """Determine attack severity based on impact."""
        
        if attack_type == AttackType.SYSTEMIC_FAILURE:
            return AttackSeverity.CRITICAL
        
        elif attack_type == AttackType.CONSENSUS_CONTROL:
            return AttackSeverity.CRITICAL if stake_compromised > 0.67 else AttackSeverity.HIGH
        
        elif attack_type == AttackType.CONSENSUS_HALT:
            return AttackSeverity.HIGH if stake_compromised > 0.33 else AttackSeverity.MEDIUM
        
        elif attack_type == AttackType.DOUBLE_SPEND:
            if stake_compromised > 0.5:
                return AttackSeverity.HIGH
            elif stake_compromised > 0.2:
                return AttackSeverity.MEDIUM
            else:
                return AttackSeverity.LOW
        
        elif attack_type == AttackType.TARGETED_THEFT:
            # Depends on value stolen
            if stake_compromised > 0.1:
                return AttackSeverity.HIGH
            elif stake_compromised > 0.01:
                return AttackSeverity.MEDIUM
            else:
                return AttackSeverity.LOW
        
        else:  # KEY_COMPROMISE
            return AttackSeverity.LOW if stake_compromised < 0.01 else AttackSeverity.MEDIUM
    
    def _calculate_execution_time(
        self,
        attack_type: AttackType,
        capability: QuantumCapability,
        validators_compromised: int
    ) -> float:
        """Calculate time to execute attack in hours."""
        
        # Base time per key break
        base_time = capability.estimated_break_time_hours
        
        # Scale by number of validators
        if attack_type in [AttackType.KEY_COMPROMISE, AttackType.TARGETED_THEFT]:
            # Can parallelize to some extent
            parallel_factor = min(validators_compromised, 10)
            total_time = base_time * validators_compromised / parallel_factor
        
        elif attack_type in [AttackType.CONSENSUS_HALT, AttackType.CONSENSUS_CONTROL]:
            # Need sequential breaks for consensus attacks
            total_time = base_time * validators_compromised * 0.5  # Some parallelization
        
        elif attack_type == AttackType.DOUBLE_SPEND:
            # Need coordinated attack
            total_time = base_time * validators_compromised * 0.3
        
        else:  # SYSTEMIC_FAILURE
            # Highly parallel attack
            total_time = base_time * validators_compromised * 0.1
        
        return max(0.1, total_time)  # Minimum 6 minutes
    
    def _can_mitigate(
        self,
        attack_type: AttackType,
        network: NetworkSnapshot,
        execution_time: float
    ) -> bool:
        """Determine if attack can be mitigated."""
        
        # If network is mostly migrated, mitigation is easier
        if network.migration_progress > 0.8:
            return True
        
        # Fast attacks are harder to mitigate
        if execution_time < 1:  # Less than 1 hour
            return False
        
        # Some attacks are easier to mitigate
        if attack_type in [AttackType.DOUBLE_SPEND, AttackType.CONSENSUS_HALT]:
            return True  # Can halt network or roll back
        
        # Targeted attacks on few validators
        if attack_type in [AttackType.KEY_COMPROMISE, AttackType.TARGETED_THEFT]:
            return execution_time > 6  # Need time to respond
        
        # Systemic attacks are hard to mitigate
        if attack_type == AttackType.SYSTEMIC_FAILURE:
            return False
        
        return execution_time > 24  # Default: need 24 hours to respond
    
    def _estimate_accounts_at_risk(
        self,
        attack_type: AttackType,
        network: NetworkSnapshot,
        stake_compromised: float
    ) -> int:
        """Estimate number of user accounts at risk."""
        
        # Base estimate: 1M accounts per 1% of stake
        base_accounts = int(stake_compromised * 100 * 1_000_000)
        
        # Adjust by attack type
        if attack_type == AttackType.TARGETED_THEFT:
            # Only high-value accounts
            return min(10_000, base_accounts)
        
        elif attack_type in [AttackType.KEY_COMPROMISE]:
            # Accounts delegated to compromised validators
            return min(100_000, base_accounts)
        
        elif attack_type in [AttackType.CONSENSUS_HALT, AttackType.CONSENSUS_CONTROL]:
            # All accounts affected by consensus issues
            return base_accounts * 10
        
        elif attack_type == AttackType.SYSTEMIC_FAILURE:
            # All accounts at risk
            return 100_000_000  # Assume 100M total accounts
        
        return base_accounts
    
    def _get_attack_vector(self, attack_type: AttackType) -> AttackVector:
        """Get primary attack vector for attack type."""
        if attack_type in [
            AttackType.KEY_COMPROMISE,
            AttackType.CONSENSUS_HALT,
            AttackType.CONSENSUS_CONTROL
        ]:
            return AttackVector.VALIDATOR_KEYS
        
        elif attack_type == AttackType.TARGETED_THEFT:
            return AttackVector.USER_ACCOUNTS
        
        elif attack_type == AttackType.DOUBLE_SPEND:
            return AttackVector.NETWORK_PROTOCOL
        
        else:
            return AttackVector.VALIDATOR_KEYS
    
    def _identify_attack_windows(
        self,
        capability: QuantumCapability,
        network: NetworkSnapshot,
        scenarios: List[AttackScenario]
    ) -> List[AttackWindow]:
        """Identify optimal attack windows."""
        windows = []
        
        # Primary window: when capability first exceeds requirements
        if scenarios:
            first_feasible = min([s.year for s in scenarios])
            
            # Window closes as migration progresses
            if network.migration_progress < 0.5:
                end_year = network.year + (1 - network.migration_progress) * 5
            else:
                end_year = network.year + 2  # Closing fast
            
            peak_year = first_feasible + 1  # Best time is shortly after capability
            
            # Score based on success probability and migration
            opportunity = max([s.success_probability for s in scenarios]) * (1 - network.migration_progress)
            
            windows.append(AttackWindow(
                start_year=first_feasible,
                end_year=end_year,
                peak_year=peak_year,
                opportunity_score=opportunity
            ))
        
        return windows
    
    def _select_primary_target(
        self,
        network: NetworkSnapshot,
        scenarios: List[AttackScenario]
    ) -> str:
        """Select primary target based on scenarios."""
        if not scenarios:
            return "none"
        
        # Count attack vectors
        vector_counts = {}
        for scenario in scenarios:
            vector = scenario.vector
            vector_counts[vector] = vector_counts.get(vector, 0) + 1
        
        # Most common vector determines target
        primary_vector = max(vector_counts, key=vector_counts.get)
        
        if primary_vector == AttackVector.VALIDATOR_KEYS:
            return "validator"
        elif primary_vector == AttackVector.USER_ACCOUNTS:
            return "user"
        else:
            return "protocol"
    
    def _generate_contingency_scenarios(
        self,
        rng: np.random.RandomState,
        primary_scenarios: List[AttackScenario],
        capability: QuantumCapability,
        network: NetworkSnapshot
    ) -> List[AttackScenario]:
        """Generate backup attack scenarios."""
        contingency = []
        
        # If primary attacks fail, try simpler attacks
        for scenario in primary_scenarios[:2]:  # Take top 2
            # Create degraded version
            degraded = AttackScenario(
                attack_type=AttackType.KEY_COMPROMISE,  # Fallback to simple attack
                vector=AttackVector.VALIDATOR_KEYS,
                year=scenario.year,
                success_probability=min(0.9, scenario.success_probability * 1.2),
                severity=AttackSeverity.LOW,
                validators_compromised=1,
                stake_compromised=scenario.stake_compromised * 0.1,
                accounts_at_risk=scenario.accounts_at_risk // 10,
                time_to_execute=scenario.time_to_execute * 0.5,
                detection_probability=scenario.detection_probability * 0.5,
                mitigation_possible=True
            )
            contingency.append(degraded)
        
        return contingency


def test_attack_model():
    """Test the attack scenarios model."""
    from src.models.quantum_timeline import QuantumDevelopmentModel
    from src.models.network_state import NetworkStateModel
    
    # Initialize models
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    rng = np.random.RandomState(42)
    
    # Generate quantum timeline
    quantum_timeline = quantum_model.sample(rng)
    
    # Generate network evolution
    network_evolution = network_model.sample(rng, {'crqc_year': quantum_timeline.crqc_year})
    
    print("Attack Scenarios Test:")
    print("=" * 50)
    
    # Test at different years
    test_years = [2030, 2035, 2040]
    
    for year in test_years:
        # Get capability and network state at this year
        capability = None
        for cap in quantum_timeline.capabilities:
            if cap.year <= year:
                capability = cap
            else:
                break
        
        if not capability:
            continue
        
        network = network_evolution.get_snapshot_at_year(year)
        
        # Generate attack plan
        attack_plan = attack_model.sample(rng, capability, network)
        
        print(f"\n{year} Attack Assessment:")
        print(f"  Quantum Qubits: {capability.logical_qubits:,}")
        print(f"  Network Migration: {network.migration_progress:.1%}")
        print(f"  Vulnerable Stake: {network.vulnerable_stake_percentage:.1%}")
        
        if attack_plan.scenarios:
            print(f"  Feasible Attacks: {len(attack_plan.scenarios)}")
            print(f"  Success Rate: {attack_plan.estimated_success_rate:.1%}")
            print(f"  Stake at Risk: {attack_plan.total_stake_at_risk:.1%}")
            
            # Show top scenario
            best = max(attack_plan.scenarios, key=lambda s: s.impact_score)
            print(f"  Best Attack: {best.attack_type.value}")
            print(f"    - Success Prob: {best.success_probability:.1%}")
            print(f"    - Severity: {best.severity.value}")
            print(f"    - Execution Time: {best.time_to_execute:.1f} hours")
            print(f"    - Detection Prob: {best.detection_probability:.1%}")
        else:
            print(f"  No feasible attacks")
    
    print("\n✓ Attack model test passed")


if __name__ == "__main__":
    test_attack_model()
