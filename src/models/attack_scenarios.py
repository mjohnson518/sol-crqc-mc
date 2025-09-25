"""
Attack scenarios model for quantum threats against Solana.

This module models different quantum attack scenarios, their success
probabilities, and potential impacts on the Solana network.

Enhanced features (controlled by config flags):
- Hybrid attacks combining Shor's, Grover's, and classical methods
- Agent-based modeling for adversarial strategies
- Grover's algorithm risks for Solana's PoH
- Exponential time distributions for attack execution
"""

import numpy as np
from scipy import stats
from scipy.stats import expon
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import math

# Import mesa for agent-based modeling if available
try:
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.datacollection import DataCollector
    MESA_AVAILABLE = True
except ImportError:
    MESA_AVAILABLE = False
    logging.info("Mesa not available - agent-based modeling will use fallback implementation")

from src.config import QuantumParameters
from src.models.quantum_timeline import QuantumCapability, QuantumThreat, GroverCapability
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
    # New enhanced attack types
    GROVER_POH = "grover_poh"             # Grover attack on Proof of History
    GROVER_HASH = "grover_hash"           # Grover attack on SHA-256 hashes
    HYBRID_SHOR_GROVER = "hybrid_shor_grover"  # Combined Shor's + Grover's
    HYBRID_QUANTUM_CLASSICAL = "hybrid_quantum_classical"  # Quantum + classical
    POH_FORGERY = "poh_forgery"           # Forge PoH timestamps


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
class HybridAttack:
    """Represents a hybrid quantum-classical attack."""
    
    quantum_component: str  # Shor, Grover, or both
    classical_component: str  # Brute-force, side-channel, social engineering
    synergy_factor: float  # How well components work together (1-3)
    time_reduction_factor: float  # Speed improvement from hybrid approach
    qubit_reduction_factor: float  # Reduced qubit requirements
    success_prob_quantum: float
    success_prob_classical: float
    success_prob_combined: float
    
    @property
    def effective_success_rate(self) -> float:
        """Calculate effective success rate from combined approach."""
        # P(success) = P(quantum) + P(classical) - P(quantum) * P(classical) + synergy bonus
        independent = self.success_prob_quantum + self.success_prob_classical - \
                     (self.success_prob_quantum * self.success_prob_classical)
        return min(0.99, independent * self.synergy_factor)


@dataclass
class GroverAttack:
    """Represents a Grover's algorithm attack on hashing."""
    
    target: str  # SHA-256, PoH, Merkle tree, etc.
    operations_required: float  # 2^n operations
    speedup_achieved: float  # Square root speedup
    qubits_required: int
    gate_count: float
    attack_time_hours: float
    poh_vulnerability: bool  # Can forge PoH
    can_rewrite_history: bool  # Can modify past blocks
    
    @property
    def effective_security_reduction(self) -> int:
        """Calculate effective security bit reduction."""
        return int(np.log2(self.speedup_achieved))


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
    # New fields for enhanced attacks
    hybrid_attack: Optional[HybridAttack] = None
    grover_attack: Optional[GroverAttack] = None
    attack_time_distribution: Optional[str] = "exponential"  # Distribution type
    parallelization_factor: float = 1.0  # For multiple quantum processors
    
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
        
        # Bonus for hybrid attacks
        hybrid_bonus = 1.2 if self.hybrid_attack else 1.0
        
        # Bonus for Grover attacks on PoH
        grover_bonus = 1.3 if (self.grover_attack and self.grover_attack.poh_vulnerability) else 1.0
        
        return base_score * (1 + stake_factor) * detection_factor * hybrid_bonus * grover_bonus / 2


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


# Agent-based modeling components (if Mesa available)
if MESA_AVAILABLE:
    class AdversaryAgent(Agent):
        """Adversary agent for agent-based attack modeling."""
        
        def __init__(self, unique_id, model, profile: AttackerProfile):
            super().__init__(unique_id, model)
            self.profile = profile
            self.resources = 1.0  # Normalized resources
            self.knowledge = 0.5  # Knowledge of vulnerabilities
            self.success_count = 0
            self.failed_count = 0
            
            # Profile-specific attributes
            if profile == AttackerProfile.NATION_STATE:
                self.resources = 10.0
                self.knowledge = 0.9
            elif profile == AttackerProfile.PROFIT_DRIVEN:
                self.resources = 2.0
                self.knowledge = 0.7
            elif profile == AttackerProfile.CHAOS_AGENT:
                self.resources = 0.5
                self.knowledge = 0.6
        
        def step(self):
            """Execute one step of the attack simulation."""
            # Decide on attack strategy
            if self.profile == AttackerProfile.NATION_STATE:
                # Strategic, patient approach
                self.execute_strategic_attack()
            elif self.profile == AttackerProfile.PROFIT_DRIVEN:
                # Opportunistic approach
                self.execute_opportunistic_attack()
            else:
                # Chaotic approach
                self.execute_chaotic_attack()
        
        def execute_strategic_attack(self):
            """Nation-state strategic attack."""
            # Patient, high-resource attack
            if self.model.quantum_capability and self.model.quantum_capability.logical_qubits > 2000:
                self.success_count += 1
                self.model.attack_success = True
        
        def execute_opportunistic_attack(self):
            """Criminal opportunistic attack."""
            # Quick profit-driven attack
            if self.model.vulnerable_stake > 0.1:
                if self.model.random.random() < 0.7:
                    self.success_count += 1
        
        def execute_chaotic_attack(self):
            """Chaos agent random attack."""
            # Random disruptive attack
            if self.model.random.random() < 0.3:
                self.success_count += 1
    
    class AttackSimulationModel(Model):
        """Agent-based model for attack simulation."""
        
        def __init__(self, n_adversaries: int = 10, quantum_capability=None, network_state=None):
            super().__init__()
            self.num_agents = n_adversaries
            self.schedule = RandomActivation(self)
            self.quantum_capability = quantum_capability
            self.network_state = network_state
            self.vulnerable_stake = network_state.vulnerable_stake_percentage if network_state else 0.5
            self.attack_success = False
            
            # Create adversary agents
            profiles = [AttackerProfile.NATION_STATE] * 2 + \
                      [AttackerProfile.PROFIT_DRIVEN] * 6 + \
                      [AttackerProfile.CHAOS_AGENT] * 2
            
            for i in range(self.num_agents):
                profile = profiles[i % len(profiles)]
                agent = AdversaryAgent(i, self, profile)
                self.schedule.add(agent)
            
            # Data collection
            self.datacollector = DataCollector(
                model_reporters={
                    "AttackSuccess": lambda m: m.attack_success,
                    "TotalAttempts": lambda m: sum([a.success_count + a.failed_count 
                                                   for a in m.schedule.agents])
                },
                agent_reporters={
                    "Success": "success_count",
                    "Resources": "resources"
                }
            )
        
        def step(self):
            """Advance the model by one step."""
            self.datacollector.collect(self)
            self.schedule.step()


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
    
    def __init__(self, params: Optional[QuantumParameters] = None,
                 enable_grover: bool = False,
                 enable_hybrid_attacks: bool = False,
                 use_agent_based_model: bool = False):
        """
        Initialize attack scenarios model.
        
        Args:
            params: Quantum parameters configuration
            enable_grover: Whether to model Grover's algorithm attacks
            enable_hybrid_attacks: Whether to model hybrid quantum-classical attacks
            use_agent_based_model: Whether to use agent-based modeling
        """
        self.params = params or QuantumParameters()
        self.enable_grover = enable_grover and getattr(self.params, 'enable_grover_modeling', False)
        self.enable_hybrid_attacks = enable_hybrid_attacks
        self.use_agent_based_model = use_agent_based_model and MESA_AVAILABLE
        
        # Attack requirements (logical qubits needed)
        self.attack_requirements = {
            AttackType.KEY_COMPROMISE: self.params.logical_qubits_for_ed25519,
            AttackType.DOUBLE_SPEND: self.params.logical_qubits_for_ed25519 * 10,  # Need many keys
            AttackType.CONSENSUS_HALT: self.params.logical_qubits_for_ed25519 * 33,  # 33% stake
            AttackType.CONSENSUS_CONTROL: self.params.logical_qubits_for_ed25519 * 67,  # 67% stake
            AttackType.TARGETED_THEFT: self.params.logical_qubits_for_ed25519,
            AttackType.SYSTEMIC_FAILURE: self.params.logical_qubits_for_ed25519 * 100,
            # New Grover attack requirements
            AttackType.GROVER_POH: self.params.grover_qubits_sha256,
            AttackType.GROVER_HASH: self.params.grover_qubits_sha256,
            AttackType.HYBRID_SHOR_GROVER: min(self.params.logical_qubits_for_ed25519,
                                              self.params.grover_qubits_sha256) // 2,
            AttackType.HYBRID_QUANTUM_CLASSICAL: self.params.logical_qubits_for_ed25519 // 2,
            AttackType.POH_FORGERY: self.params.grover_qubits_sha256
        }
        
        # Base success rates when requirements are met
        self.base_success_rates = {
            AttackType.KEY_COMPROMISE: 0.95,
            AttackType.DOUBLE_SPEND: 0.70,
            AttackType.CONSENSUS_HALT: 0.60,
            AttackType.CONSENSUS_CONTROL: 0.40,
            AttackType.TARGETED_THEFT: 0.85,
            AttackType.SYSTEMIC_FAILURE: 0.30,
            # New attack success rates
            AttackType.GROVER_POH: 0.80,
            AttackType.GROVER_HASH: 0.75,
            AttackType.HYBRID_SHOR_GROVER: 0.90,
            AttackType.HYBRID_QUANTUM_CLASSICAL: 0.85,
            AttackType.POH_FORGERY: 0.70
        }
        
        # Detection probabilities
        self.detection_rates = {
            AttackType.KEY_COMPROMISE: 0.3,      # Hard to detect single key compromise
            AttackType.DOUBLE_SPEND: 0.8,        # Double spends are obvious
            AttackType.CONSENSUS_HALT: 0.95,     # Network halt is immediately visible
            AttackType.CONSENSUS_CONTROL: 0.9,   # Consensus takeover is visible
            AttackType.TARGETED_THEFT: 0.5,      # Depends on monitoring
            AttackType.SYSTEMIC_FAILURE: 1.0,    # Systemic failure is obvious
            # New attack detection rates
            AttackType.GROVER_POH: 0.4,          # PoH attacks harder to detect
            AttackType.GROVER_HASH: 0.6,         # Hash collisions detectable
            AttackType.HYBRID_SHOR_GROVER: 0.35, # Hybrid attacks harder to detect
            AttackType.HYBRID_QUANTUM_CLASSICAL: 0.4,
            AttackType.POH_FORGERY: 0.7          # Forged timestamps eventually detected
        }
        
        # Agent-based model instance
        self.agent_model = None
    
    def sample(
        self,
        rng: np.random.RandomState,
        quantum_capability: QuantumCapability,
        network_snapshot: NetworkSnapshot,
        grover_capability: Optional[GroverCapability] = None
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
            network_snapshot,
            grover_capability
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
                network_snapshot,
                grover_capability
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
        network: NetworkSnapshot,
        grover_capability: Optional[GroverCapability] = None
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
            
            # Check Grover-specific attacks
            elif attack_type in [AttackType.GROVER_POH, AttackType.GROVER_HASH, AttackType.POH_FORGERY]:
                if not self.enable_grover or not grover_capability:
                    continue
                if not grover_capability.can_attack_sha256:
                    continue
            
            # Check hybrid attacks
            elif attack_type in [AttackType.HYBRID_SHOR_GROVER, AttackType.HYBRID_QUANTUM_CLASSICAL]:
                if not self.enable_hybrid_attacks:
                    continue
                # For Shor+Grover hybrid, need both capabilities
                if attack_type == AttackType.HYBRID_SHOR_GROVER:
                    if not grover_capability or not grover_capability.can_attack_sha256:
                        continue
            
            feasible.append(attack_type)
        
        return feasible
    
    def _generate_scenario(
        self,
        rng: np.random.RandomState,
        attack_type: AttackType,
        capability: QuantumCapability,
        network: NetworkSnapshot,
        grover_capability: Optional[GroverCapability] = None
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
        
        # Generate Grover attack if applicable
        grover_attack = None
        if self.enable_grover and attack_type in [AttackType.GROVER_POH, AttackType.GROVER_HASH,
                                                   AttackType.POH_FORGERY]:
            if grover_capability:
                grover_attack = self._generate_grover_attack(rng, grover_capability, network)
        
        # Generate hybrid attack if applicable
        hybrid_attack = None
        if self.enable_hybrid_attacks and attack_type in [AttackType.HYBRID_SHOR_GROVER,
                                                          AttackType.HYBRID_QUANTUM_CLASSICAL]:
            hybrid_attack = self._generate_hybrid_attack(
                rng, attack_type, capability, grover_capability, network
            )
        
        # Calculate execution time (hours)
        # Determine parallelization factor
        parallelization = 1.0
        if attacker_profile == AttackerProfile.NATION_STATE:
            parallelization = rng.uniform(2, 5)  # Multiple quantum processors
        elif hybrid_attack:
            parallelization = hybrid_attack.time_reduction_factor
        
        base_execution_time = self._calculate_execution_time(
            attack_type,
            capability,
            validators_compromised
        )
        
        # Use exponential distribution for time modeling
        execution_time = self._calculate_exponential_attack_time(
            base_execution_time,
            rng,
            parallelization
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
            strategic_value=strategic_value,
            hybrid_attack=hybrid_attack,
            grover_attack=grover_attack,
            attack_time_distribution="exponential",
            parallelization_factor=parallelization
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
            AttackType.CONSENSUS_CONTROL,
            AttackType.HYBRID_QUANTUM_CLASSICAL
        ]:
            return AttackVector.VALIDATOR_KEYS
        
        elif attack_type == AttackType.TARGETED_THEFT:
            return AttackVector.USER_ACCOUNTS
        
        elif attack_type in [AttackType.DOUBLE_SPEND, AttackType.GROVER_POH, 
                            AttackType.GROVER_HASH, AttackType.POH_FORGERY,
                            AttackType.HYBRID_SHOR_GROVER]:
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
    
    def _generate_grover_attack(
        self,
        rng: np.random.RandomState,
        grover_capability: GroverCapability,
        network: NetworkSnapshot
    ) -> Optional[GroverAttack]:
        """Generate Grover's algorithm attack scenario."""
        if not grover_capability or not grover_capability.can_attack_sha256:
            return None
        
        # Determine target
        targets = ["SHA-256", "PoH", "Merkle tree", "Block hash"]
        target = rng.choice(targets, p=[0.3, 0.4, 0.2, 0.1])
        
        # Calculate attack parameters
        operations = 2 ** 128  # SHA-256 with Grover speedup
        speedup = grover_capability.speedup_factor
        
        # Attack time with exponential distribution
        base_time = grover_capability.attack_time_hours
        actual_time = expon.rvs(scale=base_time, random_state=rng)
        
        # PoH specific vulnerability
        poh_vulnerable = (target == "PoH" and grover_capability.poh_vulnerability)
        
        # Can rewrite history if attacking block hashes with enough speed
        can_rewrite = (target in ["Block hash", "PoH"] and actual_time < 1.0)
        
        return GroverAttack(
            target=target,
            operations_required=operations,
            speedup_achieved=speedup,
            qubits_required=grover_capability.logical_qubits,
            gate_count=operations / speedup,
            attack_time_hours=actual_time,
            poh_vulnerability=poh_vulnerable,
            can_rewrite_history=can_rewrite
        )
    
    def _generate_hybrid_attack(
        self,
        rng: np.random.RandomState,
        attack_type: AttackType,
        quantum_capability: QuantumCapability,
        grover_capability: Optional[GroverCapability],
        network: NetworkSnapshot
    ) -> Optional[HybridAttack]:
        """Generate hybrid quantum-classical attack."""
        if not self.enable_hybrid_attacks:
            return None
        
        # Determine components based on attack type
        if attack_type == AttackType.HYBRID_SHOR_GROVER:
            if not grover_capability or not grover_capability.can_attack_sha256:
                return None
            
            quantum_component = "Shor + Grover"
            classical_component = "Brute-force residual"
            
            # Success probabilities
            shor_success = 0.9 if quantum_capability.can_break_ed25519 else 0.1
            grover_success = 0.8 if grover_capability.can_attack_sha256 else 0.1
            classical_success = 0.3
            
            # Synergy from combined approach
            synergy = 1.5  # 50% improvement from combination
            
            # Time reduction from parallelization
            time_reduction = 0.4  # 60% faster
            
            # Qubit reduction from hybrid approach
            qubit_reduction = 0.7  # 30% fewer qubits needed
            
        elif attack_type == AttackType.HYBRID_QUANTUM_CLASSICAL:
            quantum_component = "Shor's algorithm"
            classical_component = "Side-channel attack"
            
            shor_success = 0.8 if quantum_capability.can_break_ed25519 else 0.2
            classical_success = 0.4  # Side-channel attacks moderately successful
            grover_success = 0  # Not using Grover
            
            synergy = 1.3  # 30% improvement
            time_reduction = 0.5  # 50% faster
            qubit_reduction = 0.8  # 20% fewer qubits
            
        else:
            return None
        
        # Calculate combined success probability
        quantum_success = max(shor_success, grover_success)
        combined_success = quantum_success + classical_success - \
                         (quantum_success * classical_success)
        combined_success *= synergy
        combined_success = min(0.99, combined_success)
        
        return HybridAttack(
            quantum_component=quantum_component,
            classical_component=classical_component,
            synergy_factor=synergy,
            time_reduction_factor=time_reduction,
            qubit_reduction_factor=qubit_reduction,
            success_prob_quantum=quantum_success,
            success_prob_classical=classical_success,
            success_prob_combined=combined_success
        )
    
    def _calculate_exponential_attack_time(
        self,
        base_time: float,
        rng: np.random.RandomState,
        parallelization: float = 1.0
    ) -> float:
        """
        Calculate attack execution time using exponential distribution.
        
        Args:
            base_time: Mean time for attack
            rng: Random number generator
            parallelization: Parallelization factor (>1 means multiple processors)
            
        Returns:
            Attack time in hours
        """
        # Sample from exponential distribution
        scale = base_time / parallelization
        time_components = []
        
        # Model as sum of exponential random variables (phases of attack)
        n_phases = 3  # Key extraction, computation, verification
        for _ in range(n_phases):
            phase_time = expon.rvs(scale=scale/n_phases, random_state=rng)
            time_components.append(phase_time)
        
        total_time = sum(time_components)
        
        # Apply minimum time constraint
        return max(0.1, total_time)
    
    def simulate_with_agents(
        self,
        quantum_capability: QuantumCapability,
        network_state: NetworkSnapshot,
        n_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Run agent-based attack simulation.
        
        Args:
            quantum_capability: Quantum computing capability
            network_state: Network state
            n_steps: Number of simulation steps
            
        Returns:
            Dictionary with simulation results
        """
        if not self.use_agent_based_model or not MESA_AVAILABLE:
            return {"error": "Agent-based modeling not available"}
        
        # Create agent model
        self.agent_model = AttackSimulationModel(
            n_adversaries=10,
            quantum_capability=quantum_capability,
            network_state=network_state
        )
        
        # Run simulation
        for _ in range(n_steps):
            self.agent_model.step()
        
        # Collect results
        model_data = self.agent_model.datacollector.get_model_vars_dataframe()
        agent_data = self.agent_model.datacollector.get_agent_vars_dataframe()
        
        # Analyze results
        success_rate = model_data["AttackSuccess"].mean() if len(model_data) > 0 else 0
        total_attempts = model_data["TotalAttempts"].iloc[-1] if len(model_data) > 0 else 0
        
        # Get agent profiles success rates
        agent_success = {}
        if len(agent_data) > 0:
            for agent_id in agent_data.index.get_level_values(1).unique():
                agent_success[f"agent_{agent_id}"] = \
                    agent_data.xs(agent_id, level=1)["Success"].iloc[-1]
        
        return {
            "success_rate": success_rate,
            "total_attempts": total_attempts,
            "agent_success": agent_success,
            "model_data": model_data,
            "agent_data": agent_data
        }


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
