"""
Economic impact model for quantum attacks on Solana.

This module models the economic consequences of successful quantum attacks,
including direct losses, market reactions, and recovery costs.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import math

from src.config import EconomicParameters
from src.models.attack_scenarios import AttackScenario, AttackType, AttackSeverity, AttackerProfile
from src.models.network_state import NetworkSnapshot
from src.distributions.probability_dists import EconomicDistributions
from src.models.stablecoin_vulnerability import StablecoinVulnerabilityModel

logger = logging.getLogger(__name__)


class ImpactType(Enum):
    """Types of economic impacts."""
    DIRECT_LOSS = "direct_loss"          # Direct theft/loss from attacks
    MARKET_CRASH = "market_crash"        # Market reaction and panic selling
    DEFI_CASCADE = "defi_cascade"        # DeFi protocol failures
    REPUTATION = "reputation"            # Long-term reputation damage
    MIGRATION_COST = "migration_cost"    # Cost of migrating to quantum-safe
    RECOVERY_COST = "recovery_cost"      # Cost to recover from attack


class RecoverySpeed(Enum):
    """Speed of economic recovery."""
    IMMEDIATE = "immediate"    # < 1 day
    FAST = "fast"             # 1-7 days
    MODERATE = "moderate"     # 1-4 weeks
    SLOW = "slow"             # 1-3 months
    VERY_SLOW = "very_slow"   # > 3 months


@dataclass
class ImpactComponent:
    """Component of economic impact."""
    
    impact_type: ImpactType
    amount_usd: float
    percentage_of_tvl: float
    time_to_realize_days: float
    confidence_interval: Tuple[float, float]  # 95% CI
    
    @property
    def is_immediate(self) -> bool:
        """Check if impact is immediate (< 1 day)."""
        return self.time_to_realize_days < 1.0


@dataclass
class MarketReaction:
    """Market reaction to attack."""
    
    sol_price_drop_percent: float
    tvl_drop_percent: float
    daily_volume_drop_percent: float
    panic_duration_days: float
    recovery_time_days: float
    
    @property
    def total_market_impact(self) -> float:
        """Calculate overall market impact score (0-1)."""
        # Weighted average of different impacts
        weights = [0.4, 0.3, 0.2, 0.1]  # Price, TVL, Volume, Duration
        impacts = [
            self.sol_price_drop_percent / 100,
            self.tvl_drop_percent / 100,
            self.daily_volume_drop_percent / 100,
            min(self.panic_duration_days / 30, 1.0)  # Normalize to 30 days
        ]
        return sum(w * i for w, i in zip(weights, impacts))


@dataclass
class EconomicLoss:
    """Total economic loss from attack."""
    
    attack_scenario: AttackScenario
    components: List[ImpactComponent]
    market_reaction: MarketReaction
    total_loss_usd: float
    immediate_loss_usd: float
    long_term_loss_usd: float
    recovery_speed: RecoverySpeed
    recovery_timeline_days: float
    confidence_level: float
    
    def get_loss_by_type(self, impact_type: ImpactType) -> float:
        """Get loss amount for specific impact type."""
        return sum(c.amount_usd for c in self.components if c.impact_type == impact_type)
    
    def get_cumulative_loss_at_day(self, day: float) -> float:
        """Calculate cumulative loss realized by specific day."""
        total = 0.0
        for component in self.components:
            if component.time_to_realize_days <= day:
                total += component.amount_usd
            elif component.time_to_realize_days - day < 1:
                # Partial realization
                fraction = 1 - (component.time_to_realize_days - day)
                total += component.amount_usd * fraction
        return total


@dataclass
class EconomicRecovery:
    """Economic recovery trajectory."""
    
    recovery_phases: List[Dict[str, Any]]
    milestones: Dict[str, float]  # Milestone -> Days to reach
    final_tvl_percent: float  # Final TVL as % of pre-attack
    permanent_damage_percent: float
    
    def get_tvl_at_day(self, day: float, initial_tvl: float) -> float:
        """Calculate TVL at specific day during recovery."""
        if day <= 0:
            return initial_tvl
        
        # Find applicable phase
        for phase in self.recovery_phases:
            if phase['start_day'] <= day <= phase['end_day']:
                # Interpolate within phase
                phase_progress = (day - phase['start_day']) / (phase['end_day'] - phase['start_day'])
                phase_recovery = phase['start_tvl'] + (phase['end_tvl'] - phase['start_tvl']) * phase_progress
                return initial_tvl * phase_recovery
        
        # After all phases
        return initial_tvl * self.final_tvl_percent


class EconomicImpactModel:
    """
    Models economic impact of quantum attacks on Solana.
    
    Simulates:
    - Direct financial losses
    - Market reactions and panic selling
    - DeFi protocol cascading failures
    - Reputation damage
    - Migration and recovery costs
    """
    
    def __init__(self, params: Optional[EconomicParameters] = None):
        """
        Initialize economic impact model.
        
        Args:
            params: Economic parameters configuration
        """
        self.params = params or EconomicParameters()
        self.stablecoin_model = StablecoinVulnerabilityModel()
        
        # Impact multipliers by attack type
        self.attack_impact_multipliers = {
            AttackType.KEY_COMPROMISE: 0.1,
            AttackType.TARGETED_THEFT: 0.3,
            AttackType.DOUBLE_SPEND: 0.5,
            AttackType.CONSENSUS_HALT: 0.7,
            AttackType.CONSENSUS_CONTROL: 0.9,
            AttackType.SYSTEMIC_FAILURE: 1.0
        }
        
        # Market reaction severity by attack severity
        self.market_reaction_factors = {
            AttackSeverity.LOW: 0.1,
            AttackSeverity.MEDIUM: 0.3,
            AttackSeverity.HIGH: 0.6,
            AttackSeverity.CRITICAL: 1.0
        }
        
        # Recovery speed by attack severity
        self.recovery_speeds = {
            AttackSeverity.LOW: RecoverySpeed.FAST,
            AttackSeverity.MEDIUM: RecoverySpeed.MODERATE,
            AttackSeverity.HIGH: RecoverySpeed.SLOW,
            AttackSeverity.CRITICAL: RecoverySpeed.VERY_SLOW
        }
    
    def calculate_impact(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        network_snapshot: NetworkSnapshot
    ) -> EconomicLoss:
        """
        Calculate economic impact of an attack.
        
        Args:
            rng: Random number generator
            attack_scenario: Attack scenario to evaluate
            network_snapshot: Network state at time of attack
            
        Returns:
            EconomicLoss instance
        """
        components = []
        
        # 1. Direct losses from attack
        direct_loss = self._calculate_direct_loss(rng, attack_scenario)
        components.append(direct_loss)
        
        # 2. Market reaction and panic
        market_reaction = self._calculate_market_reaction(rng, attack_scenario)
        market_loss = self._market_reaction_to_loss(market_reaction)
        components.append(market_loss)
        
        # 3. DeFi cascade effects
        defi_loss = self._calculate_defi_cascade(rng, attack_scenario, market_reaction)
        components.append(defi_loss)
        
        # 4. Stablecoin vulnerability impact (if quantum capable)
        if hasattr(attack_scenario, 'attacker_profile'):
            stablecoin_loss = self._calculate_stablecoin_impact(
                rng, attack_scenario, network_snapshot
            )
            if stablecoin_loss.amount_usd > 0:
                components.append(stablecoin_loss)
        
        # 5. Reputation damage
        reputation_loss = self._calculate_reputation_damage(rng, attack_scenario)
        components.append(reputation_loss)
        
        # 6. Migration costs
        migration_cost = self._calculate_migration_cost(rng, network_snapshot)
        components.append(migration_cost)
        
        # 7. Recovery costs
        recovery_cost = self._calculate_recovery_cost(rng, attack_scenario, components)
        components.append(recovery_cost)
        
        # Calculate totals
        total_loss = sum(c.amount_usd for c in components)
        immediate_loss = sum(c.amount_usd for c in components if c.is_immediate)
        long_term_loss = total_loss - immediate_loss
        
        # Determine recovery timeline
        recovery_speed = self.recovery_speeds[attack_scenario.severity]
        recovery_timeline = self._estimate_recovery_timeline(recovery_speed, attack_scenario)
        
        return EconomicLoss(
            attack_scenario=attack_scenario,
            components=components,
            market_reaction=market_reaction,
            total_loss_usd=total_loss,
            immediate_loss_usd=immediate_loss,
            long_term_loss_usd=long_term_loss,
            recovery_speed=recovery_speed,
            recovery_timeline_days=recovery_timeline,
            confidence_level=0.95
        )
    
    def _calculate_direct_loss(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario
    ) -> ImpactComponent:
        """Calculate direct financial loss from attack."""
        
        # Base loss from compromised stake
        stake_value = attack_scenario.stake_compromised * self.params.total_value_locked_usd
        
        # Adjust by attack type
        multiplier = self.attack_impact_multipliers[attack_scenario.attack_type]
        
        # Add randomness
        loss_factor = rng.beta(2, 5) * multiplier  # Skewed towards lower losses
        direct_loss = stake_value * loss_factor
        
        # Confidence interval
        ci_lower = direct_loss * 0.7
        ci_upper = direct_loss * 1.5
        
        return ImpactComponent(
            impact_type=ImpactType.DIRECT_LOSS,
            amount_usd=direct_loss,
            percentage_of_tvl=direct_loss / self.params.total_value_locked_usd,
            time_to_realize_days=0.1,  # Nearly immediate
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _calculate_market_reaction(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario
    ) -> MarketReaction:
        """Calculate market reaction to attack."""
        
        # Base reaction from severity
        base_factor = self.market_reaction_factors[attack_scenario.severity]
        
        # Price drop
        price_drop_mean = base_factor * 50  # Up to 50% drop for critical
        price_drop = rng.normal(price_drop_mean, price_drop_mean * 0.2)
        price_drop = max(0, min(90, price_drop))  # Cap at 90% drop
        
        # TVL drop (usually larger than price drop due to withdrawals)
        tvl_drop = price_drop * rng.uniform(1.2, 1.8)
        tvl_drop = max(0, min(95, tvl_drop))  # Cap at 95% drop
        
        # Volume drop (trading may increase initially, then drop)
        if attack_scenario.severity == AttackSeverity.LOW:
            volume_drop = rng.uniform(-20, 10)  # May increase
        else:
            volume_drop = rng.uniform(30, 70)
        
        # Panic duration
        panic_days = base_factor * rng.gamma(2, 5)  # Skewed distribution
        
        # Recovery time
        recovery_days = panic_days * rng.uniform(2, 5)
        
        return MarketReaction(
            sol_price_drop_percent=price_drop,
            tvl_drop_percent=tvl_drop,
            daily_volume_drop_percent=volume_drop,
            panic_duration_days=panic_days,
            recovery_time_days=recovery_days
        )
    
    def _market_reaction_to_loss(self, market_reaction: MarketReaction) -> ImpactComponent:
        """Convert market reaction to financial loss."""
        
        # Calculate loss from price drop
        market_cap = self.params.sol_price_usd * 400_000_000  # Assume 400M SOL
        price_loss = market_cap * (market_reaction.sol_price_drop_percent / 100)
        
        # Calculate loss from TVL drop (excluding price impact)
        tvl_loss = self.params.total_value_locked_usd * (
            market_reaction.tvl_drop_percent / 100 - market_reaction.sol_price_drop_percent / 100
        )
        tvl_loss = max(0, tvl_loss)
        
        total_market_loss = price_loss + tvl_loss
        
        return ImpactComponent(
            impact_type=ImpactType.MARKET_CRASH,
            amount_usd=total_market_loss,
            percentage_of_tvl=total_market_loss / self.params.total_value_locked_usd,
            time_to_realize_days=1.0,  # Market reacts within a day
            confidence_interval=(total_market_loss * 0.5, total_market_loss * 1.5)
        )
    
    def _calculate_defi_cascade(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        market_reaction: MarketReaction
    ) -> ImpactComponent:
        """Calculate cascading DeFi protocol failures."""
        
        # Probability of cascade based on market reaction
        cascade_prob = market_reaction.total_market_impact
        
        if rng.random() > cascade_prob:
            # No cascade
            return ImpactComponent(
                impact_type=ImpactType.DEFI_CASCADE,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        # Calculate cascade impact
        affected_protocols = []
        for protocol, tvl in self.params.defi_protocols.items():
            if rng.random() < cascade_prob * 0.7:  # 70% chance if cascade occurs
                affected_protocols.append((protocol, tvl))
        
        # Loss from affected protocols
        cascade_loss = sum(tvl * rng.uniform(0.3, 0.8) for _, tvl in affected_protocols)
        
        return ImpactComponent(
            impact_type=ImpactType.DEFI_CASCADE,
            amount_usd=cascade_loss,
            percentage_of_tvl=cascade_loss / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(2, 7),  # Takes days to cascade
            confidence_interval=(cascade_loss * 0.5, cascade_loss * 2.0)
        )
    
    def _calculate_reputation_damage(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario
    ) -> ImpactComponent:
        """Calculate long-term reputation damage."""
        
        # Base reputation loss (use confidence_loss_factor * 0.2 as reputation damage)
        base_loss = self.params.total_value_locked_usd * self.params.confidence_loss_factor * 0.2
        
        # Adjust by severity
        severity_multiplier = self.market_reaction_factors[attack_scenario.severity]
        
        # Long-term impact on adoption and growth
        reputation_loss = base_loss * severity_multiplier * rng.uniform(0.5, 1.5)
        
        return ImpactComponent(
            impact_type=ImpactType.REPUTATION,
            amount_usd=reputation_loss,
            percentage_of_tvl=reputation_loss / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(30, 180),  # Long-term impact
            confidence_interval=(reputation_loss * 0.3, reputation_loss * 3.0)
        )
    
    def _calculate_migration_cost(
        self,
        rng: np.random.RandomState,
        network_snapshot: NetworkSnapshot
    ) -> ImpactComponent:
        """Calculate cost of migrating to quantum-safe cryptography.
        
        Uses component-based costing model:
        - Hardware acceleration: $22.5M
        - Development effort: $10M
        - Security auditing: $4M
        - Validator coordination: $6M
        - Contingency: $5M
        Total base cost: $47.5M
        """
        
        # Base migration cost components (in millions USD)
        BASE_MIGRATION_COST = 47.5e6  # $47.5M total
        
        # Component breakdown
        hardware_cost = 22.5e6  # GPU/FPGA infrastructure
        development_cost = 10.0e6  # 20 engineers × 2 years
        audit_cost = 4.0e6  # 3 independent audit firms
        coordination_cost = 6.0e6  # Validator incentives & support
        contingency = 5.0e6  # 15% buffer
        
        # Calculate base cost adjusted for migration progress
        # If migration already partially complete, costs are reduced
        remaining_migration = 1.0 - network_snapshot.migration_progress
        
        # Hardware and coordination scale with remaining validators
        scaled_hardware = hardware_cost * remaining_migration
        scaled_coordination = coordination_cost * remaining_migration
        
        # Development and audit costs are mostly fixed
        # But reduce slightly if migration is very advanced
        completion_discount = 1.0 if remaining_migration > 0.5 else 0.7
        scaled_development = development_cost * completion_discount
        scaled_audit = audit_cost * completion_discount
        
        # Calculate total migration cost
        migration_cost = (
            scaled_hardware +
            scaled_development +
            scaled_audit +
            scaled_coordination +
            contingency * remaining_migration  # Contingency scales with work remaining
        )
        
        # Add urgency premium if attack has occurred (20-50% premium, not 2-5x)
        if hasattr(network_snapshot, 'attack_occurred') and network_snapshot.attack_occurred:
            urgency_premium = rng.uniform(1.2, 1.5)  # More realistic urgency premium
            migration_cost *= urgency_premium
        elif network_snapshot.compromised_validators > 0:
            # If validators are compromised, add urgency premium
            urgency_premium = rng.uniform(1.1, 1.3)  # Smaller premium for partial compromise
            migration_cost *= urgency_premium
        
        # Add uncertainty based on how early we are in the timeline
        uncertainty_factor = rng.uniform(0.8, 1.2)  # ±20% uncertainty
        migration_cost *= uncertainty_factor
        
        return ImpactComponent(
            impact_type=ImpactType.MIGRATION_COST,
            amount_usd=migration_cost,
            percentage_of_tvl=migration_cost / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(180, 730),  # 6-24 months for full migration
            confidence_interval=(migration_cost * 0.7, migration_cost * 1.3)
        )
    
    def _calculate_stablecoin_impact(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        network_snapshot: NetworkSnapshot
    ) -> ImpactComponent:
        """Calculate impact from stablecoin vulnerabilities."""
        
        # Check if we have a valid year
        if not network_snapshot.year:
            return ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        # Check if attacker can break Ed25519 (need ~2330 logical qubits)
        if network_snapshot.year < 2027:  # Too early for quantum attacks
            return ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        # Estimate quantum capability (simplified)
        years_from_2025 = network_snapshot.year - 2025
        estimated_qubits = 1000 * (1.5 ** years_from_2025)  # Exponential growth
        
        if estimated_qubits < 2330:  # Not enough qubits
            return ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        # Determine attacker motivation
        attacker_motivation = "profit"
        if hasattr(attack_scenario, 'attacker_profile'):
            if attack_scenario.attacker_profile == AttackerProfile.NATION_STATE:
                attacker_motivation = "destabilize"
            elif attack_scenario.attacker_profile == AttackerProfile.CHAOS_AGENT:
                attacker_motivation = "destabilize"
        
        # Assess stablecoin vulnerabilities
        scenarios = self.stablecoin_model.assess_vulnerability(
            int(estimated_qubits),
            attacker_motivation
        )
        
        if not scenarios:
            return ImpactComponent(
                impact_type=ImpactType.DIRECT_LOSS,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        # For nation-states, might target USDC for maximum impact
        if attacker_motivation == "destabilize" and "USDC" in scenarios:
            primary_target = scenarios["USDC"]
        else:
            # Pick most valuable target
            primary_target = max(scenarios.values(), 
                                key=lambda s: s.total_economic_impact())
        
        # Calculate total impact including cascades
        total_impact = primary_target.total_economic_impact()
        
        # Add randomness
        total_impact *= rng.uniform(0.8, 1.2)
        
        return ImpactComponent(
            impact_type=ImpactType.DEFI_CASCADE,  # Stablecoin failure cascades
            amount_usd=total_impact,
            percentage_of_tvl=total_impact / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(0.1, 1),  # Very fast cascade
            confidence_interval=(total_impact * 0.5, total_impact * 2.0)
        )
    
    def _calculate_recovery_cost(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        other_components: List[ImpactComponent]
    ) -> ImpactComponent:
        """Calculate cost to recover from attack."""
        
        # Base recovery cost as percentage of other losses
        other_losses = sum(c.amount_usd for c in other_components)
        recovery_percentage = 0.1  # 10% of other losses
        
        # Adjust by attack type
        if attack_scenario.attack_type == AttackType.SYSTEMIC_FAILURE:
            recovery_percentage = 0.5  # 50% for systemic failure
        elif attack_scenario.attack_type in [AttackType.CONSENSUS_HALT, AttackType.CONSENSUS_CONTROL]:
            recovery_percentage = 0.3  # 30% for consensus attacks
        
        recovery_cost = other_losses * recovery_percentage * rng.uniform(0.7, 1.3)
        
        return ImpactComponent(
            impact_type=ImpactType.RECOVERY_COST,
            amount_usd=recovery_cost,
            percentage_of_tvl=recovery_cost / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(1, 14),  # Immediate to 2 weeks
            confidence_interval=(recovery_cost * 0.5, recovery_cost * 2.0)
        )
    
    def _estimate_recovery_timeline(
        self,
        recovery_speed: RecoverySpeed,
        attack_scenario: AttackScenario
    ) -> float:
        """Estimate days to recover from attack."""
        
        base_timelines = {
            RecoverySpeed.IMMEDIATE: 1,
            RecoverySpeed.FAST: 7,
            RecoverySpeed.MODERATE: 30,
            RecoverySpeed.SLOW: 90,
            RecoverySpeed.VERY_SLOW: 180
        }
        
        base_days = base_timelines[recovery_speed]
        
        # Adjust by attack complexity
        if attack_scenario.attack_type == AttackType.SYSTEMIC_FAILURE:
            base_days *= 2
        elif attack_scenario.mitigation_possible:
            base_days *= 0.7
        
        return base_days
    
    def simulate_recovery(
        self,
        rng: np.random.RandomState,
        economic_loss: EconomicLoss,
        initial_tvl: float
    ) -> EconomicRecovery:
        """
        Simulate economic recovery after attack.
        
        Args:
            rng: Random number generator
            economic_loss: Economic loss from attack
            initial_tvl: TVL before attack
            
        Returns:
            EconomicRecovery instance
        """
        # Define recovery phases based on severity
        if economic_loss.recovery_speed == RecoverySpeed.IMMEDIATE:
            phases = [
                {'start_day': 0, 'end_day': 1, 'start_tvl': 0.9, 'end_tvl': 0.95},
                {'start_day': 1, 'end_day': 7, 'start_tvl': 0.95, 'end_tvl': 0.98}
            ]
            final_tvl = 0.98
            
        elif economic_loss.recovery_speed == RecoverySpeed.FAST:
            phases = [
                {'start_day': 0, 'end_day': 1, 'start_tvl': 0.7, 'end_tvl': 0.75},
                {'start_day': 1, 'end_day': 7, 'start_tvl': 0.75, 'end_tvl': 0.85},
                {'start_day': 7, 'end_day': 30, 'start_tvl': 0.85, 'end_tvl': 0.95}
            ]
            final_tvl = 0.95
            
        elif economic_loss.recovery_speed == RecoverySpeed.MODERATE:
            phases = [
                {'start_day': 0, 'end_day': 1, 'start_tvl': 0.5, 'end_tvl': 0.55},
                {'start_day': 1, 'end_day': 7, 'start_tvl': 0.55, 'end_tvl': 0.65},
                {'start_day': 7, 'end_day': 30, 'start_tvl': 0.65, 'end_tvl': 0.75},
                {'start_day': 30, 'end_day': 90, 'start_tvl': 0.75, 'end_tvl': 0.85}
            ]
            final_tvl = 0.85
            
        elif economic_loss.recovery_speed == RecoverySpeed.SLOW:
            phases = [
                {'start_day': 0, 'end_day': 1, 'start_tvl': 0.3, 'end_tvl': 0.35},
                {'start_day': 1, 'end_day': 7, 'start_tvl': 0.35, 'end_tvl': 0.40},
                {'start_day': 7, 'end_day': 30, 'start_tvl': 0.40, 'end_tvl': 0.50},
                {'start_day': 30, 'end_day': 90, 'start_tvl': 0.50, 'end_tvl': 0.65},
                {'start_day': 90, 'end_day': 180, 'start_tvl': 0.65, 'end_tvl': 0.75}
            ]
            final_tvl = 0.75
            
        else:  # VERY_SLOW
            phases = [
                {'start_day': 0, 'end_day': 1, 'start_tvl': 0.1, 'end_tvl': 0.15},
                {'start_day': 1, 'end_day': 7, 'start_tvl': 0.15, 'end_tvl': 0.20},
                {'start_day': 7, 'end_day': 30, 'start_tvl': 0.20, 'end_tvl': 0.30},
                {'start_day': 30, 'end_day': 90, 'start_tvl': 0.30, 'end_tvl': 0.40},
                {'start_day': 90, 'end_day': 180, 'start_tvl': 0.40, 'end_tvl': 0.50},
                {'start_day': 180, 'end_day': 365, 'start_tvl': 0.50, 'end_tvl': 0.60}
            ]
            final_tvl = 0.60
        
        # Add randomness to phases
        for phase in phases:
            phase['start_tvl'] *= rng.uniform(0.9, 1.1)
            phase['end_tvl'] *= rng.uniform(0.9, 1.1)
        
        # Calculate milestones
        milestones = {}
        for target in [0.25, 0.50, 0.75, 0.90]:
            for phase in phases:
                if phase['start_tvl'] <= target <= phase['end_tvl']:
                    # Interpolate
                    progress = (target - phase['start_tvl']) / (phase['end_tvl'] - phase['start_tvl'])
                    day = phase['start_day'] + progress * (phase['end_day'] - phase['start_day'])
                    milestones[f"{int(target*100)}% recovery"] = day
                    break
        
        permanent_damage = (1 - final_tvl) * 100
        
        return EconomicRecovery(
            recovery_phases=phases,
            milestones=milestones,
            final_tvl_percent=final_tvl,
            permanent_damage_percent=permanent_damage
        )


def test_economic_model():
    """Test the economic impact model."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.attack_scenarios import AttackScenario, AttackType, AttackVector, AttackSeverity
    from src.models.network_state import NetworkSnapshot, ValidatorState, ValidatorTier, MigrationStatus
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    # Create test attack scenario
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
        n_validators=1032,
        total_stake=400000000,
        validators=[],
        geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.3,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.4,
        compromised_validators=100,  # Test with some compromised validators
        attack_occurred=True  # Test scenario with attack
    )
    
    print("Economic Impact Model Test:")
    print("=" * 50)
    
    # Calculate impact
    impact = model.calculate_impact(rng, attack, network)
    
    print(f"\nAttack: {attack.attack_type.value}")
    print(f"Severity: {attack.severity.value}")
    print(f"Stake Compromised: {attack.stake_compromised:.1%}")
    
    print(f"\nEconomic Impact:")
    print(f"  Total Loss: ${impact.total_loss_usd/1e9:.2f}B")
    print(f"  Immediate Loss: ${impact.immediate_loss_usd/1e9:.2f}B")
    print(f"  Long-term Loss: ${impact.long_term_loss_usd/1e9:.2f}B")
    print(f"  Recovery Time: {impact.recovery_timeline_days:.0f} days")
    
    print(f"\nLoss Components:")
    for component in impact.components:
        print(f"  {component.impact_type.value}: ${component.amount_usd/1e9:.2f}B "
              f"({component.percentage_of_tvl:.1%} of TVL)")
    
    print(f"\nMarket Reaction:")
    print(f"  SOL Price Drop: {impact.market_reaction.sol_price_drop_percent:.1f}%")
    print(f"  TVL Drop: {impact.market_reaction.tvl_drop_percent:.1f}%")
    print(f"  Panic Duration: {impact.market_reaction.panic_duration_days:.1f} days")
    
    # Simulate recovery
    recovery = model.simulate_recovery(rng, impact, model.params.total_value_locked_usd)
    
    print(f"\nRecovery Trajectory:")
    print(f"  Final TVL: {recovery.final_tvl_percent:.1%} of pre-attack")
    print(f"  Permanent Damage: {recovery.permanent_damage_percent:.1f}%")
    
    if recovery.milestones:
        print(f"\n  Milestones:")
        for milestone, days in recovery.milestones.items():
            print(f"    {milestone}: {days:.0f} days")
    
    print("\n✓ Economic model test passed")


if __name__ == "__main__":
    test_economic_model()
