"""
Economic impact model for quantum attacks on Solana.

This module models the economic consequences of successful quantum attacks,
including direct losses, market reactions, and recovery costs.

Enhanced features (controlled by config flags):
- System dynamics with stocks/flows model
- Vector Autoregression (VAR) for recovery forecasts
- Regulatory response branching
- Cross-chain contagion effects
- Grover-amplified DeFi cascades
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import math
import pandas as pd

# Import statsmodels for VAR if available
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller, acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.info("statsmodels not available - VAR models will use fallback implementation")

from src.config import EconomicParameters
from src.models.attack_scenarios import (
    AttackScenario, AttackType, AttackSeverity, AttackerProfile,
    GroverAttack, HybridAttack
)
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
    # New impact types
    CROSS_CHAIN_CONTAGION = "cross_chain_contagion"  # Spillover to other chains
    REGULATORY_RESPONSE = "regulatory_response"      # Regulatory intervention costs
    GROVER_AMPLIFICATION = "grover_amplification"    # Additional losses from Grover attacks


class RecoverySpeed(Enum):
    """Speed of economic recovery."""
    IMMEDIATE = "immediate"    # < 1 day
    FAST = "fast"             # 1-7 days
    MODERATE = "moderate"     # 1-4 weeks
    SLOW = "slow"             # 1-3 months
    VERY_SLOW = "very_slow"   # > 3 months


@dataclass
class SystemDynamicsState:
    """System dynamics state with stocks and flows."""
    
    # Stocks (accumulated values)
    total_value_locked: float  # Current TVL (stock)
    market_confidence: float   # Confidence level (0-1)
    regulatory_pressure: float # Regulatory attention level (0-1)
    cross_chain_exposure: float # Exposure to other chains (0-1)
    
    # Flows (rates of change)
    value_inflow_rate: float   # TVL inflow rate (USD/day)
    value_outflow_rate: float  # TVL outflow rate (USD/day)
    confidence_decay_rate: float  # Confidence decay rate (per day)
    contagion_spread_rate: float  # Cross-chain spread rate
    
    # Time delays
    market_reaction_delay: float  # Days for market to react
    regulatory_response_delay: float  # Days for regulatory response
    recovery_initiation_delay: float  # Days to start recovery
    
    @property
    def net_flow_rate(self) -> float:
        """Calculate net flow rate."""
        return self.value_inflow_rate - self.value_outflow_rate
    
    @property
    def is_stable(self) -> bool:
        """Check if system is in stable state."""
        return abs(self.net_flow_rate) < 0.01 * self.total_value_locked


@dataclass
class VARForecast:
    """Vector Autoregression forecast for recovery."""
    
    # Time series variables
    tvl_series: np.ndarray      # TVL time series
    price_series: np.ndarray    # Price time series
    volume_series: np.ndarray   # Volume time series
    confidence_series: np.ndarray # Confidence time series
    
    # VAR model parameters
    lag_order: int              # Optimal lag order
    coefficients: np.ndarray    # VAR coefficients
    residuals: np.ndarray       # Model residuals
    
    # Forecasts
    forecast_horizon: int       # Days to forecast
    tvl_forecast: np.ndarray   # Forecasted TVL
    price_forecast: np.ndarray # Forecasted price
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # CI for each series
    
    # Exogenous variables
    leverage_factor: float      # Leverage in system
    regulatory_intervention: bool  # Whether regulators intervene
    
    @property
    def forecast_accuracy(self) -> float:
        """Estimate forecast accuracy based on residuals."""
        if len(self.residuals) == 0:
            return 0.0
        return 1.0 - np.std(self.residuals) / np.mean(np.abs(self.tvl_series))


@dataclass
class RegulatoryResponse:
    """Regulatory response to quantum attack."""
    
    response_type: str  # "emergency_halt", "asset_freeze", "bailout", "none"
    response_time_days: float
    intervention_cost: float
    market_stabilization_effect: float  # 0-1, how much it helps
    long_term_restrictions: List[str]  # New regulations imposed
    cross_jurisdiction_coordination: bool
    
    @property
    def is_effective(self) -> bool:
        """Check if regulatory response is effective."""
        return self.market_stabilization_effect > 0.5


@dataclass
class CrossChainContagion:
    """Cross-chain contagion effects."""
    
    affected_chains: List[str]  # List of affected blockchains
    contagion_probabilities: Dict[str, float]  # Chain -> probability
    spillover_losses: Dict[str, float]  # Chain -> loss amount
    total_ecosystem_impact: float
    contagion_speed_days: float  # How fast it spreads
    
    @property
    def systemic_risk(self) -> bool:
        """Check if represents systemic risk."""
        return len(self.affected_chains) > 3 or self.total_ecosystem_impact > 1e10


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
    # New fields for enhanced modeling
    system_dynamics: Optional[SystemDynamicsState] = None
    var_forecast: Optional[VARForecast] = None
    regulatory_response: Optional[RegulatoryResponse] = None
    cross_chain_contagion: Optional[CrossChainContagion] = None
    
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
    - System dynamics with stocks and flows
    - VAR-based recovery forecasts
    - Cross-chain contagion
    - Regulatory responses
    """
    
    def __init__(self, params: Optional[EconomicParameters] = None,
                 use_system_dynamics: bool = False,
                 use_var_forecast: bool = False,
                 model_cross_chain: bool = False):
        """
        Initialize economic impact model.
        
        Args:
            params: Economic parameters configuration
            use_system_dynamics: Whether to use system dynamics modeling
            use_var_forecast: Whether to use VAR for recovery forecasting
            model_cross_chain: Whether to model cross-chain contagion
        """
        self.params = params or EconomicParameters()
        self.stablecoin_model = StablecoinVulnerabilityModel()
        self.use_system_dynamics = use_system_dynamics
        self.use_var_forecast = use_var_forecast and STATSMODELS_AVAILABLE
        self.model_cross_chain = model_cross_chain
        
        # Impact multipliers by attack type
        self.attack_impact_multipliers = {
            AttackType.KEY_COMPROMISE: 0.1,
            AttackType.TARGETED_THEFT: 0.3,
            AttackType.DOUBLE_SPEND: 0.5,
            AttackType.CONSENSUS_HALT: 0.7,
            AttackType.CONSENSUS_CONTROL: 0.9,
            AttackType.SYSTEMIC_FAILURE: 1.0,
            # New attack types
            AttackType.GROVER_POH: 0.8,
            AttackType.GROVER_HASH: 0.6,
            AttackType.HYBRID_SHOR_GROVER: 0.85,
            AttackType.HYBRID_QUANTUM_CLASSICAL: 0.75,
            AttackType.POH_FORGERY: 0.9
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
        
        # Cross-chain relationships for contagion modeling
        self.cross_chain_correlations = {
            'ethereum': 0.7,    # High correlation
            'polygon': 0.6,      # Medium-high correlation
            'avalanche': 0.5,    # Medium correlation
            'binance': 0.4,      # Medium-low correlation
            'cardano': 0.3,      # Low correlation
            'bitcoin': 0.2       # Very low correlation (different architecture)
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
        
        # 8. Grover amplification (if applicable)
        if hasattr(attack_scenario, 'grover_attack') and attack_scenario.grover_attack:
            grover_impact = self._calculate_grover_amplification(
                rng, attack_scenario, network_snapshot
            )
            if grover_impact.amount_usd > 0:
                components.append(grover_impact)
        
        # 9. Cross-chain contagion (if enabled)
        cross_chain_contagion = None
        if self.model_cross_chain:
            contagion_impact, cross_chain_contagion = self._calculate_cross_chain_contagion(
                rng, attack_scenario, market_reaction
            )
            if contagion_impact.amount_usd > 0:
                components.append(contagion_impact)
        
        # 10. Regulatory response
        regulatory_response = self._model_regulatory_response(
            rng, attack_scenario, sum(c.amount_usd for c in components)
        )
        if regulatory_response and regulatory_response.intervention_cost > 0:
            reg_component = ImpactComponent(
                impact_type=ImpactType.REGULATORY_RESPONSE,
                amount_usd=regulatory_response.intervention_cost,
                percentage_of_tvl=regulatory_response.intervention_cost / self.params.total_value_locked_usd,
                time_to_realize_days=regulatory_response.response_time_days,
                confidence_interval=(regulatory_response.intervention_cost * 0.5,
                                    regulatory_response.intervention_cost * 1.5)
            )
            components.append(reg_component)
        
        # Calculate totals
        total_loss = sum(c.amount_usd for c in components)
        immediate_loss = sum(c.amount_usd for c in components if c.is_immediate)
        long_term_loss = total_loss - immediate_loss
        
        # Determine recovery timeline
        recovery_speed = self.recovery_speeds[attack_scenario.severity]
        recovery_timeline = self._estimate_recovery_timeline(recovery_speed, attack_scenario)
        
        # Apply system dynamics if enabled
        system_dynamics = None
        if self.use_system_dynamics:
            system_dynamics = self._calculate_system_dynamics(
                total_loss, market_reaction, network_snapshot
            )
            # Adjust losses based on system dynamics
            total_loss = self._apply_system_dynamics_adjustment(
                total_loss, system_dynamics
            )
        
        # Generate VAR forecast if enabled
        var_forecast = None
        if self.use_var_forecast:
            var_forecast = self._generate_var_forecast(
                rng, market_reaction, recovery_timeline
            )
        
        return EconomicLoss(
            attack_scenario=attack_scenario,
            components=components,
            market_reaction=market_reaction,
            total_loss_usd=total_loss,
            immediate_loss_usd=immediate_loss,
            long_term_loss_usd=long_term_loss,
            recovery_speed=recovery_speed,
            recovery_timeline_days=recovery_timeline,
            confidence_level=0.95,
            system_dynamics=system_dynamics,
            var_forecast=var_forecast,
            regulatory_response=regulatory_response,
            cross_chain_contagion=cross_chain_contagion
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


    def _calculate_grover_amplification(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        network_snapshot: NetworkSnapshot
    ) -> ImpactComponent:
        """Calculate additional economic impact from Grover attacks."""
        if not hasattr(attack_scenario, 'grover_attack') or not attack_scenario.grover_attack:
            return ImpactComponent(
                impact_type=ImpactType.GROVER_AMPLIFICATION,
                amount_usd=0,
                percentage_of_tvl=0,
                time_to_realize_days=0,
                confidence_interval=(0, 0)
            )
        
        grover = attack_scenario.grover_attack
        amplification = 0
        
        # PoH vulnerability creates massive impact
        if grover.poh_vulnerability:
            # Can forge PoH, essentially breaking Solana's core consensus
            base_impact = self.params.total_value_locked_usd * 0.3  # 30% TVL at risk
            
            # Amplification factor for PoH attacks
            if grover.can_rewrite_history:
                # Can rewrite history - catastrophic
                amplification = base_impact * rng.uniform(2, 3)
            else:
                # Can forge timestamps - severe
                amplification = base_impact * rng.uniform(1.2, 1.8)
        else:
            # Other hash attacks still significant
            amplification = self.params.total_value_locked_usd * 0.1 * rng.uniform(0.5, 1.5)
        
        # Adjust for attack time - faster attacks are worse
        if grover.attack_time_hours < 1:
            amplification *= 1.5  # Very fast attack
        elif grover.attack_time_hours < 24:
            amplification *= 1.2  # Fast attack
        
        return ImpactComponent(
            impact_type=ImpactType.GROVER_AMPLIFICATION,
            amount_usd=amplification,
            percentage_of_tvl=amplification / self.params.total_value_locked_usd,
            time_to_realize_days=rng.uniform(0.1, 1),
            confidence_interval=(amplification * 0.5, amplification * 2)
        )
    
    def _calculate_cross_chain_contagion(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        market_reaction: MarketReaction
    ) -> Tuple[ImpactComponent, CrossChainContagion]:
        """Calculate cross-chain contagion effects."""
        # Determine contagion probability based on attack severity
        base_contagion_prob = market_reaction.total_market_impact
        
        affected_chains = []
        contagion_probs = {}
        spillover_losses = {}
        
        for chain, correlation in self.cross_chain_correlations.items():
            # Probability of contagion based on correlation
            contagion_prob = base_contagion_prob * correlation
            contagion_probs[chain] = contagion_prob
            
            if rng.random() < contagion_prob:
                affected_chains.append(chain)
                
                # Loss proportional to correlation and market cap
                # Assume other chains have similar or larger TVL
                chain_tvl = self.params.total_value_locked_usd * rng.uniform(0.5, 2.0)
                loss_percentage = correlation * market_reaction.tvl_drop_percent / 100
                spillover_losses[chain] = chain_tvl * loss_percentage
        
        total_spillover = sum(spillover_losses.values())
        
        contagion = CrossChainContagion(
            affected_chains=affected_chains,
            contagion_probabilities=contagion_probs,
            spillover_losses=spillover_losses,
            total_ecosystem_impact=total_spillover,
            contagion_speed_days=rng.uniform(0.5, 3)
        )
        
        impact = ImpactComponent(
            impact_type=ImpactType.CROSS_CHAIN_CONTAGION,
            amount_usd=total_spillover * 0.1,  # Solana bears 10% of ecosystem damage
            percentage_of_tvl=total_spillover * 0.1 / self.params.total_value_locked_usd,
            time_to_realize_days=contagion.contagion_speed_days,
            confidence_interval=(total_spillover * 0.05, total_spillover * 0.2)
        )
        
        return impact, contagion
    
    def _model_regulatory_response(
        self,
        rng: np.random.RandomState,
        attack_scenario: AttackScenario,
        total_loss: float
    ) -> Optional[RegulatoryResponse]:
        """Model regulatory response to attack."""
        # Response probability based on severity and loss
        response_prob = min(0.95, total_loss / (self.params.total_value_locked_usd * 0.5))
        
        if rng.random() > response_prob:
            return None
        
        # Determine response type based on severity
        if attack_scenario.severity == AttackSeverity.CRITICAL:
            response_types = ['emergency_halt', 'asset_freeze', 'bailout']
            response_type = rng.choice(response_types, p=[0.3, 0.3, 0.4])
        elif attack_scenario.severity == AttackSeverity.HIGH:
            response_types = ['emergency_halt', 'asset_freeze', 'none']
            response_type = rng.choice(response_types, p=[0.4, 0.4, 0.2])
        else:
            response_type = 'none' if rng.random() < 0.7 else 'emergency_halt'
        
        # Calculate response parameters
        if response_type == 'bailout':
            intervention_cost = total_loss * rng.uniform(0.3, 0.6)
            stabilization_effect = rng.uniform(0.6, 0.9)
            response_days = rng.uniform(1, 7)
        elif response_type == 'emergency_halt':
            intervention_cost = self.params.total_value_locked_usd * 0.01  # Cost of halt
            stabilization_effect = rng.uniform(0.4, 0.7)
            response_days = rng.uniform(0.1, 1)
        elif response_type == 'asset_freeze':
            intervention_cost = total_loss * 0.05  # Administrative costs
            stabilization_effect = rng.uniform(0.3, 0.6)
            response_days = rng.uniform(0.5, 2)
        else:
            intervention_cost = 0
            stabilization_effect = 0
            response_days = 0
        
        # Long-term restrictions
        restrictions = []
        if response_type != 'none':
            possible_restrictions = [
                'mandatory_quantum_safe_migration',
                'enhanced_validator_requirements',
                'cross_border_restrictions',
                'defi_protocol_audits',
                'stablecoin_reserves'
            ]
            n_restrictions = rng.poisson(2)
            restrictions = rng.choice(possible_restrictions, 
                                    size=min(n_restrictions, len(possible_restrictions)),
                                    replace=False).tolist()
        
        return RegulatoryResponse(
            response_type=response_type,
            response_time_days=response_days,
            intervention_cost=intervention_cost,
            market_stabilization_effect=stabilization_effect,
            long_term_restrictions=restrictions,
            cross_jurisdiction_coordination=response_type == 'bailout'
        )
    
    def _calculate_system_dynamics(
        self,
        total_loss: float,
        market_reaction: MarketReaction,
        network_snapshot: NetworkSnapshot
    ) -> SystemDynamicsState:
        """
        Calculate system dynamics state using stocks and flows.
        
        Implements: L = D + M * C + R
        Where: L = Total loss, D = Direct loss, M = Market multiplier,
               C = Cascade effects, R = Recovery costs
        """
        # Initial stocks
        tvl_stock = self.params.total_value_locked_usd * (1 - market_reaction.tvl_drop_percent / 100)
        confidence_stock = 1.0 - market_reaction.total_market_impact
        regulatory_pressure = min(1.0, total_loss / (self.params.total_value_locked_usd * 0.3))
        cross_chain_exposure = 0.3  # Base exposure
        
        # Calculate flows based on market reaction
        # Outflows increase with panic, inflows increase with confidence
        value_outflow = tvl_stock * market_reaction.tvl_drop_percent / 100 / market_reaction.panic_duration_days
        value_inflow = tvl_stock * confidence_stock * 0.1 / market_reaction.recovery_time_days
        
        # Confidence decay based on attack severity
        confidence_decay = (1 - confidence_stock) / market_reaction.panic_duration_days
        
        # Contagion spread rate
        contagion_rate = market_reaction.total_market_impact * 0.2  # 20% of impact spreads
        
        # Time delays
        market_delay = 0.5  # Half day for market reaction
        regulatory_delay = max(1, 7 - regulatory_pressure * 6)  # Faster with more pressure
        recovery_delay = market_reaction.panic_duration_days
        
        return SystemDynamicsState(
            total_value_locked=tvl_stock,
            market_confidence=confidence_stock,
            regulatory_pressure=regulatory_pressure,
            cross_chain_exposure=cross_chain_exposure,
            value_inflow_rate=value_inflow,
            value_outflow_rate=value_outflow,
            confidence_decay_rate=confidence_decay,
            contagion_spread_rate=contagion_rate,
            market_reaction_delay=market_delay,
            regulatory_response_delay=regulatory_delay,
            recovery_initiation_delay=recovery_delay
        )
    
    def _apply_system_dynamics_adjustment(
        self,
        base_loss: float,
        dynamics: SystemDynamicsState
    ) -> float:
        """
        Apply system dynamics adjustment to losses.
        
        Formula: L = D + M * C + R
        """
        # Direct loss component
        D = base_loss * 0.3  # 30% is direct
        
        # Market multiplier based on confidence
        M = 2.0 - dynamics.market_confidence  # Multiplier from 1.0 to 2.0
        
        # Cascade effects based on flows
        C = abs(dynamics.net_flow_rate) * 30  # 30 days of net flow
        
        # Recovery costs based on regulatory pressure
        R = base_loss * dynamics.regulatory_pressure * 0.2
        
        # Apply formula
        adjusted_loss = D + M * C + R
        
        # Ensure reasonable bounds
        return max(base_loss, min(adjusted_loss, base_loss * 3))
    
    def _generate_var_forecast(
        self,
        rng: np.random.RandomState,
        market_reaction: MarketReaction,
        recovery_timeline: float
    ) -> Optional[VARForecast]:
        """Generate VAR forecast for recovery using Vector Autoregression."""
        if not self.use_var_forecast:
            return None
        
        # Generate synthetic historical data for VAR
        n_historical = 100
        
        # Create correlated time series
        mean = [100, 50, 1000, 0.8]  # TVL, Price, Volume, Confidence
        cov = [[100, 30, 50, -10],
               [30, 25, 20, -5],
               [50, 20, 200, -15],
               [-10, -5, -15, 0.1]]
        
        historical_data = rng.multivariate_normal(mean, cov, n_historical)
        
        # Apply shock based on market reaction
        shock_size = int(n_historical * 0.1)
        historical_data[-shock_size:, 0] *= (1 - market_reaction.tvl_drop_percent / 100)
        historical_data[-shock_size:, 1] *= (1 - market_reaction.sol_price_drop_percent / 100)
        historical_data[-shock_size:, 2] *= (1 - market_reaction.daily_volume_drop_percent / 100)
        historical_data[-shock_size:, 3] *= 0.5  # Confidence drops
        
        # Extract series
        tvl_series = historical_data[:, 0]
        price_series = historical_data[:, 1]
        volume_series = historical_data[:, 2]
        confidence_series = historical_data[:, 3]
        
        if STATSMODELS_AVAILABLE:
            # Use statsmodels VAR
            try:
                # Create dataframe
                df = pd.DataFrame({
                    'tvl': tvl_series,
                    'price': price_series,
                    'volume': volume_series,
                    'confidence': confidence_series
                })
                
                # Fit VAR model
                model = VAR(df)
                lag_order = model.select_order(maxlags=5).selected_orders['aic']
                fitted_model = model.fit(lag_order)
                
                # Generate forecast
                forecast_horizon = int(recovery_timeline)
                forecast = fitted_model.forecast(df.values[-lag_order:], forecast_horizon)
                
                # Extract forecasts
                tvl_forecast = forecast[:, 0]
                price_forecast = forecast[:, 1]
                
                # Generate confidence intervals
                forecast_ci = fitted_model.forecast_interval(df.values[-lag_order:], forecast_horizon)
                ci_lower = forecast_ci[0]
                ci_upper = forecast_ci[2]
                
                confidence_intervals = {
                    'tvl': (ci_lower[:, 0], ci_upper[:, 0]),
                    'price': (ci_lower[:, 1], ci_upper[:, 1])
                }
                
                return VARForecast(
                    tvl_series=tvl_series,
                    price_series=price_series,
                    volume_series=volume_series,
                    confidence_series=confidence_series,
                    lag_order=lag_order,
                    coefficients=fitted_model.coefs,
                    residuals=fitted_model.resid,
                    forecast_horizon=forecast_horizon,
                    tvl_forecast=tvl_forecast,
                    price_forecast=price_forecast,
                    confidence_intervals=confidence_intervals,
                    leverage_factor=rng.uniform(1.5, 3.0),
                    regulatory_intervention=rng.random() < 0.3
                )
            except Exception as e:
                logging.warning(f"VAR model fitting failed: {e}")
                # Fall through to simple forecast
        
        # Simple forecast fallback
        forecast_horizon = int(recovery_timeline)
        
        # Exponential recovery
        recovery_rate = 1 - np.exp(-3 / recovery_timeline)  # 95% recovery over timeline
        tvl_forecast = tvl_series[-1] * (1 + recovery_rate * np.linspace(0, 1, forecast_horizon))
        price_forecast = price_series[-1] * (1 + recovery_rate * 0.8 * np.linspace(0, 1, forecast_horizon))
        
        return VARForecast(
            tvl_series=tvl_series,
            price_series=price_series,
            volume_series=volume_series,
            confidence_series=confidence_series,
            lag_order=2,
            coefficients=np.random.randn(2, 4, 4) * 0.1,  # Random small coefficients
            residuals=np.random.randn(len(tvl_series), 4) * 0.01,
            forecast_horizon=forecast_horizon,
            tvl_forecast=tvl_forecast,
            price_forecast=price_forecast,
            confidence_intervals={
                'tvl': (tvl_forecast * 0.8, tvl_forecast * 1.2),
                'price': (price_forecast * 0.7, price_forecast * 1.3)
            },
            leverage_factor=rng.uniform(1.5, 3.0),
            regulatory_intervention=rng.random() < 0.3
        )


def test_economic_model():
    """Test the economic impact model."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.attack_scenarios import (
        AttackScenario, AttackType, AttackVector, AttackSeverity,
        GroverAttack, HybridAttack
    )
    from src.models.network_state import NetworkSnapshot, ValidatorState, ValidatorTier, MigrationStatus
    
    # Test with enhanced features
    model = EconomicImpactModel(
        use_system_dynamics=True,
        use_var_forecast=True,
        model_cross_chain=True
    )
    rng = np.random.RandomState(42)
    
    # Create test attack scenario with Grover component
    grover_attack = GroverAttack(
        target="PoH",
        operations_required=2**128,
        speedup_achieved=65536,  # Square root of 2^128
        qubits_required=1000000,
        gate_count=2**128 / 65536,
        attack_time_hours=12.0,
        poh_vulnerability=True,
        can_rewrite_history=False
    )
    
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
        mitigation_possible=True,
        grover_attack=grover_attack
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
    
    # Show enhanced features if enabled
    if impact.system_dynamics:
        print(f"\nSystem Dynamics:")
        print(f"  Net Flow Rate: ${impact.system_dynamics.net_flow_rate/1e6:.1f}M/day")
        print(f"  Market Confidence: {impact.system_dynamics.market_confidence:.2%}")
        print(f"  Regulatory Pressure: {impact.system_dynamics.regulatory_pressure:.2%}")
        print(f"  System Stable: {impact.system_dynamics.is_stable}")
    
    if impact.var_forecast:
        print(f"\nVAR Forecast:")
        print(f"  Forecast Horizon: {impact.var_forecast.forecast_horizon} days")
        print(f"  Forecast Accuracy: {impact.var_forecast.forecast_accuracy:.2%}")
        print(f"  Leverage Factor: {impact.var_forecast.leverage_factor:.1f}x")
    
    if impact.regulatory_response:
        print(f"\nRegulatory Response:")
        print(f"  Type: {impact.regulatory_response.response_type}")
        print(f"  Response Time: {impact.regulatory_response.response_time_days:.1f} days")
        print(f"  Intervention Cost: ${impact.regulatory_response.intervention_cost/1e9:.2f}B")
        print(f"  Effectiveness: {impact.regulatory_response.is_effective}")
    
    if impact.cross_chain_contagion:
        print(f"\nCross-Chain Contagion:")
        print(f"  Affected Chains: {', '.join(impact.cross_chain_contagion.affected_chains)}")
        print(f"  Total Ecosystem Impact: ${impact.cross_chain_contagion.total_ecosystem_impact/1e9:.2f}B")
        print(f"  Systemic Risk: {impact.cross_chain_contagion.systemic_risk}")
    
    print("\n✓ Economic model test passed")


if __name__ == "__main__":
    test_economic_model()
