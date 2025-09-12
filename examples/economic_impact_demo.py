#!/usr/bin/env python3
"""
Demonstration of the economic impact model.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.economic_impact import (
    EconomicImpactModel,
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
    MigrationStatus
)
from src.config import EconomicParameters


def demonstrate_single_impact():
    """Demonstrate economic impact for a single attack."""
    print("=" * 60)
    print("SINGLE ATTACK ECONOMIC IMPACT")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    # Create attack scenario
    attack = AttackScenario(
        attack_type=AttackType.CONSENSUS_HALT,
        vector=AttackVector.VALIDATOR_KEYS,
        year=2035,
        success_probability=0.7,
        severity=AttackSeverity.HIGH,
        validators_compromised=100,
        stake_compromised=0.35,
        accounts_at_risk=1_000_000,
        time_to_execute=24.0,
        detection_probability=0.9,
        mitigation_possible=True
    )
    
    # Create network state
    network = NetworkSnapshot(
        year=2035,
        n_validators=1032,
        total_stake=400_000_000,
        validators=[],
        geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.3,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.4
    )
    
    print(f"\nAttack Scenario:")
    print(f"  Type: {attack.attack_type.value}")
    print(f"  Severity: {attack.severity.value}")
    print(f"  Stake Compromised: {attack.stake_compromised:.1%}")
    print(f"  Accounts at Risk: {attack.accounts_at_risk:,}")
    
    print(f"\nNetwork State:")
    print(f"  Validators: {network.n_validators:,}")
    print(f"  Migration Progress: {network.migration_progress:.1%}")
    print(f"  Network Resilience: {network.network_resilience:.2f}")
    
    # Calculate impact
    impact = model.calculate_impact(rng, attack, network)
    
    print(f"\nEconomic Impact:")
    print(f"  Total Loss: ${impact.total_loss_usd/1e9:.2f}B")
    print(f"  Immediate Loss: ${impact.immediate_loss_usd/1e9:.2f}B")
    print(f"  Long-term Loss: ${impact.long_term_loss_usd/1e9:.2f}B")
    print(f"  Recovery Speed: {impact.recovery_speed.value}")
    print(f"  Recovery Timeline: {impact.recovery_timeline_days:.0f} days")
    
    print(f"\nLoss Breakdown:")
    for component in impact.components:
        print(f"  {component.impact_type.value.replace('_', ' ').title()}:")
        print(f"    Amount: ${component.amount_usd/1e9:.2f}B")
        print(f"    % of TVL: {component.percentage_of_tvl:.1%}")
        print(f"    Time to Realize: {component.time_to_realize_days:.1f} days")
    
    print(f"\nMarket Reaction:")
    print(f"  SOL Price Drop: {impact.market_reaction.sol_price_drop_percent:.1f}%")
    print(f"  TVL Drop: {impact.market_reaction.tvl_drop_percent:.1f}%")
    print(f"  Volume Drop: {impact.market_reaction.daily_volume_drop_percent:.1f}%")
    print(f"  Panic Duration: {impact.market_reaction.panic_duration_days:.1f} days")


def demonstrate_severity_comparison():
    """Compare economic impact across different severity levels."""
    print("\n" + "=" * 60)
    print("SEVERITY IMPACT COMPARISON")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    # Fixed network state
    network = NetworkSnapshot(
        year=2035,
        n_validators=1032,
        total_stake=400_000_000,
        validators=[],
        geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.3,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.4
    )
    
    severities = [
        (AttackSeverity.LOW, AttackType.KEY_COMPROMISE, 0.01),
        (AttackSeverity.MEDIUM, AttackType.TARGETED_THEFT, 0.1),
        (AttackSeverity.HIGH, AttackType.CONSENSUS_HALT, 0.35),
        (AttackSeverity.CRITICAL, AttackType.SYSTEMIC_FAILURE, 0.8)
    ]
    
    print(f"\n{'Severity':<12} {'Attack Type':<20} {'Total Loss':<15} {'Recovery Days':<15}")
    print("-" * 62)
    
    for severity, attack_type, stake_compromised in severities:
        attack = AttackScenario(
            attack_type=attack_type,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=severity,
            validators_compromised=int(stake_compromised * 1000),
            stake_compromised=stake_compromised,
            accounts_at_risk=int(stake_compromised * 10_000_000),
            time_to_execute=24.0,
            detection_probability=0.5,
            mitigation_possible=(severity != AttackSeverity.CRITICAL)
        )
        
        impact = model.calculate_impact(rng, attack, network)
        
        print(f"{severity.value:<12} {attack_type.value:<20} "
              f"${impact.total_loss_usd/1e9:<14.2f}B "
              f"{impact.recovery_timeline_days:<15.0f}")


def demonstrate_migration_impact():
    """Show how network migration affects economic impact."""
    print("\n" + "=" * 60)
    print("MIGRATION IMPACT ON ECONOMIC LOSS")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    # Fixed attack
    attack = AttackScenario(
        attack_type=AttackType.CONSENSUS_HALT,
        vector=AttackVector.VALIDATOR_KEYS,
        year=2035,
        success_probability=0.7,
        severity=AttackSeverity.HIGH,
        validators_compromised=100,
        stake_compromised=0.35,
        accounts_at_risk=1_000_000,
        time_to_execute=24.0,
        detection_probability=0.9,
        mitigation_possible=True
    )
    
    migration_levels = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    print(f"\n{'Migration':<12} {'Total Loss':<15} {'Direct Loss':<15} {'Migration Cost':<15}")
    print("-" * 57)
    
    for migration in migration_levels:
        network = NetworkSnapshot(
            year=2035,
            n_validators=1032,
            total_stake=400_000_000,
            validators=[],
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.PARTIAL if 0 < migration < 0.95 else 
                           (MigrationStatus.COMPLETE if migration >= 0.95 else MigrationStatus.NOT_STARTED),
            migration_progress=migration,
            superminority_count=30,
            gini_coefficient=0.8,
            network_resilience=0.3 + migration * 0.5
        )
        
        impact = model.calculate_impact(rng, attack, network)
        
        direct_loss = impact.get_loss_by_type(ImpactType.DIRECT_LOSS)
        migration_cost = impact.get_loss_by_type(ImpactType.MIGRATION_COST)
        
        print(f"{migration:<12.1%} ${impact.total_loss_usd/1e9:<14.2f}B "
              f"${direct_loss/1e9:<14.2f}B ${migration_cost/1e9:<14.2f}B")


def demonstrate_recovery_trajectory():
    """Demonstrate economic recovery after attack."""
    print("\n" + "=" * 60)
    print("ECONOMIC RECOVERY TRAJECTORY")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    # Create a severe attack
    attack = AttackScenario(
        attack_type=AttackType.CONSENSUS_CONTROL,
        vector=AttackVector.VALIDATOR_KEYS,
        year=2035,
        success_probability=0.5,
        severity=AttackSeverity.CRITICAL,
        validators_compromised=500,
        stake_compromised=0.7,
        accounts_at_risk=10_000_000,
        time_to_execute=100.0,
        detection_probability=0.95,
        mitigation_possible=False
    )
    
    network = NetworkSnapshot(
        year=2035,
        n_validators=1032,
        total_stake=400_000_000,
        validators=[],
        geographic_distribution={'north_america': 1.0},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.2,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.3
    )
    
    # Calculate impact and recovery
    impact = model.calculate_impact(rng, attack, network)
    initial_tvl = model.params.total_value_locked_usd
    recovery = model.simulate_recovery(rng, impact, initial_tvl)
    
    print(f"\nAttack Impact:")
    print(f"  Total Loss: ${impact.total_loss_usd/1e9:.2f}B")
    print(f"  Recovery Speed: {impact.recovery_speed.value}")
    print(f"  Recovery Timeline: {impact.recovery_timeline_days:.0f} days")
    
    print(f"\nRecovery Milestones:")
    for milestone, days in recovery.milestones.items():
        print(f"  {milestone}: Day {days:.0f}")
    
    print(f"\nTVL Recovery Over Time:")
    print(f"{'Day':<10} {'TVL ($B)':<12} {'% of Original':<15}")
    print("-" * 37)
    
    for day in [0, 1, 7, 30, 60, 90, 180, 365]:
        tvl = recovery.get_tvl_at_day(day, initial_tvl)
        percent = (tvl / initial_tvl) * 100
        print(f"{day:<10} ${tvl/1e9:<11.1f} {percent:<14.1f}%")
    
    print(f"\nFinal Recovery:")
    print(f"  Final TVL: {recovery.final_tvl_percent:.1%} of pre-attack")
    print(f"  Permanent Damage: {recovery.permanent_damage_percent:.1f}%")


def demonstrate_cumulative_loss():
    """Show how losses accumulate over time."""
    print("\n" + "=" * 60)
    print("CUMULATIVE LOSS REALIZATION")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    attack = AttackScenario(
        attack_type=AttackType.CONSENSUS_HALT,
        vector=AttackVector.VALIDATOR_KEYS,
        year=2035,
        success_probability=0.7,
        severity=AttackSeverity.HIGH,
        validators_compromised=100,
        stake_compromised=0.35,
        accounts_at_risk=1_000_000,
        time_to_execute=24.0,
        detection_probability=0.9,
        mitigation_possible=True
    )
    
    network = NetworkSnapshot(
        year=2035,
        n_validators=1032,
        total_stake=400_000_000,
        validators=[],
        geographic_distribution={'north_america': 1.0},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.3,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.4
    )
    
    impact = model.calculate_impact(rng, attack, network)
    
    print(f"\nTotal Loss: ${impact.total_loss_usd/1e9:.2f}B")
    print(f"\nLoss Realization Timeline:")
    print(f"{'Day':<10} {'Cumulative Loss':<20} {'% of Total':<15} {'New Components':<30}")
    print("-" * 75)
    
    days_to_check = [0.1, 1, 2, 7, 14, 30, 60, 90, 180]
    previous_loss = 0
    
    for day in days_to_check:
        cumulative = impact.get_cumulative_loss_at_day(day)
        percent = (cumulative / impact.total_loss_usd) * 100 if impact.total_loss_usd > 0 else 0
        
        # Find new components realized
        new_components = []
        for component in impact.components:
            if component.time_to_realize_days <= day and component.time_to_realize_days > (days_to_check[days_to_check.index(day)-1] if day != 0.1 else 0):
                new_components.append(component.impact_type.value)
        
        components_str = ", ".join(new_components) if new_components else "-"
        
        print(f"{day:<10.1f} ${cumulative/1e9:<19.2f}B {percent:<14.1f}% {components_str:<30}")
        previous_loss = cumulative


def demonstrate_attack_type_comparison():
    """Compare economic impact across different attack types."""
    print("\n" + "=" * 60)
    print("ATTACK TYPE ECONOMIC COMPARISON")
    print("=" * 60)
    
    model = EconomicImpactModel()
    rng = np.random.RandomState(42)
    
    network = NetworkSnapshot(
        year=2035,
        n_validators=1032,
        total_stake=400_000_000,
        validators=[],
        geographic_distribution={'north_america': 1.0},
        migration_status=MigrationStatus.IN_PROGRESS,
        migration_progress=0.3,
        superminority_count=30,
        gini_coefficient=0.8,
        network_resilience=0.4
    )
    
    attack_configs = [
        (AttackType.KEY_COMPROMISE, 0.01, 1),
        (AttackType.TARGETED_THEFT, 0.05, 5),
        (AttackType.DOUBLE_SPEND, 0.15, 10),
        (AttackType.CONSENSUS_HALT, 0.35, 100),
        (AttackType.CONSENSUS_CONTROL, 0.70, 500),
        (AttackType.SYSTEMIC_FAILURE, 0.90, 1000)
    ]
    
    print(f"\n{'Attack Type':<25} {'Direct Loss':<15} {'Market Crash':<15} {'Total Loss':<15}")
    print("-" * 70)
    
    for attack_type, stake_compromised, validators_compromised in attack_configs:
        # Determine severity based on attack type
        if attack_type in [AttackType.KEY_COMPROMISE, AttackType.TARGETED_THEFT]:
            severity = AttackSeverity.LOW
        elif attack_type in [AttackType.DOUBLE_SPEND]:
            severity = AttackSeverity.MEDIUM
        elif attack_type in [AttackType.CONSENSUS_HALT]:
            severity = AttackSeverity.HIGH
        else:
            severity = AttackSeverity.CRITICAL
        
        attack = AttackScenario(
            attack_type=attack_type,
            vector=AttackVector.VALIDATOR_KEYS,
            year=2035,
            success_probability=0.7,
            severity=severity,
            validators_compromised=validators_compromised,
            stake_compromised=stake_compromised,
            accounts_at_risk=int(stake_compromised * 10_000_000),
            time_to_execute=24.0,
            detection_probability=0.5,
            mitigation_possible=(severity != AttackSeverity.CRITICAL)
        )
        
        impact = model.calculate_impact(rng, attack, network)
        
        direct_loss = impact.get_loss_by_type(ImpactType.DIRECT_LOSS)
        market_crash = impact.get_loss_by_type(ImpactType.MARKET_CRASH)
        
        print(f"{attack_type.value:<25} ${direct_loss/1e9:<14.2f}B "
              f"${market_crash/1e9:<14.2f}B ${impact.total_loss_usd/1e9:<14.2f}B")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ECONOMIC IMPACT MODEL DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_single_impact()
    demonstrate_severity_comparison()
    demonstrate_migration_impact()
    demonstrate_recovery_trajectory()
    demonstrate_cumulative_loss()
    demonstrate_attack_type_comparison()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
