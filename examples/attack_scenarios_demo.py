#!/usr/bin/env python3
"""
Demonstration of the attack scenarios model.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.attack_scenarios import (
    AttackScenariosModel,
    AttackType,
    AttackSeverity
)
from src.models.quantum_timeline import QuantumDevelopmentModel
from src.models.network_state import NetworkStateModel
from src.config import QuantumParameters, NetworkParameters


def demonstrate_single_attack_plan():
    """Demonstrate a single attack plan generation."""
    print("=" * 60)
    print("SINGLE ATTACK PLAN GENERATION")
    print("=" * 60)
    
    # Initialize models
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    rng = np.random.RandomState(42)
    
    # Generate quantum timeline
    quantum_timeline = quantum_model.sample(rng)
    crqc_year = quantum_timeline.crqc_year
    
    # Generate network evolution
    network_evolution = network_model.sample(rng, {'crqc_year': crqc_year})
    
    print(f"\nScenario Setup:")
    print(f"  CRQC Emergence: {crqc_year:.1f}")
    
    # Get capability and network at CRQC year
    capability = quantum_timeline.get_capability_at_year(crqc_year)
    network = network_evolution.get_snapshot_at_year(crqc_year)
    
    print(f"  Quantum Capability: {capability.logical_qubits:,} logical qubits")
    print(f"  Network Migration: {network.migration_progress:.1%}")
    print(f"  Vulnerable Stake: {network.vulnerable_stake_percentage:.1%}")
    
    # Generate attack plan
    attack_plan = attack_model.sample(rng, capability, network)
    
    print(f"\nAttack Plan:")
    print(f"  Feasible Attacks: {len(attack_plan.scenarios)}")
    print(f"  Primary Target: {attack_plan.primary_target}")
    print(f"  Overall Success Rate: {attack_plan.estimated_success_rate:.1%}")
    print(f"  Total Stake at Risk: {attack_plan.total_stake_at_risk:.1%}")
    
    if attack_plan.scenarios:
        print(f"\nTop Attack Scenarios:")
        # Sort by impact score
        sorted_scenarios = sorted(
            attack_plan.scenarios,
            key=lambda s: s.impact_score,
            reverse=True
        )
        
        for i, scenario in enumerate(sorted_scenarios[:3], 1):
            print(f"\n  {i}. {scenario.attack_type.value.replace('_', ' ').title()}")
            print(f"     Success Probability: {scenario.success_probability:.1%}")
            print(f"     Severity: {scenario.severity.value}")
            print(f"     Validators Compromised: {scenario.validators_compromised}")
            print(f"     Stake Compromised: {scenario.stake_compromised:.1%}")
            print(f"     Execution Time: {scenario.time_to_execute:.1f} hours")
            print(f"     Detection Probability: {scenario.detection_probability:.1%}")
            print(f"     Mitigation Possible: {'Yes' if scenario.mitigation_possible else 'No'}")
            print(f"     Impact Score: {scenario.impact_score:.2f}")


def demonstrate_attack_progression():
    """Show how attack capabilities evolve over time."""
    print("\n" + "=" * 60)
    print("ATTACK CAPABILITY PROGRESSION")
    print("=" * 60)
    
    # Initialize models
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    rng = np.random.RandomState(42)
    
    # Generate timelines
    quantum_timeline = quantum_model.sample(rng)
    network_evolution = network_model.sample(rng, {'crqc_year': quantum_timeline.crqc_year})
    
    print(f"\nAttack Capability Over Time:")
    print(f"{'Year':<6} {'Qubits':<10} {'Migration':<12} {'Feasible':<10} {'Success':<10}")
    print("-" * 48)
    
    for year in range(2025, 2046, 5):
        capability = quantum_timeline.get_capability_at_year(year)
        network = network_evolution.get_snapshot_at_year(year)
        
        if capability:
            attack_plan = attack_model.sample(rng, capability, network)
            
            print(f"{year:<6} {capability.logical_qubits:<10,} "
                  f"{network.migration_progress:<12.1%} "
                  f"{len(attack_plan.scenarios):<10} "
                  f"{attack_plan.estimated_success_rate:<10.1%}")


def demonstrate_attack_types_comparison():
    """Compare different attack types and their requirements."""
    print("\n" + "=" * 60)
    print("ATTACK TYPES COMPARISON")
    print("=" * 60)
    
    attack_model = AttackScenariosModel()
    
    print(f"\nAttack Type Requirements:")
    print(f"{'Attack Type':<25} {'Qubits Required':<20} {'Base Success Rate':<20}")
    print("-" * 65)
    
    for attack_type in AttackType:
        required = attack_model.attack_requirements[attack_type]
        base_rate = attack_model.base_success_rates[attack_type]
        
        print(f"{attack_type.value:<25} {required:<20,} {base_rate:<20.1%}")
    
    print(f"\nDetection Probabilities:")
    for attack_type in AttackType:
        detection = attack_model.detection_rates[attack_type]
        print(f"  {attack_type.value}: {detection:.1%}")


def demonstrate_network_migration_impact():
    """Show how network migration affects attack success."""
    print("\n" + "=" * 60)
    print("NETWORK MIGRATION IMPACT ON ATTACKS")
    print("=" * 60)
    
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    
    rng = np.random.RandomState(42)
    
    # Generate quantum capability
    quantum_timeline = quantum_model.sample(rng)
    capability = quantum_timeline.get_capability_at_year(2035)
    
    print(f"\nFixed Quantum Capability: {capability.logical_qubits:,} qubits (2035)")
    
    # Test different migration levels
    migration_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    print(f"\n{'Migration':<12} {'Feasible':<12} {'Success Rate':<15} {'Primary Impact':<20}")
    print("-" * 59)
    
    for migration in migration_levels:
        # Create network with specific migration level
        from src.models.network_state import NetworkSnapshot, ValidatorState, ValidatorTier, MigrationStatus
        
        # Create validators with appropriate migration
        validators = []
        for i in range(100):
            is_migrated = (i / 100) < migration
            validators.append(ValidatorState(
                validator_id=i,
                stake_amount=1000000 * (100 - i),
                stake_percentage=0.01 * (100 - i) / 50.5,
                tier=ValidatorTier.SUPERMINORITY if i < 10 else ValidatorTier.SMALL,
                location='north_america',
                is_migrated=is_migrated
            ))
        
        network = NetworkSnapshot(
            year=2035,
            n_validators=100,
            total_stake=50500000,
            validators=validators,
            geographic_distribution={'north_america': 1.0},
            migration_status=MigrationStatus.PARTIAL if 0 < migration < 0.95 else 
                           (MigrationStatus.COMPLETE if migration >= 0.95 else MigrationStatus.NOT_STARTED),
            migration_progress=migration,
            superminority_count=10,
            gini_coefficient=0.8,
            network_resilience=0.3 + migration * 0.5
        )
        
        # Generate attack plan
        attack_plan = attack_model.sample(rng, capability, network)
        
        if attack_plan.scenarios:
            primary_attack = max(attack_plan.scenarios, key=lambda s: s.impact_score)
            primary_type = primary_attack.attack_type.value
        else:
            primary_type = "None feasible"
        
        print(f"{migration:<12.1%} {len(attack_plan.scenarios):<12} "
              f"{attack_plan.estimated_success_rate:<15.1%} {primary_type:<20}")


def demonstrate_attack_windows():
    """Demonstrate attack window identification."""
    print("\n" + "=" * 60)
    print("ATTACK WINDOWS ANALYSIS")
    print("=" * 60)
    
    # Initialize models
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    # Test multiple scenarios
    print(f"\nAnalyzing 10 scenarios for attack windows...")
    
    all_windows = []
    
    for seed in range(10):
        rng = np.random.RandomState(seed)
        
        # Generate timelines
        quantum_timeline = quantum_model.sample(rng)
        network_evolution = network_model.sample(rng, {'crqc_year': quantum_timeline.crqc_year})
        
        # Check for attack windows around CRQC emergence
        crqc_year = quantum_timeline.crqc_year
        
        for year_offset in [-1, 0, 1, 2]:
            year = crqc_year + year_offset
            if year < 2025 or year > 2045:
                continue
            
            capability = quantum_timeline.get_capability_at_year(year)
            network = network_evolution.get_snapshot_at_year(year)
            
            if capability:
                attack_plan = attack_model.sample(rng, capability, network)
                
                for window in attack_plan.windows:
                    all_windows.append({
                        'scenario': seed,
                        'start': window.start_year,
                        'end': window.end_year,
                        'peak': window.peak_year,
                        'score': window.opportunity_score,
                        'duration': window.duration
                    })
    
    if all_windows:
        # Analyze windows
        avg_duration = np.mean([w['duration'] for w in all_windows])
        avg_score = np.mean([w['score'] for w in all_windows])
        
        print(f"\nWindow Statistics (n={len(all_windows)}):")
        print(f"  Average Duration: {avg_duration:.1f} years")
        print(f"  Average Opportunity Score: {avg_score:.2f}")
        
        # Find best window
        best_window = max(all_windows, key=lambda w: w['score'])
        print(f"\nBest Attack Window:")
        print(f"  Scenario: {best_window['scenario']}")
        print(f"  Period: {best_window['start']:.1f} - {best_window['end']:.1f}")
        print(f"  Peak Year: {best_window['peak']:.1f}")
        print(f"  Opportunity Score: {best_window['score']:.2f}")


def demonstrate_severity_distribution():
    """Show distribution of attack severities."""
    print("\n" + "=" * 60)
    print("ATTACK SEVERITY DISTRIBUTION")
    print("=" * 60)
    
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    # Collect severity data
    severity_counts = {s: 0 for s in AttackSeverity}
    total_scenarios = 0
    
    print(f"\nGenerating 100 attack scenarios...")
    
    for seed in range(100):
        rng = np.random.RandomState(seed)
        
        # Generate random year between 2030-2040
        year = 2030 + rng.randint(0, 11)
        
        # Generate capabilities
        quantum_timeline = quantum_model.sample(rng)
        network_evolution = network_model.sample(rng, {'crqc_year': quantum_timeline.crqc_year})
        
        capability = quantum_timeline.get_capability_at_year(year)
        network = network_evolution.get_snapshot_at_year(year)
        
        if capability:
            attack_plan = attack_model.sample(rng, capability, network)
            
            for scenario in attack_plan.scenarios:
                severity_counts[scenario.severity] += 1
                total_scenarios += 1
    
    if total_scenarios > 0:
        print(f"\nSeverity Distribution (n={total_scenarios}):")
        print(f"{'Severity':<15} {'Count':<10} {'Percentage':<12} {'Bar':<30}")
        print("-" * 67)
        
        for severity in AttackSeverity:
            count = severity_counts[severity]
            percentage = count / total_scenarios * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            
            print(f"{severity.value:<15} {count:<10} {percentage:<12.1f}% {bar:<30}")


def demonstrate_contingency_planning():
    """Show contingency scenario generation."""
    print("\n" + "=" * 60)
    print("CONTINGENCY PLANNING")
    print("=" * 60)
    
    attack_model = AttackScenariosModel()
    quantum_model = QuantumDevelopmentModel()
    network_model = NetworkStateModel()
    
    rng = np.random.RandomState(42)
    
    # Generate scenario
    quantum_timeline = quantum_model.sample(rng)
    network_evolution = network_model.sample(rng, {'crqc_year': quantum_timeline.crqc_year})
    
    year = quantum_timeline.crqc_year
    capability = quantum_timeline.get_capability_at_year(year)
    network = network_evolution.get_snapshot_at_year(year)
    
    attack_plan = attack_model.sample(rng, capability, network)
    
    print(f"\nPrimary Attack Plan:")
    if attack_plan.scenarios:
        primary = max(attack_plan.scenarios, key=lambda s: s.impact_score)
        print(f"  Type: {primary.attack_type.value}")
        print(f"  Success Rate: {primary.success_probability:.1%}")
        print(f"  Severity: {primary.severity.value}")
        print(f"  Impact Score: {primary.impact_score:.2f}")
    else:
        print(f"  No primary attacks feasible")
    
    if attack_plan.contingency_scenarios:
        print(f"\nContingency Scenarios ({len(attack_plan.contingency_scenarios)}):")
        for i, contingency in enumerate(attack_plan.contingency_scenarios, 1):
            print(f"\n  Contingency {i}:")
            print(f"    Type: {contingency.attack_type.value}")
            print(f"    Success Rate: {contingency.success_probability:.1%}")
            print(f"    Severity: {contingency.severity.value}")
            print(f"    Execution Time: {contingency.time_to_execute:.1f} hours")
    else:
        print(f"\nNo contingency scenarios generated")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ATTACK SCENARIOS MODEL DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_single_attack_plan()
    demonstrate_attack_progression()
    demonstrate_attack_types_comparison()
    demonstrate_network_migration_impact()
    demonstrate_attack_windows()
    demonstrate_severity_distribution()
    demonstrate_contingency_planning()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
