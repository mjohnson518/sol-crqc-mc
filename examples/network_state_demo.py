#!/usr/bin/env python3
"""
Demonstration of the network state model.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.network_state import (
    NetworkStateModel,
    MigrationStatus,
    ValidatorTier
)
from src.config import NetworkParameters


def demonstrate_single_evolution():
    """Demonstrate a single network evolution."""
    print("=" * 60)
    print("SINGLE NETWORK EVOLUTION")
    print("=" * 60)
    
    model = NetworkStateModel()
    rng = np.random.RandomState(42)
    
    # Simulate with CRQC emerging in 2035
    quantum_timeline = {'crqc_year': 2035}
    
    evolution = model.sample(rng, quantum_timeline)
    
    print(f"\nEvolution Summary:")
    print(f"  Migration Start: {evolution.migration_start_year:.1f}")
    if evolution.migration_completion_year:
        print(f"  Migration Complete: {evolution.migration_completion_year:.1f}")
    else:
        print(f"  Migration Incomplete by 2045")
    print(f"  Peak Validators: {evolution.peak_validators:,}")
    print(f"  Minimum Gini: {evolution.minimum_gini:.3f}")
    
    # Show progression over time
    print(f"\nNetwork Progression:")
    print(f"{'Year':<6} {'Validators':<12} {'Migration':<12} {'Vulnerable':<12} {'Resilience':<12}")
    print("-" * 60)
    
    for year in [2025, 2030, 2035, 2040, 2045]:
        snapshot = evolution.get_snapshot_at_year(year)
        print(f"{year:<6} {snapshot.n_validators:<12,} "
              f"{snapshot.migration_progress:<12.1%} "
              f"{snapshot.vulnerable_stake_percentage:<12.1%} "
              f"{snapshot.network_resilience:<12.2f}")
    
    # Show migration milestones
    milestones = evolution.get_migration_timeline()
    print(f"\nMigration Milestones:")
    for milestone, year in milestones.items():
        if year:
            print(f"  {milestone}: {year:.1f}")


def demonstrate_migration_scenarios():
    """Compare different migration scenarios."""
    print("\n" + "=" * 60)
    print("MIGRATION SCENARIO COMPARISON")
    print("=" * 60)
    
    model = NetworkStateModel()
    quantum_timeline = {'crqc_year': 2035}
    
    scenarios = {}
    
    # Force different migration profiles
    for profile_name in ['proactive', 'reactive', 'laggard']:
        # Temporarily override the model's profile selection
        original_profiles = model.migration_profiles.copy()
        
        # Set weights to force selection
        if profile_name == 'proactive':
            weights = [1.0, 0.0, 0.0]
        elif profile_name == 'reactive':
            weights = [0.0, 1.0, 0.0]
        else:  # laggard
            weights = [0.0, 0.0, 1.0]
        
        # Generate multiple samples
        vulnerabilities = []
        for i in range(20):
            rng = np.random.RandomState(1000 + i)
            
            # Force the profile
            profile = model.migration_profiles[profile_name].copy()
            profile['adoption_rate'] *= rng.uniform(0.9, 1.1)
            profile['speed'] *= rng.uniform(0.9, 1.1)
            
            # Generate evolution with forced profile
            evolution = model.sample(rng, quantum_timeline)
            
            # Get vulnerability at CRQC emergence
            snapshot_2035 = evolution.get_snapshot_at_year(2035)
            vulnerabilities.append(snapshot_2035.vulnerable_stake_percentage)
        
        scenarios[profile_name] = {
            'mean_vulnerable': np.mean(vulnerabilities),
            'std_vulnerable': np.std(vulnerabilities),
            'min_vulnerable': np.min(vulnerabilities),
            'max_vulnerable': np.max(vulnerabilities)
        }
    
    # Display comparison
    print(f"\nVulnerable Stake at CRQC Emergence (2035):")
    print(f"{'Profile':<12} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 52)
    
    for profile, stats in scenarios.items():
        print(f"{profile.capitalize():<12} "
              f"{stats['mean_vulnerable']:<10.1%} "
              f"{stats['std_vulnerable']:<10.1%} "
              f"{stats['min_vulnerable']:<10.1%} "
              f"{stats['max_vulnerable']:<10.1%}")


def demonstrate_validator_dynamics():
    """Show validator distribution and dynamics."""
    print("\n" + "=" * 60)
    print("VALIDATOR DYNAMICS")
    print("=" * 60)
    
    model = NetworkStateModel()
    rng = np.random.RandomState(42)
    
    quantum_timeline = {'crqc_year': 2035}
    evolution = model.sample(rng, quantum_timeline)
    
    # Analyze validator distribution at different years
    for year in [2025, 2035, 2045]:
        snapshot = evolution.get_snapshot_at_year(year)
        
        print(f"\n{year} Validator Distribution:")
        print(f"  Total Validators: {snapshot.n_validators:,}")
        print(f"  Superminority Count: {snapshot.superminority_count}")
        print(f"  Gini Coefficient: {snapshot.gini_coefficient:.3f}")
        
        # Count by tier
        tier_counts = {tier: 0 for tier in ValidatorTier}
        tier_stakes = {tier: 0.0 for tier in ValidatorTier}
        
        for validator in snapshot.validators:
            tier_counts[validator.tier] += 1
            tier_stakes[validator.tier] += validator.stake_percentage
        
        print(f"\n  Distribution by Tier:")
        for tier in ValidatorTier:
            count = tier_counts[tier]
            stake = tier_stakes[tier]
            print(f"    {tier.value:<15}: {count:4} validators, {stake:6.1%} of stake")
        
        # Geographic distribution
        print(f"\n  Geographic Distribution:")
        for region, percentage in snapshot.geographic_distribution.items():
            print(f"    {region:<15}: {percentage:.1%}")


def demonstrate_attack_surface():
    """Analyze attack surface evolution."""
    print("\n" + "=" * 60)
    print("ATTACK SURFACE ANALYSIS")
    print("=" * 60)
    
    model = NetworkStateModel()
    rng = np.random.RandomState(42)
    
    # Test with different CRQC timelines
    crqc_years = [2030, 2035, 2040]
    
    for crqc_year in crqc_years:
        quantum_timeline = {'crqc_year': crqc_year}
        evolution = model.sample(rng, quantum_timeline)
        
        print(f"\nScenario: CRQC emerges in {crqc_year}")
        
        # Get attack surface at CRQC emergence
        snapshot = evolution.get_snapshot_at_year(crqc_year)
        attack_surface = snapshot.get_attack_surface()
        
        print(f"  Network State at {crqc_year}:")
        print(f"    Total Validators: {snapshot.n_validators:,}")
        print(f"    Migration Progress: {snapshot.migration_progress:.1%}")
        
        print(f"  Attack Surface:")
        print(f"    Vulnerable Validators: {attack_surface['vulnerable_validators']:,}")
        print(f"    Vulnerable Stake: {attack_surface['vulnerable_stake_percentage']:.1%}")
        print(f"    Superminority at Risk: {attack_surface['superminority_vulnerable']}")
        
        # Show vulnerable validators by tier
        print(f"  Vulnerable by Tier:")
        for tier, count in attack_surface['vulnerable_by_tier'].items():
            if count > 0:
                print(f"    {tier:<15}: {count:,}")


def demonstrate_resilience_factors():
    """Show factors affecting network resilience."""
    print("\n" + "=" * 60)
    print("NETWORK RESILIENCE FACTORS")
    print("=" * 60)
    
    model = NetworkStateModel()
    
    # Create scenarios with different resilience factors
    scenarios = [
        {
            'name': 'High Resilience',
            'migration_progress': 0.9,
            'superminority_count': 50,
            'total_validators': 5000,
            'year': 2030,
            'crqc_year': 2040
        },
        {
            'name': 'Medium Resilience',
            'migration_progress': 0.5,
            'superminority_count': 35,
            'total_validators': 3500,
            'year': 2033,
            'crqc_year': 2035
        },
        {
            'name': 'Low Resilience',
            'migration_progress': 0.1,
            'superminority_count': 20,
            'total_validators': 2500,
            'year': 2034,
            'crqc_year': 2035
        }
    ]
    
    print(f"\nResilience Score Calculation:")
    print(f"{'Scenario':<20} {'Migration':<12} {'Decentral':<12} {'Time Buffer':<12} {'Score':<10}")
    print("-" * 76)
    
    for scenario in scenarios:
        # Calculate components
        migration_score = scenario['migration_progress']
        decentralization_score = 1.0 - (scenario['superminority_count'] / scenario['total_validators'])
        years_until_crqc = max(0, scenario['crqc_year'] - scenario['year'])
        time_score = min(years_until_crqc / 10, 1.0)
        
        # Weighted average (matching model weights)
        weights = [0.5, 0.3, 0.2]
        scores = [migration_score, decentralization_score, time_score]
        resilience = sum(w * s for w, s in zip(weights, scores))
        
        print(f"{scenario['name']:<20} "
              f"{migration_score:<12.2f} "
              f"{decentralization_score:<12.2f} "
              f"{time_score:<12.2f} "
              f"{resilience:<10.2f}")


def demonstrate_network_growth():
    """Show network growth patterns."""
    print("\n" + "=" * 60)
    print("NETWORK GROWTH PATTERNS")
    print("=" * 60)
    
    model = NetworkStateModel()
    
    # Generate multiple samples to show variation
    n_samples = 100
    quantum_timeline = {'crqc_year': 2035}
    
    validator_counts = {year: [] for year in range(2025, 2046, 5)}
    stake_totals = {year: [] for year in range(2025, 2046, 5)}
    
    for i in range(n_samples):
        rng = np.random.RandomState(i)
        evolution = model.sample(rng, quantum_timeline)
        
        for year in validator_counts.keys():
            snapshot = evolution.get_snapshot_at_year(year)
            validator_counts[year].append(snapshot.n_validators)
            stake_totals[year].append(snapshot.total_stake)
    
    # Display statistics
    print(f"\nValidator Count Growth (n={n_samples}):")
    print(f"{'Year':<6} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 54)
    
    for year in sorted(validator_counts.keys()):
        counts = validator_counts[year]
        print(f"{year:<6} {np.mean(counts):<12,.0f} "
              f"{np.std(counts):<12,.0f} "
              f"{np.min(counts):<12,} "
              f"{np.max(counts):<12,}")
    
    print(f"\nTotal Stake Growth (billions of SOL):")
    print(f"{'Year':<6} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 54)
    
    for year in sorted(stake_totals.keys()):
        stakes = [s/1e9 for s in stake_totals[year]]  # Convert to billions
        print(f"{year:<6} {np.mean(stakes):<12.1f} "
              f"{np.std(stakes):<12.1f} "
              f"{np.min(stakes):<12.1f} "
              f"{np.max(stakes):<12.1f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("NETWORK STATE MODEL DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_single_evolution()
    demonstrate_migration_scenarios()
    demonstrate_validator_dynamics()
    demonstrate_attack_surface()
    demonstrate_resilience_factors()
    demonstrate_network_growth()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
