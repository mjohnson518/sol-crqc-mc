#!/usr/bin/env python3
"""
Demonstration of the quantum timeline model.
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantum_timeline import (
    QuantumDevelopmentModel,
    QuantumThreat
)
from src.config import QuantumParameters
from src.distributions.probability_dists import (
    TimeDistributions,
    DistributionSampler
)


def demonstrate_single_timeline():
    """Demonstrate a single quantum timeline generation."""
    print("=" * 60)
    print("SINGLE QUANTUM TIMELINE")
    print("=" * 60)
    
    model = QuantumDevelopmentModel()
    rng = np.random.RandomState(42)
    
    timeline = model.sample(rng)
    
    print(f"\nTimeline Summary:")
    print(f"  Projection Method: {timeline.projection_method}")
    print(f"  Confidence: {timeline.confidence:.1%}")
    print(f"  CRQC Year: {timeline.crqc_year:.1f}")
    print(f"  Breakthrough Years: {len(timeline.breakthrough_years)}")
    
    # Show capability progression
    print(f"\nCapability Progression:")
    years_to_show = [2025, 2030, 2035, 2040, 2045]
    
    for year in years_to_show:
        cap = timeline.get_capability_at_year(year)
        print(f"  {year}:")
        print(f"    Logical Qubits: {cap.logical_qubits:,}")
        print(f"    Threat Level: {cap.threat_level.value}")
        print(f"    Can Break Ed25519: {cap.can_break_ed25519}")
        if cap.can_break_ed25519:
            print(f"    Break Time: {cap.estimated_break_time_hours:.1f} hours")
    
    # Show threat level timeline
    print(f"\nYears to Threat Levels:")
    for threat in [QuantumThreat.EMERGING, QuantumThreat.MODERATE, 
                   QuantumThreat.HIGH, QuantumThreat.CRITICAL]:
        years = timeline.years_until_threat_level(threat)
        if years < float('inf'):
            print(f"  {threat.value}: {2025 + years:.0f}")
        else:
            print(f"  {threat.value}: Not reached by 2045")


def demonstrate_projection_methods():
    """Compare different projection methods."""
    print("\n" + "=" * 60)
    print("PROJECTION METHOD COMPARISON")
    print("=" * 60)
    
    model = QuantumDevelopmentModel()
    
    methods = {
        'Industry': [],
        'Expert': [],
        'Breakthrough': [],
        'Historical': []
    }
    
    # Generate multiple samples for each method
    n_samples = 20
    
    for method_name in methods.keys():
        for i in range(n_samples):
            rng = np.random.RandomState(1000 + i)
            
            # Force specific method
            if method_name == 'Industry':
                timeline = model._sample_industry_projection(rng)
            elif method_name == 'Expert':
                timeline = model._sample_expert_projection(rng)
            elif method_name == 'Breakthrough':
                timeline = model._sample_breakthrough_projection(rng)
            else:  # Historical
                timeline = model._sample_historical_projection(rng)
            
            methods[method_name].append(timeline.crqc_year)
    
    # Compare statistics
    print(f"\nCRQC Year Statistics ({n_samples} samples each):")
    print(f"{'Method':<15} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    for method_name, years in methods.items():
        years_array = np.array(years)
        print(f"{method_name:<15} {np.mean(years_array):<10.1f} "
              f"{np.std(years_array):<10.1f} {np.min(years_array):<10.1f} "
              f"{np.max(years_array):<10.1f}")


def demonstrate_consensus_projection():
    """Demonstrate consensus projection across many samples."""
    print("\n" + "=" * 60)
    print("CONSENSUS PROJECTION")
    print("=" * 60)
    
    model = QuantumDevelopmentModel()
    
    print("\nGenerating consensus from 1000 samples...")
    consensus = model.create_consensus_projection(n_samples=1000, seed=42)
    
    print("\nCRQC Emergence Statistics:")
    crqc_stats = consensus['crqc_emergence']
    print(f"  Mean: {crqc_stats['mean']:.1f}")
    print(f"  Median: {crqc_stats['median']:.1f}")
    print(f"  Std Dev: {crqc_stats['std']:.1f}")
    
    print(f"\nPercentiles:")
    for p, year in crqc_stats['percentiles'].items():
        print(f"  {p:3d}%: {year:.1f}")
    
    print(f"\nThreat Level Probabilities:")
    for level, data in consensus['threat_levels'].items():
        if 'probability' in data:
            print(f"  {level}: {data['probability']:.1%} probability by 2045")
            if 'mean' in data:
                print(f"    Mean year: {data['mean']:.1f}")


def demonstrate_parameter_sensitivity():
    """Show sensitivity to key parameters."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Base case
    base_params = QuantumParameters()
    base_model = QuantumDevelopmentModel(base_params)
    
    # Variations
    variations = {
        'Base': base_params,
        'Slow Growth': QuantumParameters(qubit_growth_rate=1.2),
        'Fast Growth': QuantumParameters(qubit_growth_rate=2.0),
        'More Qubits': QuantumParameters(logical_qubits_for_ed25519=3000),
        'Fewer Qubits': QuantumParameters(logical_qubits_for_ed25519=2000)
    }
    
    print(f"\nParameter Variations (100 samples each):")
    print(f"{'Scenario':<15} {'Mean CRQC':<12} {'Std Dev':<10} {'P(2030)':<10} {'P(2035)':<10}")
    print("-" * 65)
    
    for name, params in variations.items():
        model = QuantumDevelopmentModel(params)
        
        crqc_years = []
        for i in range(100):
            rng = np.random.RandomState(i)
            timeline = model.sample(rng)
            crqc_years.append(timeline.crqc_year)
        
        crqc_array = np.array(crqc_years)
        p_2030 = np.mean(crqc_array <= 2030)
        p_2035 = np.mean(crqc_array <= 2035)
        
        print(f"{name:<15} {np.mean(crqc_array):<12.1f} {np.std(crqc_array):<10.1f} "
              f"{p_2030:<10.1%} {p_2035:<10.1%}")


def demonstrate_qubit_growth_patterns():
    """Show different qubit growth patterns."""
    print("\n" + "=" * 60)
    print("QUBIT GROWTH PATTERNS")
    print("=" * 60)
    
    model = QuantumDevelopmentModel()
    
    # Generate samples from each method
    patterns = {}
    methods = ['industry', 'expert', 'breakthrough', 'historical']
    
    for method in methods:
        rng = np.random.RandomState(42)
        
        if method == 'industry':
            timeline = model._sample_industry_projection(rng)
        elif method == 'expert':
            timeline = model._sample_expert_projection(rng)
        elif method == 'breakthrough':
            timeline = model._sample_breakthrough_projection(rng)
        else:
            timeline = model._sample_historical_projection(rng)
        
        # Extract qubit counts for specific years
        years = [2025, 2030, 2035, 2040, 2045]
        qubits = []
        for year in years:
            cap = timeline.get_capability_at_year(year)
            qubits.append(cap.logical_qubits)
        
        patterns[method] = qubits
    
    # Display growth patterns
    print(f"\nLogical Qubit Projections:")
    print(f"{'Method':<12} {'2025':<10} {'2030':<10} {'2035':<10} {'2040':<10} {'2045':<10}")
    print("-" * 62)
    
    for method, qubits in patterns.items():
        print(f"{method.capitalize():<12}", end="")
        for q in qubits:
            print(f"{q:<10,}", end="")
        print()
    
    # Calculate growth rates
    print(f"\nAnnualized Growth Rates (2025-2045):")
    for method, qubits in patterns.items():
        if qubits[0] > 0 and qubits[-1] > 0:
            growth_rate = (qubits[-1] / qubits[0]) ** (1/20) - 1
            print(f"  {method.capitalize()}: {growth_rate:.1%}")


def demonstrate_threat_evolution():
    """Show how quantum threat evolves over time."""
    print("\n" + "=" * 60)
    print("QUANTUM THREAT EVOLUTION")
    print("=" * 60)
    
    model = QuantumDevelopmentModel()
    
    # Generate multiple timelines
    n_samples = 500
    threat_probabilities = {year: {threat: 0 for threat in QuantumThreat} 
                          for year in range(2025, 2046)}
    
    for i in range(n_samples):
        rng = np.random.RandomState(i)
        timeline = model.sample(rng)
        
        for cap in timeline.capabilities:
            year = int(cap.year)
            if year <= 2045:
                threat_probabilities[year][cap.threat_level] += 1
    
    # Normalize to probabilities
    for year in threat_probabilities:
        total = sum(threat_probabilities[year].values())
        if total > 0:
            for threat in threat_probabilities[year]:
                threat_probabilities[year][threat] /= total
    
    # Display threat evolution
    print(f"\nProbability of Threat Levels by Year:")
    print(f"{'Year':<6}", end="")
    for threat in QuantumThreat:
        print(f"{threat.value.capitalize():<12}", end="")
    print()
    print("-" * 66)
    
    for year in [2025, 2030, 2035, 2040, 2045]:
        print(f"{year:<6}", end="")
        for threat in QuantumThreat:
            prob = threat_probabilities[year][threat]
            print(f"{prob:<12.1%}", end="")
        print()
    
    # Calculate cumulative high threat probability
    print(f"\nCumulative Probability of HIGH or CRITICAL Threat:")
    cumulative_high = 0
    for year in [2025, 2030, 2035, 2040, 2045]:
        for y in range(2025, year + 1):
            prob_high = threat_probabilities[y].get(QuantumThreat.HIGH, 0)
            prob_critical = threat_probabilities[y].get(QuantumThreat.CRITICAL, 0)
            cumulative_high = max(cumulative_high, prob_high + prob_critical)
        print(f"  By {year}: {cumulative_high:.1%}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("QUANTUM TIMELINE MODEL DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_single_timeline()
    demonstrate_projection_methods()
    demonstrate_consensus_projection()
    demonstrate_parameter_sensitivity()
    demonstrate_qubit_growth_patterns()
    demonstrate_threat_evolution()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
