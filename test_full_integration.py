#!/usr/bin/env python3
"""
Test full integration of all models in the Monte Carlo simulation.
Shows where to find simulation outputs.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulation import MonteCarloSimulation
from src.config import SimulationParameters
from src.models.quantum_timeline import QuantumDevelopmentModel
from src.models.network_state import NetworkStateModel
from src.models.attack_scenarios import AttackScenariosModel
from src.models.economic_impact import EconomicImpactModel


def run_full_simulation():
    """Run a complete simulation with all models integrated."""
    
    print("=" * 60)
    print("FULL MONTE CARLO SIMULATION TEST")
    print("=" * 60)
    
    # Create configuration
    config = SimulationParameters(
        n_iterations=10,  # Small number for quick test
        n_cores=1,
        random_seed=42,
        save_raw_results=True,  # Enable saving results to file
        output_dir=Path("data/output")
    )
    
    print(f"\nConfiguration:")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Save raw results: {config.save_raw_results}")
    
    # Create all models
    models = {
        'quantum_timeline': QuantumDevelopmentModel(config.quantum),
        'network_state': NetworkStateModel(config.network),
        'attack_scenarios': AttackScenariosModel(config.quantum),
        'economic_impact': EconomicImpactModel(config.economic)
    }
    
    print(f"\nModels loaded:")
    for model_name in models.keys():
        print(f"  ✓ {model_name}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    sim = MonteCarloSimulation(config, models=models)
    results = sim.run()
    
    # Display results summary
    print(f"\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    if 'metadata' in results:
        metadata = results['metadata']
        print(f"\nMetadata:")
        print(f"  Successful iterations: {metadata['successful_iterations']}")
        print(f"  Failed iterations: {metadata['failed_iterations']}")
        print(f"  Runtime: {metadata['runtime_seconds']:.2f} seconds")
        print(f"  Rate: {metadata['iterations_per_second']:.2f} iterations/second")
    
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\nKey Metrics:")
        
        if 'first_attack_year' in metrics:
            stats = metrics['first_attack_year']
            print(f"  First Attack Year:")
            print(f"    Mean: {stats.get('mean', 'N/A'):.1f}" if stats.get('mean') else "    Mean: N/A")
            print(f"    Median: {stats.get('median', 'N/A'):.1f}" if stats.get('median') else "    Median: N/A")
            print(f"    Std Dev: {stats.get('std', 'N/A'):.1f}" if stats.get('std') else "    Std Dev: N/A")
        
        if 'economic_loss_usd' in metrics:
            stats = metrics['economic_loss_usd']
            print(f"  Economic Loss (USD):")
            print(f"    Mean: ${stats.get('mean', 0)/1e9:.2f}B")
            print(f"    Median: ${stats.get('median', 0)/1e9:.2f}B")
            print(f"    95% VaR: ${stats.get('percentile_95', 0)/1e9:.2f}B")
    
    # Show sample of raw results
    if 'raw_results' in results and len(results['raw_results']) > 0:
        print(f"\nSample Result (Iteration 0):")
        sample = results['raw_results'][0]
        
        if 'quantum_timeline' in sample:
            print(f"  CRQC Year: {sample['quantum_timeline'].get('crqc_year', 'N/A'):.1f}" 
                  if sample['quantum_timeline'].get('crqc_year') else "  CRQC Year: N/A")
        
        if 'attack_results' in sample:
            attack = sample['attack_results']
            print(f"  Attack Success: {attack.get('attacks_successful', 0) > 0}")
            if attack.get('attack_type'):
                print(f"  Attack Type: {attack['attack_type']}")
            if attack.get('stake_compromised'):
                print(f"  Stake Compromised: {attack['stake_compromised']:.1%}")
        
        if 'economic_impact' in sample:
            impact = sample['economic_impact']
            print(f"  Total Loss: ${impact.get('total_loss_usd', 0)/1e9:.2f}B")
            print(f"  Recovery Time: {impact.get('recovery_time_months', 0):.1f} months")
    
    # OUTPUT LOCATIONS
    print(f"\n" + "=" * 60)
    print("OUTPUT LOCATIONS")
    print("=" * 60)
    
    print(f"\n1. RETURNED RESULTS (in memory):")
    print(f"   - Available in 'results' variable")
    print(f"   - Contains: metrics, raw_results, metadata")
    print(f"   - Access: results['metrics'], results['raw_results'], etc.")
    
    print(f"\n2. SAVED FILES (if save_raw_results=True):")
    results_dir = config.output_dir / "results"
    print(f"   - Directory: {results_dir.absolute()}")
    
    # List saved files
    if results_dir.exists():
        json_files = list(results_dir.glob("simulation_results_*.json"))
        if json_files:
            print(f"   - Files found: {len(json_files)}")
            for file in sorted(json_files)[-3:]:  # Show last 3 files
                print(f"     • {file.name} ({file.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"   - No results files found yet")
    else:
        print(f"   - Results directory not created yet")
    
    print(f"\n3. MANUAL SAVE:")
    print(f"   - You can save results manually:")
    print(f"     with open('my_results.json', 'w') as f:")
    print(f"         json.dump(results, f, indent=2, default=str)")
    
    # Optionally save to custom location
    custom_path = Path("simulation_output.json")
    save_custom = input(f"\nSave results to {custom_path}? (y/n): ").lower().strip()
    if save_custom == 'y':
        with open(custom_path, 'w') as f:
            # Only save summary, not raw results (to keep file size manageable)
            summary = {
                'metadata': results.get('metadata', {}),
                'metrics': results.get('metrics', {}),
                'sample_results': results.get('raw_results', [])[:3]  # First 3 iterations
            }
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Results saved to {custom_path}")
        print(f"  File size: {custom_path.stat().st_size / 1024:.1f} KB")
    
    return results


if __name__ == "__main__":
    results = run_full_simulation()
    print(f"\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
