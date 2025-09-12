#!/usr/bin/env python3
"""
Demonstration of the Monte Carlo simulation engine.
"""

import sys
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.simulation import MonteCarloSimulation
from src.config import SimulationParameters, get_test_config
from src.core.random_engine import RandomEngine


def demonstrate_basic_simulation():
    """Run a basic simulation with default settings."""
    print("=" * 60)
    print("BASIC MONTE CARLO SIMULATION")
    print("=" * 60)
    
    # Create configuration for quick test
    config = SimulationParameters(
        n_iterations=1000,
        n_cores=2,
        random_seed=42,
        save_raw_results=False
    )
    
    print(f"\nConfiguration:")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Cores: {config.n_cores}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Time range: {config.start_year}-{config.end_year}")
    
    # Create and run simulation
    print("\nRunning simulation...")
    sim = MonteCarloSimulation(config)
    
    start_time = time.time()
    results = sim.run()
    runtime = time.time() - start_time
    
    print(f"\n✓ Simulation complete in {runtime:.2f} seconds")
    
    # Display results
    print("\nResults Summary:")
    metadata = results['metadata']
    print(f"  Successful iterations: {metadata['successful_iterations']}")
    print(f"  Failed iterations: {metadata['failed_iterations']}")
    print(f"  Iterations/second: {metadata['iterations_per_second']:.1f}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        if metrics.get('first_attack_year'):
            print(f"\nFirst Attack Year Statistics:")
            attack_stats = metrics['first_attack_year']
            print(f"  Mean: {attack_stats['mean']:.1f}")
            print(f"  Median: {attack_stats['median']:.1f}")
            print(f"  Std Dev: {attack_stats['std']:.1f}")
            
        if metrics.get('economic_loss_usd'):
            print(f"\nEconomic Loss Statistics:")
            loss_stats = metrics['economic_loss_usd']
            print(f"  Mean: ${loss_stats['mean']/1e9:.1f}B")
            print(f"  Median: ${loss_stats['median']/1e9:.1f}B")
            if 'var_95' in metrics:
                print(f"  VaR(95%): ${metrics['var_95']/1e9:.1f}B")
    
    return results


def demonstrate_parallel_vs_sequential():
    """Compare parallel and sequential execution."""
    print("\n" + "=" * 60)
    print("PARALLEL VS SEQUENTIAL EXECUTION")
    print("=" * 60)
    
    n_iterations = 50
    
    # Sequential execution
    print(f"\nSequential execution ({n_iterations} iterations):")
    config_seq = SimulationParameters(
        n_iterations=n_iterations,
        n_cores=1,
        random_seed=42,
        save_raw_results=False
    )
    
    sim_seq = MonteCarloSimulation(config_seq)
    start = time.time()
    results_seq = sim_seq.run()
    time_seq = time.time() - start
    print(f"  Time: {time_seq:.2f} seconds")
    print(f"  Rate: {n_iterations/time_seq:.1f} iterations/second")
    
    # Parallel execution
    print(f"\nParallel execution ({n_iterations} iterations, 4 cores):")
    config_par = SimulationParameters(
        n_iterations=n_iterations,
        n_cores=4,
        random_seed=42,
        save_raw_results=False
    )
    
    sim_par = MonteCarloSimulation(config_par)
    start = time.time()
    results_par = sim_par.run()
    time_par = time.time() - start
    print(f"  Time: {time_par:.2f} seconds")
    print(f"  Rate: {n_iterations/time_par:.1f} iterations/second")
    
    # Compare
    speedup = time_seq / time_par
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Verify same results (reproducibility)
    if 'metrics' in results_seq and 'metrics' in results_par:
        seq_success = results_seq['metrics'].get('attack_success_rate', 0)
        par_success = results_par['metrics'].get('attack_success_rate', 0)
        
        if abs(seq_success - par_success) < 0.01:
            print("✓ Results are consistent (same random seed)")
        else:
            print("⚠ Results differ (this shouldn't happen with same seed)")


def demonstrate_reproducibility():
    """Demonstrate reproducibility with fixed seeds."""
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)
    
    # Run simulation twice with same seed
    config1 = SimulationParameters(
        n_iterations=20,
        n_cores=1,
        random_seed=12345,
        save_raw_results=False
    )
    
    config2 = SimulationParameters(
        n_iterations=20,
        n_cores=1,
        random_seed=12345,
        save_raw_results=False
    )
    
    print("\nRunning first simulation (seed=12345)...")
    sim1 = MonteCarloSimulation(config1)
    results1 = sim1.run()
    
    print("Running second simulation (seed=12345)...")
    sim2 = MonteCarloSimulation(config2)
    results2 = sim2.run()
    
    # Compare results
    print("\nComparing results:")
    
    if 'raw_results' in results1 and 'raw_results' in results2:
        match = True
        for i in range(min(5, len(results1['raw_results']))):
            year1 = results1['raw_results'][i].get('first_attack_year')
            year2 = results2['raw_results'][i].get('first_attack_year')
            
            if year1 != year2:
                match = False
                break
        
        if match:
            print("✓ Results are identical (reproducible)")
        else:
            print("✗ Results differ (should not happen)")
    
    # Run with different seed
    config3 = SimulationParameters(
        n_iterations=20,
        n_cores=1,
        random_seed=99999,
        save_raw_results=False
    )
    
    print("\nRunning third simulation (seed=99999)...")
    sim3 = MonteCarloSimulation(config3)
    results3 = sim3.run()
    
    if 'metrics' in results1 and 'metrics' in results3:
        rate1 = results1['metrics'].get('attack_success_rate', 0)
        rate3 = results3['metrics'].get('attack_success_rate', 0)
        
        if abs(rate1 - rate3) > 0.01:
            print("✓ Different seed produces different results (expected)")
        else:
            print("⚠ Different seeds produced very similar results (unlikely)")


def demonstrate_random_engine():
    """Demonstrate the random engine functionality."""
    print("\n" + "=" * 60)
    print("RANDOM ENGINE DEMONSTRATION")
    print("=" * 60)
    
    engine = RandomEngine(seed=42)
    
    print("\nIteration seeds (deterministic):")
    for i in range(5):
        seed = engine.get_iteration_seed(i)
        print(f"  Iteration {i}: {seed}")
    
    print("\nComponent RNGs (independent streams):")
    iteration_seed = 1000
    for component in ['quantum', 'network', 'attack', 'economic']:
        rng = engine.get_component_rng(component, iteration_seed)
        values = rng.random(3)
        print(f"  {component:10s}: {values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}")
    
    print("\nBatch seeds for parallel execution:")
    batch_seeds = engine.create_batch_seeds(batch_size=4, batch_id=0)
    for i, seed in enumerate(batch_seeds):
        print(f"  Worker {i}: {seed}")


def demonstrate_progress_tracking():
    """Demonstrate progress tracking during simulation."""
    print("\n" + "=" * 60)
    print("PROGRESS TRACKING")
    print("=" * 60)
    
    # Define progress callback
    def progress_callback(state):
        if state.completed_iterations % 10 == 0:
            print(f"  Progress: {state.progress:.1f}% "
                  f"({state.completed_iterations}/{state.total_iterations}) "
                  f"Rate: {state.iterations_per_second:.1f} it/s")
    
    config = SimulationParameters(
        n_iterations=50,
        n_cores=2,
        random_seed=42,
        save_raw_results=False
    )
    
    print("\nRunning simulation with progress tracking...")
    sim = MonteCarloSimulation(config)
    results = sim.run(progress_callback=progress_callback)
    
    print(f"\n✓ Simulation complete")
    print(f"  Total runtime: {results['metadata']['runtime_seconds']:.2f} seconds")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Run demonstrations
    results = demonstrate_basic_simulation()
    demonstrate_parallel_vs_sequential()
    demonstrate_reproducibility()
    demonstrate_random_engine()
    # demonstrate_progress_tracking()  # Commented out as callback doesn't work well with tqdm
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    
    # Optionally save results
    save = input("\nSave results to file? (y/n): ").lower().strip()
    if save == 'y':
        filepath = Path("data/output/results/demo_results.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()
