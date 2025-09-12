#!/usr/bin/env python3
"""
Main entry point for running the Solana Quantum Impact Monte Carlo Simulation.

Usage:
    python run_simulation.py                    # Run with default settings
    python run_simulation.py --iterations 5000  # Custom iteration count
    python run_simulation.py --cores 8          # Use 8 CPU cores
    python run_simulation.py --save              # Save results to file
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.simulation import MonteCarloSimulation
from src.config import SimulationParameters
from src.models.quantum_timeline import QuantumDevelopmentModel
from src.models.network_state import NetworkStateModel
from src.models.attack_scenarios import AttackScenariosModel
from src.models.economic_impact import EconomicImpactModel


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solana Quantum Impact Monte Carlo Simulation"
    )
    
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=1000,
        help='Number of Monte Carlo iterations (default: 1000)'
    )
    
    parser.add_argument(
        '--cores', '-c',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: all available)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--save', '-o',
        action='store_true',
        help='Save results to file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/output'),
        help='Output directory for results (default: data/output)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def print_header(args):
    """Print simulation header."""
    print("=" * 70)
    print("SOLANA QUANTUM IMPACT MONTE CARLO SIMULATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Iterations: {args.iterations:,}")
    print(f"  CPU Cores: {args.cores if args.cores else 'All available'}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Save Results: {'Yes' if args.save else 'No'}")
    if args.save:
        print(f"  Output Directory: {args.output_dir}")
    print()


def run_simulation(args):
    """Run the Monte Carlo simulation."""
    
    # Create configuration
    config = SimulationParameters(
        n_iterations=args.iterations,
        n_cores=args.cores if args.cores else None,
        random_seed=args.seed,
        save_raw_results=args.save,
        output_dir=args.output_dir
    )
    
    if args.verbose:
        print("Initializing models...")
    
    # Initialize all models
    models = {
        'quantum_timeline': QuantumDevelopmentModel(config.quantum),
        'network_state': NetworkStateModel(config.network),
        'attack_scenarios': AttackScenariosModel(config.quantum),
        'economic_impact': EconomicImpactModel(config.economic)
    }
    
    if args.verbose:
        print("  ✓ Quantum Timeline Model")
        print("  ✓ Network State Model")
        print("  ✓ Attack Scenarios Model")
        print("  ✓ Economic Impact Model")
        print()
    
    # Create and run simulation
    if not args.quiet:
        print("Running simulation...")
    
    start_time = time.time()
    sim = MonteCarloSimulation(config, models=models)
    results = sim.run()
    runtime = time.time() - start_time
    
    if not args.quiet:
        print(f"✓ Simulation complete in {runtime:.1f} seconds\n")
    
    return results, runtime


def print_results(results, runtime, args):
    """Print simulation results."""
    
    if args.quiet:
        return
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Metadata
    if 'metadata' in results:
        metadata = results['metadata']
        print(f"\nExecution Summary:")
        print(f"  Successful Iterations: {metadata['successful_iterations']:,}")
        print(f"  Failed Iterations: {metadata['failed_iterations']}")
        print(f"  Total Runtime: {runtime:.1f} seconds")
        print(f"  Speed: {metadata['iterations_per_second']:.1f} iterations/second")
    
    # Key metrics
    if 'metrics' in results:
        metrics = results['metrics']
        
        print(f"\n" + "-" * 70)
        print("KEY FINDINGS")
        print("-" * 70)
        
        # CRQC emergence
        if 'first_attack_year' in metrics and metrics['first_attack_year']:
            attack_stats = metrics['first_attack_year']
            print(f"\nCRQC Emergence (First Successful Attack):")
            print(f"  Mean Year: {attack_stats.get('mean', 'N/A'):.1f}" if attack_stats.get('mean') else "  Mean Year: N/A")
            print(f"  Median Year: {attack_stats.get('median', 'N/A'):.1f}" if attack_stats.get('median') else "  Median Year: N/A")
            print(f"  Std Deviation: {attack_stats.get('std', 'N/A'):.1f} years" if attack_stats.get('std') else "  Std Deviation: N/A")
            
            if 'percentile_5' in attack_stats and 'percentile_95' in attack_stats:
                print(f"  90% Confidence Interval: [{attack_stats['percentile_5']:.0f}, {attack_stats['percentile_95']:.0f}]")
        
        # Attack success rate
        if 'attack_success_rate' in metrics:
            print(f"\nAttack Success Rate: {metrics['attack_success_rate']:.1%}")
        
        # Economic impact
        if 'economic_loss_usd' in metrics and metrics['economic_loss_usd']:
            loss_stats = metrics['economic_loss_usd']
            print(f"\nEconomic Impact (USD):")
            print(f"  Mean Loss: ${loss_stats.get('mean', 0)/1e9:.1f}B")
            print(f"  Median Loss: ${loss_stats.get('median', 0)/1e9:.1f}B")
            
            if 'percentile_95' in loss_stats:
                print(f"  95% VaR: ${loss_stats['percentile_95']/1e9:.1f}B")
            if 'percentile_99' in loss_stats:
                print(f"  99% VaR: ${loss_stats['percentile_99']/1e9:.1f}B")
        
        # Recovery time
        if 'recovery_time_months' in metrics and metrics['recovery_time_months']:
            recovery_stats = metrics['recovery_time_months']
            if isinstance(recovery_stats, dict):
                print(f"\nRecovery Time:")
                print(f"  Mean: {recovery_stats.get('mean', 0):.1f} months")
                print(f"  Median: {recovery_stats.get('median', 0):.1f} months")
    
    # Risk summary
    print(f"\n" + "-" * 70)
    print("RISK SUMMARY")
    print("-" * 70)
    
    if 'metrics' in results:
        # Calculate risk levels
        if 'first_attack_year' in metrics and metrics['first_attack_year']:
            mean_year = metrics['first_attack_year'].get('mean', 2040)
            if mean_year < 2030:
                risk_level = "CRITICAL"
                risk_color = "🔴"
            elif mean_year < 2035:
                risk_level = "HIGH"
                risk_color = "🟠"
            elif mean_year < 2040:
                risk_level = "MODERATE"
                risk_color = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
            
            print(f"\n{risk_color} Quantum Risk Level: {risk_level}")
            print(f"   Expected CRQC emergence: ~{mean_year:.0f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        print(f"  1. Begin quantum-safe migration planning immediately")
        print(f"  2. Target <30% vulnerable stake by 2030")
        print(f"  3. Implement continuous quantum threat monitoring")
        print(f"  4. Develop incident response procedures")
    
    print("\n" + "=" * 70)


def save_results(results, args):
    """Save results to file."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_{timestamp}.json"
    filepath = args.output_dir / "results" / filename
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results (excluding raw data for file size)
    save_data = {
        'metadata': results.get('metadata', {}),
        'metrics': results.get('metrics', {}),
        'timestamp': timestamp,
        'parameters': {
            'iterations': args.iterations,
            'cores': args.cores,
            'seed': args.seed
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    if not args.quiet:
        print(f"Results saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.1f} KB")
    
    return filepath


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    if not args.quiet:
        print_header(args)
    
    try:
        # Run simulation
        results, runtime = run_simulation(args)
        
        # Print results
        print_results(results, runtime, args)
        
        # Save if requested
        if args.save:
            filepath = save_results(results, args)
            if args.verbose:
                print(f"\nFull results available at: {filepath}")
        
        # Success
        return 0
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
