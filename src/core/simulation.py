"""
Core Monte Carlo simulation engine for Solana CRQC analysis.

This module contains the main simulation orchestrator that coordinates
all components and manages the Monte Carlo iterations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from src.config import SimulationParameters
from src.core.random_engine import RandomEngine
from src.core.results_collector import ResultsCollector

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for a single simulation iteration result."""
    
    iteration_id: int
    quantum_timeline: Dict[str, Any]
    network_state: Dict[str, Any]
    attack_results: Dict[str, Any]
    economic_impact: Dict[str, Any]
    first_attack_year: Optional[float] = None
    runtime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'iteration_id': self.iteration_id,
            'quantum_timeline': self.quantum_timeline,
            'network_state': self.network_state,
            'attack_results': self.attack_results,
            'economic_impact': self.economic_impact,
            'first_attack_year': self.first_attack_year,
            'runtime_seconds': self.runtime_seconds
        }


@dataclass
class SimulationState:
    """Tracks the overall state of the simulation."""
    
    total_iterations: int
    completed_iterations: int = 0
    failed_iterations: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_iterations == 0:
            return 0.0
        return (self.completed_iterations / self.total_iterations) * 100
    
    @property
    def runtime(self) -> float:
        """Calculate total runtime in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def iterations_per_second(self) -> float:
        """Calculate iteration rate."""
        if self.runtime == 0:
            return 0.0
        return self.completed_iterations / self.runtime


class MonteCarloSimulation:
    """
    Main Monte Carlo simulation engine.
    
    Orchestrates the simulation by:
    1. Managing random number generation
    2. Running iterations in parallel
    3. Coordinating model components
    4. Collecting and aggregating results
    """
    
    def __init__(
        self,
        config: SimulationParameters,
        models: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration parameters
            models: Optional dictionary of model instances
        """
        self.config = config
        self.random_engine = RandomEngine(seed=config.random_seed)
        self.results_collector = ResultsCollector()
        self.state = SimulationState(total_iterations=config.n_iterations)
        
        # Model instances (will be properly initialized when models are implemented)
        self.models = models or {}
        
        # Validate configuration
        config.validate()
        
        logger.info(f"Initialized simulation with {config.n_iterations} iterations")
    
    def run(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute the complete Monte Carlo simulation.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing results and analysis
        """
        logger.info(f"Starting Monte Carlo simulation with {self.config.n_iterations} iterations")
        logger.info(f"Using {self.config.n_cores} CPU cores for parallel processing")
        
        self.state.start_time = time.time()
        
        try:
            if self.config.n_cores == 1:
                results = self._run_sequential(progress_callback)
            else:
                results = self._run_parallel(progress_callback)
            
            self.state.end_time = time.time()
            
            # Aggregate results
            aggregated = self._aggregate_results(results)
            
            # Save if configured
            if self.config.save_raw_results:
                self._save_results(aggregated)
            
            logger.info(
                f"Simulation complete: {self.state.completed_iterations} successful, "
                f"{self.state.failed_iterations} failed, "
                f"runtime: {self.state.runtime:.2f}s"
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def _run_sequential(self, progress_callback: Optional[Callable]) -> List[SimulationResult]:
        """
        Run simulation iterations sequentially.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of simulation results
        """
        results = []
        
        with tqdm(total=self.config.n_iterations, desc="Simulating") as pbar:
            for i in range(self.config.n_iterations):
                try:
                    # Generate seed for this iteration
                    iteration_seed = self.random_engine.get_iteration_seed(i)
                    
                    # Run single iteration
                    result = self._run_single_iteration(i, iteration_seed)
                    results.append(result)
                    
                    self.state.completed_iterations += 1
                    
                except Exception as e:
                    logger.warning(f"Iteration {i} failed: {e}")
                    self.state.failed_iterations += 1
                
                # Update progress
                pbar.update(1)
                if progress_callback:
                    progress_callback(self.state)
        
        return results
    
    def _run_parallel(self, progress_callback: Optional[Callable]) -> List[SimulationResult]:
        """
        Run simulation iterations in parallel.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of simulation results
        """
        results = []
        
        # Prepare batches for parallel execution
        batch_size = max(1, self.config.n_iterations // (self.config.n_cores * 10))
        batches = self._create_batches(batch_size)
        
        with ProcessPoolExecutor(max_workers=self.config.n_cores) as executor:
            # Submit all batches
            futures = {}
            for batch_start, batch_end in batches:
                future = executor.submit(
                    self._run_batch,
                    batch_start,
                    batch_end
                )
                futures[future] = (batch_start, batch_end)
            
            # Process completed batches
            with tqdm(total=self.config.n_iterations, desc="Simulating") as pbar:
                for future in as_completed(futures):
                    batch_start, batch_end = futures[future]
                    
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        
                        completed = len(batch_results)
                        self.state.completed_iterations += completed
                        self.state.failed_iterations += (batch_end - batch_start) - completed
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                        self.state.failed_iterations += (batch_end - batch_start)
                    
                    # Update progress
                    pbar.update(batch_end - batch_start)
                    if progress_callback:
                        progress_callback(self.state)
        
        return results
    
    def _create_batches(self, batch_size: int) -> List[Tuple[int, int]]:
        """
        Create batches of iteration indices.
        
        Args:
            batch_size: Size of each batch
            
        Returns:
            List of (start, end) tuples for each batch
        """
        batches = []
        for i in range(0, self.config.n_iterations, batch_size):
            batch_end = min(i + batch_size, self.config.n_iterations)
            batches.append((i, batch_end))
        return batches
    
    def _run_batch(self, start_idx: int, end_idx: int) -> List[SimulationResult]:
        """
        Run a batch of simulation iterations.
        
        Args:
            start_idx: Starting iteration index
            end_idx: Ending iteration index (exclusive)
            
        Returns:
            List of simulation results for the batch
        """
        results = []
        
        for i in range(start_idx, end_idx):
            try:
                iteration_seed = self.random_engine.get_iteration_seed(i)
                result = self._run_single_iteration(i, iteration_seed)
                results.append(result)
            except Exception as e:
                logger.warning(f"Iteration {i} failed in batch: {e}")
                # Continue with other iterations in batch
        
        return results
    
    def _run_single_iteration(self, iteration_id: int, seed: int) -> SimulationResult:
        """
        Execute one complete simulation iteration.
        
        Args:
            iteration_id: Unique identifier for this iteration
            seed: Random seed for this iteration
            
        Returns:
            SimulationResult containing all outputs
        """
        start_time = time.time()
        
        # Create RNG for this iteration
        rng = np.random.RandomState(seed)
        
        # Step 1: Sample quantum development timeline
        quantum_timeline = self._sample_quantum_timeline(rng)
        
        # Step 2: Sample network state evolution
        network_state = self._sample_network_evolution(rng, quantum_timeline)
        
        # Step 3: Identify attack opportunities
        attack_opportunities = self._identify_attack_windows(
            quantum_timeline,
            network_state
        )
        
        # Step 4: Simulate attacks
        attack_results = self._simulate_attacks(
            rng,
            attack_opportunities,
            network_state
        )
        
        # Step 5: Calculate economic impact
        economic_impact = self._calculate_economic_impact(
            attack_results,
            network_state
        )
        
        # Determine first successful attack year
        first_attack_year = self._find_first_attack_year(attack_results)
        
        runtime = time.time() - start_time
        
        return SimulationResult(
            iteration_id=iteration_id,
            quantum_timeline=quantum_timeline,
            network_state=network_state,
            attack_results=attack_results,
            economic_impact=economic_impact,
            first_attack_year=first_attack_year,
            runtime_seconds=runtime
        )
    
    def _sample_quantum_timeline(self, rng: np.random.RandomState) -> Dict[str, Any]:
        """
        Sample quantum computing development timeline.
        
        Args:
            rng: Random number generator
            
        Returns:
            Dictionary containing quantum development trajectory
        """
        # Use real quantum model if available
        if 'quantum_timeline' in self.models:
            timeline = self.models['quantum_timeline'].sample(rng)
            return {
                'crqc_year': timeline.crqc_year,
                'qubit_trajectory': [cap.logical_qubits for cap in timeline.capabilities],
                'breakthrough_years': timeline.breakthrough_years,
                'capabilities': timeline.capabilities,
                'projection_method': timeline.projection_method,
                'confidence': timeline.confidence
            }
        
        # Simple placeholder (only if model not provided)
        crqc_year = 2025 + rng.exponential(scale=8)  # Mean ~2033
        return {
            'crqc_year': crqc_year,
            'qubit_trajectory': [],
            'breakthrough_year': int(crqc_year)
        }
    
    def _sample_network_evolution(
        self,
        rng: np.random.RandomState,
        quantum_timeline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sample Solana network state evolution.
        
        Args:
            rng: Random number generator
            quantum_timeline: Quantum development timeline
            
        Returns:
            Dictionary containing network state over time
        """
        # Placeholder implementation - will be replaced with actual model
        if 'network_state' in self.models:
            return self.models['network_state'].sample(
                rng,
                self.config.network,
                quantum_timeline
            )
        
        # Simple placeholder
        return {
            'validators': self.config.network.n_validators,
            'total_stake': self.config.network.total_stake_sol,
            'migration_progress': rng.uniform(0.3, 0.9)
        }
    
    def _identify_attack_windows(
        self,
        quantum_timeline: Dict[str, Any],
        network_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify potential attack windows.
        
        Args:
            quantum_timeline: Quantum development timeline
            network_state: Network state evolution
            
        Returns:
            List of attack opportunity windows
        """
        # Placeholder implementation
        if 'attack_model' in self.models:
            return self.models['attack_model'].identify_windows(
                quantum_timeline,
                network_state
            )
        
        # Simple placeholder
        if quantum_timeline['crqc_year'] < self.config.end_year:
            return [{
                'start_year': quantum_timeline['crqc_year'],
                'end_year': quantum_timeline['crqc_year'] + 1,
                'attack_type': 'validator_compromise'
            }]
        return []
    
    def _simulate_attacks(
        self,
        rng: np.random.RandomState,
        attack_opportunities: List[Dict[str, Any]],
        network_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate quantum attacks on the network.
        
        Args:
            rng: Random number generator
            attack_opportunities: List of attack windows
            network_state: Network state
            
        Returns:
            Dictionary containing attack results
        """
        # Placeholder implementation
        if 'attack_model' in self.models:
            return self.models['attack_model'].simulate(
                rng,
                attack_opportunities,
                network_state,
                self.config
            )
        
        # Simple placeholder
        if attack_opportunities:
            success = rng.random() < 0.7  # 70% success rate
            return {
                'attacks_attempted': len(attack_opportunities),
                'attacks_successful': 1 if success else 0,
                'first_success_year': attack_opportunities[0]['start_year'] if success else None
            }
        return {
            'attacks_attempted': 0,
            'attacks_successful': 0,
            'first_success_year': None
        }
    
    def _calculate_economic_impact(
        self,
        attack_results: Dict[str, Any],
        network_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate economic impact of attacks.
        
        Args:
            attack_results: Results of attack simulation
            network_state: Network state
            
        Returns:
            Dictionary containing economic impact metrics
        """
        # Placeholder implementation
        if 'economic_model' in self.models:
            return self.models['economic_model'].calculate_impact(
                attack_results,
                network_state,
                self.config.economic
            )
        
        # Simple placeholder
        if attack_results['attacks_successful'] > 0:
            direct_loss = self.config.economic.total_value_locked_usd * 0.3
            total_loss = direct_loss * self.config.economic.attack_market_impact_multiplier
        else:
            direct_loss = 0
            total_loss = 0
        
        return {
            'direct_loss_usd': direct_loss,
            'total_loss_usd': total_loss,
            'recovery_time_months': 12 if total_loss > 0 else 0
        }
    
    def _find_first_attack_year(self, attack_results: Dict[str, Any]) -> Optional[float]:
        """
        Find the year of the first successful attack.
        
        Args:
            attack_results: Attack simulation results
            
        Returns:
            Year of first successful attack, or None
        """
        return attack_results.get('first_success_year', None)
    
    def _aggregate_results(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Aggregate simulation results.
        
        Args:
            results: List of individual simulation results
            
        Returns:
            Dictionary containing aggregated results and statistics
        """
        # Use results collector for aggregation
        for result in results:
            self.results_collector.add_result(result)
        
        aggregated = self.results_collector.get_summary()
        
        # Add metadata
        aggregated['metadata'] = {
            'total_iterations': self.config.n_iterations,
            'successful_iterations': self.state.completed_iterations,
            'failed_iterations': self.state.failed_iterations,
            'runtime_seconds': self.state.runtime,
            'iterations_per_second': self.state.iterations_per_second,
            'config': self.config._to_serializable_dict()
        }
        
        # Add raw results if not too large
        if len(results) <= 10000:  # Only include raw results for smaller simulations
            aggregated['raw_results'] = [r.to_dict() for r in results]
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any]) -> Path:
        """
        Save simulation results to file.
        
        Args:
            results: Aggregated results dictionary
            
        Returns:
            Path to saved results file
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.json"
        filepath = self.config.output_dir / "results" / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def validate_models(self) -> bool:
        """
        Validate that all required models are available.
        
        Returns:
            True if all models are valid
        """
        required_models = [
            'quantum_timeline',
            'network_state',
            'attack_model',
            'economic_model'
        ]
        
        missing = [m for m in required_models if m not in self.models]
        
        if missing:
            logger.warning(f"Missing models: {missing}")
            logger.warning("Simulation will run with placeholder implementations")
        
        return len(missing) == 0
