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
from src.analysis.convergence_analyzer import ConvergenceAnalyzer, ConvergenceReport

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
        models: Optional[Dict[str, Any]] = None,
        enable_convergence_tracking: bool = True
    ):
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration parameters
            models: Optional dictionary of model instances
            enable_convergence_tracking: Whether to enable convergence monitoring
        """
        self.config = config
        self.random_engine = RandomEngine(seed=config.random_seed)
        self.results_collector = ResultsCollector()
        self.state = SimulationState(total_iterations=config.n_iterations)
        
        # Model instances (will be properly initialized when models are implemented)
        self.models = models or {}
        
        # Initialize convergence analyzer if enabled
        self.enable_convergence_tracking = enable_convergence_tracking
        self.convergence_analyzer = None
        if enable_convergence_tracking:
            self.convergence_analyzer = ConvergenceAnalyzer(
                convergence_threshold=0.01,
                confidence_level=config.confidence_level,
                min_iterations=100,
                check_interval=max(100, config.n_iterations // 20)  # Check 20 times during simulation
            )
            logger.info("Convergence tracking enabled")
        
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
            
            # Generate convergence report if enabled
            if self.convergence_analyzer:
                convergence_report = self.get_convergence_report()
                if convergence_report:
                    aggregated['convergence_report'] = {
                        'quality_score': convergence_report.quality_score,
                        'overall_convergence': convergence_report.overall_convergence,
                        'recommended_iterations': convergence_report.recommended_iterations,
                        'converged_variables': convergence_report.converged_variables,
                        'warnings': convergence_report.warnings
                    }
                    
                    # Save convergence report
                    if self.config.save_raw_results:
                        report_path = Path(self.config.output_dir) / "convergence_report.json"
                        convergence_report.to_json(report_path)
                
                # Warn if not converged
                if not self.check_convergence():
                    logger.warning(
                        "Simulation ended before convergence. "
                        f"Consider increasing iterations to {convergence_report.recommended_iterations:,}"
                    )
            
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
                    
                    # Track convergence if enabled
                    if self.convergence_analyzer and result:
                        self._track_convergence(i, result)
                        
                        # Log convergence status periodically
                        if (i + 1) % self.convergence_analyzer.check_interval == 0:
                            self._log_convergence_status()
                    
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
                        
                        # Track convergence for batch results
                        if self.convergence_analyzer:
                            for result in batch_results:
                                self._track_convergence(result.iteration_id, result)
                            
                            # Log convergence status periodically
                            if self.state.completed_iterations % self.convergence_analyzer.check_interval == 0:
                                self._log_convergence_status()
                        
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
        # Use real network model if available
        if 'network_state' in self.models:
            evolution = self.models['network_state'].sample(rng, quantum_timeline)
            
            # Get snapshot at CRQC year for summary
            crqc_year = quantum_timeline.get('crqc_year', 2035)
            snapshot = evolution.get_snapshot_at_year(crqc_year)
            
            return {
                'evolution': evolution,
                'snapshots': evolution.snapshots,
                'validators': snapshot.n_validators,
                'total_stake': snapshot.total_stake,
                'migration_progress': snapshot.migration_progress,
                'vulnerable_stake_percentage': snapshot.vulnerable_stake_percentage,
                'migration_timeline': evolution.get_migration_timeline(),
                'network_resilience': snapshot.network_resilience
            }
        
        # Simple placeholder (only if model not provided)
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
        # Use real attack model if available
        if 'attack_scenarios' in self.models and 'capabilities' in quantum_timeline:
            windows = []
            evolution = network_state.get('evolution')
            
            if evolution:
                # Check each year for attack opportunities
                for capability in quantum_timeline['capabilities']:
                    year = capability.year
                    if year > self.config.end_year:
                        break
                    
                    snapshot = evolution.get_snapshot_at_year(year)
                    attack_plan = self.models['attack_scenarios'].sample(
                        np.random.RandomState(int(year)),  # Deterministic for reproducibility
                        capability,
                        snapshot
                    )
                    
                    for window in attack_plan.windows:
                        windows.append({
                            'start_year': window.start_year,
                            'end_year': window.end_year,
                            'peak_year': window.peak_year,
                            'opportunity_score': window.opportunity_score,
                            'attack_plan': attack_plan
                        })
            
            return windows
        
        # Simple placeholder (only if model not provided)
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
        # Use real attack scenarios if available
        if attack_opportunities and 'attack_plan' in attack_opportunities[0]:
            # Get the best attack opportunity
            best_window = max(attack_opportunities, key=lambda w: w.get('opportunity_score', 0))
            attack_plan = best_window['attack_plan']
            
            if attack_plan.scenarios:
                # Select best scenario
                best_scenario = max(attack_plan.scenarios, key=lambda s: s.impact_score)
                
                # Simulate attack execution
                success = rng.random() < best_scenario.success_probability
                
                return {
                    'attacks_attempted': len(attack_opportunities),
                    'attacks_successful': 1 if success else 0,
                    'first_success_year': best_scenario.year if success else None,
                    'attack_type': best_scenario.attack_type.value,
                    'attack_severity': best_scenario.severity.value,
                    'validators_compromised': best_scenario.validators_compromised,
                    'stake_compromised': best_scenario.stake_compromised,
                    'accounts_at_risk': best_scenario.accounts_at_risk,
                    'impact_score': best_scenario.impact_score
                }
            
            return {
                'attacks_attempted': 0,
                'attacks_successful': 0,
                'first_success_year': None,
                'reason': 'No feasible attack scenarios'
            }
        
        # Simple placeholder (only if model not provided)
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
        # Use real economic model if available
        if 'economic_impact' in self.models and 'attack_type' in attack_results:
            # Create an attack scenario from results
            from src.models.attack_scenarios import AttackScenario, AttackType, AttackVector, AttackSeverity
            
            # Map attack type string to enum
            attack_type_map = {
                'key_compromise': AttackType.KEY_COMPROMISE,
                'double_spend': AttackType.DOUBLE_SPEND,
                'consensus_halt': AttackType.CONSENSUS_HALT,
                'consensus_control': AttackType.CONSENSUS_CONTROL,
                'targeted_theft': AttackType.TARGETED_THEFT,
                'systemic_failure': AttackType.SYSTEMIC_FAILURE
            }
            
            attack_severity_map = {
                'low': AttackSeverity.LOW,
                'medium': AttackSeverity.MEDIUM,
                'high': AttackSeverity.HIGH,
                'critical': AttackSeverity.CRITICAL
            }
            
            # Create attack scenario from results
            attack_scenario = AttackScenario(
                attack_type=attack_type_map.get(attack_results.get('attack_type'), AttackType.KEY_COMPROMISE),
                vector=AttackVector.VALIDATOR_KEYS,
                year=attack_results.get('first_success_year', 2035),
                success_probability=0.7,
                severity=attack_severity_map.get(attack_results.get('attack_severity'), AttackSeverity.MEDIUM),
                validators_compromised=attack_results.get('validators_compromised', 1),
                stake_compromised=attack_results.get('stake_compromised', 0.01),
                accounts_at_risk=attack_results.get('accounts_at_risk', 100000),
                time_to_execute=24.0,
                detection_probability=0.5,
                mitigation_possible=True
            )
            
            # Get network snapshot
            evolution = network_state.get('evolution')
            if evolution:
                network_snapshot = evolution.get_snapshot_at_year(attack_scenario.year)
            else:
                # Create minimal snapshot
                from src.models.network_state import NetworkSnapshot, MigrationStatus
                network_snapshot = NetworkSnapshot(
                    year=attack_scenario.year,
                    n_validators=network_state.get('validators', 1032),
                    total_stake=network_state.get('total_stake', 400000000),
                    validators=[],
                    geographic_distribution={},
                    migration_status=MigrationStatus.IN_PROGRESS,
                    migration_progress=network_state.get('migration_progress', 0.3),
                    superminority_count=30,
                    gini_coefficient=0.8,
                    network_resilience=network_state.get('network_resilience', 0.5),
                    compromised_validators=network_state.get('compromised_validators', 0),
                    attack_occurred=attack_scenario.attack_success if attack_scenario else False
                )
            
            # Calculate economic impact
            rng = np.random.RandomState(int(attack_scenario.year))
            economic_loss = self.models['economic_impact'].calculate_impact(
                rng,
                attack_scenario,
                network_snapshot
            )
            
            return {
                'direct_loss_usd': economic_loss.immediate_loss_usd,
                'total_loss_usd': economic_loss.total_loss_usd,
                'recovery_time_months': economic_loss.recovery_timeline_days / 30,
                'economic_loss': economic_loss,
                'market_reaction': economic_loss.market_reaction,
                'loss_components': economic_loss.components
            }
        
        # Simple placeholder (only if model not provided)
        if attack_results.get('attacks_successful', 0) > 0:
            direct_loss = self.config.economic.total_value_locked_usd * 0.3
            total_loss = direct_loss * 1.5  # Add market impact
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
        
        # Extract only essential metrics for analysis (not full raw results)
        # This dramatically reduces file size while preserving necessary data
        if len(results) <= 100:  # Only include full raw results for very small simulations
            aggregated['raw_results'] = [r.to_dict() for r in results]
        else:
            # For larger simulations, only save essential metrics
            aggregated['essential_metrics'] = {
                'first_attack_years': [r.first_attack_year for r in results if r.first_attack_year],
                'economic_losses': [r.economic_impact.get('total_loss_usd', 0) for r in results],
                'attack_success_rates': [r.attack_results.get('success_rate', 0) for r in results],
                'quantum_capabilities': [r.quantum_timeline.get('2030', {}).get('logical_qubits', 0) for r in results],
                'network_compromised_pct': [r.network_state.get('compromised_percentage', 0) for r in results],
                'validators_migrated': [r.network_state.get('migrated_validators', 0) for r in results],
                'recovery_times': [r.economic_impact.get('recovery_time_months', 0) for r in results],
                'iteration_ids': [r.iteration_id for r in results]
            }
        
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
    
    def _track_convergence(self, iteration: int, result: SimulationResult) -> None:
        """
        Track convergence metrics for the current iteration.
        
        Args:
            iteration: Current iteration number
            result: Simulation result for this iteration
        """
        if not self.convergence_analyzer:
            return
        
        # Extract key metrics to track
        metrics = {}
        
        # CRQC emergence year
        if 'crqc_year' in result.quantum_timeline:
            metrics['crqc_year'] = result.quantum_timeline['crqc_year']
        
        # First attack year
        if result.first_attack_year:
            metrics['first_attack_year'] = result.first_attack_year
        
        # Economic impact metrics
        if 'total_loss_usd' in result.economic_impact:
            metrics['total_economic_loss'] = result.economic_impact['total_loss_usd']
        
        if 'direct_loss_usd' in result.economic_impact:
            metrics['direct_economic_loss'] = result.economic_impact['direct_loss_usd']
        
        # Attack probability
        if 'successful_attacks' in result.attack_results:
            total_attacks = result.attack_results.get('total_attempts', 1)
            success_rate = result.attack_results['successful_attacks'] / max(1, total_attacks)
            metrics['attack_success_rate'] = success_rate
        
        # Network vulnerability
        if 'vulnerability_score' in result.network_state:
            metrics['network_vulnerability'] = result.network_state['vulnerability_score']
        
        # Track metrics
        if metrics:
            self.convergence_analyzer.track(iteration, metrics)
    
    def _log_convergence_status(self) -> None:
        """Log current convergence status."""
        if not self.convergence_analyzer:
            return
        
        # Check key metrics
        key_metrics = ['crqc_year', 'total_economic_loss', 'attack_success_rate']
        converged_count = 0
        
        for metric in key_metrics:
            if metric in self.convergence_analyzer.data:
                metrics = self.convergence_analyzer.get_current_metrics(metric)
                if metrics and metrics.is_converged:
                    converged_count += 1
                    logger.info(
                        f"{metric} converged: mean={metrics.running_mean:.4f}, "
                        f"SE={metrics.standard_error:.4f}, CV={metrics.coefficient_of_variation:.4f}"
                    )
                elif metrics:
                    logger.debug(
                        f"{metric} not converged: CV={metrics.coefficient_of_variation:.4f}"
                    )
        
        if converged_count == len(key_metrics):
            logger.info("All key metrics have converged!")
        else:
            logger.info(f"{converged_count}/{len(key_metrics)} key metrics converged")
    
    def get_convergence_report(self, save_path: Optional[Path] = None) -> Optional[ConvergenceReport]:
        """
        Generate convergence report for the simulation.
        
        Args:
            save_path: Optional path to save JSON report
            
        Returns:
            ConvergenceReport or None if convergence tracking disabled
        """
        if not self.convergence_analyzer:
            logger.warning("Convergence tracking not enabled")
            return None
        
        # Generate report
        report = self.convergence_analyzer.generate_report(save_path)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("CONVERGENCE REPORT")
        logger.info("=" * 60)
        logger.info(f"Quality Score: {report.quality_score}")
        logger.info(f"Overall Convergence: {'YES' if report.overall_convergence else 'NO'}")
        logger.info(f"Converged Variables: {len(report.converged_variables)}")
        logger.info(f"Non-Converged Variables: {len(report.non_converged_variables)}")
        logger.info(f"Recommended Iterations: {report.recommended_iterations:,}")
        
        if report.warnings:
            logger.warning("Convergence Warnings:")
            for warning in report.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("=" * 60)
        
        return report
    
    def check_convergence(self) -> bool:
        """
        Check if simulation has converged.
        
        Returns:
            True if all tracked metrics have converged
        """
        if not self.convergence_analyzer:
            return False
        
        return self.convergence_analyzer.is_converged()
