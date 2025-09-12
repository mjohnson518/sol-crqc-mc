"""
Random number generation engine for Monte Carlo simulation.

Provides reproducible random number generation with proper seeding
for parallel execution.
"""

import numpy as np
from typing import Optional, Tuple, List
import hashlib
import logging

logger = logging.getLogger(__name__)


class RandomEngine:
    """
    Manages random number generation for the simulation.
    
    Features:
    - Reproducible results with seed control
    - Independent seeds for parallel iterations
    - Multiple RNG streams for different components
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random engine.
        
        Args:
            seed: Master seed for reproducibility. If None, uses random seed.
        """
        if seed is None:
            # Use truly random seed
            seed = np.random.randint(0, 2**31 - 1)
            logger.info(f"Using random seed: {seed}")
        else:
            logger.info(f"Using fixed seed: {seed}")
        
        self.master_seed = seed
        self.master_rng = np.random.RandomState(seed)
        
        # Pre-generate seeds for different components
        self._generate_component_seeds()
    
    def _generate_component_seeds(self):
        """Generate seeds for different simulation components."""
        # Use master RNG to generate component seeds
        self.component_seeds = {
            'quantum': self.master_rng.randint(0, 2**31 - 1),
            'network': self.master_rng.randint(0, 2**31 - 1),
            'attack': self.master_rng.randint(0, 2**31 - 1),
            'economic': self.master_rng.randint(0, 2**31 - 1),
        }
    
    def get_iteration_seed(self, iteration_id: int) -> int:
        """
        Get a unique seed for a specific iteration.
        
        Uses hashing to ensure independence between iterations
        even in parallel execution.
        
        Args:
            iteration_id: Unique identifier for the iteration
            
        Returns:
            Seed value for this iteration
        """
        # Create unique string for this iteration
        unique_str = f"{self.master_seed}_{iteration_id}"
        
        # Hash to get reproducible seed
        hash_obj = hashlib.sha256(unique_str.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert first 4 bytes to integer seed
        seed = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**31 - 1)
        
        return seed
    
    def get_component_rng(self, component: str, iteration_seed: int) -> np.random.RandomState:
        """
        Get RNG for a specific component within an iteration.
        
        Args:
            component: Component name ('quantum', 'network', 'attack', 'economic')
            iteration_seed: Seed for the current iteration
            
        Returns:
            Configured RandomState for this component
        """
        if component not in self.component_seeds:
            raise ValueError(f"Unknown component: {component}")
        
        # Combine iteration and component seeds
        combined_seed = (iteration_seed + self.component_seeds[component]) % (2**31 - 1)
        
        return np.random.RandomState(combined_seed)
    
    def create_batch_seeds(self, batch_size: int, batch_id: int) -> List[int]:
        """
        Create seeds for a batch of iterations.
        
        Args:
            batch_size: Number of iterations in the batch
            batch_id: Unique identifier for the batch
            
        Returns:
            List of seeds for each iteration in the batch
        """
        seeds = []
        start_idx = batch_id * batch_size
        
        for i in range(batch_size):
            iteration_id = start_idx + i
            seeds.append(self.get_iteration_seed(iteration_id))
        
        return seeds
    
    def split_seed(self, seed: int, n_streams: int) -> List[int]:
        """
        Split a seed into multiple independent streams.
        
        Useful for creating independent RNGs within a single iteration.
        
        Args:
            seed: Base seed to split
            n_streams: Number of independent streams needed
            
        Returns:
            List of independent seeds
        """
        rng = np.random.RandomState(seed)
        return [rng.randint(0, 2**31 - 1) for _ in range(n_streams)]
    
    @staticmethod
    def generate_correlation_matrix(
        n_vars: int,
        correlation: float,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a correlation matrix for correlated random variables.
        
        Args:
            n_vars: Number of variables
            correlation: Base correlation coefficient
            seed: Random seed
            
        Returns:
            Correlation matrix
        """
        rng = np.random.RandomState(seed)
        
        # Start with identity matrix
        corr_matrix = np.eye(n_vars)
        
        # Add correlations
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Add some randomness to correlation
                corr_value = correlation * (0.8 + 0.4 * rng.random())
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.01)  # Ensure positive
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Normalize to correlation matrix
        D = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
        corr_matrix = D @ corr_matrix @ D
        
        return corr_matrix
    
    @staticmethod
    def generate_correlated_samples(
        mean: np.ndarray,
        cov: np.ndarray,
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate correlated random samples.
        
        Args:
            mean: Mean vector
            cov: Covariance matrix
            n_samples: Number of samples to generate
            seed: Random seed
            
        Returns:
            Array of correlated samples (n_samples x n_vars)
        """
        rng = np.random.RandomState(seed)
        return rng.multivariate_normal(mean, cov, size=n_samples)


class RandomStreamManager:
    """
    Manages multiple random streams for complex simulations.
    
    Useful when different parts of the simulation need independent
    but reproducible random streams.
    """
    
    def __init__(self, base_seed: int):
        """
        Initialize stream manager.
        
        Args:
            base_seed: Base seed for all streams
        """
        self.base_seed = base_seed
        self.streams = {}
        self.stream_counter = 0
    
    def create_stream(self, name: str) -> np.random.RandomState:
        """
        Create a new named random stream.
        
        Args:
            name: Name for the stream
            
        Returns:
            New RandomState instance
        """
        if name in self.streams:
            return self.streams[name]
        
        # Generate unique seed for this stream
        stream_seed = (self.base_seed + self.stream_counter * 1000) % (2**31 - 1)
        self.stream_counter += 1
        
        # Create and store stream
        stream = np.random.RandomState(stream_seed)
        self.streams[name] = stream
        
        return stream
    
    def get_stream(self, name: str) -> np.random.RandomState:
        """
        Get existing stream by name.
        
        Args:
            name: Stream name
            
        Returns:
            RandomState instance
            
        Raises:
            KeyError: If stream doesn't exist
        """
        if name not in self.streams:
            raise KeyError(f"Stream '{name}' not found. Create it first.")
        return self.streams[name]
    
    def reset_stream(self, name: str):
        """
        Reset a stream to its initial state.
        
        Args:
            name: Stream name
        """
        if name in self.streams:
            # Recreate with same seed
            original_seed = (self.base_seed + 
                           list(self.streams.keys()).index(name) * 1000) % (2**31 - 1)
            self.streams[name] = np.random.RandomState(original_seed)
    
    def reset_all(self):
        """Reset all streams to their initial states."""
        for name in list(self.streams.keys()):
            self.reset_stream(name)


def test_reproducibility():
    """Test that random engine produces reproducible results."""
    # Create two engines with same seed
    engine1 = RandomEngine(seed=42)
    engine2 = RandomEngine(seed=42)
    
    # Generate iteration seeds
    seeds1 = [engine1.get_iteration_seed(i) for i in range(10)]
    seeds2 = [engine2.get_iteration_seed(i) for i in range(10)]
    
    assert seeds1 == seeds2, "Seeds should be identical with same master seed"
    
    # Test component RNGs
    rng1 = engine1.get_component_rng('quantum', 100)
    rng2 = engine2.get_component_rng('quantum', 100)
    
    vals1 = rng1.random(5)
    vals2 = rng2.random(5)
    
    assert np.allclose(vals1, vals2), "Component RNGs should produce same values"
    
    print("✓ Reproducibility test passed")


if __name__ == "__main__":
    # Run tests
    test_reproducibility()
    
    # Demonstrate usage
    engine = RandomEngine(seed=123)
    
    print("\nIteration seeds:")
    for i in range(5):
        seed = engine.get_iteration_seed(i)
        print(f"  Iteration {i}: {seed}")
    
    print("\nComponent RNGs:")
    for component in ['quantum', 'network', 'attack', 'economic']:
        rng = engine.get_component_rng(component, 1000)
        value = rng.random()
        print(f"  {component}: {value:.6f}")
