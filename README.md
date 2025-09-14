# Solana Quantum Impact Monte Carlo Simulation

## Overview

This Monte Carlo simulation models the probabilistic impact of quantum computing advances on the Solana blockchain network. It evaluates when cryptographically relevant quantum computers (CRQC) might emerge, how they could compromise Solana's security, and what economic impacts would result.

## Key Questions Addressed

1. **When will quantum computers threaten Solana?** - Probabilistic modeling of CRQC emergence (2030-2045)
2. **What is the attack success probability?** - Based on quantum capabilities vs. network migration progress
3. **What are the economic consequences?** - Direct losses, market crashes, and recovery trajectories
4. **How effective is migration to quantum-safe cryptography?** - Impact of proactive vs. reactive migration strategies

## Project Structure

```
sol-qv-mc/
├── src/
│   ├── config.py              # Central configuration system
│   ├── core/
│   │   ├── simulation.py      # Main Monte Carlo engine (with convergence)
│   │   ├── random_engine.py   # Reproducible random generation
│   │   └── results_collector.py # Results aggregation
│   ├── models/
│   │   ├── quantum_timeline.py # Quantum development model
│   │   ├── network_state.py    # Network evolution model
│   │   ├── attack_scenarios.py # Attack simulation model
│   │   └── economic_impact.py  # Economic impact model
│   ├── analysis/
│   │   ├── convergence_analyzer.py # Convergence monitoring
│   │   ├── pdf_generator.py    # PDF report generation
│   │   └── report_generator.py # Report generation
│   ├── distributions/
│   │   └── probability_dists.py # Statistical distributions
│   └── utils/
│       ├── validators.py       # Parameter validation
│       ├── pre_simulation_checks.py  # Pre-run validation
│       └── post_simulation_checks.py # Post-run quality checks
├── visualization/
│   ├── executive_dashboard.py  # Executive summary visualization
│   └── technical_report_plots.py # Detailed technical plots
├── tests/                      # Unit tests for all components
├── examples/                   # Demonstration scripts
├── data/
│   ├── input/                 # Input data files
│   └── output/
│       ├── results/           # Simulation results
│       └── figures/           # Generated visualizations
└── docs/
    ├── parameters.md          # Parameter documentation
    └── convergence_analysis.md # Convergence methodology
```

## Core Models

### 1. Quantum Timeline Model
- Projects CRQC emergence using multiple methods (industry, expert, breakthrough, historical)
- Models qubit growth rates and technological breakthroughs
- Estimates time to break Ed25519 signatures

### 2. Network State Model  
- Simulates validator dynamics and stake distribution
- Models migration to quantum-safe cryptography
- Tracks network resilience over time

### 3. Attack Scenarios Model
- Evaluates feasible attack types based on quantum capabilities
- Calculates success probabilities for different attacks
- Identifies optimal attack windows

### 4. Economic Impact Model
- Quantifies direct losses from successful attacks
- Models market reactions and panic selling
- Simulates recovery trajectories

## Installation

```bash
# Clone the repository
git clone https://github.com/mjohnson518/sol-crqc-mc.git
cd sol-qv-mc

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run a Basic Simulation

```python
from src.core.simulation import MonteCarloSimulation
from src.config import SimulationParameters
from src.models import *

# Configure simulation
config = SimulationParameters(
    n_iterations=1000,
    n_cores=4,
    random_seed=42
)

# Initialize models
models = {
    'quantum_timeline': QuantumDevelopmentModel(config.quantum),
    'network_state': NetworkStateModel(config.network),
    'attack_scenarios': AttackScenariosModel(config.quantum),
    'economic_impact': EconomicImpactModel(config.economic)
}

# Run simulation
sim = MonteCarloSimulation(config, models=models)
results = sim.run()

# Access results
print(f"Mean CRQC emergence: {results['metrics']['first_attack_year']['mean']:.1f}")
print(f"Economic loss (95% VaR): ${results['metrics']['economic_loss_usd']['percentile_95']/1e9:.1f}B")
```

### Run Example Demonstrations

```bash
# Test individual models
python examples/quantum_timeline_demo.py
python examples/network_state_demo.py
python examples/attack_scenarios_demo.py
python examples/economic_impact_demo.py

# Run full simulation demo
python examples/simulation_demo.py
```

## Key Findings (Preliminary)

Based on initial simulations with default parameters:

- **CRQC Emergence**: Most likely between 2033-2038 (mean ~2035)
- **Attack Success Rate**: 30-70% depending on network migration progress
- **Economic Impact**: $20-60B potential losses for successful attacks
- **Migration Effectiveness**: Reducing vulnerable stake below 30% significantly decreases attack feasibility
- **Recovery Time**: 30-180 days depending on attack severity

## Configuration

Key parameters can be adjusted in `src/config.py` or via environment variables:

```python
SimulationParameters(
    n_iterations=10000,        # Number of Monte Carlo iterations
    random_seed=42,           # For reproducibility
    n_cores=8,                # Parallel processing cores
    start_year=2025,          # Simulation start
    end_year=2045,            # Simulation end
)
```

See [docs/parameters.md](docs/parameters.md) for detailed parameter documentation.

## Statistical Robustness

### Convergence Analysis
Our simulation includes comprehensive convergence monitoring to ensure statistical reliability. See [Convergence Documentation](docs/convergence_analysis.md) for details.

- **Automatic Convergence Checking**: Real-time monitoring of metric stability
- **Statistical Significance**: All results include 95% confidence intervals
- **Recommended Iterations**: 10,000 for publication-quality results

### Visualization Suite
Generate executive and technical reports with a single command:

```bash
# Generate executive dashboard
python -m visualization.executive_dashboard --results-dir simulation_results/latest

# Generate technical plots
python -m visualization.technical_report_plots --results-path results.json --plot-type all
```

### Quality Assurance
Every simulation run includes:
- Pre-simulation resource and parameter validation
- Real-time convergence monitoring  
- Post-simulation quality scoring
- Automated statistical significance testing

### Pre-Simulation Validation
```python
from src.utils.pre_simulation_checks import PreSimulationValidator

validator = PreSimulationValidator(config)
is_valid, report = validator.validate_all()
# Checks parameters, resources, runtime estimates
```

### Post-Simulation Quality Scoring
```python
from src.utils.post_simulation_checks import PostSimulationValidator

validator = PostSimulationValidator(results)
grade, report = validator.validate_all()
# Grades: A (publication), B (internal), C (preliminary), D (testing), F (invalid)
```

### Convergence Tracking
```python
from src.analysis.convergence_analyzer import ConvergenceAnalyzer

analyzer = ConvergenceAnalyzer()
# Tracks running statistics during simulation
# Determines optimal iteration count
# Generates convergence report
```

## Output

Simulation results are available in three ways:

1. **Direct Return**: Results dictionary from `sim.run()`
2. **Automatic Saving**: JSON files in `data/output/results/` (if `save_raw_results=True`)
3. **Manual Export**: Save results programmatically to custom locations

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_quantum_timeline.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- [Parameter Documentation](docs/parameters.md) - Detailed explanation of all model parameters
- [API Reference](docs/api.md) - Complete API documentation (coming soon)
- [Methodology](docs/methodology.md) - Mathematical models and assumptions (coming soon)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this simulation in your research, please cite:

```bibtex
@software{solana_quantum_mc_2025,
  title = {Solana Quantum Impact Monte Carlo Simulation},
  author = {Johnson, Marc},
  year = {2025},
  url = {https://github.com/mjohnson518/sol-crqc-mc}
}
```

## Contact

For questions or collaboration: [GitHub Issues](https://github.com/mjohnson518/sol-crqc-mc/issues)

## Acknowledgments

This simulation incorporates research from:
- IBM Quantum Network predictions
- Google Quantum AI roadmaps  
- NIST Post-Quantum Cryptography standardization
- Solana Foundation technical documentation

---

**Disclaimer**: This is a research simulation. Results should not be used as financial advice. Quantum computing timelines and impacts are highly uncertain.