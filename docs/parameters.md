# Simulation Parameters Documentation

This document provides detailed explanations for all parameters used in the Solana Quantum Impact Monte Carlo Simulation, including rationale and references for each value.

## Quick Reference: Key Economic Parameters

> **⚠️ Note**: These values require regular updates. Last updated: December 2024

| Parameter | Current Value | Source | Update Frequency |
|-----------|--------------|--------|-----------------|
| **SOL Price** | $235 USD | [CoinGecko](https://www.coingecko.com/en/coins/solana) | Monthly |
| **Total Value Locked** | $8.5B | [DeFiLlama](https://defillama.com/chain/Solana) | Monthly |
| **Daily Volume** | $3.8B | CoinGecko | Monthly |
| **Total Staked SOL** | 380M SOL | [Solana Compass](https://solanacompass.com/) | Quarterly |
| **Validator Count** | 1,017 | [Solana Beach](https://solanabeach.io/validators) | Daily |
| **Total Value at Risk** | $97.8B | Calculated (Staked + TVL) | Monthly |

### Economic Impact Calculation
- **Direct Risk**: (380M SOL × $235) + $8.5B TVL = **$97.8B**
- **With Market Impact (3x)**: $97.8B × 3 = **$293.4B potential loss**
- **Migration Cost (2% of protected value)**: $97.8B × 0.02 = **$1.96B**
- **ROI of Migration**: $293.4B / $1.96B = **~150x**

## Table of Contents

1. [Quantum Computing Parameters](#quantum-computing-parameters)
2. [Network State Parameters](#network-state-parameters)
3. [Economic Impact Parameters](#economic-impact-parameters)
4. [Simulation Control Parameters](#simulation-control-parameters)

---

## Quantum Computing Parameters

### Core Quantum Requirements

#### `logical_qubits_for_ed25519 = 2330`
- **Description**: Number of logical qubits required to break Ed25519 encryption
- **Rationale**: Based on Shor's algorithm requirements for 256-bit elliptic curve cryptography
- **Reference**: [Roetteler et al. (2017)](https://arxiv.org/abs/1706.06752) - "Quantum Resource Estimates for Computing Elliptic Curve Discrete Logarithms"
- **Note**: This assumes optimal implementation of Shor's algorithm for the Ed25519 curve

#### `physical_to_logical_ratio = 1000`
- **Description**: Ratio of physical qubits to logical qubits needed for error correction
- **Rationale**: Current quantum computers require significant overhead for error correction
- **Reference**: [Google Quantum AI (2023)](https://www.nature.com/articles/s41586-023-05782-6) - Surface code error correction demonstrations
- **Note**: This ratio may decrease as error correction improves

#### `gate_speed_hz = 1e6`
- **Description**: Quantum gate operation speed (1 MHz)
- **Rationale**: Conservative estimate based on current superconducting qubit gate speeds
- **Reference**: [IBM Quantum Network (2024)](https://www.ibm.com/quantum/roadmap) specifications

#### `circuit_depth = 1.42e9`
- **Description**: Number of gate operations required for Ed25519 attack
- **Rationale**: Derived from quantum circuit complexity for elliptic curve discrete log
- **Reference**: [Häner et al. (2020)](https://arxiv.org/abs/2001.09580) - Circuit depth analysis

#### `error_correction_distance = 15`
- **Description**: Surface code distance for quantum error correction
- **Rationale**: Balance between error suppression and resource requirements
- **Reference**: [Fowler et al. (2012)](https://arxiv.org/abs/1208.0928) - Surface code thresholds

### Quantum Development Projections

#### `qubit_growth_rate_mean = 1.5`
- **Description**: Annual multiplicative growth rate for logical qubits (50% per year)
- **Rationale**: Historical analysis of quantum computer scaling
- **Reference**: Analysis of IBM, Google, and IonQ roadmaps (2019-2024)
- **Calculation**: Based on growth from ~50 qubits (2019) to ~1000+ qubits (2024)

#### `historical_qubit_growth_rate = 1.993`
- **Description**: Observed historical growth rate (doubling annually)
- **Rationale**: Empirical fit to actual quantum computer development
- **Reference**: [Quantum Computing Report](https://quantumcomputingreport.com/) historical data

#### `historical_qubit_initial = 972`
- **Description**: Estimated logical qubits available in 2025
- **Rationale**: Extrapolation from current capabilities with error correction
- **Reference**: Based on IBM Condor (1121 physical qubits) and error correction overhead

#### `breakthrough_prob_base = 0.05`
- **Description**: Annual probability of major quantum breakthrough
- **Rationale**: Historical rate of major advances in quantum computing
- **Reference**: Analysis of major milestones (quantum supremacy, error correction demos)

#### CRQC Emergence Timeline (breakthrough_years)
```python
{
    2026: 0.01,  # 1% chance
    2027: 0.02,  # 2% chance
    2028: 0.05,  # 5% chance
    2029: 0.08,  # 8% chance
    2030: 0.12,  # 12% chance
    2031: 0.15,  # 15% chance
    2032: 0.18,  # 18% chance
    2033: 0.20,  # 20% chance
    2034: 0.25,  # 25% chance
    2035: 0.30,  # 30% chance
}
```
- **Rationale**: Expert consensus from quantum computing surveys
- **Reference**: [Global Risk Institute (2024)](https://globalriskinstitute.org/publication/quantum-threat-timeline-report-2023/) quantum threat timeline

---

## Network State Parameters

### Solana Network Metrics

#### `validators = 1,017`
- **Description**: Current number of Solana validators
- **Rationale**: Actual Solana network statistics
- **Reference**: [Solana Beach](https://solanabeach.io/validators) (December 2024)
- **Note**: Superminority consists of 19 validators
- **Update Note**: Network has grown significantly with more validators joining

#### `total_stake_sol = 380,000,000`
- **Description**: Total staked SOL (380M)
- **Rationale**: Current staking statistics (~67% of circulating supply)
- **Reference**: [Solana Compass](https://solanacompass.com/) staking metrics
- **Update Note**: Should be verified with current network data

#### `stake_gini_coefficient = 0.82`
- **Description**: Measure of stake concentration (0=perfect equality, 1=perfect inequality)
- **Rationale**: Calculated from actual validator stake distribution
- **Reference**: [Solana Validator Info](https://www.validators.app/cluster-stats)
- **Note**: Slightly improved distribution but still high concentration

#### Geographic Distribution
```python
{
    'north_america': 0.40,  # 40%
    'europe': 0.30,         # 30%
    'asia': 0.20,           # 20%
    'other': 0.10           # 10%
}
```
- **Rationale**: Based on validator geographic analysis
- **Reference**: [Solana Beach Geographic Distribution](https://solanabeach.io/validators)

#### Consensus Thresholds
```python
{
    'halt': 0.333,    # 33.3% - Network halt threshold
    'control': 0.667  # 66.7% - Consensus control threshold
}
```
- **Rationale**: Solana's Tower BFT consensus mechanism requirements (used with PoH/PoS)
- **Reference**: [Solana Documentation](https://docs.solana.com/cluster/stake-delegation-and-rewards)

### Migration Parameters

#### `migration_start_year = 2030`
- **Description**: Expected start of serious quantum-safe migration efforts
- **Rationale**: Allowing 5-year preparation window from 2025
- **Reference**: [NIST Post-Quantum Migration Timeline](https://www.nist.gov/pqc)

#### `migration_end_year = 2040`
- **Description**: Target completion for full migration
- **Rationale**: 10-year migration window is realistic for large-scale crypto migration
- **Reference**: Historical analysis of major cryptographic transitions (SHA-1 to SHA-256)

#### `migration_rate_mean = 0.8`
- **Description**: Mean adoption rate for quantum-safe cryptography (80%)
- **Rationale**: Some validators may be slow or unable to migrate
- **Reference**: Analysis of previous Solana network upgrades

#### `migration_cost_per_sol = 0.001`
- **Description**: Cost in USD to migrate 1 SOL to quantum-safe
- **Rationale**: Estimated transaction and operational costs
- **Reference**: Current Solana transaction fees and estimated development costs

---

## Economic Impact Parameters

### Market Values

#### `sol_price_usd = 235.0`
- **Description**: SOL token price in USD
- **Rationale**: Market price as of December 2024
- **Reference**: [CoinGecko](https://www.coingecko.com/en/coins/solana)
- **Update Note**: SOL has seen significant appreciation in late 2024
- **Volatility Note**: Crypto prices are volatile; update regularly

#### `total_value_locked_usd = 8,500,000,000`
- **Description**: Total Value Locked in Solana DeFi ($8.5B)
- **Rationale**: Current TVL across Solana protocols
- **Reference**: [DeFiLlama](https://defillama.com/chain/Solana)
- **Update Note**: TVL fluctuates with market conditions and protocol activity
- **As of**: December 2024

#### `daily_volume_usd = 3,800,000,000`
- **Description**: Daily trading volume ($3.8B)
- **Rationale**: Average daily DEX + CEX volume
- **Reference**: [CoinGecko](https://www.coingecko.com/en/coins/solana) volume data
- **Update Note**: Volume has increased with market activity

#### DeFi Protocol Distribution
```python
{
    'lending': 3_200_000_000,      # $3.2B - Lending protocols (Marginfi, Kamino, etc.)
    'dex': 3_500_000_000,          # $3.5B - DEXs (Jupiter, Raydium, Orca)
    'liquid_staking': 1_800_000_000 # $1.8B - Liquid staking (Marinade, Jito)
}
```
- **Rationale**: Current distribution of TVL across protocol types
- **Reference**: [DeFiLlama](https://defillama.com/chain/Solana) protocol breakdown
- **Update Note**: DEX volume has grown with Jupiter's dominance

### Impact Multipliers

#### `attack_market_impact_multiplier = 3.0`
- **Description**: Total economic impact = direct loss × multiplier
- **Rationale**: Historical crypto hack impacts show 2-5x multiplier effect
- **Reference**: Analysis of Terra Luna, FTX, and major DeFi hack impacts

#### `confidence_loss_factor = 0.7`
- **Description**: Network value reduction after successful attack (30% permanent loss)
- **Rationale**: Historical recovery patterns from major crypto security breaches
- **Reference**: Post-hack valuations of compromised networks

#### `recovery_time_months = 12`
- **Description**: Expected time to recover from major attack
- **Rationale**: Historical recovery times from major crypto incidents
- **Reference**: Mt. Gox, DAO hack, and other major incident recoveries

---

## Simulation Control Parameters

### Computation Settings

#### `n_iterations = 10,000`
- **Description**: Number of Monte Carlo simulation iterations
- **Rationale**: Balance between statistical significance and computation time
- **Reference**: [Monte Carlo convergence analysis](https://en.wikipedia.org/wiki/Monte_Carlo_method#Convergence)
- **Note**: 10,000 iterations provides ~1% standard error for most metrics

#### `n_cores = os.cpu_count()`
- **Description**: Number of CPU cores for parallel processing
- **Rationale**: Maximize computational efficiency
- **Note**: Uses all available cores by default

#### `random_seed = 42`
- **Description**: Random number generator seed for reproducibility
- **Rationale**: Ensures reproducible results across runs
- **Reference**: Standard practice in scientific computing

#### `confidence_level = 0.95`
- **Description**: Confidence level for statistical intervals
- **Rationale**: Standard 95% confidence level used in risk analysis
- **Reference**: [Value at Risk methodology](https://en.wikipedia.org/wiki/Value_at_risk)

### Time Parameters

#### `start_year = 2025`
- **Description**: Simulation start year
- **Rationale**: Current year, representing present-day conditions

#### `end_year = 2045`
- **Description**: Simulation end year
- **Rationale**: 20-year horizon captures likely CRQC emergence window
- **Reference**: Expert consensus on quantum threat timeline

#### `time_step_days = 30`
- **Description**: Simulation time step (monthly)
- **Rationale**: Balance between temporal resolution and computation
- **Note**: Monthly steps capture important dynamics without excessive computation

---

## Validation and Sensitivity

### Parameter Validation

All parameters undergo validation checks:
- Range validation (e.g., probabilities ∈ [0,1])
- Consistency checks (e.g., start_year < end_year)
- Type validation (integers, floats, etc.)

### Sensitivity Analysis

Key parameters for sensitivity analysis:
1. `logical_qubits_for_ed25519` - Determines CRQC threshold
2. `qubit_growth_rate_mean` - Controls timeline uncertainty
3. `migration_rate_mean` - Affects network vulnerability
4. `stake_gini_coefficient` - Impacts attack feasibility

### Environment Variable Overrides

Parameters can be overridden via environment variables:
```bash
export N_ITERATIONS=50000
export RANDOM_SEED=123
export QUANTUM_LOGICAL_QUBITS_FOR_ED25519=2000
```

---

## References

### Academic Papers
1. Roetteler et al. (2017) - Quantum Resource Estimates for Computing Elliptic Curve Discrete Logarithms
2. Häner et al. (2020) - Improved Quantum Circuits for Elliptic Curve Discrete Logarithms
3. Fowler et al. (2012) - Surface codes: Towards practical large-scale quantum computation

### Industry Reports
1. IBM Quantum Network Roadmap (2024)
2. Google Quantum AI Publications (2023-2024)
3. Global Risk Institute Quantum Threat Timeline (2024)
4. NIST Post-Quantum Cryptography Standards (2024)

### Data Sources
1. Solana Beach - Network statistics
2. DeFiLlama - TVL and protocol metrics
3. CoinGecko - Price and volume data
4. Solana Compass - Staking metrics

---

## Update History

- **September 2024**: Initial parameter documentation
- **December 2024**: Updated market values (SOL price, TVL, validators)
- **Last Review**: December 2024
- **Next Review**: Monthly updates recommended for market values

### Important Notes on Parameter Updates

**Market Data Requirements:**
- SOL price: Update monthly (high volatility)
- TVL: Update monthly (market dependent)
- Validator count: Update quarterly (slower changes)
- Staking metrics: Update quarterly

**Data Sources for Updates:**
1. **Price & Volume**: [CoinGecko](https://www.coingecko.com/en/coins/solana)
2. **TVL & DeFi**: [DeFiLlama](https://defillama.com/chain/Solana)
3. **Validators**: [Solana Beach](https://solanabeach.io/validators)
4. **Staking**: [Solana Compass](https://solanacompass.com/)

**Verification Checklist:**
- [ ] Verify SOL price from multiple sources
- [ ] Cross-check TVL between DeFiLlama and DefiLlama
- [ ] Confirm validator count from Solana Beach
- [ ] Update stake distribution metrics
- [ ] Review quantum timeline assumptions

---

## Contact

For parameter questions or suggestions: [GitHub Issues](https://github.com/mjohnson518/sol-crqc-mc/issues)
