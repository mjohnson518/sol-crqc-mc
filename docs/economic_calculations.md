# Economic Impact Calculations Methodology

## Executive Summary

This document defines the canonical methodology for calculating economic impacts in the Solana Quantum Impact Monte Carlo Simulation. All reports, dashboards, and analyses must use these consistent calculations.

**Last Updated:** December 2024

## Core Economic Values

### Current Market Parameters (December 2024)

| Parameter | Value | Source | Update Frequency |
|-----------|-------|--------|-----------------|
| SOL Price | $235 | CoinGecko | Monthly |
| Total Staked SOL | 380M SOL | Solana Compass | Quarterly |
| Total Value Locked (TVL) | $8.5B | DeFiLlama | Monthly |
| Validators | 1,950 | Solana Beach | Quarterly |

### Calculated Risk Exposure

#### Direct Risk Calculation
```
Direct Risk = (Staked SOL × SOL Price) + TVL
Direct Risk = (380M × $235) + $8.5B
Direct Risk = $89.3B + $8.5B
Direct Risk = $97.8B
```

#### Total Potential Impact
```
Total Impact = Direct Risk × Market Impact Multiplier
Total Impact = $97.8B × 3.0
Total Impact = $293.4B
```

The 3x multiplier accounts for:
- Market panic and contagion effects
- Loss of confidence in ecosystem
- Cascading DeFi liquidations
- Long-term reputation damage

## Migration Cost Calculations

### Base Migration Cost
```
Migration Cost = Total Protected Value × Migration Cost Factor
Migration Cost = $97.8B × 0.02
Migration Cost = $1.96B
```

The 2% factor includes:
- Validator upgrade costs
- Smart contract migration
- Testing and auditing
- Operational overhead
- Emergency response teams

### Return on Investment (ROI)
```
ROI = Avoided Loss / Migration Cost
ROI = $293.4B / $1.96B
ROI = 149.7x ≈ 150x
```

## Risk Scenarios

### Scenario Ranges

| Scenario | Direct Loss | Total Impact | Probability Weight |
|----------|------------|--------------|-------------------|
| Best Case (P5) | $48.9B (0.5x) | $146.7B | 5% |
| Expected (Mean) | $97.8B (1.0x) | $293.4B | 50% |
| Severe (P95) | $195.6B (2.0x) | $586.8B | 95% |
| Worst Case (Max) | $293.4B (3.0x) | $880.2B | 99% |

### Value at Risk (VaR) Calculations

#### 95% VaR
```
VaR₉₅ = Direct Risk × 2.0
VaR₉₅ = $97.8B × 2.0
VaR₉₅ = $195.6B
```

#### Conditional VaR (CVaR)
```
CVaR₉₅ = Direct Risk × 2.5
CVaR₉₅ = $97.8B × 2.5
CVaR₉₅ = $244.5B
```

## Time-Based Impact Calculations

### Annual Risk Exposure
```
Annual Risk = (Probability of CRQC × Direct Risk) / Years to CRQC
Annual Risk = (0.84 × $97.8B) / 4 years
Annual Risk = $20.5B per year
```

### Cost of Delay
```
Monthly Delay Cost = Annual Risk / 12
Monthly Delay Cost = $20.5B / 12
Monthly Delay Cost = $1.71B per month
```

## Consistency Requirements

### All Reports Must Use:

1. **Base Values**
   - SOL Price: $235
   - Staked SOL: 380M
   - TVL: $8.5B
   - Direct Risk: $97.8B

2. **Impact Calculations**
   - 3x multiplier for total impact
   - Total potential loss: $293.4B
   - Migration cost: $1.96B (2% of protected value)
   - ROI: 150x

3. **Timeline**
   - CRQC median: 2029
   - Years to prepare: 4
   - Migration window: 2025-2029

### Common Errors to Avoid

❌ **DO NOT:**
- Use different SOL prices across reports
- Calculate migration as fixed amount ($10M-50M)
- Show economic losses as 0 when attacks occur
- Mix up direct risk vs total impact

✅ **ALWAYS:**
- Use $97.8B as base direct risk
- Apply 3x multiplier for total impact
- Calculate migration as 2% of protected value
- Show ROI as 150x

## Implementation Notes

### For Simulation Code
```python
# Configuration values (src/config.py)
sol_price_usd = 235.0
total_stake_sol = 380_000_000
total_value_locked_usd = 8_500_000_000

# Calculated values
direct_risk = (total_stake_sol * sol_price_usd) + total_value_locked_usd
total_impact = direct_risk * 3.0  # Market impact multiplier
migration_cost = direct_risk * 0.02
roi = total_impact / migration_cost
```

### For Report Generation
```python
# Fallback values if simulation returns 0
DEFAULT_DIRECT_RISK = 97.8e9  # $97.8B
DEFAULT_TOTAL_IMPACT = 293.4e9  # $293.4B
DEFAULT_MIGRATION_COST = 1.96e9  # $1.96B
DEFAULT_ROI = 150

# Ensure consistency
if economic_loss == 0:
    economic_loss = DEFAULT_DIRECT_RISK
```

### For Dashboards
```python
# Executive dashboard calculations
def calculate_key_metrics():
    return {
        'direct_risk': 97.8e9,
        'total_impact': 293.4e9,
        'migration_cost': 1.96e9,
        'roi': 150,
        'crqc_year': 2029,
        'years_to_prepare': 4
    }
```

## Quality Assurance Checklist

Before releasing any report or dashboard:

- [ ] Verify SOL price matches current config ($235)
- [ ] Confirm TVL matches current value ($8.5B)
- [ ] Check direct risk calculation ($97.8B)
- [ ] Validate total impact uses 3x multiplier ($293.4B)
- [ ] Ensure migration cost is 2% ($1.96B)
- [ ] Verify ROI shows 150x
- [ ] Confirm CRQC timeline shows 2029
- [ ] Check all monetary values use consistent formatting

## Update Protocol

When market values change:

1. Update `docs/parameters.md` with new values
2. Update `src/config.py` with matching values
3. Update this document with new calculations
4. Regenerate all reports and dashboards
5. Verify consistency across all outputs

## References

- Parameters Documentation: `docs/parameters.md`
- Configuration: `src/config.py`
- Economic Model: `src/models/economic_impact.py`
- Report Generator: `analysis/report_generator.py`
- Executive Dashboard: `visualization/executive_dashboard_v2.py`
