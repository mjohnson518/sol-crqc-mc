"""
Dynamic migration cost calculator based on network parameters.
"""
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MigrationCostBreakdown:
    """Detailed breakdown of migration costs."""
    hardware_acceleration: float
    development_effort: float
    security_auditing: float
    validator_coordination: float
    contingency_reserve: float
    total_cost: float
    
    # Economic metrics
    protected_value: float
    avoided_losses: float
    benefit_cost_ratio: float
    roi_percentage: float
    npv: float
    irr: float
    payback_years: float
    break_even_probability: float
    
    # Component details
    cost_per_validator_hardware: float
    validators_requiring_upgrade: int
    engineering_team_size: int
    audit_firms_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'costs': {
                'hardware_acceleration': self.hardware_acceleration,
                'development_effort': self.development_effort,
                'security_auditing': self.security_auditing,
                'validator_coordination': self.validator_coordination,
                'contingency_reserve': self.contingency_reserve,
                'total': self.total_cost
            },
            'economics': {
                'protected_value': self.protected_value,
                'avoided_losses': self.avoided_losses,
                'benefit_cost_ratio': self.benefit_cost_ratio,
                'roi_percentage': self.roi_percentage,
                'npv': self.npv,
                'irr': self.irr,
                'payback_years': self.payback_years,
                'break_even_probability': self.break_even_probability
            },
            'details': {
                'cost_per_validator_hardware': self.cost_per_validator_hardware,
                'validators_requiring_upgrade': self.validators_requiring_upgrade,
                'engineering_team_size': self.engineering_team_size,
                'audit_firms_count': self.audit_firms_count
            }
        }


def calculate_migration_cost(
    n_validators: int,
    sol_price_usd: float,
    total_stake_sol: float,
    tvl_usd: float,
    migration_progress: float = 0.0,
    urgency_factor: float = 1.0,
    discount_rate: float = 0.15
) -> MigrationCostBreakdown:
    """
    Calculate detailed migration cost based on network parameters.
    
    Args:
        n_validators: Number of validators in the network
        sol_price_usd: Current SOL price in USD
        total_stake_sol: Total staked SOL
        tvl_usd: Total value locked in USD
        migration_progress: Current migration progress (0-1)
        urgency_factor: Urgency multiplier (1.0 = normal, >1 = urgent)
        discount_rate: Discount rate for NPV calculation
    
    Returns:
        Detailed cost breakdown and economic metrics
    """
    
    # Calculate protected value
    staked_value_usd = total_stake_sol * sol_price_usd
    protected_value = staked_value_usd + tvl_usd
    
    # 1. Hardware Acceleration Cost
    # Base cost per validator for GPU/FPGA
    BASE_HARDWARE_PER_VALIDATOR = 25_000  # $25k per validator for acceleration hardware
    
    # Not all validators need hardware - only the ones not yet migrated
    validators_needing_hardware = int(n_validators * (1 - migration_progress))
    
    # Economies of scale - bulk purchases reduce per-unit cost
    scale_discount = 1.0
    if validators_needing_hardware > 500:
        scale_discount = 0.85  # 15% discount for bulk
    elif validators_needing_hardware > 200:
        scale_discount = 0.92  # 8% discount
    
    cost_per_validator_hardware = BASE_HARDWARE_PER_VALIDATOR * scale_discount
    hardware_cost = validators_needing_hardware * cost_per_validator_hardware
    
    # 2. Development Effort
    # Engineering team sizing based on network scale
    if n_validators > 1500:
        engineering_team_size = 25
    elif n_validators > 1000:
        engineering_team_size = 20
    else:
        engineering_team_size = 15
    
    # Cost per engineer per year
    ENGINEER_ANNUAL_COST = 250_000
    DEVELOPMENT_DURATION_YEARS = 2.0
    
    # Reduce development cost if migration is already advanced
    dev_completion_factor = 1.0 if migration_progress < 0.3 else 0.7 if migration_progress < 0.7 else 0.4
    
    development_cost = (engineering_team_size * ENGINEER_ANNUAL_COST * 
                       DEVELOPMENT_DURATION_YEARS * dev_completion_factor)
    
    # 3. Security Auditing
    # Number of audit firms based on protected value
    if protected_value > 100e9:  # >$100B
        audit_firms_count = 4
        audit_cost_per_firm = 1.5e6
    elif protected_value > 50e9:  # >$50B
        audit_firms_count = 3
        audit_cost_per_firm = 1.3e6
    else:
        audit_firms_count = 2
        audit_cost_per_firm = 1.0e6
    
    # Audit costs don't reduce much with progress (always need full audit)
    audit_completion_factor = 1.0 if migration_progress < 0.5 else 0.8
    
    security_audit_cost = audit_firms_count * audit_cost_per_firm * audit_completion_factor
    
    # 4. Validator Coordination
    # Incentive program scales with number of validators
    BASE_INCENTIVE_PER_VALIDATOR = 5_000  # $5k incentive per validator
    
    # Support and documentation costs
    SUPPORT_BASE_COST = 2_000_000  # $2M base for support infrastructure
    
    validator_incentives = validators_needing_hardware * BASE_INCENTIVE_PER_VALIDATOR
    coordination_cost = validator_incentives + SUPPORT_BASE_COST
    
    # Scale coordination cost based on network value
    if protected_value > 100e9:
        coordination_cost *= 1.2  # 20% premium for larger networks
    
    # 5. Contingency Reserve
    # 15% of total costs
    base_costs = hardware_cost + development_cost + security_audit_cost + coordination_cost
    contingency = base_costs * 0.15
    
    # Apply urgency factor to all costs
    hardware_cost *= urgency_factor
    development_cost *= urgency_factor
    security_audit_cost *= urgency_factor
    coordination_cost *= urgency_factor
    contingency *= urgency_factor
    
    # Total migration cost
    total_cost = hardware_cost + development_cost + security_audit_cost + coordination_cost + contingency
    
    # Calculate economic benefits
    # Avoided losses based on protected value
    # Conservative estimate: 3x multiplier for total impact
    potential_catastrophic_loss = protected_value * 3.0
    
    # Probability-weighted avoided losses
    # Based on industry consensus of >5% CRQC probability by 2035
    CRQC_PROBABILITY_10Y = 0.15  # 15% probability in 10 years
    avoided_losses = potential_catastrophic_loss * CRQC_PROBABILITY_10Y
    
    # Additional benefits (conservative estimates)
    institutional_adoption_value = protected_value * 0.008  # 0.8% value preservation
    first_mover_advantage = protected_value * 0.002  # 0.2% competitive advantage
    crisis_migration_avoided = total_cost * 3  # Crisis migration costs 3x more
    
    total_benefits = avoided_losses + institutional_adoption_value + first_mover_advantage + (crisis_migration_avoided * 0.1)
    
    # Calculate financial metrics
    benefit_cost_ratio = total_benefits / total_cost if total_cost > 0 else 0
    roi_percentage = ((total_benefits - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # NPV calculation (simplified - assumes benefits realized over 5 years)
    BENEFIT_REALIZATION_YEARS = 5
    annual_benefit = total_benefits / BENEFIT_REALIZATION_YEARS
    
    npv = -total_cost  # Initial investment
    for year in range(1, BENEFIT_REALIZATION_YEARS + 1):
        npv += annual_benefit / ((1 + discount_rate) ** year)
    
    # IRR calculation (simplified approximation)
    # Using the formula: IRR ≈ (Benefits/Costs)^(1/n) - 1
    if total_cost > 0 and total_benefits > total_cost:
        irr = (total_benefits / total_cost) ** (1 / BENEFIT_REALIZATION_YEARS) - 1
    else:
        irr = 0
    
    # Payback period
    payback_years = total_cost / annual_benefit if annual_benefit > 0 else float('inf')
    
    # Break-even probability
    # What CRQC probability makes NPV = 0?
    break_even_probability = total_cost / potential_catastrophic_loss if potential_catastrophic_loss > 0 else 0
    
    return MigrationCostBreakdown(
        hardware_acceleration=hardware_cost,
        development_effort=development_cost,
        security_auditing=security_audit_cost,
        validator_coordination=coordination_cost,
        contingency_reserve=contingency,
        total_cost=total_cost,
        protected_value=protected_value,
        avoided_losses=avoided_losses,
        benefit_cost_ratio=benefit_cost_ratio,
        roi_percentage=roi_percentage,
        npv=npv,
        irr=irr,
        payback_years=payback_years,
        break_even_probability=break_even_probability,
        cost_per_validator_hardware=cost_per_validator_hardware,
        validators_requiring_upgrade=validators_needing_hardware,
        engineering_team_size=engineering_team_size,
        audit_firms_count=audit_firms_count
    )


def format_cost_breakdown_for_report(breakdown: MigrationCostBreakdown) -> str:
    """
    Format the cost breakdown for inclusion in reports.
    
    Args:
        breakdown: Migration cost breakdown
        
    Returns:
        Formatted string for report inclusion
    """
    
    report = []
    report.append("## Migration Cost Calculation Details\n")
    
    # Network parameters used
    report.append("### Input Parameters")
    report.append(f"- **Protected Value**: ${breakdown.protected_value/1e9:.1f}B")
    report.append(f"- **Validators Requiring Upgrade**: {breakdown.validators_requiring_upgrade:,}")
    report.append("")
    
    # Cost breakdown
    report.append("### Component Costs")
    report.append("")
    report.append("| Component | Calculation | Amount |")
    report.append("|-----------|-------------|--------|")
    report.append(f"| Hardware Acceleration | {breakdown.validators_requiring_upgrade:,} validators × ${breakdown.cost_per_validator_hardware:,.0f} | ${breakdown.hardware_acceleration/1e6:.1f}M |")
    report.append(f"| Development Effort | {breakdown.engineering_team_size} engineers × 2 years × $250k | ${breakdown.development_effort/1e6:.1f}M |")
    report.append(f"| Security Auditing | {breakdown.audit_firms_count} firms × audit cost | ${breakdown.security_auditing/1e6:.1f}M |")
    report.append(f"| Validator Coordination | Incentives + Support | ${breakdown.validator_coordination/1e6:.1f}M |")
    report.append(f"| Contingency Reserve | 15% of base costs | ${breakdown.contingency_reserve/1e6:.1f}M |")
    report.append(f"| **Total Cost** | **Sum of components** | **${breakdown.total_cost/1e6:.1f}M** |")
    report.append("")
    
    # Economic benefits
    report.append("### Economic Benefits Calculation")
    report.append("")
    report.append(f"- **Potential Catastrophic Loss**: ${breakdown.protected_value/1e9:.1f}B × 3.0 multiplier = ${breakdown.protected_value*3/1e9:.1f}B")
    report.append(f"- **Probability-Weighted Avoided Losses**: ${breakdown.protected_value*3/1e9:.1f}B × 15% CRQC probability = ${breakdown.avoided_losses/1e9:.1f}B")
    report.append("")
    
    # Financial metrics
    report.append("### Financial Metrics")
    report.append("")
    report.append(f"- **Benefit-Cost Ratio**: {breakdown.benefit_cost_ratio:.1f}:1")
    report.append(f"- **Return on Investment**: {breakdown.roi_percentage:,.0f}%")
    report.append(f"- **Net Present Value (15% discount)**: ${breakdown.npv/1e9:.2f}B")
    report.append(f"- **Internal Rate of Return**: {breakdown.irr*100:.0f}%")
    
    # Format payback period more intelligently
    if breakdown.payback_years < 0.1:  # Less than ~1 month
        payback_days = breakdown.payback_years * 365
        if payback_days < 1:
            payback_hours = payback_days * 24
            report.append(f"- **Payback Period**: {payback_hours:.1f} hours")
        else:
            report.append(f"- **Payback Period**: {payback_days:.1f} days")
    elif breakdown.payback_years < 1:
        payback_months = breakdown.payback_years * 12
        report.append(f"- **Payback Period**: {payback_months:.1f} months")
    else:
        report.append(f"- **Payback Period**: {breakdown.payback_years:.1f} years")
    
    # Format break-even probability more intelligently
    if breakdown.break_even_probability < 0.001:  # Less than 0.1%
        report.append(f"- **Break-even CRQC Probability**: {breakdown.break_even_probability*100:.3f}%")
        report.append(f"  - *Investment justified with just {breakdown.break_even_probability*100:.3f}% probability of quantum computers emerging*")
    elif breakdown.break_even_probability < 0.01:  # Less than 1%
        report.append(f"- **Break-even CRQC Probability**: {breakdown.break_even_probability*100:.2f}%")
        report.append(f"  - *Investment justified with just {breakdown.break_even_probability*100:.3f}% probability of quantum computers emerging*")
    else:
        report.append(f"- **Break-even CRQC Probability**: {breakdown.break_even_probability*100:.1f}%")
        report.append(f"  - *Investment justified with just {breakdown.break_even_probability*100:.3f}% probability of quantum computers emerging*")
    
    # Add context about industry consensus for comparison
    report.append("  - *Industry consensus: >5% probability by 2035*")
    report.append(f"  - *Safety margin: {5.0/(breakdown.break_even_probability*100):.0f}x above break-even*")
    report.append("")
    
    report.append("*Note: All calculations are dynamic based on current network parameters and market conditions.*")
    
    return "\n".join(report)


# Test the calculator
if __name__ == "__main__":
    # Example with current Solana parameters
    breakdown = calculate_migration_cost(
        n_validators=1017,
        sol_price_usd=235.0,
        total_stake_sol=380_000_000,
        tvl_usd=8_500_000_000,
        migration_progress=0.0,
        urgency_factor=1.0
    )
    
    print(format_cost_breakdown_for_report(breakdown))
    print("\nTotal Migration Cost: ${:.1f}M".format(breakdown.total_cost / 1e6))
    print("ROI: {:.0f}x".format(breakdown.benefit_cost_ratio))
