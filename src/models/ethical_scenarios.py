"""
Ethical scenarios module for modeling societal and geopolitical impacts.

This module extends the simulation to consider ethical dimensions of quantum threats,
including state-sponsored attacks, privacy violations, and broader societal impacts.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import random
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


class ActorType(Enum):
    """Types of threat actors with different ethical implications."""
    CRIMINAL_ORGANIZATION = "criminal_org"
    STATE_SPONSORED = "state_sponsored"
    HACKTIVIST = "hacktivist"
    CORPORATE_ESPIONAGE = "corporate"
    ROGUE_INSIDER = "insider"
    TERRORIST_GROUP = "terrorist"
    RESEARCH_ACCIDENT = "accidental"


class GeopoliticalContext(Enum):
    """Geopolitical scenarios affecting quantum threat landscape."""
    PEACEFUL_COOPERATION = "peaceful"
    ECONOMIC_COMPETITION = "competition"
    COLD_WAR = "cold_war"
    REGIONAL_CONFLICT = "regional_conflict"
    GLOBAL_CRISIS = "global_crisis"
    QUANTUM_ARMS_RACE = "arms_race"


class PrivacyImpactLevel(Enum):
    """Levels of privacy impact from quantum attacks."""
    MINIMAL = 1  # Limited individual impact
    MODERATE = 2  # Some personal data exposed
    SIGNIFICANT = 3  # Major personal data breach
    SEVERE = 4  # Mass surveillance capability
    CATASTROPHIC = 5  # Complete privacy erosion


@dataclass
class EthicalImpact:
    """Represents ethical and societal impacts of a quantum attack."""
    
    # Privacy impacts
    privacy_impact_level: PrivacyImpactLevel
    individuals_affected: int
    data_types_compromised: List[str]
    surveillance_capability_gained: float  # 0-1 scale
    
    # Societal impacts
    trust_erosion_percentage: float
    social_unrest_probability: float
    democratic_impact: float  # Impact on democratic institutions
    human_rights_violations: int  # Severity 0-10
    
    # Economic justice
    wealth_redistribution_effect: float  # Gini coefficient change
    vulnerable_populations_impact: float  # 0-1, higher = worse
    financial_inclusion_damage: float
    
    # Geopolitical
    international_stability_impact: float
    arms_race_acceleration: float
    cooperation_breakdown_probability: float
    
    # Long-term consequences
    generational_impact_years: int
    reversibility_score: float  # 0 = irreversible, 1 = fully reversible
    
    def calculate_ethical_severity(self) -> float:
        """Calculate overall ethical severity score."""
        weights = {
            'privacy': 0.25,
            'societal': 0.25,
            'justice': 0.20,
            'geopolitical': 0.15,
            'long_term': 0.15
        }
        
        privacy_score = self.privacy_impact_level.value / 5
        societal_score = (self.trust_erosion_percentage + 
                          self.social_unrest_probability + 
                          self.democratic_impact +
                          self.human_rights_violations / 10) / 4
        justice_score = (abs(self.wealth_redistribution_effect) + 
                        self.vulnerable_populations_impact + 
                        self.financial_inclusion_damage) / 3
        geo_score = (self.international_stability_impact + 
                    self.arms_race_acceleration + 
                    self.cooperation_breakdown_probability) / 3
        long_term_score = (self.generational_impact_years / 50 + 
                          (1 - self.reversibility_score)) / 2
        
        return (weights['privacy'] * privacy_score +
                weights['societal'] * societal_score +
                weights['justice'] * justice_score +
                weights['geopolitical'] * geo_score +
                weights['long_term'] * long_term_score)


@dataclass
class StateActorProfile:
    """Profile of a state-sponsored attacker."""
    nation_state: str
    quantum_capability_level: int  # 1-10
    gdp_percentage_invested: float
    motivation: str
    target_nations: List[str]
    ethical_constraints: float  # 0 = no constraints, 1 = highly constrained
    international_law_compliance: float
    cyber_doctrine: str  # "offensive", "defensive", "hybrid"
    alliance_network: List[str]
    nuclear_power: bool
    quantum_supremacy_timeline: int  # Years until quantum supremacy


@dataclass
class SocietalResponse:
    """Societal response to quantum threats."""
    public_awareness_level: float
    media_coverage_intensity: float
    political_pressure_score: float
    regulatory_response_speed: float
    civil_society_mobilization: float
    tech_industry_cooperation: float
    academic_involvement: float
    international_coordination: float


@dataclass
class EthicalScenario:
    """Complete ethical scenario including attack and impacts."""
    actor_type: ActorType
    actor_profile: Optional[StateActorProfile]
    geopolitical_context: GeopoliticalContext
    attack_type: str
    target_sector: str
    ethical_impact: EthicalImpact
    societal_response: SocietalResponse
    
    # Narrative elements
    scenario_name: str
    description: str
    ethical_dilemmas: List[str]
    policy_recommendations: List[str]
    
    # Probabilities and timeline
    likelihood_score: float
    emergence_year: int
    escalation_risk: float
    
    def generate_narrative(self) -> str:
        """Generate a narrative description of the scenario."""
        narrative = f"## {self.scenario_name}\n\n"
        narrative += f"**Actor**: {self.actor_type.value}\n"
        narrative += f"**Context**: {self.geopolitical_context.value}\n"
        narrative += f"**Target**: {self.target_sector}\n\n"
        narrative += f"{self.description}\n\n"
        
        if self.ethical_dilemmas:
            narrative += "### Ethical Dilemmas\n"
            for dilemma in self.ethical_dilemmas:
                narrative += f"- {dilemma}\n"
            narrative += "\n"
        
        narrative += f"### Impact Assessment\n"
        narrative += f"- Privacy Impact: {self.ethical_impact.privacy_impact_level.name}\n"
        narrative += f"- Individuals Affected: {self.ethical_impact.individuals_affected:,}\n"
        narrative += f"- Trust Erosion: {self.ethical_impact.trust_erosion_percentage:.1%}\n"
        narrative += f"- Human Rights Score: {self.ethical_impact.human_rights_violations}/10\n"
        narrative += f"- Reversibility: {self.ethical_impact.reversibility_score:.1%}\n\n"
        
        if self.policy_recommendations:
            narrative += "### Policy Recommendations\n"
            for rec in self.policy_recommendations:
                narrative += f"- {rec}\n"
        
        return narrative


class EthicalScenariosModel:
    """Model for generating and analyzing ethical scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ethical scenarios model.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.scenarios_db = self._initialize_scenarios()
        self.state_actors = self._initialize_state_actors()
        
    def _initialize_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize database of ethical scenario templates."""
        return [
            {
                'name': 'Operation Digital Silk Road',
                'actor': ActorType.STATE_SPONSORED,
                'description': 'A major power uses quantum computers to break encryption protecting global financial communications, gaining unprecedented economic intelligence.',
                'sector': 'Financial Infrastructure',
                'dilemmas': [
                    'Should nations develop quantum weapons for deterrence?',
                    'Is preemptive cyber defense justified?',
                    'How to balance security with privacy rights?'
                ]
            },
            {
                'name': 'The Privacy Apocalypse',
                'actor': ActorType.STATE_SPONSORED,
                'description': 'Mass decryption of historical encrypted data exposes decades of personal communications, medical records, and private transactions.',
                'sector': 'Personal Data',
                'dilemmas': [
                    'Should there be a "right to quantum immunity"?',
                    'How to protect vulnerable populations?',
                    'Is retroactive privacy protection possible?'
                ]
            },
            {
                'name': 'Cryptocurrency Collapse',
                'actor': ActorType.CRIMINAL_ORGANIZATION,
                'description': 'Criminal organizations use quantum computers to steal cryptocurrency worth trillions, causing global economic chaos.',
                'sector': 'Digital Assets',
                'dilemmas': [
                    'Who bears responsibility for quantum-vulnerable systems?',
                    'Should victims be compensated?',
                    'How to prevent wealth concentration through quantum advantage?'
                ]
            },
            {
                'name': 'The Quantum Divide',
                'actor': ActorType.CORPORATE_ESPIONAGE,
                'description': 'Corporations with quantum access dominate those without, creating unprecedented inequality.',
                'sector': 'Corporate Competition',
                'dilemmas': [
                    'Should quantum computing be regulated as a utility?',
                    'How to ensure equitable access?',
                    'Can competitive markets survive quantum asymmetry?'
                ]
            },
            {
                'name': 'Democratic Interference 2.0',
                'actor': ActorType.STATE_SPONSORED,
                'description': 'Quantum computers break election infrastructure encryption, enabling unprecedented vote manipulation.',
                'sector': 'Democratic Institutions',
                'dilemmas': [
                    'Can democracy survive in a post-quantum world?',
                    'Should elections revert to paper-only systems?',
                    'How to verify election integrity?'
                ]
            },
            {
                'name': 'Medical Records Massacre',
                'actor': ActorType.TERRORIST_GROUP,
                'description': 'Terrorist groups decrypt and weaponize medical records for targeted biological attacks.',
                'sector': 'Healthcare',
                'dilemmas': [
                    'Should medical data be quantum-protected at all costs?',
                    'How to balance medical research with security?',
                    'Who decides access to quantum-safe medical systems?'
                ]
            },
            {
                'name': 'The Quantum Cold War',
                'actor': ActorType.STATE_SPONSORED,
                'description': 'Multiple nations achieve quantum supremacy simultaneously, creating a new arms race with unclear rules.',
                'sector': 'National Security',
                'dilemmas': [
                    'Should there be quantum weapons treaties?',
                    'How to verify quantum disarmament?',
                    'Can mutual quantum deterrence work?'
                ]
            },
            {
                'name': 'Accidental Apocalypse',
                'actor': ActorType.RESEARCH_ACCIDENT,
                'description': 'A research quantum computer accidentally breaks critical infrastructure encryption during an experiment.',
                'sector': 'Critical Infrastructure',
                'dilemmas': [
                    'Should quantum research be restricted?',
                    'Who is liable for quantum accidents?',
                    'How much safety is enough?'
                ]
            }
        ]
    
    def _initialize_state_actors(self) -> List[StateActorProfile]:
        """Initialize profiles of potential state actors."""
        return [
            StateActorProfile(
                nation_state="Superpower A",
                quantum_capability_level=9,
                gdp_percentage_invested=0.5,
                motivation="Maintain hegemony",
                target_nations=["Rival powers"],
                ethical_constraints=0.3,
                international_law_compliance=0.4,
                cyber_doctrine="offensive",
                alliance_network=["NATO-equivalent"],
                nuclear_power=True,
                quantum_supremacy_timeline=2
            ),
            StateActorProfile(
                nation_state="Rising Power B",
                quantum_capability_level=8,
                gdp_percentage_invested=0.8,
                motivation="Challenge status quo",
                target_nations=["Regional rivals", "Superpower A"],
                ethical_constraints=0.2,
                international_law_compliance=0.3,
                cyber_doctrine="hybrid",
                alliance_network=["Regional bloc"],
                nuclear_power=True,
                quantum_supremacy_timeline=3
            ),
            StateActorProfile(
                nation_state="Tech Leader C",
                quantum_capability_level=7,
                gdp_percentage_invested=0.3,
                motivation="Economic advantage",
                target_nations=["Economic competitors"],
                ethical_constraints=0.7,
                international_law_compliance=0.8,
                cyber_doctrine="defensive",
                alliance_network=["Democratic alliance"],
                nuclear_power=False,
                quantum_supremacy_timeline=4
            ),
            StateActorProfile(
                nation_state="Rogue State D",
                quantum_capability_level=4,
                gdp_percentage_invested=2.0,  # Disproportionate investment
                motivation="Regime survival",
                target_nations=["Perceived enemies"],
                ethical_constraints=0.1,
                international_law_compliance=0.1,
                cyber_doctrine="offensive",
                alliance_network=[],
                nuclear_power=True,
                quantum_supremacy_timeline=10
            )
        ]
    
    def generate_scenario(
        self,
        quantum_capability: Dict[str, Any],
        geopolitical_context: Optional[GeopoliticalContext] = None
    ) -> EthicalScenario:
        """
        Generate an ethical scenario based on current capabilities.
        
        Args:
            quantum_capability: Current quantum computing capabilities
            geopolitical_context: Current geopolitical situation
            
        Returns:
            Generated ethical scenario
        """
        if geopolitical_context is None:
            geopolitical_context = random.choice(list(GeopoliticalContext))
        
        # Select scenario template
        template = random.choice(self.scenarios_db)
        
        # Select actor
        actor_type = template['actor']
        actor_profile = None
        if actor_type == ActorType.STATE_SPONSORED:
            actor_profile = random.choice(self.state_actors)
        
        # Generate ethical impact
        ethical_impact = self._generate_ethical_impact(
            actor_type,
            template['sector'],
            quantum_capability
        )
        
        # Generate societal response
        societal_response = self._generate_societal_response(
            ethical_impact.calculate_ethical_severity(),
            geopolitical_context
        )
        
        # Calculate likelihood based on context
        likelihood = self._calculate_likelihood(
            actor_type,
            geopolitical_context,
            quantum_capability
        )
        
        # Determine emergence year
        base_year = 2025
        if quantum_capability.get('logical_qubits', 0) > 1000:
            emergence_year = base_year + random.randint(0, 5)
        else:
            emergence_year = base_year + random.randint(5, 15)
        
        # Generate policy recommendations
        recommendations = self._generate_policy_recommendations(
            template,
            ethical_impact,
            societal_response
        )
        
        return EthicalScenario(
            actor_type=actor_type,
            actor_profile=actor_profile,
            geopolitical_context=geopolitical_context,
            attack_type=quantum_capability.get('attack_type', 'Unknown'),
            target_sector=template['sector'],
            ethical_impact=ethical_impact,
            societal_response=societal_response,
            scenario_name=template['name'],
            description=template['description'],
            ethical_dilemmas=template['dilemmas'],
            policy_recommendations=recommendations,
            likelihood_score=likelihood,
            emergence_year=emergence_year,
            escalation_risk=self._calculate_escalation_risk(actor_type, geopolitical_context)
        )
    
    def _generate_ethical_impact(
        self,
        actor: ActorType,
        sector: str,
        capability: Dict[str, Any]
    ) -> EthicalImpact:
        """Generate ethical impact based on actor and target."""
        
        # Base impact scales with quantum capability
        severity_multiplier = min(capability.get('logical_qubits', 100) / 1000, 2.0)
        
        # Actor-specific impacts
        if actor == ActorType.STATE_SPONSORED:
            privacy_level = PrivacyImpactLevel.SEVERE
            individuals = int(1e6 * severity_multiplier)
            surveillance = 0.8 * severity_multiplier
            human_rights = int(7 * severity_multiplier)
        elif actor == ActorType.CRIMINAL_ORGANIZATION:
            privacy_level = PrivacyImpactLevel.SIGNIFICANT
            individuals = int(1e5 * severity_multiplier)
            surveillance = 0.3
            human_rights = int(4 * severity_multiplier)
        elif actor == ActorType.TERRORIST_GROUP:
            privacy_level = PrivacyImpactLevel.MODERATE
            individuals = int(1e4 * severity_multiplier)
            surveillance = 0.1
            human_rights = int(8 * severity_multiplier)
        else:
            privacy_level = PrivacyImpactLevel.MODERATE
            individuals = int(1e3 * severity_multiplier)
            surveillance = 0.2
            human_rights = int(3 * severity_multiplier)
        
        # Sector-specific impacts
        if sector == "Financial Infrastructure":
            data_types = ["Transaction records", "Account details", "Credit histories"]
            trust_erosion = 0.7
            wealth_redistribution = 0.15  # Gini increase
            vulnerable_impact = 0.8
        elif sector == "Personal Data":
            data_types = ["Communications", "Medical records", "Personal identifiers"]
            trust_erosion = 0.9
            wealth_redistribution = 0.05
            vulnerable_impact = 0.9
        elif sector == "Democratic Institutions":
            data_types = ["Voting records", "Political affiliations", "Campaign data"]
            trust_erosion = 0.95
            wealth_redistribution = 0.1
            vulnerable_impact = 0.7
        else:
            data_types = ["Various"]
            trust_erosion = 0.5
            wealth_redistribution = 0.05
            vulnerable_impact = 0.5
        
        return EthicalImpact(
            privacy_impact_level=privacy_level,
            individuals_affected=individuals,
            data_types_compromised=data_types,
            surveillance_capability_gained=min(surveillance, 1.0),
            trust_erosion_percentage=trust_erosion,
            social_unrest_probability=min(trust_erosion * 0.7, 1.0),
            democratic_impact=min(trust_erosion * 0.8, 1.0),
            human_rights_violations=min(human_rights, 10),
            wealth_redistribution_effect=wealth_redistribution,
            vulnerable_populations_impact=vulnerable_impact,
            financial_inclusion_damage=vulnerable_impact * 0.7,
            international_stability_impact=0.6 * severity_multiplier,
            arms_race_acceleration=0.7 * severity_multiplier,
            cooperation_breakdown_probability=0.5 * severity_multiplier,
            generational_impact_years=int(20 * severity_multiplier),
            reversibility_score=max(0.1, 1.0 - severity_multiplier * 0.4)
        )
    
    def _generate_societal_response(
        self,
        severity: float,
        context: GeopoliticalContext
    ) -> SocietalResponse:
        """Generate societal response based on severity and context."""
        
        # Context modifiers
        if context == GeopoliticalContext.GLOBAL_CRISIS:
            awareness_mult = 1.5
            political_mult = 1.8
        elif context == GeopoliticalContext.PEACEFUL_COOPERATION:
            awareness_mult = 0.7
            political_mult = 0.6
        else:
            awareness_mult = 1.0
            political_mult = 1.0
        
        return SocietalResponse(
            public_awareness_level=min(severity * 0.8 * awareness_mult, 1.0),
            media_coverage_intensity=min(severity * 0.9 * awareness_mult, 1.0),
            political_pressure_score=min(severity * 0.7 * political_mult, 1.0),
            regulatory_response_speed=min(severity * 0.6, 1.0),
            civil_society_mobilization=min(severity * 0.5, 1.0),
            tech_industry_cooperation=min(0.4 + severity * 0.3, 1.0),
            academic_involvement=min(0.5 + severity * 0.4, 1.0),
            international_coordination=min(severity * 0.4 * political_mult, 1.0)
        )
    
    def _calculate_likelihood(
        self,
        actor: ActorType,
        context: GeopoliticalContext,
        capability: Dict[str, Any]
    ) -> float:
        """Calculate likelihood of scenario occurring."""
        
        # Base likelihood by actor
        actor_base = {
            ActorType.STATE_SPONSORED: 0.7,
            ActorType.CRIMINAL_ORGANIZATION: 0.5,
            ActorType.CORPORATE_ESPIONAGE: 0.4,
            ActorType.TERRORIST_GROUP: 0.3,
            ActorType.HACKTIVIST: 0.2,
            ActorType.ROGUE_INSIDER: 0.2,
            ActorType.RESEARCH_ACCIDENT: 0.1
        }
        
        base = actor_base.get(actor, 0.3)
        
        # Context modifier
        context_mult = {
            GeopoliticalContext.QUANTUM_ARMS_RACE: 2.0,
            GeopoliticalContext.GLOBAL_CRISIS: 1.8,
            GeopoliticalContext.COLD_WAR: 1.5,
            GeopoliticalContext.REGIONAL_CONFLICT: 1.3,
            GeopoliticalContext.ECONOMIC_COMPETITION: 1.1,
            GeopoliticalContext.PEACEFUL_COOPERATION: 0.5
        }
        
        mult = context_mult.get(context, 1.0)
        
        # Capability modifier
        if capability.get('logical_qubits', 0) > 4719:
            capability_mult = 1.5
        elif capability.get('logical_qubits', 0) > 1000:
            capability_mult = 1.2
        else:
            capability_mult = 0.8
        
        return min(base * mult * capability_mult, 0.95)
    
    def _calculate_escalation_risk(
        self,
        actor: ActorType,
        context: GeopoliticalContext
    ) -> float:
        """Calculate risk of scenario escalating."""
        
        # High escalation risk scenarios
        high_risk_actors = [
            ActorType.STATE_SPONSORED,
            ActorType.TERRORIST_GROUP
        ]
        
        high_risk_contexts = [
            GeopoliticalContext.QUANTUM_ARMS_RACE,
            GeopoliticalContext.COLD_WAR,
            GeopoliticalContext.GLOBAL_CRISIS
        ]
        
        risk = 0.3  # Base risk
        
        if actor in high_risk_actors:
            risk += 0.3
        
        if context in high_risk_contexts:
            risk += 0.3
        
        return min(risk, 0.9)
    
    def _generate_policy_recommendations(
        self,
        template: Dict[str, Any],
        impact: EthicalImpact,
        response: SocietalResponse
    ) -> List[str]:
        """Generate policy recommendations based on scenario."""
        
        recommendations = []
        
        # Universal recommendations
        recommendations.append("Accelerate post-quantum cryptography deployment")
        recommendations.append("Establish international quantum governance frameworks")
        
        # Impact-based recommendations
        if impact.privacy_impact_level.value >= 4:
            recommendations.append("Implement quantum-safe privacy legislation")
            recommendations.append("Create data sanctuary programs for vulnerable populations")
        
        if impact.human_rights_violations >= 7:
            recommendations.append("Establish quantum-era human rights protections")
            recommendations.append("Create international quantum crimes tribunal")
        
        if impact.wealth_redistribution_effect > 0.1:
            recommendations.append("Implement quantum inequality mitigation programs")
            recommendations.append("Create universal quantum access initiatives")
        
        # Response-based recommendations
        if response.international_coordination < 0.5:
            recommendations.append("Strengthen international quantum cooperation")
            recommendations.append("Create quantum threat information sharing networks")
        
        if response.tech_industry_cooperation < 0.5:
            recommendations.append("Incentivize private sector quantum defense investment")
            recommendations.append("Mandate quantum risk assessments for critical infrastructure")
        
        # Sector-specific recommendations
        sector_recs = {
            "Financial Infrastructure": [
                "Require quantum-safe banking systems",
                "Create quantum attack insurance frameworks"
            ],
            "Democratic Institutions": [
                "Implement quantum-resistant voting systems",
                "Establish election integrity quantum task forces"
            ],
            "Healthcare": [
                "Mandate quantum-safe medical records",
                "Create health data quantum sanctuaries"
            ]
        }
        
        if template['sector'] in sector_recs:
            recommendations.extend(sector_recs[template['sector']])
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def analyze_ethical_implications(
        self,
        scenarios: List[EthicalScenario]
    ) -> Dict[str, Any]:
        """
        Analyze ethical implications across multiple scenarios.
        
        Args:
            scenarios: List of ethical scenarios
            
        Returns:
            Analysis report
        """
        if not scenarios:
            return {'error': 'No scenarios to analyze'}
        
        analysis = {
            'total_scenarios': len(scenarios),
            'ethical_severity_distribution': {},
            'most_likely_scenario': None,
            'highest_risk_scenario': None,
            'policy_priority_matrix': {},
            'human_impact_summary': {},
            'geopolitical_risks': {}
        }
        
        # Calculate distributions
        severities = [s.ethical_impact.calculate_ethical_severity() for s in scenarios]
        analysis['ethical_severity_distribution'] = {
            'mean': np.mean(severities) if NUMPY_AVAILABLE else sum(severities)/len(severities),
            'max': max(severities),
            'min': min(severities),
            'high_severity_count': sum(1 for s in severities if s > 0.7)
        }
        
        # Find most likely and highest risk
        analysis['most_likely_scenario'] = max(scenarios, key=lambda s: s.likelihood_score).scenario_name
        analysis['highest_risk_scenario'] = max(scenarios, key=lambda s: s.escalation_risk).scenario_name
        
        # Aggregate human impact
        total_affected = sum(s.ethical_impact.individuals_affected for s in scenarios)
        analysis['human_impact_summary'] = {
            'total_individuals_at_risk': total_affected,
            'average_trust_erosion': sum(s.ethical_impact.trust_erosion_percentage for s in scenarios) / len(scenarios),
            'democracy_threat_level': sum(s.ethical_impact.democratic_impact for s in scenarios) / len(scenarios),
            'human_rights_concern_level': sum(s.ethical_impact.human_rights_violations for s in scenarios) / len(scenarios)
        }
        
        # Policy priorities (count recommendations)
        policy_counts = {}
        for scenario in scenarios:
            for rec in scenario.policy_recommendations:
                policy_counts[rec] = policy_counts.get(rec, 0) + 1
        
        # Sort by frequency
        analysis['policy_priority_matrix'] = dict(
            sorted(policy_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Geopolitical risk assessment
        context_counts = {}
        for scenario in scenarios:
            context = scenario.geopolitical_context.value
            context_counts[context] = context_counts.get(context, 0) + 1
        
        analysis['geopolitical_risks'] = context_counts
        
        return analysis


def generate_ethical_report(scenarios: List[EthicalScenario]) -> str:
    """
    Generate a comprehensive ethical impact report.
    
    Args:
        scenarios: List of ethical scenarios
        
    Returns:
        Formatted report string
    """
    report = "# Ethical Impact Assessment Report\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Executive Summary\n\n"
    report += "This report analyzes the ethical implications of quantum computing threats "
    report += "to blockchain systems, with focus on societal, privacy, and geopolitical impacts.\n\n"
    
    # Analyze scenarios
    model = EthicalScenariosModel({})
    analysis = model.analyze_ethical_implications(scenarios)
    
    report += "## Key Findings\n\n"
    report += f"- **Scenarios Analyzed**: {analysis['total_scenarios']}\n"
    report += f"- **Individuals at Risk**: {analysis['human_impact_summary']['total_individuals_at_risk']:,}\n"
    report += f"- **Average Trust Erosion**: {analysis['human_impact_summary']['average_trust_erosion']:.1%}\n"
    report += f"- **Democracy Threat Level**: {analysis['human_impact_summary']['democracy_threat_level']:.2f}/1.0\n"
    report += f"- **Human Rights Concern**: {analysis['human_impact_summary']['human_rights_concern_level']:.1f}/10\n\n"
    
    report += "## Critical Scenarios\n\n"
    report += f"**Most Likely**: {analysis['most_likely_scenario']}\n"
    report += f"**Highest Risk**: {analysis['highest_risk_scenario']}\n\n"
    
    report += "## Policy Priorities\n\n"
    report += "Top recommendations based on scenario analysis:\n\n"
    for i, (rec, count) in enumerate(analysis['policy_priority_matrix'].items(), 1):
        report += f"{i}. {rec} (appears in {count} scenarios)\n"
    
    report += "\n## Individual Scenario Narratives\n\n"
    
    # Add top 3 scenarios by severity
    top_scenarios = sorted(scenarios, 
                          key=lambda s: s.ethical_impact.calculate_ethical_severity(), 
                          reverse=True)[:3]
    
    for scenario in top_scenarios:
        report += scenario.generate_narrative()
        report += "\n---\n\n"
    
    report += "## Ethical Guidelines\n\n"
    report += "1. **Precautionary Principle**: Act to prevent harm even under uncertainty\n"
    report += "2. **Equity**: Ensure quantum defenses don't create new inequalities\n"
    report += "3. **Transparency**: Public has right to know quantum risks\n"
    report += "4. **Accountability**: Clear responsibility chains for quantum security\n"
    report += "5. **Human Rights**: Quantum capabilities must not violate fundamental rights\n\n"
    
    report += "## Conclusion\n\n"
    report += "The quantum threat to blockchain systems raises profound ethical questions "
    report += "that extend beyond technical solutions. Addressing these challenges requires "
    report += "coordinated action across technological, policy, and social domains, with "
    report += "careful consideration of human rights, equity, and democratic values."
    
    return report


if __name__ == "__main__":
    # Test the module
    model = EthicalScenariosModel({'quantum_capability_level': 7})
    
    # Generate sample scenarios
    scenarios = []
    for context in [GeopoliticalContext.QUANTUM_ARMS_RACE, 
                   GeopoliticalContext.PEACEFUL_COOPERATION,
                   GeopoliticalContext.COLD_WAR]:
        scenario = model.generate_scenario(
            {'logical_qubits': 5000, 'attack_type': 'Shor'},
            context
        )
        scenarios.append(scenario)
    
    # Generate report
    report = generate_ethical_report(scenarios)
    print(report)
