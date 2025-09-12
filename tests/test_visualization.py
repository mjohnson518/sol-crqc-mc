"""
Unit tests for visualization module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.config import SimulationParameters
from src.models.quantum_timeline import QuantumDevelopmentModel, QuantumCapability, QuantumThreat
from src.models.network_state import NetworkStateModel, NetworkSnapshot, ValidatorState, ValidatorTier, MigrationStatus
from src.models.attack_scenarios import AttackScenario, AttackWindow, AttackType, AttackSeverity, AttackVector
from src.models.economic_impact import EconomicLoss, ImpactComponent, ImpactType, MarketReaction

from visualization.timeline_plots import TimelinePlotter, plot_quantum_timeline
from visualization.network_plots import NetworkPlotter, plot_network_evolution
from visualization.attack_plots import AttackPlotter, plot_attack_windows
from visualization.economic_plots import EconomicPlotter, plot_economic_impact
from visualization.statistical_plots import StatisticalPlotter, plot_distributions
from visualization.dashboard import DashboardCreator


class TestTimelinePlots:
    """Test timeline visualization functions."""
    
    def test_timeline_plotter_initialization(self):
        """Test TimelinePlotter initialization."""
        plotter = TimelinePlotter()
        assert plotter is not None
        assert hasattr(plotter, 'colors')
    
    def test_plot_quantum_timeline(self):
        """Test basic quantum timeline plotting."""
        config = SimulationParameters()
        model = QuantumDevelopmentModel(config.quantum)
        rng = np.random.RandomState(42)
        
        timeline = model.project_timeline(rng, 2025, 2030)
        
        # Should not raise any exceptions
        fig = plot_quantum_timeline(timeline)
        assert fig is not None
        plt.close(fig)
    
    def test_timeline_ensemble_plot(self):
        """Test ensemble timeline plotting."""
        config = SimulationParameters()
        model = QuantumDevelopmentModel(config.quantum)
        
        timelines = []
        for i in range(10):
            rng = np.random.RandomState(42 + i)
            timeline = model.project_timeline(rng, 2025, 2030)
            timelines.append(timeline)
        
        plotter = TimelinePlotter()
        fig = plotter.plot_timeline_ensemble(timelines)
        assert fig is not None
        plt.close(fig)


class TestNetworkPlots:
    """Test network visualization functions."""
    
    def test_network_plotter_initialization(self):
        """Test NetworkPlotter initialization."""
        plotter = NetworkPlotter()
        assert plotter is not None
        assert hasattr(plotter, 'tier_colors')
        assert hasattr(plotter, 'migration_colors')
    
    def test_plot_network_evolution(self):
        """Test network evolution plotting."""
        config = SimulationParameters()
        model = NetworkStateModel(config.network)
        rng = np.random.RandomState(42)
        
        evolution = model.simulate_evolution(rng, 2025, 2030)
        
        fig = plot_network_evolution(evolution, metric="validators")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_validator_distribution(self):
        """Test validator distribution plotting."""
        plotter = NetworkPlotter()
        
        # Create sample validators
        validators = []
        for i in range(100):
            validator = ValidatorState(
                validator_id=f"val_{i}",
                stake=np.random.exponential(1000),
                tier=np.random.choice(list(ValidatorTier)),
                quantum_safe=np.random.random() > 0.5,
                migration_year=2025 + np.random.randint(0, 10),
                geographic_region=np.random.choice(['north_america', 'europe', 'asia'])
            )
            validators.append(validator)
        
        snapshot = NetworkSnapshot(
            year=2030,
            n_validators=len(validators),
            total_stake=sum(v.stake for v in validators),
            validators=validators,
            geographic_distribution={'north_america': 0.4, 'europe': 0.3, 'asia': 0.3},
            migration_status=MigrationStatus.IN_PROGRESS,
            migration_progress=0.5,
            superminority_count=22,
            gini_coefficient=0.8,
            network_resilience=0.6
        )
        
        fig = plotter.plot_validator_tier_distribution(snapshot)
        assert fig is not None
        plt.close(fig)


class TestAttackPlots:
    """Test attack visualization functions."""
    
    def test_attack_plotter_initialization(self):
        """Test AttackPlotter initialization."""
        plotter = AttackPlotter()
        assert plotter is not None
        assert hasattr(plotter, 'attack_colors')
        assert hasattr(plotter, 'severity_colors')
    
    def test_plot_attack_windows(self):
        """Test attack windows plotting."""
        windows = []
        for i in range(5):
            window = AttackWindow(
                start_year=2030 + i,
                end_year=2032 + i,
                threat_level=np.random.choice([QuantumThreat.EMERGING, 
                                             QuantumThreat.MODERATE, 
                                             QuantumThreat.HIGH]),
                vulnerability_score=0.5 + np.random.random() * 0.5
            )
            windows.append(window)
        
        fig = plot_attack_windows(windows)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_attack_scenarios(self):
        """Test attack scenario plotting."""
        plotter = AttackPlotter()
        
        scenarios = []
        for i in range(20):
            scenario = AttackScenario(
                attack_type=np.random.choice(list(AttackType)),
                vector=np.random.choice(list(AttackVector)),
                year=2030 + np.random.randint(0, 10),
                success_probability=np.random.random(),
                severity=np.random.choice(list(AttackSeverity)),
                validators_compromised=np.random.randint(1, 100),
                stake_compromised=np.random.random() * 0.3,
                accounts_at_risk=np.random.randint(1000, 100000),
                time_to_execute=np.random.exponential(24),
                detection_probability=np.random.random(),
                mitigation_possible=np.random.random() > 0.3
            )
            scenarios.append(scenario)
        
        fig = plotter.plot_attack_matrix(scenarios)
        assert fig is not None
        plt.close(fig)


class TestEconomicPlots:
    """Test economic visualization functions."""
    
    def test_economic_plotter_initialization(self):
        """Test EconomicPlotter initialization."""
        plotter = EconomicPlotter()
        assert plotter is not None
        assert hasattr(plotter, 'impact_colors')
        assert hasattr(plotter, 'recovery_colors')
    
    def test_plot_economic_impact(self):
        """Test economic impact plotting."""
        losses = np.random.lognormal(23, 1.5, 100)  # Sample losses
        
        fig = plot_economic_impact(losses)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_loss_cascade(self):
        """Test loss cascade plotting."""
        plotter = EconomicPlotter()
        
        # Create sample economic loss
        components = [
            ImpactComponent(
                impact_type=ImpactType.DIRECT_THEFT,
                amount_usd=10e9,
                description="Direct theft"
            ),
            ImpactComponent(
                impact_type=ImpactType.MARKET_PANIC,
                amount_usd=25e9,
                description="Market panic"
            ),
            ImpactComponent(
                impact_type=ImpactType.DEFI_CASCADE,
                amount_usd=15e9,
                description="DeFi cascade"
            )
        ]
        
        market_reaction = MarketReaction(
            price_drop_percent=30,
            tvl_drop_percent=40,
            volume_spike_multiplier=5,
            confidence_loss_factor=0.6,
            contagion_spread=0.3
        )
        
        loss = EconomicLoss(
            attack_year=2035,
            immediate_loss_usd=10e9,
            total_loss_usd=50e9,
            recovery_timeline_days=180,
            components=components,
            market_reaction=market_reaction,
            validators_affected=100,
            accounts_affected=50000,
            protocols_affected=20
        )
        
        fig = plotter.plot_loss_cascade(loss)
        assert fig is not None
        plt.close(fig)


class TestStatisticalPlots:
    """Test statistical visualization functions."""
    
    def test_statistical_plotter_initialization(self):
        """Test StatisticalPlotter initialization."""
        plotter = StatisticalPlotter()
        assert plotter is not None
        assert hasattr(plotter, 'default_colors')
    
    def test_plot_distributions(self):
        """Test distribution plotting."""
        data = np.random.normal(100, 15, 1000)
        
        fig = plot_distributions(data)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_convergence(self):
        """Test convergence plotting."""
        plotter = StatisticalPlotter()
        
        results = np.random.normal(100, 15, 500)
        fig = plotter.plot_monte_carlo_convergence(results)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_sensitivity(self):
        """Test sensitivity analysis plotting."""
        plotter = StatisticalPlotter()
        
        sensitivity_results = {
            'Parameter A': 0.35,
            'Parameter B': -0.22,
            'Parameter C': 0.18,
            'Parameter D': -0.45,
            'Parameter E': 0.12
        }
        
        fig = plotter.plot_sensitivity_analysis(sensitivity_results)
        assert fig is not None
        plt.close(fig)


class TestDashboard:
    """Test dashboard creation."""
    
    def test_dashboard_creator_initialization(self):
        """Test DashboardCreator initialization."""
        creator = DashboardCreator()
        assert creator is not None
        assert hasattr(creator, 'risk_colors')
    
    def test_create_executive_summary(self):
        """Test executive summary dashboard creation."""
        creator = DashboardCreator()
        
        # Create sample results
        results = {
            'n_iterations': 1000,
            'random_seed': 42,
            'quantum_timeline': {
                'median_crqc_year': 2035,
                'crqc_probabilities': [0.01 * i for i in range(26)]
            },
            'network_state': {
                'peak_validators': 1500,
                'final_migration': 0.75,
                'years': list(range(2025, 2051)),
                'migration_progress': [0.1 + 0.03 * i for i in range(26)]
            },
            'attack_scenarios': {
                'avg_success_rate': 0.42,
                'years': list(range(2025, 2051)),
                'success_rates': [0.1 + 0.02 * i for i in range(26)]
            },
            'economic_impact': {
                'mean_loss': 75e9,
                'total_losses': np.random.lognormal(24, 1.2, 100)
            }
        }
        
        fig = creator.create_executive_summary(results)
        assert fig is not None
        plt.close(fig)
    
    def test_extract_key_metrics(self):
        """Test key metrics extraction."""
        creator = DashboardCreator()
        
        results = {
            'quantum_timeline': {'median_crqc_year': 2035},
            'economic_impact': {'mean_loss': 50e9},
            'network_state': {'peak_validators': 1500},
            'attack_scenarios': {'avg_success_rate': 0.35}
        }
        
        metrics = creator._extract_key_metrics(results)
        assert 'CRQC Year (Median)' in metrics
        assert 'Avg Economic Loss' in metrics
        assert 'Peak Validators' in metrics
        assert 'Avg Attack Success' in metrics
    
    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        creator = DashboardCreator()
        
        results = {
            'quantum_timeline': {'median_crqc_year': 2033},
            'economic_impact': {'mean_loss': 120e9},
            'network_state': {'final_migration': 0.2}
        }
        
        score = creator._calculate_risk_score(results)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high risk given the parameters
