from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from hybrid_power.analytics import ComparisonReport, MetricsCalculator, StrategyComparison
from hybrid_power.config import GeneratorConfig, ProfileConfig, ScenarioConfig, SolarConfig
from hybrid_power.domain.dispatch import StepRecord
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.optimization import (
    DispatchStrategy,
    GridGeneratorBaseline,
    RuleBasedOptimizer,
    SolarFirstStrategy,
)
from hybrid_power.service import SimulationService
from hybrid_power.simulation import (
    LoggingObserver,
    ProgressObserver,
    Scenario,
    SimulationEngine,
    SimulationObserver,
    SimulationResult,
)


class RecordingObserver(SimulationObserver):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_start(self, strategy: DispatchStrategy, total_steps: int) -> None:
        self.events.append(f"start:{total_steps}")

    def on_step(self, record: StepRecord) -> None:
        self.events.append("step")

    def on_complete(self, result: SimulationResult) -> None:
        self.events.append(f"done:{result.strategy_name}")


@pytest.fixture
def scenario(small_config: ScenarioConfig) -> Scenario:
    return Scenario(small_config)


class TestSimulationEngine:
    def test_runs_whole_horizon(self, scenario: Scenario) -> None:
        result = SimulationEngine(scenario).run(RuleBasedOptimizer())
        assert len(result) == len(scenario.profile) == 72
        assert result.frame.shape[0] == 72
        assert result.strategy_name == "optimizer"

    def test_observers_are_notified(self, scenario: Scenario) -> None:
        observer = RecordingObserver()
        engine = SimulationEngine(scenario)
        engine.add_observer(observer)
        engine.run(GridGeneratorBaseline())
        assert observer.events[0] == "start:72"
        assert observer.events.count("step") == 72
        assert observer.events[-1] == "done:baseline"

    def test_progress_observer(self, scenario: Scenario) -> None:
        progress: list[float] = []
        engine = SimulationEngine(scenario, [ProgressObserver(lambda p, _n: progress.append(p))])
        engine.run(SolarFirstStrategy())
        assert progress[0] == 0.0
        assert progress[-1] == 1.0
        assert progress == sorted(progress)
        with pytest.raises(ConfigurationError):
            ProgressObserver(lambda _p, _n: None, every=0)

    def test_logging_observer_reports_unmet(
        self, small_config: ScenarioConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        starved = replace(
            small_config,
            profile=replace(small_config.profile, outages_per_day=12, mean_outage_hours=24),
            generator=GeneratorConfig(initial_fuel_l=0, refuel_interval_hours=None),
        )
        engine = SimulationEngine(Scenario(starved), [LoggingObserver()])
        with caplog.at_level("INFO"):
            engine.run(GridGeneratorBaseline())
        assert "unmet" in caplog.text
        assert "finished" in caplog.text

    def test_runs_are_independent_and_deterministic(self, scenario: Scenario) -> None:
        engine = SimulationEngine(scenario)
        first = engine.run(RuleBasedOptimizer())
        second = engine.run(RuleBasedOptimizer())
        assert first.records == second.records

    def test_duplicate_strategy_names(self, scenario: Scenario) -> None:
        with pytest.raises(ConfigurationError, match="Duplicate"):
            SimulationEngine(scenario).run_all([SolarFirstStrategy(), SolarFirstStrategy()])

    def test_result_window(self, scenario: Scenario) -> None:
        result = SimulationEngine(scenario).run(SolarFirstStrategy())
        start = result.records[0].timestamp
        window = result.between(start, start + timedelta(hours=24))
        assert len(window) == 24
        with pytest.raises(ConfigurationError):
            result.between(start - timedelta(days=9), start - timedelta(days=8))


class TestMetrics:
    def test_energy_balance_and_kpis(self, scenario: Scenario) -> None:
        result = SimulationEngine(scenario).run(RuleBasedOptimizer())
        metrics = MetricsCalculator().calculate(result)
        frame = result.frame
        assert metrics.energy_demand_kwh == pytest.approx(frame["demand_kwh"].sum())
        assert metrics.energy_served_kwh + metrics.unmet_energy_kwh == pytest.approx(
            metrics.energy_demand_kwh
        )
        assert metrics.total_cost == pytest.approx(
            metrics.grid_cost + metrics.generator_cost + metrics.battery_cost
        )
        assert 0 <= metrics.renewable_fraction <= 1
        assert 0 <= metrics.minimum_soc <= metrics.average_soc <= 1
        assert metrics.generator_starts <= metrics.generator_hours
        assert metrics.cost_per_kwh_served > 0
        assert set(metrics.to_dict()) >= {"lpsp", "renewable_fraction", "cost_per_kwh_served"}

    def test_baseline_uses_no_solar(self, scenario: Scenario) -> None:
        metrics = MetricsCalculator().calculate(
            SimulationEngine(scenario).run(GridGeneratorBaseline())
        )
        assert metrics.solar_used_kwh == 0
        assert metrics.renewable_fraction == 0
        assert metrics.battery_throughput_kwh == 0


class TestComparison:
    def test_report_against_baseline(self, small_config: ScenarioConfig) -> None:
        outcome = SimulationService().run(small_config)
        report = outcome.report
        assert report.baseline_name == "baseline"
        assert set(report.metrics) == {"baseline", "solar_first", "optimizer"}
        assert set(report.comparisons()) == {"solar_first", "optimizer"}
        assert report.to_frame().shape[0] == 3
        assert report.ranking()[0] == report.best()

    def test_optimizer_beats_baseline_on_default_month(self) -> None:
        report = SimulationService().run(ScenarioConfig()).report
        comparison = report.comparison("optimizer")
        assert comparison.is_improvement
        assert comparison.cost_savings_pct is not None
        assert comparison.cost_savings_pct > 20
        assert comparison.fuel_savings_l > 0
        assert comparison.emissions_reduction_kg > 0
        assert comparison.generator_hours_reduction > 0
        assert report.best().strategy_name == "optimizer"

    def test_percentages_undefined_for_zero_baseline(self, small_config: ScenarioConfig) -> None:
        reliable = replace(small_config, profile=replace(small_config.profile, outages_per_day=0))
        report = SimulationService().run(reliable).report
        comparison: StrategyComparison = report.comparison("optimizer")
        assert report.metrics["baseline"].fuel_used_l == 0
        assert comparison.fuel_savings_pct is None

    def test_unknown_names(self, small_config: ScenarioConfig) -> None:
        outcome = SimulationService().run(small_config, strategies=["optimizer"])
        assert set(outcome.results) == {"baseline", "optimizer"}
        with pytest.raises(ConfigurationError):
            outcome.report.comparison("nope")
        with pytest.raises(ConfigurationError):
            ComparisonReport(outcome.results, baseline="nope")
        with pytest.raises(ConfigurationError):
            SimulationService().run(small_config, baseline="nope")


class TestOperatingScenarios:
    """Scenarios named in the proposal's test plan."""

    def run(self, config: ScenarioConfig) -> dict[str, float]:
        report = SimulationService().run(config).report
        return {name: m.unmet_energy_kwh for name, m in report.metrics.items()}

    def test_permanent_outage_with_fuel(self, small_config: ScenarioConfig) -> None:
        config = replace(
            small_config,
            profile=replace(small_config.profile, outages_per_day=24, mean_outage_hours=72),
        )
        assert all(unmet == 0 for unmet in self.run(config).values())

    def test_no_solar(self, small_config: ScenarioConfig) -> None:
        config = replace(small_config, solar=SolarConfig(rated_kw=0))
        report = SimulationService().run(config).report
        assert report.metrics["optimizer"].solar_used_kwh == 0
        assert report.metrics["optimizer"].unmet_energy_kwh == 0

    def test_insufficient_fuel_leads_to_unmet_demand(self, small_config: ScenarioConfig) -> None:
        config = replace(
            small_config,
            profile=ProfileConfig(days=3, outages_per_day=24, mean_outage_hours=72),
            generator=GeneratorConfig(initial_fuel_l=5, refuel_interval_hours=None),
        )
        unmet = self.run(config)
        assert all(value > 0 for value in unmet.values())
        # Solar and storage keep the hybrid strategies far more reliable.
        assert unmet["optimizer"] < unmet["baseline"]
