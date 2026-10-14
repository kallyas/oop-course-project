from __future__ import annotations

import pytest

from hybrid_power.config import GeneratorMode, OptimizerConfig, ScenarioConfig
from hybrid_power.domain.dispatch import DispatchPlan
from hybrid_power.exceptions import ConfigurationError, DispatchError, StrategyNotFoundError
from hybrid_power.optimization import (
    DispatchStrategy,
    GridGeneratorBaseline,
    PlanBuilder,
    PowerOptimizer,
    RuleBasedOptimizer,
    SolarFirstStrategy,
    StrategyRegistry,
)
from tests.conftest import ContextFactory

PEAK, OFFPEAK, SHOULDER = 0.24, 0.12, 0.16


def served(plan: DispatchPlan) -> float:
    return plan.served_load


# --------------------------------------------------------------------------- #
# DispatchPlan & PlanBuilder
# --------------------------------------------------------------------------- #
class TestDispatchPlan:
    def test_totals(self) -> None:
        plan = DispatchPlan(
            solar_to_load=1,
            solar_to_battery=2,
            grid_to_load=3,
            generator_to_load=4,
            generator_dumped=0.5,
        )
        assert plan.solar_total == 3
        assert plan.grid_total == 3
        assert plan.generator_total == 4.5
        assert plan.battery_charge == 2
        assert plan.served_load == 8

    def test_rejects_negative_flows(self) -> None:
        with pytest.raises(DispatchError):
            DispatchPlan(grid_to_load=-1)

    def test_rejects_simultaneous_charge_and_discharge(self) -> None:
        with pytest.raises(DispatchError, match="same step"):
            DispatchPlan(solar_to_battery=1, battery_to_load=1)


class TestPlanBuilder:
    def test_allocations_never_exceed_demand(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=3.0, solar=2.0))
        assert builder.serve_from_solar() == 2.0
        assert builder.serve_from_grid() == pytest.approx(1.0)
        assert builder.serve_from_grid() == 0.0
        assert builder.demand_met

    def test_charging_blocks_discharging(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=1.0, solar=5.0))
        builder.serve_from_solar()
        assert builder.charge_from_solar() == pytest.approx(4.0)
        assert builder.is_charging
        assert builder.discharge_headroom() == 0.0

    def test_discharging_blocks_charging(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=2.0, grid=True))
        builder.serve_from_battery()
        assert builder.is_discharging
        assert builder.charge_from_grid() == 0.0

    def test_generator_respects_minimum_load(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=1.0, grid=False, generator_min=3.0))
        output = builder.run_generator(store_surplus=False)
        plan = builder.build()
        assert output == pytest.approx(3.0)
        assert plan.generator_to_load == pytest.approx(1.0)
        assert plan.generator_dumped == pytest.approx(2.0)

    def test_generator_surplus_is_stored(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=1.0, grid=False, generator_min=3.0))
        builder.run_generator()
        plan = builder.build()
        assert plan.generator_to_battery == pytest.approx(2.0)
        assert plan.generator_dumped == 0

    def test_generator_target_is_capped_by_absorption(self, make_context: ContextFactory) -> None:
        context = make_context(demand=2.0, grid=False, soc=0.94, generator_min=1.0)
        builder = PlanBuilder(context)
        output = builder.run_generator(target_output_kwh=8.0)
        headroom = context.battery.charge_limit(1.0)
        assert output == pytest.approx(2.0 + headroom)
        assert builder.build().generator_dumped == pytest.approx(0)

    def test_generator_runs_once_per_step(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(demand=12.0, grid=False))
        assert builder.run_generator() == pytest.approx(10.0)
        assert builder.run_generator() == 0.0

    def test_generator_without_fuel(self, make_context: ContextFactory) -> None:
        builder = PlanBuilder(make_context(grid=False, generator_available=0.0))
        assert builder.run_generator() == 0.0


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
ALL_STRATEGIES: list[DispatchStrategy] = [
    GridGeneratorBaseline(),
    SolarFirstStrategy(),
    RuleBasedOptimizer(),
]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
@pytest.mark.parametrize("grid", [True, False])
@pytest.mark.parametrize("solar", [0.0, 2.0, 8.0])
def test_every_strategy_serves_load_when_resources_suffice(
    strategy: DispatchStrategy, grid: bool, solar: float, make_context: ContextFactory
) -> None:
    plan = strategy.decide(make_context(demand=4.0, solar=solar, grid=grid))
    assert served(plan) == pytest.approx(4.0)
    assert plan.reasons


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_unmet_demand_is_explained(
    strategy: DispatchStrategy, make_context: ContextFactory
) -> None:
    context = make_context(demand=4.0, grid=False, soc=0.2, generator_available=0.0)
    plan = strategy.decide(context)
    assert served(plan) == pytest.approx(0.0)
    assert any("could not be served" in reason for reason in plan.reasons)


class TestBaseline:
    def test_ignores_solar_and_battery(self, make_context: ContextFactory) -> None:
        plan = GridGeneratorBaseline().decide(make_context(demand=3, solar=5, soc=0.9))
        assert plan.grid_to_load == 3
        assert plan.solar_total == 0
        assert plan.battery_to_load == 0

    def test_generator_during_outage(self, make_context: ContextFactory) -> None:
        plan = GridGeneratorBaseline().decide(make_context(demand=3, grid=False, soc=0.9))
        assert plan.generator_to_load == 3
        assert plan.battery_charge == 0


class TestSolarFirst:
    def test_merit_order(self, make_context: ContextFactory) -> None:
        plan = SolarFirstStrategy().decide(make_context(demand=5, solar=1, soc=0.3))
        battery_limit = 0.1 * 30 * 0.95
        assert plan.solar_to_load == 1
        assert plan.battery_to_load == pytest.approx(battery_limit)
        assert plan.grid_to_load == pytest.approx(4 - battery_limit)

    def test_surplus_solar_is_stored(self, make_context: ContextFactory) -> None:
        plan = SolarFirstStrategy().decide(make_context(demand=2, solar=6, soc=0.5))
        assert plan.solar_to_battery == pytest.approx(4)


class TestRuleBasedOptimizer:
    def test_alias_matches_proposal_name(self) -> None:
        assert PowerOptimizer is RuleBasedOptimizer

    def test_peak_tariff_uses_battery_above_reserve(self, make_context: ContextFactory) -> None:
        optimizer = RuleBasedOptimizer(OptimizerConfig(reserve_soc=0.5))
        plan = optimizer.decide(make_context(demand=3, tariff=PEAK, soc=0.8))
        assert plan.battery_to_load == pytest.approx(3)
        assert plan.grid_to_load == 0

    def test_outage_reserve_is_protected(self, make_context: ContextFactory) -> None:
        optimizer = RuleBasedOptimizer(OptimizerConfig(reserve_soc=0.5))
        plan = optimizer.decide(make_context(demand=3, tariff=PEAK, soc=0.5))
        assert plan.battery_to_load == 0
        assert plan.grid_to_load == 3

    def test_shoulder_tariff_uses_grid(self, make_context: ContextFactory) -> None:
        plan = RuleBasedOptimizer().decide(make_context(demand=3, tariff=SHOULDER, soc=0.9))
        assert plan.grid_to_load == 3
        assert plan.battery_to_load == 0
        assert plan.grid_to_battery == 0

    def test_offpeak_night_charging(self, make_context: ContextFactory) -> None:
        optimizer = RuleBasedOptimizer(OptimizerConfig(grid_charge_target_soc=0.7))
        plan = optimizer.decide(make_context(demand=3, tariff=OFFPEAK, soc=0.5))
        assert plan.grid_to_load == 3
        assert plan.grid_to_battery == pytest.approx(min(10.0, 0.2 * 30 / 0.95))

    def test_no_grid_charging_while_sun_shines(self, make_context: ContextFactory) -> None:
        plan = RuleBasedOptimizer().decide(make_context(demand=3, solar=1, tariff=OFFPEAK, soc=0.3))
        assert plan.grid_to_battery == 0

    def test_outage_battery_carries_load(self, make_context: ContextFactory) -> None:
        plan = RuleBasedOptimizer().decide(make_context(demand=3, grid=False, soc=0.8))
        assert plan.battery_to_load == pytest.approx(3)
        assert plan.generator_total == 0

    def test_outage_cycle_charging(self, make_context: ContextFactory) -> None:
        optimizer = RuleBasedOptimizer(OptimizerConfig(cycle_charge_load_ratio=0.8))
        plan = optimizer.decide(make_context(demand=3, grid=False, soc=0.21))
        assert plan.generator_total == pytest.approx(8.0)
        assert plan.generator_to_load == pytest.approx(3)
        assert plan.generator_to_battery == pytest.approx(5)
        assert plan.battery_to_load == 0

    def test_outage_load_following(self, make_context: ContextFactory) -> None:
        config = OptimizerConfig(generator_mode=GeneratorMode.LOAD_FOLLOWING)
        plan = RuleBasedOptimizer(config).decide(make_context(demand=4, grid=False, soc=0.21))
        assert plan.generator_total == pytest.approx(4)
        assert plan.generator_to_battery == 0

    def test_generator_hysteresis(self, make_context: ContextFactory) -> None:
        optimizer = RuleBasedOptimizer(OptimizerConfig(generator_stop_soc=0.6))
        running = make_context(demand=3, grid=False, soc=0.5, generator_running=True)
        assert optimizer.decide(running).generator_total > 0
        stopped = make_context(demand=3, grid=False, soc=0.5, generator_running=False)
        assert optimizer.decide(stopped).generator_total == 0

    def test_outage_without_fuel_falls_back_to_battery(self, make_context: ContextFactory) -> None:
        context = make_context(demand=6, grid=False, soc=0.3, generator_available=0.0)
        plan = RuleBasedOptimizer().decide(context)
        assert plan.battery_to_load == pytest.approx(context.battery.discharge_limit(1.0))
        assert any("generator unavailable" in r for r in plan.reasons)

    def test_description_reflects_mode(self) -> None:
        config = OptimizerConfig(generator_mode=GeneratorMode.LOAD_FOLLOWING)
        assert "load following" in RuleBasedOptimizer(config).description
        assert "RuleBasedOptimizer" in repr(RuleBasedOptimizer())


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestStrategyRegistry:
    def test_default_registry(self) -> None:
        registry = StrategyRegistry.default()
        assert registry.keys() == ["baseline", "solar_first", "optimizer"]
        assert "optimizer" in registry
        assert len(registry) == 3
        assert list(registry) == registry.keys()
        strategies = registry.create_all(ScenarioConfig())
        assert [s.name for s in strategies] == registry.keys()

    def test_optimizer_receives_scenario_rules(self) -> None:
        config = ScenarioConfig(optimizer=OptimizerConfig(reserve_soc=0.33))
        strategy = StrategyRegistry.default().create("optimizer", config)
        assert isinstance(strategy, RuleBasedOptimizer)
        assert strategy.config.reserve_soc == 0.33

    def test_unknown_strategy(self) -> None:
        with pytest.raises(StrategyNotFoundError, match="choose from"):
            StrategyRegistry.default().create("magic", ScenarioConfig())

    def test_duplicate_registration(self) -> None:
        registry = StrategyRegistry.default()
        with pytest.raises(ConfigurationError, match="already registered"):
            registry.register("baseline", lambda _c: GridGeneratorBaseline())
        registry.register("baseline", lambda _c: SolarFirstStrategy(), overwrite=True)
        assert registry.create("baseline", ScenarioConfig()).name == "solar_first"
