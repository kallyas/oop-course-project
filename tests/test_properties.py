"""Property-based tests: physical invariants must hold for *any* valid scenario."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hybrid_power.config import (
    BatteryConfig,
    GeneratorConfig,
    GeneratorMode,
    OptimizerConfig,
    ProfileConfig,
    ScenarioConfig,
    SolarConfig,
)
from hybrid_power.optimization import StrategyRegistry
from hybrid_power.simulation import Scenario, SimulationEngine

TOLERANCE = 1e-6


@st.composite
def scenarios(draw: st.DrawFn) -> ScenarioConfig:
    min_soc = draw(st.floats(0.0, 0.5))
    max_soc = draw(st.floats(min_soc + 0.1, 1.0))
    tank = draw(st.floats(10, 300))
    base = draw(st.floats(0.5, 6))
    return ScenarioConfig(
        profile=ProfileConfig(
            days=2,
            base_load_kw=base,
            peak_load_kw=base + draw(st.floats(0, 6)),
            solar_mean_clearness=draw(st.floats(0.1, 1.0)),
            outages_per_day=draw(st.floats(0, 8)),
            mean_outage_hours=draw(st.floats(1, 12)),
            seed=draw(st.integers(0, 10_000)),
        ),
        solar=SolarConfig(rated_kw=draw(st.floats(0, 20))),
        battery=BatteryConfig(
            capacity_kwh=draw(st.floats(5, 80)),
            min_soc=min_soc,
            max_soc=max_soc,
            initial_soc=draw(st.floats(0.0, max_soc)),
            max_charge_kw=draw(st.floats(1, 20)),
            max_discharge_kw=draw(st.floats(1, 20)),
        ),
        generator=GeneratorConfig(
            rated_kw=draw(st.floats(3, 20)),
            tank_capacity_l=tank,
            initial_fuel_l=draw(st.floats(0, tank)),
            min_load_ratio=draw(st.floats(0, 0.5)),
            refuel_interval_hours=draw(st.none() | st.integers(6, 48)),
        ),
        optimizer=OptimizerConfig(
            reserve_soc=draw(st.floats(0, 1)),
            generator_mode=draw(st.sampled_from(GeneratorMode)),
        ),
    )


@pytest.mark.parametrize("strategy_name", StrategyRegistry.default().keys())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(config=scenarios())
def test_physical_invariants_hold(strategy_name: str, config: ScenarioConfig) -> None:
    strategy = StrategyRegistry.default().create(strategy_name, config)
    result = SimulationEngine(Scenario(config)).run(strategy)
    for record in result.records:
        supplied = (
            record.solar_to_load
            + record.grid_to_load
            + record.generator_to_load
            + record.battery_to_load
        )
        # Energy balance at the load closes exactly.
        assert supplied + record.unmet_kwh == pytest.approx(record.demand_kwh, abs=TOLERANCE)
        # Physical limits are respected.
        assert -TOLERANCE <= record.soc <= config.battery.max_soc + TOLERANCE
        assert record.fuel_level_l >= -TOLERANCE
        assert record.unmet_kwh >= 0
        assert record.solar_curtailed >= -TOLERANCE
        if not record.grid_available:
            assert record.grid_to_load + record.grid_to_battery == 0
        assert not (record.battery_charge > TOLERANCE and record.battery_to_load > TOLERANCE)
