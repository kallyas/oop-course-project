from __future__ import annotations

import pytest

from hybrid_power.config import BatteryConfig, GeneratorConfig, ScenarioConfig
from hybrid_power.domain.dispatch import DispatchPlan
from hybrid_power.domain.station import BaseStation
from hybrid_power.exceptions import DispatchError
from tests.conftest import ConditionsFactory


@pytest.fixture
def station() -> BaseStation:
    return BaseStation.from_config(
        ScenarioConfig(
            battery=BatteryConfig(initial_soc=0.5, degradation_cost_per_kwh=0.01),
            generator=GeneratorConfig(maintenance_cost_per_hour=0.0, refuel_interval_hours=None),
        )
    )


def test_factory_builds_components(station: BaseStation) -> None:
    assert station.solar.rated_kw == ScenarioConfig().solar.rated_kw
    assert station.generator.rated_kw == ScenarioConfig().generator.rated_kw
    assert station.battery.soc == 0.5
    assert len(station.sources) == 3
    assert "BaseStation" in repr(station)
    assert station.name == ScenarioConfig().name


def test_snapshot_reflects_conditions(
    station: BaseStation, make_conditions: ConditionsFactory
) -> None:
    context = station.snapshot(make_conditions(solar_factor=0.5, grid_available=False))
    assert context.solar_available_kwh == pytest.approx(4.0)
    assert context.grid_available_kwh == 0
    assert not context.grid_is_available
    assert context.generator_min_kwh == pytest.approx(3.0)
    assert context.battery.soc == 0.5


def test_execute_updates_state_and_records(
    station: BaseStation, make_conditions: ConditionsFactory
) -> None:
    conditions = make_conditions(demand_kw=4.0, solar_factor=0.5, tariff=0.2)
    plan = DispatchPlan(solar_to_load=3.0, solar_to_battery=1.0, grid_to_load=1.0)
    record = station.execute(plan, conditions)
    assert record.unmet_kwh == 0
    assert record.served_kwh == 4.0
    assert record.solar_curtailed == pytest.approx(0.0)
    assert record.grid_cost == pytest.approx(0.2)
    assert record.battery_cost == pytest.approx(0.01)
    assert record.soc > 0.5
    assert record.total_cost == pytest.approx(0.21)
    assert record.to_dict()["total_cost"] == pytest.approx(0.21)


def test_unmet_demand_and_curtailment(
    station: BaseStation, make_conditions: ConditionsFactory
) -> None:
    record = station.execute(
        DispatchPlan(solar_to_load=1.0), make_conditions(demand_kw=3, solar_factor=0.5)
    )
    assert record.unmet_kwh == pytest.approx(2.0)
    assert record.solar_curtailed == pytest.approx(3.0)


def test_generator_fuel_is_recorded(
    station: BaseStation, make_conditions: ConditionsFactory
) -> None:
    conditions = make_conditions(demand_kw=5, grid_available=False)
    record = station.execute(DispatchPlan(generator_to_load=5.0), conditions)
    assert record.generator_on
    assert record.fuel_used_l == pytest.approx(station.generator.fuel_for(5.0, 1.0))
    assert record.emissions_kg == pytest.approx(record.fuel_used_l * 2.68)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (DispatchPlan(grid_to_load=10.0), "demand"),
        (DispatchPlan(solar_to_load=3.0), "Solar PV"),
        (DispatchPlan(generator_to_load=1.0), "minimum stable output"),
        (DispatchPlan(grid_to_load=3.0, grid_to_battery=12.0), "charge limit"),
        (DispatchPlan(battery_to_load=3.0), "discharge limit"),
    ],
)
def test_infeasible_plans_are_rejected_without_side_effects(
    station: BaseStation, make_conditions: ConditionsFactory, plan: DispatchPlan, message: str
) -> None:
    station.battery.soc = 0.25 if plan.battery_to_load else 0.5  # type: ignore[misc]
    soc_before = station.battery.soc
    fuel_before = station.generator.fuel_level_l
    with pytest.raises(DispatchError, match=message):
        station.execute(plan, make_conditions(demand_kw=3.0, solar_factor=0.1))
    assert station.battery.soc == soc_before
    assert station.generator.fuel_level_l == fuel_before
    assert station.grid.energy_supplied_kwh == 0
