from __future__ import annotations

import pytest

from hybrid_power.domain.sources import (
    GeneratorSource,
    GridSource,
    PowerSource,
    SolarSource,
    SourceType,
)
from hybrid_power.exceptions import CapacityExceededError, ConfigurationError, OperatingLimitError
from tests.conftest import ConditionsFactory


def make_generator(**overrides: float | int | None) -> GeneratorSource:
    params: dict[str, float | int | None] = {
        "rated_kw": 10.0,
        "tank_capacity_l": 100.0,
        "initial_fuel_l": 100.0,
        "fuel_price_per_l": 1.5,
    }
    params.update(overrides)
    rated = params.pop("rated_kw")
    tank = params.pop("tank_capacity_l")
    fuel = params.pop("initial_fuel_l")
    price = params.pop("fuel_price_per_l")
    return GeneratorSource(rated, tank, fuel, price, **params)  # type: ignore[arg-type]


def test_power_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        PowerSource("abstract")  # type: ignore[abstract]


def test_sources_are_polymorphic(make_conditions: ConditionsFactory) -> None:
    conditions = make_conditions(solar_factor=0.5, grid_available=True, tariff=0.2)
    sources: list[PowerSource] = [
        SolarSource(rated_kw=8),
        GridSource(max_import_kw=15),
        make_generator(),
    ]
    available = {source.source_type: source.available_energy(conditions) for source in sources}
    assert available == {SourceType.SOLAR: 4.0, SourceType.GRID: 15.0, SourceType.GENERATOR: 10.0}
    assert [s.is_renewable for s in sources] == [True, False, False]


class TestSolarSource:
    def test_output_scales_with_solar_factor(self, make_conditions: ConditionsFactory) -> None:
        solar = SolarSource(rated_kw=10, om_cost_per_kwh=0.01)
        conditions = make_conditions(solar_factor=0.3)
        assert solar.available_energy(conditions) == pytest.approx(3.0)
        result = solar.supply(2.0, conditions)
        assert result.cost == pytest.approx(0.02)
        assert result.emissions_kg == 0
        assert solar.energy_supplied_kwh == pytest.approx(2.0)

    def test_cannot_exceed_available_energy(self, make_conditions: ConditionsFactory) -> None:
        solar = SolarSource(rated_kw=10)
        with pytest.raises(CapacityExceededError):
            solar.supply(5.0, make_conditions(solar_factor=0.2))

    def test_negative_request_is_rejected(self, make_conditions: ConditionsFactory) -> None:
        with pytest.raises(OperatingLimitError):
            SolarSource(rated_kw=10).supply(-1.0, make_conditions(solar_factor=0.5))

    def test_invalid_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            SolarSource(rated_kw=-5)
        with pytest.raises(ConfigurationError):
            SolarSource(rated_kw=5, name="  ")


class TestGridSource:
    def test_outage_means_no_energy(self, make_conditions: ConditionsFactory) -> None:
        grid = GridSource(max_import_kw=15)
        assert grid.available_energy(make_conditions(grid_available=False)) == 0
        with pytest.raises(CapacityExceededError):
            grid.supply(1.0, make_conditions(grid_available=False))

    def test_cost_follows_tariff(self, make_conditions: ConditionsFactory) -> None:
        grid = GridSource(max_import_kw=15, emission_factor_kg_per_kwh=0.5)
        result = grid.supply(4.0, make_conditions(tariff=0.25))
        assert result.cost == pytest.approx(1.0)
        assert result.emissions_kg == pytest.approx(2.0)
        assert grid.total_cost == pytest.approx(1.0)
        assert grid.total_emissions_kg == pytest.approx(2.0)


class TestGeneratorSource:
    def test_fuel_curve(self) -> None:
        gen = make_generator()
        assert gen.fuel_rate_l_per_h(0) == 0
        assert gen.fuel_rate_l_per_h(5) == pytest.approx(0.08415 * 10 + 0.246 * 5)

    def test_supply_burns_fuel_and_tracks_hours(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator(maintenance_cost_per_hour=0.5)
        conditions = make_conditions()
        expected_fuel = gen.fuel_for(6.0, 1.0)
        result = gen.supply(6.0, conditions)
        assert gen.fuel_level_l == pytest.approx(100 - expected_fuel)
        assert gen.fuel_consumed_l == pytest.approx(expected_fuel)
        assert result.cost == pytest.approx(expected_fuel * 1.5 + 0.5)
        assert result.emissions_kg == pytest.approx(expected_fuel * 2.68)
        assert gen.run_hours == 1
        assert gen.is_running

    def test_starts_are_counted_per_run(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator()
        for energy in (5.0, 5.0, 0.0, 5.0):
            gen.supply(energy, make_conditions())
        assert gen.starts == 2
        assert gen.run_hours == 3

    def test_minimum_stable_load_is_enforced(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator(min_load_ratio=0.3)
        with pytest.raises(OperatingLimitError, match="minimum stable output"):
            gen.supply(1.0, make_conditions())
        gen.supply(3.0, make_conditions())  # exactly the minimum is fine

    def test_insufficient_fuel_limits_output(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator(initial_fuel_l=2.0)
        available = gen.available_energy(make_conditions())
        assert 0 < available < 10
        assert gen.fuel_for(available, 1.0) == pytest.approx(2.0)
        gen.supply(available, make_conditions())
        assert gen.fuel_level_l == pytest.approx(0.0, abs=1e-9)

    def test_empty_tank_means_unavailable(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator(initial_fuel_l=0.5)
        assert gen.available_energy(make_conditions()) == 0

    def test_refuel(self) -> None:
        gen = make_generator(initial_fuel_l=40.0)
        assert gen.refuel(10) == 10
        assert gen.refuel() == 50
        assert gen.fuel_level_l == 100
        assert gen.refuel(5) == 0
        assert gen.fuel_delivered_l == 60

    def test_scheduled_refuel(self, make_conditions: ConditionsFactory) -> None:
        gen = make_generator(initial_fuel_l=10.0, refuel_interval_hours=24)
        gen.begin_step(make_conditions(index=0))
        assert gen.fuel_level_l == 10
        gen.begin_step(make_conditions(index=24))
        assert gen.fuel_level_l == 100

    def test_invalid_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            make_generator(initial_fuel_l=200.0)
        with pytest.raises(ConfigurationError):
            make_generator(rated_kw=0.0)
        with pytest.raises(ConfigurationError):
            make_generator(refuel_interval_hours=0)

    def test_repr_mentions_name(self) -> None:
        assert "Diesel generator" in repr(make_generator())
