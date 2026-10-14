from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from hybrid_power.config import (
    BatteryConfig,
    GeneratorConfig,
    GeneratorMode,
    OptimizerConfig,
    ProfileConfig,
    ScenarioConfig,
    SolarConfig,
)
from hybrid_power.exceptions import ConfigurationError


def test_default_scenario_is_valid() -> None:
    config = ScenarioConfig()
    assert config.profile.hours == 30 * 24
    assert config.optimizer.generator_mode is GeneratorMode.CYCLE_CHARGING


def test_json_round_trip_preserves_every_value(tmp_path: Path) -> None:
    original = ScenarioConfig(
        name="Gulu site",
        profile=ProfileConfig(days=10, start_date=date(2026, 1, 5), seed=99),
        optimizer=OptimizerConfig(generator_mode=GeneratorMode.LOAD_FOLLOWING),
        generator=GeneratorConfig(refuel_interval_hours=None),
    )
    path = tmp_path / "scenario.json"
    original.save(path)
    assert ScenarioConfig.load(path) == original
    payload = json.loads(path.read_text())
    assert payload["profile"]["start_date"] == "2026-01-05"
    assert payload["optimizer"]["generator_mode"] == "load_following"


def test_partial_dictionary_uses_defaults() -> None:
    config = ScenarioConfig.from_dict({"battery": {"capacity_kwh": 50}})
    assert config.battery.capacity_kwh == 50
    assert config.battery.min_soc == BatteryConfig().min_soc
    assert config.solar == SolarConfig()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BatteryConfig(min_soc=0.9, max_soc=0.5), "min_soc"),
        (lambda: BatteryConfig(initial_soc=0.99, max_soc=0.9), "initial_soc"),
        (lambda: BatteryConfig(capacity_kwh=0), "capacity_kwh"),
        (lambda: GeneratorConfig(initial_fuel_l=500, tank_capacity_l=100), "initial_fuel_l"),
        (lambda: GeneratorConfig(refuel_interval_hours=0), "refuel_interval_hours"),
        (lambda: ProfileConfig(days=0), "days"),
        (lambda: ProfileConfig(base_load_kw=5, peak_load_kw=4), "peak_load_kw"),
        (lambda: ProfileConfig(peak_tariff=float("nan")), "peak_tariff"),
        (lambda: SolarConfig(rated_kw=-1), "rated_kw"),
        (lambda: OptimizerConfig(reserve_soc=1.5), "reserve_soc"),
    ],
)
def test_invalid_values_are_rejected(factory: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        factory()  # type: ignore[operator]


def test_wrong_types_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="must be a number"):
        SolarConfig(rated_kw="ten")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="integer"):
        ProfileConfig(days=2.5)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="number"):
        SolarConfig(rated_kw=True)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unknown BatteryConfig field"):
        ScenarioConfig.from_dict({"battery": {"capacity": 10}})


def test_bad_enum_and_date_strings_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="generator_mode"):
        OptimizerConfig.from_dict({"generator_mode": "turbo"})
    with pytest.raises(ConfigurationError, match="ISO date"):
        ProfileConfig.from_dict({"start_date": "yesterday"})


def test_invalid_json_is_reported() -> None:
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        ScenarioConfig.from_json("{not json")
    with pytest.raises(ConfigurationError, match="expects a mapping"):
        ScenarioConfig.from_dict([1, 2])  # type: ignore[arg-type]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read"):
        ScenarioConfig.load(tmp_path / "missing.json")


def test_configs_are_immutable() -> None:
    config = ScenarioConfig()
    with pytest.raises(AttributeError):
        config.name = "changed"  # type: ignore[misc]
