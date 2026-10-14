from __future__ import annotations

import pytest

from hybrid_power.domain.battery import Battery, EnergyStorage
from hybrid_power.exceptions import CapacityExceededError, ConfigurationError


def make_battery(**overrides: float) -> Battery:
    params: dict[str, float] = {
        "initial_soc": 0.5,
        "min_soc": 0.2,
        "max_soc": 0.9,
        "charge_efficiency": 0.9,
        "discharge_efficiency": 0.9,
        "max_charge_kw": 5.0,
        "max_discharge_kw": 5.0,
    }
    params.update(overrides)
    return Battery(20.0, **params)


def test_battery_implements_storage_interface() -> None:
    assert isinstance(make_battery(), EnergyStorage)
    with pytest.raises(TypeError):
        EnergyStorage()  # type: ignore[abstract]


def test_charge_applies_efficiency() -> None:
    battery = make_battery()
    stored = battery.charge(4.0, step_hours=1)
    assert stored == pytest.approx(3.6)
    assert battery.soc == pytest.approx(0.5 + 3.6 / 20)
    assert battery.throughput_kwh == pytest.approx(4.0)


def test_discharge_applies_efficiency() -> None:
    battery = make_battery()
    drawn = battery.discharge(3.6, step_hours=1)
    assert drawn == pytest.approx(4.0)
    assert battery.soc == pytest.approx(0.5 - 4.0 / 20)


def test_power_limit_caps_energy_per_step() -> None:
    battery = make_battery()
    state = battery.state()
    assert state.charge_limit(1) == pytest.approx(5.0)
    assert state.charge_limit(0.5) == pytest.approx(2.5)
    with pytest.raises(CapacityExceededError):
        battery.charge(6.0, step_hours=1)


def test_soc_window_caps_energy() -> None:
    battery = make_battery(initial_soc=0.85)
    headroom = (0.9 - 0.85) * 20 / 0.9
    assert battery.state().charge_limit(1) == pytest.approx(headroom)
    battery.charge(headroom, 1)
    assert battery.soc == pytest.approx(0.9)

    low = make_battery(initial_soc=0.25)
    usable = (0.25 - 0.2) * 20 * 0.9
    assert low.state().discharge_limit(1) == pytest.approx(usable)
    with pytest.raises(CapacityExceededError):
        low.discharge(usable + 0.1, 1)


def test_floor_and_ceiling_overrides_never_widen_the_window() -> None:
    state = make_battery(initial_soc=0.6).state()
    assert state.discharge_limit(1, floor_soc=0.5) == pytest.approx(min(5.0, 0.1 * 20 * 0.9))
    assert state.discharge_limit(1, floor_soc=0.0) == state.discharge_limit(1)
    assert state.charge_limit(1, ceiling_soc=0.65) == pytest.approx(0.05 * 20 / 0.9)
    assert state.charge_limit(1, ceiling_soc=1.0) == state.charge_limit(1)


def test_depleted_battery_below_min_soc_cannot_discharge() -> None:
    battery = make_battery(initial_soc=0.1)
    assert battery.state().discharge_limit(1) == 0


def test_soc_setter_validates() -> None:
    battery = make_battery()
    battery.soc = 0.7
    assert battery.soc == 0.7
    with pytest.raises(ConfigurationError):
        battery.soc = 0.95


def test_cycling_cost() -> None:
    battery = Battery(10, degradation_cost_per_kwh=0.05)
    assert battery.cycling_cost(4) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_soc": 0.9, "max_soc": 0.5},
        {"charge_efficiency": 1.2},
        {"max_charge_kw": 0.0},
        {"initial_soc": -0.1},
    ],
)
def test_invalid_construction(overrides: dict[str, float]) -> None:
    with pytest.raises(ConfigurationError):
        make_battery(**overrides)


def test_default_power_limits_are_half_c_rate() -> None:
    state = Battery(40).state()
    assert state.max_charge_kw == 20
    assert state.max_discharge_kw == 20
    assert state.stored_energy_kwh == pytest.approx(32)
    assert "Battery" in repr(Battery(40))
