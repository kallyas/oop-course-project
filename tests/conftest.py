"""Shared fixtures and factories for the test suite."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from hybrid_power.config import ProfileConfig, ScenarioConfig
from hybrid_power.domain.battery import StorageState
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.domain.dispatch import DispatchContext

ConditionsFactory = Callable[..., OperatingConditions]
ContextFactory = Callable[..., DispatchContext]


@pytest.fixture
def make_conditions() -> ConditionsFactory:
    """Build :class:`OperatingConditions` with overridable defaults."""

    def factory(
        *,
        index: int = 0,
        hour: int = 12,
        demand_kw: float = 3.0,
        solar_factor: float = 0.0,
        grid_available: bool = True,
        tariff: float = 0.15,
        step_hours: float = 1.0,
    ) -> OperatingConditions:
        return OperatingConditions(
            index=index,
            timestamp=datetime(2026, 10, 1, hour),
            demand_kw=demand_kw,
            solar_factor=solar_factor,
            grid_available=grid_available,
            tariff_per_kwh=tariff,
            step_hours=step_hours,
        )

    return factory


@pytest.fixture
def make_context(make_conditions: ConditionsFactory) -> ContextFactory:
    """Build a :class:`DispatchContext` for strategy tests without a station."""

    def factory(
        *,
        demand: float = 3.0,
        solar: float = 0.0,
        grid: bool = True,
        tariff: float = 0.15,
        soc: float = 0.6,
        generator_available: float = 10.0,
        generator_min: float = 3.0,
        generator_running: bool = False,
        capacity: float = 30.0,
    ) -> DispatchContext:
        conditions = make_conditions(
            demand_kw=demand,
            solar_factor=min(1.0, solar / 10.0),
            grid_available=grid,
            tariff=tariff,
        )
        return DispatchContext(
            conditions=conditions,
            solar_available_kwh=solar,
            grid_available_kwh=15.0 if grid else 0.0,
            generator_available_kwh=generator_available,
            generator_min_kwh=generator_min,
            generator_rated_kwh=10.0,
            generator_running=generator_running,
            battery=StorageState(
                capacity_kwh=capacity,
                soc=soc,
                min_soc=0.2,
                max_soc=0.95,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                max_charge_kw=10.0,
                max_discharge_kw=10.0,
            ),
        )

    return factory


@pytest.fixture
def small_config() -> ScenarioConfig:
    """A short (3-day) default scenario for fast integration tests."""
    return ScenarioConfig(profile=ProfileConfig(days=3, seed=7))
