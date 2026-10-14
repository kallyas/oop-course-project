"""Validated, immutable configuration objects for a simulation scenario.

Each section is a frozen dataclass whose ``__post_init__`` enforces its
invariants, so an invalid configuration can never be constructed. The
:class:`ConfigSection` mixin adds dictionary/JSON round-tripping, which the
CLI and the dashboard use to save and reload scenarios.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import Field, dataclass, field, fields
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from hybrid_power._validation import (
    require_fraction,
    require_in_range,
    require_int_in_range,
    require_non_negative,
    require_positive,
    require_text,
)
from hybrid_power.exceptions import ConfigurationError

SectionT = TypeVar("SectionT", bound="ConfigSection")


def _serialise(value: object) -> object:
    """Convert a configuration value into a JSON-compatible primitive."""
    if isinstance(value, ConfigSection):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_serialise(item) for item in value]
    return value


class ConfigSection:
    """Mixin that gives frozen configuration dataclasses (de)serialisation.

    Subclasses override :meth:`_coerce` when a field needs converting from its
    JSON representation (for example an ISO date string or an enum value).
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @classmethod
    def from_dict(cls: type[SectionT], data: Mapping[str, Any]) -> SectionT:
        """Build the section from a mapping, rejecting unknown keys."""
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigurationError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}")
        factory: Callable[..., SectionT] = cls
        try:
            return factory(**cls._coerce(dict(data)))
        except TypeError as exc:
            raise ConfigurationError(f"Invalid {cls.__name__}: {exc}") from exc

    @classmethod
    def _coerce(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Hook for converting JSON primitives into rich field types."""
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary of this section."""
        return {f.name: _serialise(getattr(self, f.name)) for f in fields(self)}

    def to_json(self, indent: int = 2) -> str:
        """Serialise the section to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls: type[SectionT], text: str) -> SectionT:
        """Parse the section from a JSON string."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Configuration is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)


class GeneratorMode(str, Enum):
    """How the optimiser loads the diesel generator once it is started.

    ``LOAD_FOLLOWING`` produces only what the load needs, while
    ``CYCLE_CHARGING`` runs the engine at an efficient set-point and stores the
    surplus in the battery, reducing the number of generator run hours.
    """

    LOAD_FOLLOWING = "load_following"
    CYCLE_CHARGING = "cycle_charging"


@dataclass(frozen=True)
class ProfileConfig(ConfigSection):
    """Parameters for the synthetic hourly operating profile."""

    days: int = 30
    start_date: date = date(2026, 10, 1)
    base_load_kw: float = 2.2
    peak_load_kw: float = 4.0
    load_noise: float = 0.05
    solar_mean_clearness: float = 0.7
    solar_variability: float = 0.3
    outages_per_day: float = 1.2
    mean_outage_hours: float = 3.0
    peak_tariff: float = 0.24
    offpeak_tariff: float = 0.12
    peak_start_hour: int = 18
    peak_end_hour: int = 23
    seed: int = 42

    def __post_init__(self) -> None:
        require_int_in_range("days", self.days, 1, 366)
        if not isinstance(self.start_date, date):
            raise ConfigurationError("start_date must be a datetime.date")
        require_non_negative("base_load_kw", self.base_load_kw)
        require_positive("peak_load_kw", self.peak_load_kw)
        if self.peak_load_kw < self.base_load_kw:
            raise ConfigurationError("peak_load_kw must be >= base_load_kw")
        require_in_range("load_noise", self.load_noise, 0.0, 0.5)
        require_in_range("solar_mean_clearness", self.solar_mean_clearness, 0.01, 1.0)
        require_in_range("solar_variability", self.solar_variability, 0.0, 0.95)
        require_in_range("outages_per_day", self.outages_per_day, 0.0, 24.0)
        require_in_range("mean_outage_hours", self.mean_outage_hours, 1.0, 72.0)
        require_non_negative("peak_tariff", self.peak_tariff)
        require_non_negative("offpeak_tariff", self.offpeak_tariff)
        require_int_in_range("peak_start_hour", self.peak_start_hour, 0, 23)
        require_int_in_range("peak_end_hour", self.peak_end_hour, 0, 24)
        require_int_in_range("seed", self.seed, 0, 2**32 - 1)

    @classmethod
    def _coerce(cls, data: dict[str, Any]) -> dict[str, Any]:
        raw = data.get("start_date")
        if isinstance(raw, str):
            try:
                data["start_date"] = date.fromisoformat(raw)
            except ValueError as exc:
                raise ConfigurationError(f"start_date is not an ISO date: {raw!r}") from exc
        return data

    @property
    def hours(self) -> int:
        """Total number of hourly steps in the profile."""
        return self.days * 24


@dataclass(frozen=True)
class SolarConfig(ConfigSection):
    """Solar PV array parameters."""

    rated_kw: float = 8.0
    om_cost_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        require_non_negative("rated_kw", self.rated_kw)
        require_non_negative("om_cost_per_kwh", self.om_cost_per_kwh)


@dataclass(frozen=True)
class GridConfig(ConfigSection):
    """Utility grid connection parameters."""

    max_import_kw: float = 15.0
    emission_factor_kg_per_kwh: float = 0.1

    def __post_init__(self) -> None:
        require_non_negative("max_import_kw", self.max_import_kw)
        require_non_negative("emission_factor_kg_per_kwh", self.emission_factor_kg_per_kwh)


@dataclass(frozen=True)
class BatteryConfig(ConfigSection):
    """Battery energy storage parameters."""

    capacity_kwh: float = 30.0
    initial_soc: float = 0.8
    min_soc: float = 0.2
    max_soc: float = 0.95
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_charge_kw: float = 10.0
    max_discharge_kw: float = 10.0
    degradation_cost_per_kwh: float = 0.02

    def __post_init__(self) -> None:
        require_positive("capacity_kwh", self.capacity_kwh)
        require_fraction("min_soc", self.min_soc)
        require_fraction("max_soc", self.max_soc)
        if self.min_soc >= self.max_soc:
            raise ConfigurationError("min_soc must be lower than max_soc")
        require_in_range("initial_soc", self.initial_soc, 0.0, self.max_soc)
        require_in_range("charge_efficiency", self.charge_efficiency, 0.5, 1.0)
        require_in_range("discharge_efficiency", self.discharge_efficiency, 0.5, 1.0)
        require_positive("max_charge_kw", self.max_charge_kw)
        require_positive("max_discharge_kw", self.max_discharge_kw)
        require_non_negative("degradation_cost_per_kwh", self.degradation_cost_per_kwh)


@dataclass(frozen=True)
class GeneratorConfig(ConfigSection):
    """Diesel generator parameters.

    The fuel model is the widely used linear approximation
    ``F = a * P_out + b * P_rated`` (litres/hour) with default coefficients
    ``a = 0.246`` and ``b = 0.08415`` from the hybrid-system literature.
    """

    rated_kw: float = 10.0
    tank_capacity_l: float = 150.0
    initial_fuel_l: float = 150.0
    fuel_price_per_l: float = 1.45
    min_load_ratio: float = 0.3
    fuel_curve_slope: float = 0.246
    fuel_curve_intercept: float = 0.08415
    maintenance_cost_per_hour: float = 0.4
    co2_kg_per_litre: float = 2.68
    refuel_interval_hours: int | None = 168

    def __post_init__(self) -> None:
        require_positive("rated_kw", self.rated_kw)
        require_positive("tank_capacity_l", self.tank_capacity_l)
        require_in_range("initial_fuel_l", self.initial_fuel_l, 0.0, self.tank_capacity_l)
        require_non_negative("fuel_price_per_l", self.fuel_price_per_l)
        require_in_range("min_load_ratio", self.min_load_ratio, 0.0, 0.9)
        require_positive("fuel_curve_slope", self.fuel_curve_slope)
        require_non_negative("fuel_curve_intercept", self.fuel_curve_intercept)
        require_non_negative("maintenance_cost_per_hour", self.maintenance_cost_per_hour)
        require_non_negative("co2_kg_per_litre", self.co2_kg_per_litre)
        if self.refuel_interval_hours is not None:
            require_int_in_range("refuel_interval_hours", self.refuel_interval_hours, 1, 10_000)


@dataclass(frozen=True)
class OptimizerConfig(ConfigSection):
    """Tunable rules of the rule-based dispatch optimiser."""

    reserve_soc: float = 0.5
    peak_tariff_threshold: float = 0.2
    grid_charge_tariff_threshold: float = 0.13
    grid_charge_target_soc: float = 0.7
    generator_mode: GeneratorMode = GeneratorMode.CYCLE_CHARGING
    cycle_charge_load_ratio: float = 0.8
    generator_stop_soc: float = 0.6

    def __post_init__(self) -> None:
        require_fraction("reserve_soc", self.reserve_soc)
        require_non_negative("peak_tariff_threshold", self.peak_tariff_threshold)
        require_non_negative("grid_charge_tariff_threshold", self.grid_charge_tariff_threshold)
        require_fraction("grid_charge_target_soc", self.grid_charge_target_soc)
        if not isinstance(self.generator_mode, GeneratorMode):
            raise ConfigurationError("generator_mode must be a GeneratorMode")
        require_in_range("cycle_charge_load_ratio", self.cycle_charge_load_ratio, 0.3, 1.0)
        require_fraction("generator_stop_soc", self.generator_stop_soc)

    @classmethod
    def _coerce(cls, data: dict[str, Any]) -> dict[str, Any]:
        raw = data.get("generator_mode")
        if isinstance(raw, str):
            try:
                data["generator_mode"] = GeneratorMode(raw)
            except ValueError as exc:
                options = ", ".join(mode.value for mode in GeneratorMode)
                raise ConfigurationError(f"generator_mode must be one of: {options}") from exc
        return data


FloatRange = tuple[float, float]


def _require_range(name: str, value: object, low: float, high: float) -> None:
    """Validate a ``(lower, upper)`` pair lying inside ``[low, high]``."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigurationError(f"{name} must be a (lower, upper) pair")
    lower = require_in_range(f"{name} lower bound", value[0], low, high)
    upper = require_in_range(f"{name} upper bound", value[1], low, high)
    if lower > upper:
        raise ConfigurationError(f"{name} lower bound must not exceed the upper bound")


@dataclass(frozen=True)
class FleetConfig(ConfigSection):
    """How a fleet of base stations is sampled around a template scenario.

    Every site shares the template's equipment (the same hybrid kit rolled out
    across the network) but draws its own demand level, grid reliability and
    solar resource uniformly from the given ranges.
    """

    sites: int = 50
    fleet_sizes: tuple[int, ...] = (1, 5, 10, 25, 50)
    load_scale: FloatRange = (0.7, 1.4)
    outages_per_day: FloatRange = (0.2, 3.0)
    mean_outage_hours: FloatRange = (1.0, 6.0)
    solar_mean_clearness: FloatRange = (0.5, 0.85)
    seed: int = 2026

    _RANGE_FIELDS: ClassVar[tuple[str, ...]] = (
        "fleet_sizes",
        "load_scale",
        "outages_per_day",
        "mean_outage_hours",
        "solar_mean_clearness",
    )

    def __post_init__(self) -> None:
        require_int_in_range("sites", self.sites, 1, 1000)
        if not isinstance(self.fleet_sizes, tuple) or not self.fleet_sizes:
            raise ConfigurationError("fleet_sizes must be a non-empty tuple")
        for size in self.fleet_sizes:
            require_int_in_range("fleet size", size, 1, self.sites)
        if list(self.fleet_sizes) != sorted(set(self.fleet_sizes)):
            raise ConfigurationError("fleet_sizes must be strictly increasing")
        _require_range("load_scale", self.load_scale, 0.1, 5.0)
        _require_range("outages_per_day", self.outages_per_day, 0.0, 24.0)
        _require_range("mean_outage_hours", self.mean_outage_hours, 1.0, 72.0)
        _require_range("solar_mean_clearness", self.solar_mean_clearness, 0.01, 1.0)
        require_int_in_range("seed", self.seed, 0, 2**32 - 1)

    @classmethod
    def _coerce(cls, data: dict[str, Any]) -> dict[str, Any]:
        for key in cls._RANGE_FIELDS:
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        return data

    @classmethod
    def with_sites(cls, sites: int, **overrides: object) -> FleetConfig:
        """Build a config whose standard fleet sizes (1, 5, 10, 25, 50, ...) fit ``sites``."""
        ladder = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
        sizes = (*(size for size in ladder if size < sites), sites)
        return cls.from_dict({"sites": sites, "fleet_sizes": sizes, **overrides})


@dataclass(frozen=True)
class ScenarioConfig(ConfigSection):
    """Complete description of one simulation scenario (site + environment)."""

    name: str = "Default telecom site"
    currency: str = "USD"
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    _SECTIONS: ClassVar[Mapping[str, type[ConfigSection]]] = {
        "profile": ProfileConfig,
        "solar": SolarConfig,
        "grid": GridConfig,
        "battery": BatteryConfig,
        "generator": GeneratorConfig,
        "optimizer": OptimizerConfig,
    }

    def __post_init__(self) -> None:
        require_text("name", self.name)
        require_text("currency", self.currency)
        for key, section_type in self._SECTIONS.items():
            if not isinstance(getattr(self, key), section_type):
                raise ConfigurationError(f"{key} must be a {section_type.__name__}")

    @classmethod
    def _coerce(cls, data: dict[str, Any]) -> dict[str, Any]:
        for key, section_type in cls._SECTIONS.items():
            if key in data and isinstance(data[key], Mapping):
                data[key] = section_type.from_dict(data[key])
        return data

    @classmethod
    def load(cls, path: Path) -> ScenarioConfig:
        """Read a scenario from a JSON file."""
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"Cannot read scenario file {path}: {exc}") from exc

    def save(self, path: Path) -> None:
        """Write the scenario to a JSON file."""
        path.write_text(self.to_json() + "\n", encoding="utf-8")
