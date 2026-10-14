"""Power generating assets of a base station.

:class:`PowerSource` is the abstraction shared by every generator. It
implements :meth:`PowerSource.supply` as a *template method*: the common
bookkeeping (validation against availability, cost and emission accounting)
lives in the base class, while subclasses supply the physics through
abstract methods and optional hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from typing_extensions import override

from hybrid_power._validation import (
    ENERGY_TOLERANCE,
    require_in_range,
    require_int_in_range,
    require_non_negative,
    require_positive,
    require_text,
)
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.exceptions import CapacityExceededError, OperatingLimitError


class SourceType(str, Enum):
    """Kinds of generating assets known to the system."""

    SOLAR = "solar"
    GRID = "grid"
    GENERATOR = "generator"


@dataclass(frozen=True)
class SupplyResult:
    """Outcome of asking a source to deliver energy for one step."""

    source: SourceType
    energy_kwh: float
    cost: float
    emissions_kg: float


class PowerSource(ABC):
    """Abstract base class for every energy source feeding the site bus."""

    def __init__(self, name: str) -> None:
        self._name = require_text("name", name)
        self._energy_supplied_kwh = 0.0
        self._total_cost = 0.0
        self._total_emissions_kg = 0.0

    # ------------------------------------------------------------------ #
    # Public read-only state
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Human readable name of the asset."""
        return self._name

    @property
    def energy_supplied_kwh(self) -> float:
        """Cumulative energy delivered since construction."""
        return self._energy_supplied_kwh

    @property
    def total_cost(self) -> float:
        """Cumulative operating cost since construction."""
        return self._total_cost

    @property
    def total_emissions_kg(self) -> float:
        """Cumulative CO2 emissions since construction."""
        return self._total_emissions_kg

    @property
    def is_renewable(self) -> bool:
        """Whether the energy counts towards the renewable fraction."""
        return False

    # ------------------------------------------------------------------ #
    # Behaviour that each concrete source must define
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Category of the source."""

    @abstractmethod
    def available_energy(self, conditions: OperatingConditions) -> float:
        """Maximum energy (kWh) the source can deliver during the step."""

    @abstractmethod
    def operating_cost(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        """Cost of delivering ``energy_kwh`` during the step."""

    @abstractmethod
    def emissions(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        """CO2 emitted (kg) when delivering ``energy_kwh`` during the step."""

    # ------------------------------------------------------------------ #
    # Hooks with sensible defaults
    # ------------------------------------------------------------------ #
    def begin_step(self, conditions: OperatingConditions) -> None:
        """Perform time-based housekeeping before a step is dispatched."""

    def minimum_energy(self, conditions: OperatingConditions) -> float:
        """Smallest non-zero energy the source can deliver (0 if unconstrained)."""
        return 0.0

    def _commit(self, energy_kwh: float, conditions: OperatingConditions) -> None:
        """Update internal state after a successful supply (override as needed)."""

    # ------------------------------------------------------------------ #
    # Template method
    # ------------------------------------------------------------------ #
    def validate_request(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        """Check that ``energy_kwh`` is deliverable and return it clamped to availability.

        Raises:
            OperatingLimitError: if the request is negative or below the
                minimum stable output.
            CapacityExceededError: if the request exceeds availability.

        """
        if energy_kwh < -ENERGY_TOLERANCE:
            raise OperatingLimitError(f"{self.name}: requested energy must be >= 0")
        available = self.available_energy(conditions)
        if energy_kwh > available + ENERGY_TOLERANCE:
            raise CapacityExceededError(
                f"{self.name}: requested {energy_kwh:.3f} kWh, only {available:.3f} kWh available"
            )
        energy = min(max(energy_kwh, 0.0), available)
        minimum = self.minimum_energy(conditions)
        if ENERGY_TOLERANCE < energy < minimum - ENERGY_TOLERANCE:
            raise OperatingLimitError(
                f"{self.name}: {energy:.3f} kWh is below the minimum stable output "
                f"of {minimum:.3f} kWh"
            )
        return energy

    def supply(self, energy_kwh: float, conditions: OperatingConditions) -> SupplyResult:
        """Deliver ``energy_kwh`` for the step, updating cumulative accounting."""
        energy = self.validate_request(energy_kwh, conditions)
        cost = self.operating_cost(energy, conditions)
        emitted = self.emissions(energy, conditions)
        self._commit(energy, conditions)
        self._energy_supplied_kwh += energy
        self._total_cost += cost
        self._total_emissions_kg += emitted
        return SupplyResult(self.source_type, energy, cost, emitted)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class SolarSource(PowerSource):
    """Photovoltaic array whose output follows the hourly solar factor."""

    def __init__(
        self, rated_kw: float, om_cost_per_kwh: float = 0.0, name: str = "Solar PV"
    ) -> None:
        super().__init__(name)
        self._rated_kw = require_non_negative("rated_kw", rated_kw)
        self._om_cost_per_kwh = require_non_negative("om_cost_per_kwh", om_cost_per_kwh)

    @property
    def rated_kw(self) -> float:
        """Peak (rated) output of the array."""
        return self._rated_kw

    @override
    @property
    def source_type(self) -> SourceType:
        return SourceType.SOLAR

    @override
    @property
    def is_renewable(self) -> bool:
        return True

    @override
    def available_energy(self, conditions: OperatingConditions) -> float:
        return self._rated_kw * conditions.solar_factor * conditions.step_hours

    @override
    def operating_cost(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        return energy_kwh * self._om_cost_per_kwh

    @override
    def emissions(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        return 0.0


class GridSource(PowerSource):
    """Utility grid connection with a time-varying tariff and outages."""

    def __init__(
        self,
        max_import_kw: float,
        emission_factor_kg_per_kwh: float = 0.1,
        name: str = "Utility grid",
    ) -> None:
        super().__init__(name)
        self._max_import_kw = require_non_negative("max_import_kw", max_import_kw)
        self._emission_factor = require_non_negative(
            "emission_factor_kg_per_kwh", emission_factor_kg_per_kwh
        )

    @property
    def max_import_kw(self) -> float:
        """Contracted import limit of the connection."""
        return self._max_import_kw

    @override
    @property
    def source_type(self) -> SourceType:
        return SourceType.GRID

    @override
    def available_energy(self, conditions: OperatingConditions) -> float:
        if not conditions.grid_available:
            return 0.0
        return self._max_import_kw * conditions.step_hours

    @override
    def operating_cost(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        return energy_kwh * conditions.tariff_per_kwh

    @override
    def emissions(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        return energy_kwh * self._emission_factor


class GeneratorSource(PowerSource):
    """Diesel generator with a fuel tank, minimum load and linear fuel curve."""

    def __init__(
        self,
        rated_kw: float,
        tank_capacity_l: float,
        initial_fuel_l: float,
        fuel_price_per_l: float,
        *,
        min_load_ratio: float = 0.3,
        fuel_curve_slope: float = 0.246,
        fuel_curve_intercept: float = 0.08415,
        maintenance_cost_per_hour: float = 0.0,
        co2_kg_per_litre: float = 2.68,
        refuel_interval_hours: int | None = None,
        name: str = "Diesel generator",
    ) -> None:
        super().__init__(name)
        self._rated_kw = require_positive("rated_kw", rated_kw)
        self._tank_capacity_l = require_positive("tank_capacity_l", tank_capacity_l)
        self._fuel_level_l = require_in_range(
            "initial_fuel_l", initial_fuel_l, 0.0, self._tank_capacity_l
        )
        self._fuel_price_per_l = require_non_negative("fuel_price_per_l", fuel_price_per_l)
        self._min_load_ratio = require_in_range("min_load_ratio", min_load_ratio, 0.0, 0.9)
        self._slope = require_positive("fuel_curve_slope", fuel_curve_slope)
        self._intercept = require_non_negative("fuel_curve_intercept", fuel_curve_intercept)
        self._maintenance_cost_per_hour = require_non_negative(
            "maintenance_cost_per_hour", maintenance_cost_per_hour
        )
        self._co2_kg_per_litre = require_non_negative("co2_kg_per_litre", co2_kg_per_litre)
        if refuel_interval_hours is not None:
            require_int_in_range("refuel_interval_hours", refuel_interval_hours, 1, 10_000)
        self._refuel_interval_hours = refuel_interval_hours
        self._run_hours = 0.0
        self._starts = 0
        self._is_running = False
        self._fuel_consumed_l = 0.0
        self._fuel_delivered_l = 0.0

    # ------------------------------------------------------------------ #
    # Read-only state
    # ------------------------------------------------------------------ #
    @property
    def rated_kw(self) -> float:
        """Nameplate electrical output."""
        return self._rated_kw

    @property
    def min_output_kw(self) -> float:
        """Minimum stable electrical output while running."""
        return self._rated_kw * self._min_load_ratio

    @property
    def fuel_level_l(self) -> float:
        """Fuel currently in the tank."""
        return self._fuel_level_l

    @property
    def tank_capacity_l(self) -> float:
        """Usable tank volume."""
        return self._tank_capacity_l

    @property
    def fuel_consumed_l(self) -> float:
        """Cumulative fuel burned since construction."""
        return self._fuel_consumed_l

    @property
    def fuel_delivered_l(self) -> float:
        """Cumulative fuel added by refuelling."""
        return self._fuel_delivered_l

    @property
    def run_hours(self) -> float:
        """Cumulative engine running hours."""
        return self._run_hours

    @property
    def starts(self) -> int:
        """Number of engine starts."""
        return self._starts

    @property
    def is_running(self) -> bool:
        """Whether the engine ran during the most recent step."""
        return self._is_running

    @override
    @property
    def source_type(self) -> SourceType:
        return SourceType.GENERATOR

    # ------------------------------------------------------------------ #
    # Fuel model
    # ------------------------------------------------------------------ #
    def fuel_rate_l_per_h(self, output_kw: float) -> float:
        """Fuel consumption at a given electrical output (0 when stopped)."""
        output = require_non_negative("output_kw", output_kw)
        if output <= ENERGY_TOLERANCE:
            return 0.0
        return self._intercept * self._rated_kw + self._slope * output

    def fuel_for(self, energy_kwh: float, step_hours: float) -> float:
        """Litres needed to deliver ``energy_kwh`` over ``step_hours``."""
        return self.fuel_rate_l_per_h(energy_kwh / step_hours) * step_hours

    def refuel(self, litres: float | None = None) -> float:
        """Add fuel (default: fill the tank) and return the litres actually added."""
        headroom = self._tank_capacity_l - self._fuel_level_l
        requested = headroom if litres is None else require_non_negative("litres", litres)
        added = min(requested, headroom)
        self._fuel_level_l += added
        self._fuel_delivered_l += added
        return added

    # ------------------------------------------------------------------ #
    # PowerSource implementation
    # ------------------------------------------------------------------ #
    @override
    def begin_step(self, conditions: OperatingConditions) -> None:
        interval = self._refuel_interval_hours
        if interval is not None and conditions.index > 0 and conditions.index % interval == 0:
            self.refuel()

    @override
    def minimum_energy(self, conditions: OperatingConditions) -> float:
        return self.min_output_kw * conditions.step_hours

    @override
    def available_energy(self, conditions: OperatingConditions) -> float:
        step = conditions.step_hours
        idle_rate = self._intercept * self._rated_kw
        max_by_fuel_kw = (self._fuel_level_l / step - idle_rate) / self._slope
        output_kw = min(self._rated_kw, max_by_fuel_kw)
        if output_kw <= 0 or output_kw < self.min_output_kw - ENERGY_TOLERANCE:
            return 0.0
        return output_kw * step

    @override
    def operating_cost(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        if energy_kwh <= ENERGY_TOLERANCE:
            return 0.0
        fuel = self.fuel_for(energy_kwh, conditions.step_hours)
        return (
            fuel * self._fuel_price_per_l + self._maintenance_cost_per_hour * conditions.step_hours
        )

    @override
    def emissions(self, energy_kwh: float, conditions: OperatingConditions) -> float:
        return self.fuel_for(energy_kwh, conditions.step_hours) * self._co2_kg_per_litre

    def _commit(self, energy_kwh: float, conditions: OperatingConditions) -> None:
        if energy_kwh <= ENERGY_TOLERANCE:
            self._is_running = False
            return
        fuel = min(self.fuel_for(energy_kwh, conditions.step_hours), self._fuel_level_l)
        self._fuel_level_l -= fuel
        self._fuel_consumed_l += fuel
        self._run_hours += conditions.step_hours
        if not self._is_running:
            self._starts += 1
        self._is_running = True


__all__ = [
    "GeneratorSource",
    "GridSource",
    "PowerSource",
    "SolarSource",
    "SourceType",
    "SupplyResult",
]
