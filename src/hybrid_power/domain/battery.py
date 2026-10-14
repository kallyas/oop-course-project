"""Energy storage abstractions and the lithium battery implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import override

from hybrid_power._validation import (
    ENERGY_TOLERANCE,
    require_fraction,
    require_in_range,
    require_non_negative,
    require_positive,
)
from hybrid_power.exceptions import CapacityExceededError, ConfigurationError


@dataclass(frozen=True)
class StorageState:
    """Immutable snapshot of a storage device used for planning decisions.

    Strategies receive this snapshot instead of the live :class:`Battery`, so
    they can reason about limits without being able to mutate the device.
    """

    capacity_kwh: float
    soc: float
    min_soc: float
    max_soc: float
    charge_efficiency: float
    discharge_efficiency: float
    max_charge_kw: float
    max_discharge_kw: float

    @property
    def stored_energy_kwh(self) -> float:
        """Energy currently held in the device."""
        return self.soc * self.capacity_kwh

    def discharge_limit(self, step_hours: float, floor_soc: float | None = None) -> float:
        """Maximum energy deliverable to the bus without going below ``floor_soc``.

        The floor can never be lower than the device's own ``min_soc``.
        """
        floor = self.min_soc if floor_soc is None else max(self.min_soc, floor_soc)
        usable = max(0.0, (self.soc - floor) * self.capacity_kwh) * self.discharge_efficiency
        return min(self.max_discharge_kw * step_hours, usable)

    def charge_limit(self, step_hours: float, ceiling_soc: float | None = None) -> float:
        """Maximum energy the device can absorb from the bus without exceeding ``ceiling_soc``.

        The ceiling can never be higher than the device's own ``max_soc``.
        """
        ceiling = self.max_soc if ceiling_soc is None else min(self.max_soc, ceiling_soc)
        headroom = max(0.0, (ceiling - self.soc) * self.capacity_kwh) / self.charge_efficiency
        return min(self.max_charge_kw * step_hours, headroom)


class EnergyStorage(ABC):
    """Interface for any device that can store and release energy."""

    @property
    @abstractmethod
    def soc(self) -> float:
        """State of charge as a fraction of capacity."""

    @abstractmethod
    def state(self) -> StorageState:
        """Return an immutable snapshot of the device."""

    @abstractmethod
    def charge(self, energy_kwh: float, step_hours: float) -> float:
        """Absorb ``energy_kwh`` from the bus; return the energy actually stored."""

    @abstractmethod
    def discharge(self, energy_kwh: float, step_hours: float) -> float:
        """Deliver ``energy_kwh`` to the bus; return the energy drawn from storage."""

    @abstractmethod
    def cycling_cost(self, energy_kwh: float) -> float:
        """Wear cost attributed to moving ``energy_kwh`` through the device."""


class Battery(EnergyStorage):
    """Battery bank with efficiency losses, power limits and safe SoC window."""

    def __init__(
        self,
        capacity_kwh: float,
        *,
        initial_soc: float = 0.8,
        min_soc: float = 0.2,
        max_soc: float = 0.95,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        max_charge_kw: float | None = None,
        max_discharge_kw: float | None = None,
        degradation_cost_per_kwh: float = 0.0,
    ) -> None:
        self._capacity_kwh = require_positive("capacity_kwh", capacity_kwh)
        self._min_soc = require_fraction("min_soc", min_soc)
        self._max_soc = require_fraction("max_soc", max_soc)
        if self._min_soc >= self._max_soc:
            raise ConfigurationError("min_soc must be lower than max_soc")
        self._charge_efficiency = require_in_range("charge_efficiency", charge_efficiency, 0.5, 1)
        self._discharge_efficiency = require_in_range(
            "discharge_efficiency", discharge_efficiency, 0.5, 1
        )
        default_power = capacity_kwh / 2
        self._max_charge_kw = require_positive(
            "max_charge_kw", default_power if max_charge_kw is None else max_charge_kw
        )
        self._max_discharge_kw = require_positive(
            "max_discharge_kw", default_power if max_discharge_kw is None else max_discharge_kw
        )
        self._degradation_cost_per_kwh = require_non_negative(
            "degradation_cost_per_kwh", degradation_cost_per_kwh
        )
        self._soc = 0.0
        self.soc = initial_soc
        self._throughput_kwh = 0.0

    # ------------------------------------------------------------------ #
    # Encapsulated state
    # ------------------------------------------------------------------ #
    @override
    @property
    def soc(self) -> float:
        return self._soc

    @soc.setter
    def soc(self, value: float) -> None:
        """Set the state of charge (e.g. for a reset); must lie in ``[0, max_soc]``."""
        self._soc = require_in_range("soc", value, 0.0, self._max_soc)

    @property
    def capacity_kwh(self) -> float:
        """Nominal energy capacity."""
        return self._capacity_kwh

    @property
    def min_soc(self) -> float:
        """Lowest permitted state of charge during normal operation."""
        return self._min_soc

    @property
    def max_soc(self) -> float:
        """Highest permitted state of charge."""
        return self._max_soc

    @property
    def throughput_kwh(self) -> float:
        """Cumulative energy charged plus discharged at the terminals."""
        return self._throughput_kwh

    @override
    def state(self) -> StorageState:
        return StorageState(
            capacity_kwh=self._capacity_kwh,
            soc=self._soc,
            min_soc=self._min_soc,
            max_soc=self._max_soc,
            charge_efficiency=self._charge_efficiency,
            discharge_efficiency=self._discharge_efficiency,
            max_charge_kw=self._max_charge_kw,
            max_discharge_kw=self._max_discharge_kw,
        )

    # ------------------------------------------------------------------ #
    # Behaviour
    # ------------------------------------------------------------------ #
    @override
    def charge(self, energy_kwh: float, step_hours: float) -> float:
        energy = require_non_negative("energy_kwh", energy_kwh)
        limit = self.state().charge_limit(step_hours)
        if energy > limit + ENERGY_TOLERANCE:
            raise CapacityExceededError(
                f"Battery can absorb at most {limit:.3f} kWh this step, requested {energy:.3f} kWh"
            )
        stored = energy * self._charge_efficiency
        self._soc = min(self._max_soc, self._soc + stored / self._capacity_kwh)
        self._throughput_kwh += energy
        return stored

    @override
    def discharge(self, energy_kwh: float, step_hours: float) -> float:
        energy = require_non_negative("energy_kwh", energy_kwh)
        limit = self.state().discharge_limit(step_hours)
        if energy > limit + ENERGY_TOLERANCE:
            raise CapacityExceededError(
                f"Battery can deliver at most {limit:.3f} kWh this step, requested {energy:.3f} kWh"
            )
        drawn = energy / self._discharge_efficiency
        self._soc = max(0.0, self._soc - drawn / self._capacity_kwh)
        self._throughput_kwh += energy
        return drawn

    @override
    def cycling_cost(self, energy_kwh: float) -> float:
        return require_non_negative("energy_kwh", energy_kwh) * self._degradation_cost_per_kwh

    def __repr__(self) -> str:
        return f"Battery(capacity_kwh={self._capacity_kwh}, soc={self._soc:.3f})"
