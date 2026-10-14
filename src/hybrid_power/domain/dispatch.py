"""Value objects exchanged between the site model and dispatch strategies.

* :class:`DispatchContext` - a read-only view of the site at the start of a
  step, handed to a strategy.
* :class:`DispatchPlan` - the strategy's decision: how much energy flows
  along each path.
* :class:`StepRecord` - what actually happened once the plan was executed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

from hybrid_power._validation import ENERGY_TOLERANCE, require_non_negative
from hybrid_power.domain.battery import StorageState
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.exceptions import DispatchError


@dataclass(frozen=True)
class DispatchContext:
    """Everything a strategy may know when deciding one step."""

    conditions: OperatingConditions
    solar_available_kwh: float
    grid_available_kwh: float
    generator_available_kwh: float
    generator_min_kwh: float
    generator_rated_kwh: float
    generator_running: bool
    battery: StorageState

    @property
    def demand_kwh(self) -> float:
        """Energy the load requires this step."""
        return self.conditions.demand_kwh

    @property
    def step_hours(self) -> float:
        """Step length in hours."""
        return self.conditions.step_hours

    @property
    def tariff(self) -> float:
        """Grid price this step."""
        return self.conditions.tariff_per_kwh

    @property
    def grid_is_available(self) -> bool:
        """Whether grid import is possible this step."""
        return self.conditions.grid_available and self.grid_available_kwh > ENERGY_TOLERANCE


@dataclass(frozen=True)
class DispatchPlan:
    """Energy flows (kWh) chosen by a strategy for one step.

    Flows are expressed at the AC bus. Generator output that can be neither
    consumed nor stored (because of the minimum stable load) is recorded as
    ``generator_dumped`` so that the energy balance always closes.
    """

    solar_to_load: float = 0.0
    solar_to_battery: float = 0.0
    grid_to_load: float = 0.0
    grid_to_battery: float = 0.0
    generator_to_load: float = 0.0
    generator_to_battery: float = 0.0
    generator_dumped: float = 0.0
    battery_to_load: float = 0.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name != "reasons":
                require_non_negative(item.name, getattr(self, item.name), DispatchError)
        if self.battery_charge > ENERGY_TOLERANCE and self.battery_to_load > ENERGY_TOLERANCE:
            raise DispatchError("The battery cannot charge and discharge in the same step")

    @property
    def solar_total(self) -> float:
        """Total PV energy taken from the array."""
        return self.solar_to_load + self.solar_to_battery

    @property
    def grid_total(self) -> float:
        """Total energy imported from the grid."""
        return self.grid_to_load + self.grid_to_battery

    @property
    def generator_total(self) -> float:
        """Total generator electrical output (including dumped energy)."""
        return self.generator_to_load + self.generator_to_battery + self.generator_dumped

    @property
    def battery_charge(self) -> float:
        """Energy sent into the battery."""
        return self.solar_to_battery + self.grid_to_battery + self.generator_to_battery

    @property
    def served_load(self) -> float:
        """Energy delivered to the load."""
        return (
            self.solar_to_load + self.grid_to_load + self.generator_to_load + self.battery_to_load
        )


@dataclass(frozen=True)
class StepRecord:
    """Executed outcome of one simulation step (energies in kWh)."""

    index: int
    timestamp: datetime
    demand_kwh: float
    solar_available_kwh: float
    grid_available: bool
    tariff: float
    solar_to_load: float
    solar_to_battery: float
    solar_curtailed: float
    grid_to_load: float
    grid_to_battery: float
    generator_to_load: float
    generator_to_battery: float
    generator_dumped: float
    battery_to_load: float
    unmet_kwh: float
    soc: float
    fuel_used_l: float
    fuel_level_l: float
    grid_cost: float
    generator_cost: float
    battery_cost: float
    emissions_kg: float
    reasons: tuple[str, ...]

    @property
    def served_kwh(self) -> float:
        """Energy delivered to the load."""
        return self.demand_kwh - self.unmet_kwh

    @property
    def battery_charge(self) -> float:
        """Energy sent into the battery."""
        return self.solar_to_battery + self.grid_to_battery + self.generator_to_battery

    @property
    def generator_output(self) -> float:
        """Total generator electrical output."""
        return self.generator_to_load + self.generator_to_battery + self.generator_dumped

    @property
    def generator_on(self) -> bool:
        """Whether the generator ran during the step."""
        return self.generator_output > ENERGY_TOLERANCE

    @property
    def total_cost(self) -> float:
        """Operating cost of the step."""
        return self.grid_cost + self.generator_cost + self.battery_cost

    def to_dict(self) -> dict[str, object]:
        """Flatten the record for tabular export."""
        row: dict[str, object] = {f.name: getattr(self, f.name) for f in fields(self)}
        row["reasons"] = " | ".join(self.reasons)
        row["total_cost"] = self.total_cost
        row["generator_on"] = self.generator_on
        return row
