"""Incremental, limit-aware construction of a :class:`DispatchPlan`."""

from __future__ import annotations

from hybrid_power._validation import ENERGY_TOLERANCE
from hybrid_power.domain.dispatch import DispatchContext, DispatchPlan


class PlanBuilder:
    """Allocate energy step by step while respecting every limit in a context.

    Strategies express their priorities as a sequence of calls
    (``serve_from_solar``, ``serve_from_battery``, ...). The builder tracks the
    remaining demand and each resource's remaining headroom, so a strategy
    cannot accidentally produce an infeasible plan. Every allocation method
    returns the energy actually allocated.
    """

    def __init__(self, context: DispatchContext) -> None:
        self._ctx = context
        self._remaining = context.demand_kwh
        self._solar_left = context.solar_available_kwh
        self._grid_left = context.grid_available_kwh
        self._generator_used = False
        self._solar_to_load = 0.0
        self._solar_to_battery = 0.0
        self._grid_to_load = 0.0
        self._grid_to_battery = 0.0
        self._generator_to_load = 0.0
        self._generator_to_battery = 0.0
        self._generator_dumped = 0.0
        self._battery_to_load = 0.0
        self._reasons: list[str] = []

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def context(self) -> DispatchContext:
        """The context the plan is built for."""
        return self._ctx

    @property
    def remaining_demand(self) -> float:
        """Demand not yet allocated to any source."""
        return self._remaining

    @property
    def demand_met(self) -> bool:
        """Whether all demand has been allocated."""
        return self._remaining <= ENERGY_TOLERANCE

    @property
    def battery_charged(self) -> float:
        """Energy already allocated to charging the battery."""
        return self._solar_to_battery + self._grid_to_battery + self._generator_to_battery

    @property
    def is_charging(self) -> bool:
        """Whether any charging has been allocated (prevents discharging)."""
        return self.battery_charged > ENERGY_TOLERANCE

    @property
    def is_discharging(self) -> bool:
        """Whether any discharging has been allocated (prevents charging)."""
        return self._battery_to_load > ENERGY_TOLERANCE

    def charge_headroom(self, ceiling_soc: float | None = None) -> float:
        """Energy the battery can still absorb this step."""
        if self.is_discharging:
            return 0.0
        limit = self._ctx.battery.charge_limit(self._ctx.step_hours, ceiling_soc)
        return max(0.0, limit - self.battery_charged)

    def discharge_headroom(self, floor_soc: float | None = None) -> float:
        """Energy the battery can still deliver this step."""
        if self.is_charging:
            return 0.0
        limit = self._ctx.battery.discharge_limit(self._ctx.step_hours, floor_soc)
        return max(0.0, limit - self._battery_to_load)

    # ------------------------------------------------------------------ #
    # Allocation commands
    # ------------------------------------------------------------------ #
    def note(self, reason: str) -> None:
        """Record a human-readable justification for the decision log."""
        self._reasons.append(reason)

    def serve_from_solar(self) -> float:
        """Serve as much remaining demand as possible from PV."""
        energy = min(self._remaining, self._solar_left)
        self._solar_left -= energy
        self._remaining -= energy
        self._solar_to_load += energy
        return energy

    def charge_from_solar(self, ceiling_soc: float | None = None) -> float:
        """Store surplus PV in the battery."""
        energy = min(self._solar_left, self.charge_headroom(ceiling_soc))
        self._solar_left -= energy
        self._solar_to_battery += energy
        return energy

    def serve_from_battery(self, floor_soc: float | None = None) -> float:
        """Serve remaining demand from the battery down to ``floor_soc``."""
        energy = min(self._remaining, self.discharge_headroom(floor_soc))
        self._remaining -= energy
        self._battery_to_load += energy
        return energy

    def serve_from_grid(self) -> float:
        """Serve remaining demand from the grid (zero during an outage)."""
        energy = min(self._remaining, self._grid_left)
        self._grid_left -= energy
        self._remaining -= energy
        self._grid_to_load += energy
        return energy

    def charge_from_grid(self, ceiling_soc: float | None = None) -> float:
        """Charge the battery from the grid up to ``ceiling_soc``."""
        energy = min(self._grid_left, self.charge_headroom(ceiling_soc))
        self._grid_left -= energy
        self._grid_to_battery += energy
        return energy

    def run_generator(
        self, target_output_kwh: float | None = None, *, store_surplus: bool = True
    ) -> float:
        """Run the generator and return its total output for the step.

        The output is at least the remaining demand and the engine's minimum
        stable load, and at least ``target_output_kwh`` when given (capped so
        it never exceeds what the load and battery can take). Surplus beyond
        the load is stored when ``store_surplus`` is true; anything that still
        cannot be absorbed is dumped.
        """
        if self._generator_used:
            return 0.0
        available = self._ctx.generator_available_kwh
        if available <= ENERGY_TOLERANCE:
            return 0.0
        headroom = self.charge_headroom() if store_surplus else 0.0
        desired = self._remaining
        if target_output_kwh is not None:
            desired = max(desired, min(target_output_kwh, self._remaining + headroom))
        if desired <= ENERGY_TOLERANCE:
            return 0.0
        output = min(max(desired, self._ctx.generator_min_kwh), available)
        to_load = min(output, self._remaining)
        surplus = output - to_load
        to_battery = min(surplus, headroom)
        self._remaining -= to_load
        self._generator_to_load += to_load
        self._generator_to_battery += to_battery
        self._generator_dumped += surplus - to_battery
        self._generator_used = True
        return output

    def build(self) -> DispatchPlan:
        """Freeze the allocations into an immutable plan."""
        return DispatchPlan(
            solar_to_load=self._solar_to_load,
            solar_to_battery=self._solar_to_battery,
            grid_to_load=self._grid_to_load,
            grid_to_battery=self._grid_to_battery,
            generator_to_load=self._generator_to_load,
            generator_to_battery=self._generator_to_battery,
            generator_dumped=self._generator_dumped,
            battery_to_load=self._battery_to_load,
            reasons=tuple(self._reasons),
        )
