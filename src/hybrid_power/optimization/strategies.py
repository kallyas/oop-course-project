"""Interchangeable dispatch strategies (Strategy design pattern).

All strategies share the :class:`DispatchStrategy` interface, so the
simulation engine can run any of them polymorphically on identical input
data and the analytics layer can compare their outcomes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import override

from hybrid_power._validation import ENERGY_TOLERANCE
from hybrid_power.config import GeneratorMode, OptimizerConfig
from hybrid_power.domain.dispatch import DispatchContext, DispatchPlan
from hybrid_power.optimization.builder import PlanBuilder


class DispatchStrategy(ABC):
    """Decide how the site's energy sources serve the load each step."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name used in reports."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence explanation of the strategy."""

    def reset(self) -> None:
        """Clear any internal state before a new simulation run."""

    def decide(self, context: DispatchContext) -> DispatchPlan:
        """Return the dispatch plan for one step (template method)."""
        builder = PlanBuilder(context)
        self._allocate(builder)
        if not builder.demand_met:
            builder.note(f"{builder.remaining_demand:.2f} kWh of demand could not be served")
        return builder.build()

    @abstractmethod
    def _allocate(self, builder: PlanBuilder) -> None:
        """Allocate energy for the step using ``builder``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class GridGeneratorBaseline(DispatchStrategy):
    """Legacy operation: grid when available, diesel generator otherwise.

    Solar and battery assets are ignored, which mirrors how many sites run
    before a hybrid controller is installed. It is the reference against
    which savings are reported.
    """

    @override
    @property
    def name(self) -> str:
        return "baseline"

    @override
    @property
    def description(self) -> str:
        return "Legacy grid-first operation with diesel backup; ignores solar and battery."

    def _allocate(self, builder: PlanBuilder) -> None:
        grid = builder.serve_from_grid()
        if grid > ENERGY_TOLERANCE:
            builder.note(f"Grid supplies {grid:.2f} kWh (legacy grid-first rule)")
        if not builder.demand_met:
            output = builder.run_generator(store_surplus=False)
            if output > ENERGY_TOLERANCE:
                builder.note(f"Grid unavailable: generator runs at {output:.2f} kWh")


class SolarFirstStrategy(DispatchStrategy):
    """Greedy merit order: solar, battery, grid, then generator.

    The strategy uses renewable energy and storage whenever possible but is
    blind to tariffs and outage risk, so it can empty the battery just before
    an outage and buy grid energy at peak prices.
    """

    @override
    @property
    def name(self) -> str:
        return "solar_first"

    @override
    @property
    def description(self) -> str:
        return "Greedy merit order (solar, battery, grid, generator) without tariff awareness."

    def _allocate(self, builder: PlanBuilder) -> None:
        solar = builder.serve_from_solar()
        if solar > ENERGY_TOLERANCE:
            builder.note(f"Solar supplies {solar:.2f} kWh")
        stored = builder.charge_from_solar()
        if stored > ENERGY_TOLERANCE:
            builder.note(f"Surplus solar charges battery with {stored:.2f} kWh")
        battery = builder.serve_from_battery()
        if battery > ENERGY_TOLERANCE:
            builder.note(f"Battery supplies {battery:.2f} kWh")
        grid = builder.serve_from_grid()
        if grid > ENERGY_TOLERANCE:
            builder.note(f"Grid supplies {grid:.2f} kWh")
        if not builder.demand_met:
            output = builder.run_generator()
            if output > ENERGY_TOLERANCE:
                builder.note(f"Generator runs at {output:.2f} kWh to cover the remainder")


class RuleBasedOptimizer(DispatchStrategy):
    """Tariff- and resilience-aware rule-based optimiser.

    Rules, in order of priority:

    1. Solar serves the load first; surplus charges the battery.
    2. **Grid connected** - at peak tariff the battery discharges, but only
       down to an *outage reserve*, so the site keeps backup energy for the
       next outage; the grid covers the rest. At off-peak tariff and without
       PV output the battery is topped up from the grid (energy arbitrage +
       resilience), leaving headroom for the next day's solar.
    3. **Grid outage** - if the battery alone can carry the load it does so,
       avoiding a generator start. Otherwise the generator runs; in
       *cycle-charging* mode it is loaded to an efficient set-point and the
       surplus recharges the battery, and it is kept running (hysteresis)
       until the battery reaches the stop SoC to avoid short cycling.
    4. Any residual demand is reported as unmet.
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self._config = config or OptimizerConfig()

    @property
    def config(self) -> OptimizerConfig:
        """Rule parameters."""
        return self._config

    @override
    @property
    def name(self) -> str:
        return "optimizer"

    @override
    @property
    def description(self) -> str:
        return (
            "Rule-based optimiser: tariff-aware battery use with outage reserve and "
            f"{self._config.generator_mode.value.replace('_', ' ')} generator control."
        )

    def _allocate(self, builder: PlanBuilder) -> None:
        solar = builder.serve_from_solar()
        if solar > ENERGY_TOLERANCE:
            builder.note(f"Solar supplies {solar:.2f} kWh (zero fuel cost and emissions)")
        if builder.demand_met:
            stored = builder.charge_from_solar()
            if stored > ENERGY_TOLERANCE:
                builder.note(f"Surplus solar charges battery with {stored:.2f} kWh")

        if builder.context.grid_is_available:
            self._dispatch_grid_connected(builder)
        else:
            self._dispatch_islanded(builder)

    def _dispatch_grid_connected(self, builder: PlanBuilder) -> None:
        ctx, cfg = builder.context, self._config
        if not builder.demand_met and ctx.tariff >= cfg.peak_tariff_threshold:
            battery = builder.serve_from_battery(floor_soc=cfg.reserve_soc)
            if battery > ENERGY_TOLERANCE:
                builder.note(
                    f"Peak tariff {ctx.tariff:.3f}: battery supplies {battery:.2f} kWh, "
                    f"keeping {cfg.reserve_soc:.0%} outage reserve"
                )
        if not builder.demand_met:
            grid = builder.serve_from_grid()
            if grid > ENERGY_TOLERANCE:
                builder.note(f"Grid supplies {grid:.2f} kWh at {ctx.tariff:.3f}/kWh")
        if not builder.demand_met:
            output = builder.run_generator()
            if output > ENERGY_TOLERANCE:
                builder.note(f"Grid import limit reached: generator adds {output:.2f} kWh")
        # Grid charging is reserved for hours without PV, leaving battery headroom
        # for free solar energy later in the day.
        if (
            ctx.tariff <= cfg.grid_charge_tariff_threshold
            and ctx.battery.soc < cfg.grid_charge_target_soc
            and ctx.solar_available_kwh <= ENERGY_TOLERANCE
        ):
            charged = builder.charge_from_grid(ceiling_soc=cfg.grid_charge_target_soc)
            if charged > ENERGY_TOLERANCE:
                builder.note(
                    f"Off-peak tariff {ctx.tariff:.3f}: grid charges battery with "
                    f"{charged:.2f} kWh towards {cfg.grid_charge_target_soc:.0%}"
                )

    def _dispatch_islanded(self, builder: PlanBuilder) -> None:
        if builder.demand_met:
            return
        ctx, cfg = builder.context, self._config
        cycle_charging = cfg.generator_mode is GeneratorMode.CYCLE_CHARGING
        keep_running = (
            cycle_charging and ctx.generator_running and ctx.battery.soc < cfg.generator_stop_soc
        )
        if not keep_running and builder.discharge_headroom() >= builder.remaining_demand:
            battery = builder.serve_from_battery()
            builder.note(f"Grid outage: battery carries the load ({battery:.2f} kWh), no start")
            return

        target = ctx.generator_rated_kwh * cfg.cycle_charge_load_ratio if cycle_charging else None
        output = builder.run_generator(target_output_kwh=target)
        if output > ENERGY_TOLERANCE:
            reason = "kept running until battery recovers" if keep_running else "battery too low"
            builder.note(
                f"Grid outage, {reason}: generator runs at {output:.2f} kWh "
                f"({cfg.generator_mode.value.replace('_', ' ')})"
            )
        else:
            battery = builder.serve_from_battery()
            builder.note(
                f"Grid outage and generator unavailable (fuel): battery supplies {battery:.2f} kWh"
            )
