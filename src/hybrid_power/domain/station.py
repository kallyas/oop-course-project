"""The base station aggregate: composes sources and storage on one AC bus."""

from __future__ import annotations

from hybrid_power._validation import ENERGY_TOLERANCE, require_text
from hybrid_power.config import ScenarioConfig
from hybrid_power.domain.battery import Battery, EnergyStorage
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.domain.dispatch import DispatchContext, DispatchPlan, StepRecord
from hybrid_power.domain.sources import GeneratorSource, GridSource, PowerSource, SolarSource
from hybrid_power.exceptions import DispatchError, OperatingLimitError


class BaseStation:
    """A telecom site with solar, grid, diesel generator and battery storage.

    The station owns all mutable component state. Strategies never touch the
    components directly: they receive a :class:`DispatchContext` snapshot and
    return a :class:`DispatchPlan`, which :meth:`execute` validates in full
    before committing any change. This keeps every step transactional - a
    rejected plan leaves the station untouched.
    """

    def __init__(
        self,
        solar: SolarSource,
        grid: GridSource,
        generator: GeneratorSource,
        battery: EnergyStorage,
        name: str = "Telecom base station",
    ) -> None:
        self._name = require_text("name", name)
        self._solar = solar
        self._grid = grid
        self._generator = generator
        self._battery = battery

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> BaseStation:
        """Factory method that assembles a fresh station from a scenario."""
        gen = config.generator
        bat = config.battery
        return cls(
            solar=SolarSource(config.solar.rated_kw, config.solar.om_cost_per_kwh),
            grid=GridSource(config.grid.max_import_kw, config.grid.emission_factor_kg_per_kwh),
            generator=GeneratorSource(
                gen.rated_kw,
                gen.tank_capacity_l,
                gen.initial_fuel_l,
                gen.fuel_price_per_l,
                min_load_ratio=gen.min_load_ratio,
                fuel_curve_slope=gen.fuel_curve_slope,
                fuel_curve_intercept=gen.fuel_curve_intercept,
                maintenance_cost_per_hour=gen.maintenance_cost_per_hour,
                co2_kg_per_litre=gen.co2_kg_per_litre,
                refuel_interval_hours=gen.refuel_interval_hours,
            ),
            battery=Battery(
                bat.capacity_kwh,
                initial_soc=bat.initial_soc,
                min_soc=bat.min_soc,
                max_soc=bat.max_soc,
                charge_efficiency=bat.charge_efficiency,
                discharge_efficiency=bat.discharge_efficiency,
                max_charge_kw=bat.max_charge_kw,
                max_discharge_kw=bat.max_discharge_kw,
                degradation_cost_per_kwh=bat.degradation_cost_per_kwh,
            ),
            name=config.name,
        )

    # ------------------------------------------------------------------ #
    # Components (read-only access)
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Site name."""
        return self._name

    @property
    def solar(self) -> SolarSource:
        """PV array."""
        return self._solar

    @property
    def grid(self) -> GridSource:
        """Grid connection."""
        return self._grid

    @property
    def generator(self) -> GeneratorSource:
        """Diesel generator."""
        return self._generator

    @property
    def battery(self) -> EnergyStorage:
        """Battery storage."""
        return self._battery

    @property
    def sources(self) -> tuple[PowerSource, ...]:
        """All generating sources, in merit order."""
        return (self._solar, self._grid, self._generator)

    # ------------------------------------------------------------------ #
    # Step lifecycle
    # ------------------------------------------------------------------ #
    def begin_step(self, conditions: OperatingConditions) -> None:
        """Let every source perform time-based housekeeping (e.g. refuelling)."""
        for source in self.sources:
            source.begin_step(conditions)

    def snapshot(self, conditions: OperatingConditions) -> DispatchContext:
        """Build the read-only view given to a dispatch strategy."""
        return DispatchContext(
            conditions=conditions,
            solar_available_kwh=self._solar.available_energy(conditions),
            grid_available_kwh=self._grid.available_energy(conditions),
            generator_available_kwh=self._generator.available_energy(conditions),
            generator_min_kwh=self._generator.minimum_energy(conditions),
            generator_rated_kwh=self._generator.rated_kw * conditions.step_hours,
            generator_running=self._generator.is_running,
            battery=self._battery.state(),
        )

    def validate(self, plan: DispatchPlan, conditions: OperatingConditions) -> None:
        """Raise :class:`DispatchError` if ``plan`` is infeasible for this step."""
        step = conditions.step_hours
        if plan.served_load > conditions.demand_kwh + ENERGY_TOLERANCE:
            raise DispatchError(
                f"Step {conditions.index}: plan serves {plan.served_load:.3f} kWh but demand "
                f"is only {conditions.demand_kwh:.3f} kWh"
            )
        battery_state = self._battery.state()
        try:
            self._solar.validate_request(plan.solar_total, conditions)
            self._grid.validate_request(plan.grid_total, conditions)
            self._generator.validate_request(plan.generator_total, conditions)
        except OperatingLimitError as exc:
            raise DispatchError(f"Step {conditions.index}: {exc}") from exc
        if plan.battery_charge > battery_state.charge_limit(step) + ENERGY_TOLERANCE:
            raise DispatchError(f"Step {conditions.index}: battery charge limit exceeded")
        if plan.battery_to_load > battery_state.discharge_limit(step) + ENERGY_TOLERANCE:
            raise DispatchError(f"Step {conditions.index}: battery discharge limit exceeded")

    def execute(self, plan: DispatchPlan, conditions: OperatingConditions) -> StepRecord:
        """Validate and commit ``plan``, returning the resulting :class:`StepRecord`."""
        self.validate(plan, conditions)
        step = conditions.step_hours
        solar_available = self._solar.available_energy(conditions)
        fuel_before = self._generator.fuel_level_l

        self._solar.supply(plan.solar_total, conditions)
        grid = self._grid.supply(plan.grid_total, conditions)
        generator = self._generator.supply(plan.generator_total, conditions)
        battery_cost = 0.0
        if plan.battery_charge > ENERGY_TOLERANCE:
            self._battery.charge(plan.battery_charge, step)
            battery_cost = self._battery.cycling_cost(plan.battery_charge)
        elif plan.battery_to_load > ENERGY_TOLERANCE:
            self._battery.discharge(plan.battery_to_load, step)
            battery_cost = self._battery.cycling_cost(plan.battery_to_load)

        unmet = conditions.demand_kwh - plan.served_load
        return StepRecord(
            index=conditions.index,
            timestamp=conditions.timestamp,
            demand_kwh=conditions.demand_kwh,
            solar_available_kwh=solar_available,
            grid_available=conditions.grid_available,
            tariff=conditions.tariff_per_kwh,
            solar_to_load=plan.solar_to_load,
            solar_to_battery=plan.solar_to_battery,
            solar_curtailed=max(0.0, solar_available - plan.solar_total),
            grid_to_load=plan.grid_to_load,
            grid_to_battery=plan.grid_to_battery,
            generator_to_load=plan.generator_to_load,
            generator_to_battery=plan.generator_to_battery,
            generator_dumped=plan.generator_dumped,
            battery_to_load=plan.battery_to_load,
            unmet_kwh=unmet if unmet > ENERGY_TOLERANCE else 0.0,
            soc=self._battery.soc,
            fuel_used_l=fuel_before - self._generator.fuel_level_l,
            fuel_level_l=self._generator.fuel_level_l,
            grid_cost=grid.cost,
            generator_cost=generator.cost,
            battery_cost=battery_cost,
            emissions_kg=grid.emissions_kg + generator.emissions_kg,
            reasons=plan.reasons,
        )

    def __repr__(self) -> str:
        return f"BaseStation(name={self._name!r}, battery_soc={self._battery.soc:.3f})"
