"""Key performance indicators computed from a simulation result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hybrid_power.simulation.engine import SimulationResult


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` or 0 when the denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class PerformanceMetrics:
    """Aggregated technical, economic and environmental outcomes of one run."""

    strategy_name: str
    currency: str
    total_cost: float
    grid_cost: float
    generator_cost: float
    battery_cost: float
    energy_demand_kwh: float
    energy_served_kwh: float
    unmet_energy_kwh: float
    unmet_hours: int
    solar_used_kwh: float
    solar_curtailed_kwh: float
    grid_energy_kwh: float
    generator_energy_kwh: float
    generator_dumped_kwh: float
    battery_throughput_kwh: float
    fuel_used_l: float
    generator_hours: float
    generator_starts: int
    emissions_kg: float
    average_soc: float
    minimum_soc: float

    @property
    def loss_of_power_supply_probability(self) -> float:
        """LPSP: share of demand that was not served (0 = fully reliable)."""
        return _ratio(self.unmet_energy_kwh, self.energy_demand_kwh)

    @property
    def renewable_fraction(self) -> float:
        """Share of generated energy that came from solar."""
        generated = self.solar_used_kwh + self.grid_energy_kwh + self.generator_energy_kwh
        return _ratio(self.solar_used_kwh, generated)

    @property
    def cost_per_kwh_served(self) -> float:
        """Levelised operating cost of the energy actually delivered."""
        return _ratio(self.total_cost, self.energy_served_kwh)

    def to_dict(self) -> dict[str, float | int | str]:
        """Flatten the metrics, including derived indicators, for tabular display."""
        data: dict[str, float | int | str] = asdict(self)
        data["renewable_fraction"] = self.renewable_fraction
        data["lpsp"] = self.loss_of_power_supply_probability
        data["cost_per_kwh_served"] = self.cost_per_kwh_served
        return data


class MetricsCalculator:
    """Compute :class:`PerformanceMetrics` from a :class:`SimulationResult`."""

    def calculate(self, result: SimulationResult) -> PerformanceMetrics:
        """Aggregate the step records of ``result``."""
        frame = result.frame
        generator_output = frame["generator_to_load"] + frame["generator_to_battery"]
        generator_total = generator_output + frame["generator_dumped"]
        running = generator_total > 0
        starts = int((running & ~running.shift(1, fill_value=False)).sum())
        step_hours = self._step_hours(result)

        return PerformanceMetrics(
            strategy_name=result.strategy_name,
            currency=result.currency,
            total_cost=float(frame["total_cost"].sum()),
            grid_cost=float(frame["grid_cost"].sum()),
            generator_cost=float(frame["generator_cost"].sum()),
            battery_cost=float(frame["battery_cost"].sum()),
            energy_demand_kwh=float(frame["demand_kwh"].sum()),
            energy_served_kwh=float((frame["demand_kwh"] - frame["unmet_kwh"]).sum()),
            unmet_energy_kwh=float(frame["unmet_kwh"].sum()),
            unmet_hours=int((frame["unmet_kwh"] > 1e-6).sum()),
            solar_used_kwh=float((frame["solar_to_load"] + frame["solar_to_battery"]).sum()),
            solar_curtailed_kwh=float(frame["solar_curtailed"].sum()),
            grid_energy_kwh=float((frame["grid_to_load"] + frame["grid_to_battery"]).sum()),
            generator_energy_kwh=float(generator_output.sum()),
            generator_dumped_kwh=float(frame["generator_dumped"].sum()),
            battery_throughput_kwh=float(
                (
                    frame["battery_to_load"]
                    + frame["solar_to_battery"]
                    + frame["grid_to_battery"]
                    + frame["generator_to_battery"]
                ).sum()
            ),
            fuel_used_l=float(frame["fuel_used_l"].sum()),
            generator_hours=float(running.sum()) * step_hours,
            generator_starts=starts,
            emissions_kg=float(frame["emissions_kg"].sum()),
            average_soc=float(frame["soc"].mean()),
            minimum_soc=float(frame["soc"].min()),
        )

    @staticmethod
    def _step_hours(result: SimulationResult) -> float:
        records = result.records
        if len(records) < 2:
            return 1.0
        return (records[1].timestamp - records[0].timestamp).total_seconds() / 3600.0
