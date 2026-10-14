"""Compare strategies against a baseline and rank them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from hybrid_power.analytics.metrics import MetricsCalculator, PerformanceMetrics
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.simulation.engine import SimulationResult

#: Unmet energy (kWh) below which two strategies are considered equally reliable.
RELIABILITY_TOLERANCE_KWH = 0.01


def _percent_change(baseline: float, candidate: float) -> float | None:
    """Reduction of ``candidate`` relative to ``baseline`` in percent (None if undefined)."""
    if baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


@dataclass(frozen=True)
class StrategyComparison:
    """Savings of a candidate strategy relative to the baseline."""

    baseline: PerformanceMetrics
    candidate: PerformanceMetrics

    @property
    def cost_savings(self) -> float:
        """Absolute operating-cost saving (positive = cheaper than baseline)."""
        return self.baseline.total_cost - self.candidate.total_cost

    @property
    def cost_savings_pct(self) -> float | None:
        """Relative operating-cost saving in percent."""
        return _percent_change(self.baseline.total_cost, self.candidate.total_cost)

    @property
    def fuel_savings_l(self) -> float:
        """Diesel saved in litres."""
        return self.baseline.fuel_used_l - self.candidate.fuel_used_l

    @property
    def fuel_savings_pct(self) -> float | None:
        """Diesel saved in percent."""
        return _percent_change(self.baseline.fuel_used_l, self.candidate.fuel_used_l)

    @property
    def emissions_reduction_kg(self) -> float:
        """CO2 avoided in kilograms."""
        return self.baseline.emissions_kg - self.candidate.emissions_kg

    @property
    def emissions_reduction_pct(self) -> float | None:
        """CO2 avoided in percent."""
        return _percent_change(self.baseline.emissions_kg, self.candidate.emissions_kg)

    @property
    def generator_hours_reduction(self) -> float:
        """Generator run-hours avoided."""
        return self.baseline.generator_hours - self.candidate.generator_hours

    @property
    def unmet_energy_change_kwh(self) -> float:
        """Change in unmet energy (positive = candidate is less reliable)."""
        return self.candidate.unmet_energy_kwh - self.baseline.unmet_energy_kwh

    @property
    def is_improvement(self) -> bool:
        """Cheaper than the baseline without serving less of the load."""
        return self.cost_savings > 0 and self.unmet_energy_change_kwh <= RELIABILITY_TOLERANCE_KWH


class ComparisonReport:
    """Metrics for several strategies plus comparisons against a baseline."""

    def __init__(
        self,
        results: Mapping[str, SimulationResult],
        baseline: str = "baseline",
        calculator: MetricsCalculator | None = None,
    ) -> None:
        if baseline not in results:
            raise ConfigurationError(f"Baseline strategy {baseline!r} has no result")
        calculator = calculator or MetricsCalculator()
        self._baseline = baseline
        self._metrics: dict[str, PerformanceMetrics] = {
            name: calculator.calculate(result) for name, result in results.items()
        }

    @property
    def baseline_name(self) -> str:
        """Name of the reference strategy."""
        return self._baseline

    @property
    def metrics(self) -> Mapping[str, PerformanceMetrics]:
        """Metrics keyed by strategy name."""
        return dict(self._metrics)

    def comparison(self, candidate: str) -> StrategyComparison:
        """Compare ``candidate`` with the baseline."""
        if candidate not in self._metrics:
            raise ConfigurationError(f"No result for strategy {candidate!r}")
        return StrategyComparison(self._metrics[self._baseline], self._metrics[candidate])

    def comparisons(self) -> dict[str, StrategyComparison]:
        """Comparisons for every non-baseline strategy."""
        return {name: self.comparison(name) for name in self._metrics if name != self._baseline}

    def ranking(self) -> list[PerformanceMetrics]:
        """Strategies ordered by reliability first, then operating cost."""
        return sorted(
            self._metrics.values(),
            key=lambda m: (round(m.unmet_energy_kwh / RELIABILITY_TOLERANCE_KWH), m.total_cost),
        )

    def best(self) -> PerformanceMetrics:
        """The most reliable, then cheapest, strategy."""
        return self.ranking()[0]

    def to_frame(self) -> pd.DataFrame:
        """Metrics table with one row per strategy."""
        frame: pd.DataFrame = pd.DataFrame([m.to_dict() for m in self._metrics.values()])
        indexed: pd.DataFrame = frame.set_index("strategy_name")
        return indexed
