"""Simulate every site of a fleet and collect per-site metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from hybrid_power._validation import require_int_in_range
from hybrid_power.analytics.comparison import RELIABILITY_TOLERANCE_KWH
from hybrid_power.analytics.metrics import PerformanceMetrics
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.fleet.generator import SiteSpec
from hybrid_power.service import SimulationService

FloatArray = npt.NDArray[np.float64]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class SiteOutcome:
    """Site characteristics and the metrics of every strategy at that site.

    Only aggregated metrics are kept (not hourly records), so large fleets
    stay light in memory.
    """

    site_id: str
    load_scale: float
    grid_availability: float
    outage_hours: float
    average_load_kw: float
    metrics: Mapping[str, PerformanceMetrics]

    def cost_savings_pct(self, candidate: str, baseline: str) -> float:
        """Cost saving of ``candidate`` relative to ``baseline`` in percent."""
        base = self.metrics[baseline].total_cost
        if base <= 0:
            return 0.0
        return (base - self.metrics[candidate].total_cost) / base * 100.0

    def fuel_savings_l(self, candidate: str, baseline: str) -> float:
        """Diesel saved by ``candidate`` relative to ``baseline``."""
        return self.metrics[baseline].fuel_used_l - self.metrics[candidate].fuel_used_l

    def beats(self, candidate: str, baseline: str) -> bool:
        """Cheaper than the baseline without serving less load."""
        cand, base = self.metrics[candidate], self.metrics[baseline]
        return (
            cand.total_cost < base.total_cost
            and cand.unmet_energy_kwh <= base.unmet_energy_kwh + RELIABILITY_TOLERANCE_KWH
        )


class FleetResult:
    """Outcomes for an ordered fleet of sites."""

    def __init__(self, outcomes: Sequence[SiteOutcome], baseline: str = "baseline") -> None:
        if not outcomes:
            raise ConfigurationError("A fleet result needs at least one site")
        strategies = set(outcomes[0].metrics)
        if baseline not in strategies:
            raise ConfigurationError(f"Baseline strategy {baseline!r} has no metrics")
        if any(set(outcome.metrics) != strategies for outcome in outcomes):
            raise ConfigurationError("Every site must be simulated with the same strategies")
        self._outcomes = tuple(outcomes)
        self._baseline = baseline

    @property
    def outcomes(self) -> tuple[SiteOutcome, ...]:
        """Per-site outcomes in sampling order."""
        return self._outcomes

    @property
    def baseline(self) -> str:
        """Reference strategy name."""
        return self._baseline

    @property
    def strategy_names(self) -> list[str]:
        """Simulated strategies (baseline first)."""
        names = list(self._outcomes[0].metrics)
        return [self._baseline] + [name for name in names if name != self._baseline]

    @property
    def candidates(self) -> list[str]:
        """Strategies other than the baseline."""
        return self.strategy_names[1:]

    def __len__(self) -> int:
        return len(self._outcomes)

    def subset(self, size: int) -> FleetResult:
        """The first ``size`` sites as a smaller fleet."""
        require_int_in_range("size", size, 1, len(self))
        return FleetResult(self._outcomes[:size], self._baseline)

    def savings_pct(self, candidate: str) -> FloatArray:
        """Per-site cost saving (%) of ``candidate`` versus the baseline."""
        self._check(candidate)
        return np.array(
            [o.cost_savings_pct(candidate, self._baseline) for o in self._outcomes],
            dtype=np.float64,
        )

    def fuel_savings_l(self, candidate: str) -> FloatArray:
        """Per-site diesel saved (L) by ``candidate`` versus the baseline."""
        self._check(candidate)
        return np.array(
            [o.fuel_savings_l(candidate, self._baseline) for o in self._outcomes],
            dtype=np.float64,
        )

    def win_rate(self, candidate: str) -> float:
        """Share of sites where ``candidate`` beats the baseline."""
        self._check(candidate)
        wins = sum(o.beats(candidate, self._baseline) for o in self._outcomes)
        return wins / len(self._outcomes)

    def best_strategy(self) -> str:
        """Cheapest strategy over the fleet that is at least as reliable as the baseline."""
        totals = {
            name: (
                sum(o.metrics[name].unmet_energy_kwh for o in self._outcomes),
                sum(o.metrics[name].total_cost for o in self._outcomes),
            )
            for name in self.strategy_names
        }
        baseline_unmet = totals[self._baseline][0]
        eligible = [
            name
            for name, (unmet, _) in totals.items()
            if unmet <= baseline_unmet + RELIABILITY_TOLERANCE_KWH * len(self._outcomes)
        ]
        return min(eligible, key=lambda name: totals[name][1])

    def to_frame(self) -> pd.DataFrame:
        """Long table: one row per site and strategy."""
        rows = [
            {
                "site_id": o.site_id,
                "strategy": name,
                "load_scale": o.load_scale,
                "grid_availability": o.grid_availability,
                "outage_hours": o.outage_hours,
                "average_load_kw": o.average_load_kw,
                "total_cost": m.total_cost,
                "cost_savings_pct": o.cost_savings_pct(name, self._baseline),
                "fuel_used_l": m.fuel_used_l,
                "generator_hours": m.generator_hours,
                "emissions_kg": m.emissions_kg,
                "renewable_fraction": m.renewable_fraction,
                "unmet_energy_kwh": m.unmet_energy_kwh,
            }
            for o in self._outcomes
            for name, m in o.metrics.items()
        ]
        frame: pd.DataFrame = pd.DataFrame(rows)
        return frame

    def _check(self, candidate: str) -> None:
        if candidate not in self._outcomes[0].metrics:
            raise ConfigurationError(f"No results for strategy {candidate!r}")


class FleetSimulator:
    """Run the full strategy comparison at every site of a fleet."""

    def __init__(self, service: SimulationService | None = None) -> None:
        self._service = service or SimulationService()

    def run(
        self,
        sites: Sequence[SiteSpec],
        *,
        baseline: str = "baseline",
        strategies: Sequence[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> FleetResult:
        """Simulate ``sites`` and return their outcomes in order."""
        if not sites:
            raise ConfigurationError("The fleet contains no sites")
        outcomes: list[SiteOutcome] = []
        for done, site in enumerate(sites, start=1):
            outcome = self._service.run(site.config, strategies=strategies, baseline=baseline)
            profile = outcome.profile
            outcomes.append(
                SiteOutcome(
                    site_id=site.site_id,
                    load_scale=site.load_scale,
                    grid_availability=profile.grid_availability_ratio,
                    outage_hours=profile.outage_hours,
                    average_load_kw=profile.load.average_kw,
                    metrics=dict(outcome.report.metrics),
                )
            )
            if progress is not None:
                progress(done, len(sites))
        return FleetResult(outcomes, baseline=baseline)
