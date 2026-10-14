"""Do conclusions drawn from a few sites hold for many? (fleet-size study)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from hybrid_power._validation import require_in_range
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.fleet.simulator import FleetResult
from hybrid_power.fleet.statistics import SummaryStatistics


def describe_fleets(sizes: Sequence[int]) -> str:
    """Name fleet sizes in plain English, e.g. "fleets of 1, 5 and 10 sites"."""
    if not sizes:
        raise ConfigurationError("sizes must not be empty")
    if len(sizes) == 1:
        size = sizes[0]
        return f"a fleet of {size} site{'' if size == 1 else 's'}"
    listed = ", ".join(str(size) for size in sizes[:-1])
    return f"fleets of {listed} and {sizes[-1]} sites"


@dataclass(frozen=True)
class FleetSizeSummary:
    """Statistics of one candidate strategy for one fleet size."""

    size: int
    savings_pct: SummaryStatistics
    fuel_savings_l: SummaryStatistics
    win_rate: float
    recommended_strategy: str


@dataclass(frozen=True)
class ConsistencyReport:
    """Results of the study, ordered from the smallest to the largest fleet.

    Two kinds of consistency are distinguished:

    * **direction** - every fleet size recommends the same strategy and the
      candidate wins at the majority of sites;
    * **magnitude** - every fleet's mean saving lies inside the confidence
      interval of the largest (reference) fleet.
    """

    candidate: str
    baseline: str
    summaries: tuple[FleetSizeSummary, ...]

    def __post_init__(self) -> None:
        if not self.summaries:
            raise ConfigurationError("A consistency report needs at least one fleet size")

    @property
    def reference(self) -> FleetSizeSummary:
        """The largest fleet, treated as the best estimate."""
        return self.summaries[-1]

    def agrees_with_reference(self, summary: FleetSizeSummary) -> bool:
        """Whether a fleet's mean saving lies inside the reference interval."""
        if summary is self.reference:
            return True
        return self.reference.savings_pct.contains(summary.savings_pct.mean)

    @property
    def direction_consistent(self) -> bool:
        """Same recommendation at every size and a majority of wins."""
        recommended = {s.recommended_strategy for s in self.summaries}
        return len(recommended) == 1 and all(s.win_rate > 0.5 for s in self.summaries)

    @property
    def magnitude_consistent(self) -> bool:
        """All fleet means within the reference confidence interval."""
        return all(self.agrees_with_reference(s) for s in self.summaries)

    @property
    def smallest_reliable_size(self) -> int | None:
        """Smallest size from which every larger fleet agrees with the reference."""
        result: int | None = None
        for summary in reversed(self.summaries):
            if not self.agrees_with_reference(summary):
                break
            result = summary.size
        return result

    def verdict(self) -> str:
        """Plain-language conclusion of the study."""
        ref = self.reference
        ci = ref.savings_pct
        interval = (
            f" (95% CI {ci.ci_low:.1f}% to {ci.ci_high:.1f}%)"
            if ci.ci_low is not None and ci.ci_high is not None
            else ""
        )
        lines = [
            f"With {ref.size} sites, {self.candidate} saves {ci.mean:.1f}% on average "
            f"versus {self.baseline}{interval} and wins at {ref.win_rate:.0%} of sites."
        ]
        if self.direction_consistent:
            lines.append(
                f"Direction is consistent: every fleet size recommends {ref.recommended_strategy}."
            )
        else:
            choices = ", ".join(f"{s.size}: {s.recommended_strategy}" for s in self.summaries)
            lines.append(f"Direction is NOT consistent across fleet sizes ({choices}).")
        if self.magnitude_consistent:
            lines.append("Magnitude is consistent: every fleet mean lies inside that interval.")
        else:
            outliers = [s.size for s in self.summaries if not self.agrees_with_reference(s)]
            reliable = self.smallest_reliable_size
            tail = f" Estimates are stable from {reliable} sites upward." if reliable else ""
            lines.append(
                f"Magnitude differs for {describe_fleets(outliers)}: small samples "
                f"misstate the size of the saving.{tail}"
            )
        return " ".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per fleet size."""
        rows = [
            {
                "sites": s.size,
                "mean_saving_pct": s.savings_pct.mean,
                "std_saving_pct": s.savings_pct.std,
                "ci_low_pct": s.savings_pct.ci_low,
                "ci_high_pct": s.savings_pct.ci_high,
                "min_saving_pct": s.savings_pct.minimum,
                "max_saving_pct": s.savings_pct.maximum,
                "mean_fuel_saved_l": s.fuel_savings_l.mean,
                "win_rate": s.win_rate,
                "recommended": s.recommended_strategy,
                "agrees_with_reference": self.agrees_with_reference(s),
            }
            for s in self.summaries
        ]
        frame: pd.DataFrame = pd.DataFrame(rows).set_index("sites")
        return frame


class ConsistencyStudy:
    """Compare a candidate strategy across nested fleets of increasing size."""

    def __init__(
        self,
        sizes: Sequence[int],
        candidate: str = "optimizer",
        *,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> None:
        if not sizes or list(sizes) != sorted(set(sizes)) or sizes[0] < 1:
            raise ConfigurationError("sizes must be positive and strictly increasing")
        self._sizes = tuple(sizes)
        self._candidate = candidate
        self._confidence = require_in_range("confidence", confidence, 0.5, 0.999)
        self._seed = seed

    @property
    def sizes(self) -> tuple[int, ...]:
        """Fleet sizes analysed."""
        return self._sizes

    def analyse(self, fleet: FleetResult) -> ConsistencyReport:
        """Summarise ``fleet`` at every configured size."""
        if self._sizes[-1] > len(fleet):
            raise ConfigurationError(
                f"Largest fleet size {self._sizes[-1]} exceeds the {len(fleet)} simulated sites"
            )
        summaries = tuple(self._summarise(fleet.subset(size)) for size in self._sizes)
        return ConsistencyReport(self._candidate, fleet.baseline, summaries)

    def _summarise(self, fleet: FleetResult) -> FleetSizeSummary:
        return FleetSizeSummary(
            size=len(fleet),
            savings_pct=SummaryStatistics.from_values(
                fleet.savings_pct(self._candidate), confidence=self._confidence, seed=self._seed
            ),
            fuel_savings_l=SummaryStatistics.from_values(
                fleet.fuel_savings_l(self._candidate), confidence=self._confidence, seed=self._seed
            ),
            win_rate=fleet.win_rate(self._candidate),
            recommended_strategy=fleet.best_strategy(),
        )
