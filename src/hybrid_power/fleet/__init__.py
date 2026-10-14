"""Fleet layer: simulate many base stations and test whether conclusions generalise."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_power.config import FleetConfig, ScenarioConfig
from hybrid_power.fleet.consistency import ConsistencyReport, ConsistencyStudy, FleetSizeSummary
from hybrid_power.fleet.generator import FleetGenerator, SiteSpec
from hybrid_power.fleet.simulator import FleetResult, FleetSimulator, ProgressCallback, SiteOutcome
from hybrid_power.fleet.statistics import SummaryStatistics


@dataclass(frozen=True)
class FleetStudyOutcome:
    """A simulated fleet together with its consistency analysis."""

    fleet: FleetResult
    report: ConsistencyReport


def run_fleet_study(
    template: ScenarioConfig,
    fleet_config: FleetConfig,
    *,
    candidate: str = "optimizer",
    progress: ProgressCallback | None = None,
) -> FleetStudyOutcome:
    """Generate, simulate and analyse a fleet in one call (facade for UI code)."""
    sites = FleetGenerator(template, fleet_config).generate()
    fleet = FleetSimulator().run(sites, progress=progress)
    report = ConsistencyStudy(fleet_config.fleet_sizes, candidate, seed=fleet_config.seed)
    return FleetStudyOutcome(fleet=fleet, report=report.analyse(fleet))


__all__ = [
    "ConsistencyReport",
    "ConsistencyStudy",
    "FleetGenerator",
    "FleetResult",
    "FleetSimulator",
    "FleetSizeSummary",
    "FleetStudyOutcome",
    "SiteOutcome",
    "SiteSpec",
    "SummaryStatistics",
    "run_fleet_study",
]
