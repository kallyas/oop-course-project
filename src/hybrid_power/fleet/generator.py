"""Sample a fleet of base stations around a template scenario."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hybrid_power._validation import require_int_in_range
from hybrid_power.config import FleetConfig, FloatRange, ScenarioConfig


@dataclass(frozen=True)
class SiteSpec:
    """One sampled site: its identifier, configuration and sampled load factor."""

    site_id: str
    config: ScenarioConfig
    load_scale: float


class FleetGenerator:
    """Create reproducible, randomly varied site configurations.

    Sites are drawn one after another from a single seeded random stream, so
    the first *n* sites of a large fleet are identical to a fleet of *n* sites.
    Smaller fleets are therefore nested samples of larger ones, which lets a
    consistency study compare fleet sizes without re-simulating any site.
    """

    def __init__(self, template: ScenarioConfig, fleet: FleetConfig) -> None:
        self._template = template
        self._fleet = fleet

    @property
    def template(self) -> ScenarioConfig:
        """Scenario whose equipment every site shares."""
        return self._template

    @property
    def fleet(self) -> FleetConfig:
        """Sampling ranges and fleet sizes."""
        return self._fleet

    def generate(self, count: int | None = None) -> list[SiteSpec]:
        """Return ``count`` sites (default: ``fleet.sites``)."""
        count = self._fleet.sites if count is None else count
        require_int_in_range("count", count, 1, 10_000)
        rng = np.random.default_rng(self._fleet.seed)
        return [self._sample_site(rng, index) for index in range(count)]

    def _sample_site(self, rng: np.random.Generator, index: int) -> SiteSpec:
        fleet, profile = self._fleet, self._template.profile
        load_scale = self._uniform(rng, fleet.load_scale)
        site_profile = replace(
            profile,
            base_load_kw=profile.base_load_kw * load_scale,
            peak_load_kw=profile.peak_load_kw * load_scale,
            outages_per_day=self._uniform(rng, fleet.outages_per_day),
            mean_outage_hours=self._uniform(rng, fleet.mean_outage_hours),
            solar_mean_clearness=self._uniform(rng, fleet.solar_mean_clearness),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        site_id = f"site-{index + 1:03d}"
        config = replace(
            self._template, name=f"{self._template.name} / {site_id}", profile=site_profile
        )
        return SiteSpec(site_id=site_id, config=config, load_scale=load_scale)

    @staticmethod
    def _uniform(rng: np.random.Generator, bounds: FloatRange) -> float:
        low, high = bounds
        return float(rng.uniform(low, high)) if high > low else float(low)
