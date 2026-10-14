"""Reproducible synthetic profile generation with realistic daily patterns."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from hybrid_power.config import ProfileConfig
from hybrid_power.data.profiles import ProfileProvider, ProfileSet
from hybrid_power.domain.load import SiteLoad

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

#: Normalised hourly mobile-traffic shape (0 = quietest, 1 = busiest hour).
#: Traffic is lowest before dawn and peaks in the evening, which drives the
#: radio equipment's power draw.
TRAFFIC_SHAPE: tuple[float, ...] = (
    0.30, 0.22, 0.16, 0.12, 0.12, 0.18, 0.32, 0.50, 0.64, 0.72, 0.76, 0.80,
    0.84, 0.82, 0.78, 0.78, 0.82, 0.88, 0.95, 1.00, 1.00, 0.92, 0.72, 0.48,
)  # fmt: skip


class SyntheticProfileGenerator(ProfileProvider):
    """Generate hourly demand, solar, outage and tariff series from a seed.

    * **Demand** follows :data:`TRAFFIC_SHAPE` between base and peak load with
      Gaussian noise.
    * **Solar** follows a clear-sky half-sine between 06:00 and 18:00, scaled
      by a Beta-distributed daily clearness index (cloudy vs sunny days).
    * **Grid outages** follow a two-state Markov chain whose failure and
      repair probabilities reproduce the configured outage frequency and mean
      duration.
    * **Tariff** is a two-level time-of-use price.

    Identical configurations (including the seed) always produce identical
    profiles, which makes simulation results reproducible.
    """

    def __init__(self, config: ProfileConfig) -> None:
        self._config = config

    @property
    def config(self) -> ProfileConfig:
        """Generation parameters."""
        return self._config

    @override
    def get_profile(self) -> ProfileSet:
        cfg = self._config
        rng = np.random.default_rng(cfg.seed)
        timestamps = pd.date_range(start=pd.Timestamp(cfg.start_date), periods=cfg.hours, freq="h")
        hours: IntArray = timestamps.hour.to_numpy(dtype=np.int_)
        return ProfileSet(
            timestamps=timestamps,
            load=SiteLoad(self._demand(rng, hours)),
            solar_factor=self._solar(rng, hours),
            grid_available=self._grid(rng, cfg.hours),
            tariff_per_kwh=self._tariff(hours),
        )

    def _demand(self, rng: np.random.Generator, hours: IntArray) -> FloatArray:
        cfg = self._config
        shape = np.asarray(TRAFFIC_SHAPE, dtype=np.float64)[hours]
        demand = cfg.base_load_kw + (cfg.peak_load_kw - cfg.base_load_kw) * shape
        noise = rng.normal(0.0, cfg.load_noise, size=hours.size)
        result: FloatArray = np.clip(demand * (1.0 + noise), 0.0, None)
        return result

    def _solar(self, rng: np.random.Generator, hours: IntArray) -> FloatArray:
        daylight = (hours > 6) & (hours < 18)
        clear_sky = np.where(daylight, np.sin(np.pi * (hours - 6) / 12.0), 0.0)
        clearness = self._daily_clearness(rng)
        hourly_clearness = np.repeat(clearness, 24)[: hours.size]
        flicker = 1.0 + rng.normal(0.0, 0.05, size=hours.size)
        result: FloatArray = np.clip(clear_sky * hourly_clearness * flicker, 0.0, 1.0)
        return result

    def _daily_clearness(self, rng: np.random.Generator) -> FloatArray:
        cfg = self._config
        mean, variability = cfg.solar_mean_clearness, cfg.solar_variability
        if variability == 0 or mean >= 1.0:
            return np.full(cfg.days, mean, dtype=np.float64)
        # Beta(a, b) with the requested mean; lower concentration -> more spread.
        concentration = 1.0 + 4.0 * (1.0 - variability) / variability
        draws: FloatArray = rng.beta(mean * concentration, (1.0 - mean) * concentration, cfg.days)
        return draws

    def _grid(self, rng: np.random.Generator, steps: int) -> BoolArray:
        cfg = self._config
        failure_probability = min(1.0, cfg.outages_per_day / 24.0)
        repair_probability = 1.0 / cfg.mean_outage_hours
        draws = rng.random(steps)
        available = np.empty(steps, dtype=np.bool_)
        state = True
        for i in range(steps):
            if state and draws[i] < failure_probability:
                state = False
            elif not state and draws[i] < repair_probability:
                state = True
            available[i] = state
        return available

    def _tariff(self, hours: IntArray) -> FloatArray:
        cfg = self._config
        start, end = cfg.peak_start_hour, cfg.peak_end_hour
        if start <= end:
            in_peak = (hours >= start) & (hours < end)
        else:  # window wraps around midnight
            in_peak = (hours >= start) | (hours < end)
        tariff: FloatArray = np.where(in_peak, cfg.peak_tariff, cfg.offpeak_tariff).astype(
            np.float64
        )
        return tariff
