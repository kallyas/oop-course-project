"""Descriptive statistics with bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hybrid_power._validation import require_in_range, require_int_in_range
from hybrid_power.exceptions import ConfigurationError


@dataclass(frozen=True)
class SummaryStatistics:
    """Mean, spread and a confidence interval of a sample.

    The interval is a *percentile bootstrap*: the sample is resampled with
    replacement many times and the interval is read from the distribution of
    resampled means. It needs no distributional assumption and no SciPy, and
    is undefined (``None``) for fewer than two observations.
    """

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    ci_low: float | None
    ci_high: float | None
    confidence: float

    @classmethod
    def from_values(
        cls,
        values: Sequence[float] | npt.NDArray[np.float64],
        *,
        confidence: float = 0.95,
        resamples: int = 2000,
        seed: int = 0,
    ) -> SummaryStatistics:
        """Summarise ``values`` (must be non-empty and finite)."""
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
            raise ConfigurationError("values must be a non-empty, finite 1-D sample")
        require_in_range("confidence", confidence, 0.5, 0.999)
        require_int_in_range("resamples", resamples, 100, 1_000_000)
        ci_low: float | None = None
        ci_high: float | None = None
        if data.size >= 2:
            rng = np.random.default_rng(seed)
            means = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
            tail = (1.0 - confidence) / 2.0 * 100.0
            ci_low = float(np.percentile(means, tail))
            ci_high = float(np.percentile(means, 100.0 - tail))
        return cls(
            count=int(data.size),
            mean=float(np.mean(data)),
            std=float(np.std(data, ddof=1)) if data.size >= 2 else 0.0,
            minimum=float(np.min(data)),
            maximum=float(np.max(data)),
            ci_low=ci_low,
            ci_high=ci_high,
            confidence=confidence,
        )

    @property
    def has_interval(self) -> bool:
        """Whether a confidence interval could be computed."""
        return self.ci_low is not None and self.ci_high is not None

    @property
    def ci_half_width(self) -> float | None:
        """Half the interval width: the +/- precision of the mean."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_high - self.ci_low) / 2.0

    @property
    def coefficient_of_variation(self) -> float | None:
        """Standard deviation relative to the mean (``None`` if the mean is ~0)."""
        if math.isclose(self.mean, 0.0, abs_tol=1e-12):
            return None
        return self.std / abs(self.mean)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the confidence interval."""
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low <= value <= self.ci_high
