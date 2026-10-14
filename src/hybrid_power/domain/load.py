"""Site demand model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from hybrid_power._validation import require_non_negative, require_text
from hybrid_power.exceptions import ProfileDataError

FloatArray = npt.NDArray[np.float64]


class SiteLoad:
    """Immutable hourly electrical demand of a base station (kW)."""

    def __init__(
        self, demand_kw: Sequence[float] | FloatArray, name: str = "Base station load"
    ) -> None:
        self._name = require_text("name", name)
        try:
            values = np.array(demand_kw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ProfileDataError(f"Demand values must be numeric: {exc}") from exc
        if values.ndim != 1 or values.size == 0:
            raise ProfileDataError("Demand must be a non-empty one-dimensional series")
        if not np.all(np.isfinite(values)):
            raise ProfileDataError("Demand contains NaN or infinite values")
        if np.any(values < 0):
            raise ProfileDataError("Demand values must be non-negative")
        values.setflags(write=False)
        self._values: FloatArray = values

    @property
    def name(self) -> str:
        """Descriptive name of the load."""
        return self._name

    @property
    def values(self) -> FloatArray:
        """Read-only array of hourly demand (kW)."""
        return self._values

    @property
    def peak_kw(self) -> float:
        """Maximum demand."""
        return float(np.max(self._values))

    @property
    def average_kw(self) -> float:
        """Mean demand."""
        return float(self._values.mean())

    @property
    def minimum_kw(self) -> float:
        """Minimum demand."""
        return float(np.min(self._values))

    def demand_at(self, index: int) -> float:
        """Demand during step ``index``."""
        if not 0 <= index < len(self):
            raise ProfileDataError(f"Step {index} is outside the load horizon of {len(self)}")
        return float(self._values[index])

    def total_energy_kwh(self, step_hours: float = 1.0) -> float:
        """Total energy demand over the horizon."""
        return float(self._values.sum()) * step_hours

    def scaled(self, factor: float) -> SiteLoad:
        """Return a new load with every value multiplied by ``factor``."""
        factor = require_non_negative("factor", factor)
        return SiteLoad(self._values * factor, self._name)

    def __len__(self) -> int:
        return int(self._values.size)

    def __iter__(self) -> Iterator[float]:
        return (float(value) for value in self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SiteLoad):
            return NotImplemented
        return bool(np.array_equal(self._values, other._values))

    def __hash__(self) -> int:
        return hash(self._values.tobytes())

    def __repr__(self) -> str:
        return f"SiteLoad(steps={len(self)}, peak_kw={self.peak_kw:.2f})"
