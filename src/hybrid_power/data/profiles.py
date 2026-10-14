"""Time-series container describing the operating environment of a site."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from hybrid_power._validation import require_positive
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.domain.load import SiteLoad
from hybrid_power.exceptions import ProfileDataError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Columns required in a tabular profile, in canonical order.
PROFILE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "demand_kw",
    "solar_factor",
    "grid_available",
    "tariff_per_kwh",
)


def _frozen_float(name: str, values: Sequence[float] | FloatArray, size: int) -> FloatArray:
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProfileDataError(f"{name} must be numeric: {exc}") from exc
    if array.shape != (size,):
        raise ProfileDataError(f"{name} has {array.size} values, expected {size}")
    if not np.all(np.isfinite(array)):
        raise ProfileDataError(f"{name} contains NaN or infinite values")
    array.setflags(write=False)
    return array


class ProfileSet:
    """Aligned hourly series of demand, solar resource, grid status and tariff.

    The class is immutable: every array is copied and frozen on construction,
    so a profile can be shared safely by several simulation runs.
    """

    def __init__(
        self,
        timestamps: pd.DatetimeIndex,
        load: SiteLoad,
        solar_factor: Sequence[float] | FloatArray,
        grid_available: Sequence[bool] | BoolArray,
        tariff_per_kwh: Sequence[float] | FloatArray,
        step_hours: float = 1.0,
    ) -> None:
        size = len(load)
        if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) != size:
            raise ProfileDataError(f"timestamps must be a DatetimeIndex with {size} entries")
        if size > 1 and not timestamps.is_monotonic_increasing:
            raise ProfileDataError("timestamps must be in increasing order")
        self._timestamps = timestamps
        self._load = load
        self._solar = _frozen_float("solar_factor", solar_factor, size)
        if np.any((self._solar < 0) | (self._solar > 1)):
            raise ProfileDataError("solar_factor values must lie between 0 and 1")
        grid = np.array(grid_available, dtype=np.bool_)
        if grid.shape != (size,):
            raise ProfileDataError(f"grid_available has {grid.size} values, expected {size}")
        grid.setflags(write=False)
        self._grid: BoolArray = grid
        self._tariff = _frozen_float("tariff_per_kwh", tariff_per_kwh, size)
        if np.any(self._tariff < 0):
            raise ProfileDataError("tariff_per_kwh values must be non-negative")
        self._step_hours = require_positive("step_hours", step_hours, ProfileDataError)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Start time of every step."""
        return self._timestamps

    @property
    def load(self) -> SiteLoad:
        """Site demand series."""
        return self._load

    @property
    def solar_factor(self) -> FloatArray:
        """PV output as a fraction of rated capacity."""
        return self._solar

    @property
    def grid_available(self) -> BoolArray:
        """Grid status per step."""
        return self._grid

    @property
    def tariff_per_kwh(self) -> FloatArray:
        """Grid price per step."""
        return self._tariff

    @property
    def step_hours(self) -> float:
        """Length of each step in hours."""
        return self._step_hours

    @property
    def grid_availability_ratio(self) -> float:
        """Fraction of steps with the grid energised."""
        return float(self._grid.mean())

    @property
    def outage_hours(self) -> float:
        """Total hours without grid supply."""
        return float((~self._grid).sum()) * self._step_hours

    def conditions_at(self, index: int) -> OperatingConditions:
        """Return the operating conditions of step ``index``."""
        if not 0 <= index < len(self):
            raise ProfileDataError(f"Step {index} is outside the profile of {len(self)} steps")
        return OperatingConditions(
            index=index,
            timestamp=self._timestamps[index].to_pydatetime(),
            demand_kw=float(self._load.values[index]),
            solar_factor=float(self._solar[index]),
            grid_available=bool(self._grid[index]),
            tariff_per_kwh=float(self._tariff[index]),
            step_hours=self._step_hours,
        )

    def __len__(self) -> int:
        return len(self._load)

    def __iter__(self) -> Iterator[OperatingConditions]:
        return (self.conditions_at(i) for i in range(len(self)))

    # ------------------------------------------------------------------ #
    # Tabular conversion
    # ------------------------------------------------------------------ #
    def to_frame(self) -> pd.DataFrame:
        """Return the profile as a DataFrame with :data:`PROFILE_COLUMNS`."""
        frame: pd.DataFrame = pd.DataFrame(
            {
                "timestamp": self._timestamps,
                "demand_kw": self._load.values,
                "solar_factor": self._solar,
                "grid_available": self._grid,
                "tariff_per_kwh": self._tariff,
            }
        )
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> ProfileSet:
        """Build a profile from a DataFrame containing :data:`PROFILE_COLUMNS`.

        The step length is inferred from the spacing of the timestamps.
        """
        missing = [column for column in PROFILE_COLUMNS if column not in frame.columns]
        if missing:
            raise ProfileDataError(f"Profile is missing column(s): {', '.join(missing)}")
        if frame.empty:
            raise ProfileDataError("Profile contains no rows")
        try:
            timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
        except (ValueError, TypeError) as exc:
            raise ProfileDataError(f"Invalid timestamp values: {exc}") from exc
        step_hours = cls._infer_step_hours(timestamps)
        return cls(
            timestamps=timestamps,
            load=SiteLoad(frame["demand_kw"].to_numpy(dtype=np.float64)),
            solar_factor=frame["solar_factor"].to_numpy(dtype=np.float64),
            grid_available=_parse_bool_column(frame["grid_available"]),
            tariff_per_kwh=frame["tariff_per_kwh"].to_numpy(dtype=np.float64),
            step_hours=step_hours,
        )

    @staticmethod
    def _infer_step_hours(timestamps: pd.DatetimeIndex) -> float:
        if len(timestamps) < 2:
            return 1.0
        deltas = np.diff(timestamps.to_numpy()) / np.timedelta64(1, "h")
        if not np.allclose(deltas, deltas[0]) or deltas[0] <= 0:
            raise ProfileDataError("timestamps must be evenly spaced and increasing")
        return float(deltas[0])

    def __repr__(self) -> str:
        return (
            f"ProfileSet(steps={len(self)}, step_hours={self._step_hours}, "
            f"grid_availability={self.grid_availability_ratio:.1%})"
        )


_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off"})


def _parse_bool_column(column: pd.Series) -> BoolArray:
    """Interpret booleans written as bool, 0/1 or common true/false strings."""
    if column.dtype == np.bool_:
        return column.to_numpy(dtype=np.bool_)
    parsed: list[bool] = []
    for raw in column.astype(str).str.strip().str.lower():
        if raw in _TRUE_STRINGS:
            parsed.append(True)
        elif raw in _FALSE_STRINGS:
            parsed.append(False)
        else:
            raise ProfileDataError(f"grid_available value {raw!r} is not a boolean")
    return np.array(parsed, dtype=np.bool_)


class ProfileProvider(ABC):
    """Abstract source of a :class:`ProfileSet` (synthetic, file, API, ...)."""

    @abstractmethod
    def get_profile(self) -> ProfileSet:
        """Produce the operating profile."""
