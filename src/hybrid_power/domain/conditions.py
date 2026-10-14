"""The environment a base station experiences during one simulation step."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from hybrid_power._validation import require_fraction, require_non_negative, require_positive
from hybrid_power.exceptions import ProfileDataError


@dataclass(frozen=True)
class OperatingConditions:
    """Exogenous inputs for one time step.

    Attributes:
        index: Zero-based position of the step in the simulation horizon.
        timestamp: Start time of the step.
        demand_kw: Average site demand during the step.
        solar_factor: PV output as a fraction of rated capacity (0-1).
        grid_available: Whether the utility grid is energised.
        tariff_per_kwh: Grid energy price during the step.
        step_hours: Step length in hours.

    """

    index: int
    timestamp: datetime
    demand_kw: float
    solar_factor: float
    grid_available: bool
    tariff_per_kwh: float
    step_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ProfileDataError(f"index must be >= 0, got {self.index}")
        require_non_negative("demand_kw", self.demand_kw, ProfileDataError)
        require_fraction("solar_factor", self.solar_factor, ProfileDataError)
        require_non_negative("tariff_per_kwh", self.tariff_per_kwh, ProfileDataError)
        require_positive("step_hours", self.step_hours, ProfileDataError)

    @property
    def demand_kwh(self) -> float:
        """Energy the site consumes during the step."""
        return self.demand_kw * self.step_hours

    @property
    def hour_of_day(self) -> int:
        """Clock hour (0-23) at the start of the step."""
        return self.timestamp.hour
