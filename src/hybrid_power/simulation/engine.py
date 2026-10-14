"""Hourly simulation engine that runs dispatch strategies against a scenario."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from functools import cached_property

import pandas as pd
from typing_extensions import override

from hybrid_power._validation import require_text
from hybrid_power.config import ScenarioConfig
from hybrid_power.data.profiles import ProfileProvider, ProfileSet
from hybrid_power.data.synthetic import SyntheticProfileGenerator
from hybrid_power.domain.dispatch import StepRecord
from hybrid_power.domain.station import BaseStation
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.optimization.strategies import DispatchStrategy

logger = logging.getLogger(__name__)


class Scenario:
    """A site configuration paired with the operating profile it will face.

    The profile is produced once, so every strategy is evaluated on exactly
    the same demand, weather, outage and tariff data. A fresh
    :class:`BaseStation` is built for each run, so runs never share state.
    """

    def __init__(self, config: ScenarioConfig, provider: ProfileProvider | None = None) -> None:
        self._config = config
        self._provider = provider or SyntheticProfileGenerator(config.profile)
        self._profile = self._provider.get_profile()

    @property
    def config(self) -> ScenarioConfig:
        """Site and environment configuration."""
        return self._config

    @property
    def profile(self) -> ProfileSet:
        """Operating profile shared by all runs."""
        return self._profile

    def build_station(self) -> BaseStation:
        """Create a new station in its initial state."""
        return BaseStation.from_config(self._config)


@dataclass(frozen=True)
class SimulationResult:
    """Immutable outcome of running one strategy over a scenario."""

    strategy_name: str
    strategy_description: str
    records: tuple[StepRecord, ...]
    currency: str = "USD"

    def __post_init__(self) -> None:
        require_text("strategy_name", self.strategy_name)
        if not self.records:
            raise ConfigurationError("A simulation result needs at least one step record")

    @cached_property
    def frame(self) -> pd.DataFrame:
        """Step records as a DataFrame indexed by timestamp."""
        frame: pd.DataFrame = pd.DataFrame([record.to_dict() for record in self.records])
        indexed: pd.DataFrame = frame.set_index("timestamp")
        return indexed

    def between(self, start: datetime, end: datetime) -> SimulationResult:
        """Return a result restricted to records with ``start <= timestamp < end``."""
        records = tuple(r for r in self.records if start <= r.timestamp < end)
        return replace(self, records=records)

    def __len__(self) -> int:
        return len(self.records)


class SimulationObserver:
    """Receives notifications as a simulation progresses (Observer pattern).

    All callbacks are optional no-ops so observers override only what they
    need.
    """

    def on_start(self, strategy: DispatchStrategy, total_steps: int) -> None:
        """Called once before the first step."""

    def on_step(self, record: StepRecord) -> None:
        """Called after every executed step."""

    def on_complete(self, result: SimulationResult) -> None:
        """Called once after the last step."""


class ProgressObserver(SimulationObserver):
    """Report fractional progress (0-1) to a callback, e.g. a progress bar."""

    def __init__(self, callback: Callable[[float, str], None], every: int = 24) -> None:
        if every < 1:
            raise ConfigurationError("every must be >= 1")
        self._callback = callback
        self._every = every
        self._total = 1
        self._label = ""

    @override
    def on_start(self, strategy: DispatchStrategy, total_steps: int) -> None:
        self._total = max(1, total_steps)
        self._label = strategy.name
        self._callback(0.0, self._label)

    @override
    def on_step(self, record: StepRecord) -> None:
        done = record.index + 1
        if done % self._every == 0 or done == self._total:
            self._callback(done / self._total, self._label)


class LoggingObserver(SimulationObserver):
    """Log unmet-demand events and a summary through :mod:`logging`."""

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logger

    @override
    def on_step(self, record: StepRecord) -> None:
        if record.unmet_kwh > 0:
            self._log.warning(
                "Step %d (%s): %.2f kWh unmet", record.index, record.timestamp, record.unmet_kwh
            )

    @override
    def on_complete(self, result: SimulationResult) -> None:
        self._log.info("Strategy %s finished %d steps", result.strategy_name, len(result))


class SimulationEngine:
    """Drive the step loop: housekeeping -> snapshot -> decide -> execute -> notify."""

    def __init__(self, scenario: Scenario, observers: Iterable[SimulationObserver] = ()) -> None:
        self._scenario = scenario
        self._observers: list[SimulationObserver] = list(observers)

    @property
    def scenario(self) -> Scenario:
        """Scenario being simulated."""
        return self._scenario

    def add_observer(self, observer: SimulationObserver) -> None:
        """Subscribe ``observer`` to future runs."""
        self._observers.append(observer)

    def run(self, strategy: DispatchStrategy) -> SimulationResult:
        """Simulate the whole horizon with ``strategy`` and return the result."""
        profile = self._scenario.profile
        station = self._scenario.build_station()
        strategy.reset()
        for observer in self._observers:
            observer.on_start(strategy, len(profile))

        records: list[StepRecord] = []
        for conditions in profile:
            station.begin_step(conditions)
            plan = strategy.decide(station.snapshot(conditions))
            record = station.execute(plan, conditions)
            records.append(record)
            for observer in self._observers:
                observer.on_step(record)

        result = SimulationResult(
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            records=tuple(records),
            currency=self._scenario.config.currency,
        )
        for observer in self._observers:
            observer.on_complete(result)
        return result

    def run_all(self, strategies: Iterable[DispatchStrategy]) -> dict[str, SimulationResult]:
        """Run several strategies on the same scenario, keyed by strategy name."""
        results: dict[str, SimulationResult] = {}
        for strategy in strategies:
            if strategy.name in results:
                raise ConfigurationError(f"Duplicate strategy name {strategy.name!r}")
            results[strategy.name] = self.run(strategy)
        return results
