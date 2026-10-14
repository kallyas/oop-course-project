"""High-level facade used by the CLI and the dashboard.

The facade hides the wiring of scenario, registry, engine and analytics
behind one call, so user-interface code stays thin and free of domain logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hybrid_power.analytics.comparison import ComparisonReport
from hybrid_power.config import ScenarioConfig
from hybrid_power.data.profiles import ProfileProvider, ProfileSet
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.optimization.registry import StrategyRegistry
from hybrid_power.simulation.engine import (
    Scenario,
    SimulationEngine,
    SimulationObserver,
    SimulationResult,
)


@dataclass(frozen=True)
class SimulationOutcome:
    """Everything produced by one comparative simulation."""

    config: ScenarioConfig
    profile: ProfileSet
    results: dict[str, SimulationResult]
    report: ComparisonReport


class SimulationService:
    """Run a set of strategies on a scenario and analyse the results."""

    def __init__(self, registry: StrategyRegistry | None = None) -> None:
        self._registry = registry or StrategyRegistry.default()

    @property
    def registry(self) -> StrategyRegistry:
        """Available strategies."""
        return self._registry

    def run(
        self,
        config: ScenarioConfig,
        *,
        provider: ProfileProvider | None = None,
        strategies: Sequence[str] | None = None,
        baseline: str = "baseline",
        observers: Iterable[SimulationObserver] = (),
    ) -> SimulationOutcome:
        """Simulate ``strategies`` (default: all registered) and compare them.

        The baseline strategy is always included so savings can be reported.
        """
        keys = list(strategies) if strategies else self._registry.keys()
        if baseline not in self._registry:
            raise ConfigurationError(f"Baseline strategy {baseline!r} is not registered")
        if baseline not in keys:
            keys.insert(0, baseline)
        scenario = Scenario(config, provider)
        engine = SimulationEngine(scenario, observers)
        results = engine.run_all(self._registry.create(key, config) for key in keys)
        return SimulationOutcome(
            config=config,
            profile=scenario.profile,
            results=results,
            report=ComparisonReport(results, baseline=baseline),
        )
