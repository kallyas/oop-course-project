"""Registry of named strategy factories (Registry + Factory patterns)."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from hybrid_power._validation import require_text
from hybrid_power.config import ScenarioConfig
from hybrid_power.exceptions import ConfigurationError, StrategyNotFoundError
from hybrid_power.optimization.strategies import (
    DispatchStrategy,
    GridGeneratorBaseline,
    RuleBasedOptimizer,
    SolarFirstStrategy,
)

StrategyFactory = Callable[[ScenarioConfig], DispatchStrategy]


class StrategyRegistry:
    """Map strategy names to factories so new strategies plug in without code changes.

    Factories receive the scenario so strategies can read their parameters
    (for example the optimiser rules) from configuration.
    """

    def __init__(self) -> None:
        self._factories: dict[str, StrategyFactory] = {}

    def register(self, key: str, factory: StrategyFactory, *, overwrite: bool = False) -> None:
        """Register ``factory`` under ``key``."""
        key = require_text("key", key)
        if key in self._factories and not overwrite:
            raise ConfigurationError(f"Strategy {key!r} is already registered")
        self._factories[key] = factory

    def create(self, key: str, config: ScenarioConfig) -> DispatchStrategy:
        """Instantiate the strategy registered as ``key``."""
        try:
            factory = self._factories[key]
        except KeyError as exc:
            options = ", ".join(self.keys())
            raise StrategyNotFoundError(
                f"Unknown strategy {key!r}; choose from: {options}"
            ) from exc
        return factory(config)

    def create_all(self, config: ScenarioConfig) -> list[DispatchStrategy]:
        """Instantiate every registered strategy."""
        return [self.create(key, config) for key in self.keys()]

    def keys(self) -> list[str]:
        """Registered names in registration order."""
        return list(self._factories)

    def __contains__(self, key: object) -> bool:
        return key in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)

    @classmethod
    def default(cls) -> StrategyRegistry:
        """Registry pre-populated with the built-in strategies."""
        registry = cls()
        registry.register("baseline", lambda _config: GridGeneratorBaseline())
        registry.register("solar_first", lambda _config: SolarFirstStrategy())
        registry.register("optimizer", lambda config: RuleBasedOptimizer(config.optimizer))
        return registry
