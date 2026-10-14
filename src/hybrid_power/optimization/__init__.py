"""Optimisation layer: dispatch strategies and supporting infrastructure."""

from hybrid_power.optimization.builder import PlanBuilder
from hybrid_power.optimization.registry import StrategyFactory, StrategyRegistry
from hybrid_power.optimization.strategies import (
    DispatchStrategy,
    GridGeneratorBaseline,
    RuleBasedOptimizer,
    SolarFirstStrategy,
)

#: The optimiser is called ``PowerOptimizer`` in the project proposal.
PowerOptimizer = RuleBasedOptimizer

__all__ = [
    "DispatchStrategy",
    "GridGeneratorBaseline",
    "PlanBuilder",
    "PowerOptimizer",
    "RuleBasedOptimizer",
    "SolarFirstStrategy",
    "StrategyFactory",
    "StrategyRegistry",
]
