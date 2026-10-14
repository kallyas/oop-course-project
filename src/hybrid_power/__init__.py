"""Object-oriented hybrid power optimisation system for telecom base stations.

The package is organised in layers, each depending only on the ones below it::

    presentation (dashboard, charts, CLI)
        -> fleet (multi-site generation, simulation, consistency study)
        -> service facade
        -> analytics (metrics, comparison)
        -> simulation (scenario, engine, observers)
        -> optimization (strategies, plan builder, registry)
        -> data (profiles, synthetic generator, CSV repository)
        -> domain (sources, battery, load, station)
        -> config / exceptions
"""

from hybrid_power.analytics import (
    ComparisonReport,
    MetricsCalculator,
    PerformanceMetrics,
    StrategyComparison,
)
from hybrid_power.config import (
    BatteryConfig,
    FleetConfig,
    GeneratorConfig,
    GeneratorMode,
    GridConfig,
    OptimizerConfig,
    ProfileConfig,
    ScenarioConfig,
    SolarConfig,
)
from hybrid_power.data import CsvProfileRepository, ProfileSet, SyntheticProfileGenerator
from hybrid_power.domain import (
    BaseStation,
    Battery,
    GeneratorSource,
    GridSource,
    PowerSource,
    SiteLoad,
    SolarSource,
)
from hybrid_power.exceptions import HybridPowerError
from hybrid_power.optimization import (
    DispatchStrategy,
    GridGeneratorBaseline,
    PowerOptimizer,
    RuleBasedOptimizer,
    SolarFirstStrategy,
    StrategyRegistry,
)
from hybrid_power.service import SimulationOutcome, SimulationService
from hybrid_power.simulation import Scenario, SimulationEngine, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "BaseStation",
    "Battery",
    "BatteryConfig",
    "ComparisonReport",
    "CsvProfileRepository",
    "DispatchStrategy",
    "FleetConfig",
    "GeneratorConfig",
    "GeneratorMode",
    "GeneratorSource",
    "GridConfig",
    "GridGeneratorBaseline",
    "GridSource",
    "HybridPowerError",
    "MetricsCalculator",
    "OptimizerConfig",
    "PerformanceMetrics",
    "PowerOptimizer",
    "PowerSource",
    "ProfileConfig",
    "ProfileSet",
    "RuleBasedOptimizer",
    "Scenario",
    "ScenarioConfig",
    "SimulationEngine",
    "SimulationOutcome",
    "SimulationResult",
    "SimulationService",
    "SiteLoad",
    "SolarConfig",
    "SolarFirstStrategy",
    "SolarSource",
    "StrategyComparison",
    "StrategyRegistry",
    "SyntheticProfileGenerator",
    "__version__",
]
