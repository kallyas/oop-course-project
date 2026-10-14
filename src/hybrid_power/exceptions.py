"""Exception hierarchy for the hybrid power optimisation system.

Every error raised deliberately by the package derives from
:class:`HybridPowerError`, so callers such as the CLI and the dashboard can
catch one base type and present a friendly message, while tests can assert on
the precise subclass.
"""

from __future__ import annotations


class HybridPowerError(Exception):
    """Base class for all errors raised by :mod:`hybrid_power`."""


class ConfigurationError(HybridPowerError, ValueError):
    """A configuration value or constructor argument is invalid."""


class ProfileDataError(HybridPowerError, ValueError):
    """A time-series profile (generated or loaded from file) is malformed."""


class OperatingLimitError(HybridPowerError):
    """A physical operating limit of a component would be violated."""


class CapacityExceededError(OperatingLimitError):
    """More energy was requested than a component can deliver or absorb."""


class DispatchError(HybridPowerError):
    """A dispatch plan is internally inconsistent or infeasible for the site."""


class StrategyNotFoundError(HybridPowerError, KeyError):
    """A dispatch strategy name is not registered."""
