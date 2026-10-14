"""Reusable guard functions that enforce value invariants.

Centralising validation keeps constructors short and guarantees that every
class reports invalid input with the same exception type and message style.
Each guard returns the validated value (normalised to ``float``/``int``) so it
can be used inline in assignments.
"""

from __future__ import annotations

import math
import numbers

from hybrid_power.exceptions import ConfigurationError, HybridPowerError

#: Numerical tolerance (kWh) used when comparing energy quantities.
ENERGY_TOLERANCE = 1e-6

ErrorType = type[HybridPowerError]


def require_finite(name: str, value: object, error: ErrorType = ConfigurationError) -> float:
    """Return ``value`` as a float, rejecting non-numeric, NaN and infinite input."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise error(f"{name} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{name} must be finite, got {value!r}")
    return result


def require_non_negative(name: str, value: float, error: ErrorType = ConfigurationError) -> float:
    """Return ``value`` if it is a finite number greater than or equal to zero."""
    result = require_finite(name, value, error)
    if result < 0:
        raise error(f"{name} must be >= 0, got {result}")
    return result


def require_positive(name: str, value: float, error: ErrorType = ConfigurationError) -> float:
    """Return ``value`` if it is a finite number strictly greater than zero."""
    result = require_finite(name, value, error)
    if result <= 0:
        raise error(f"{name} must be > 0, got {result}")
    return result


def require_in_range(
    name: str,
    value: float,
    low: float,
    high: float,
    error: ErrorType = ConfigurationError,
) -> float:
    """Return ``value`` if ``low <= value <= high``."""
    result = require_finite(name, value, error)
    if not low <= result <= high:
        raise error(f"{name} must be between {low} and {high}, got {result}")
    return result


def require_fraction(name: str, value: float, error: ErrorType = ConfigurationError) -> float:
    """Return ``value`` if it lies in the closed unit interval ``[0, 1]``."""
    return require_in_range(name, value, 0.0, 1.0, error)


def require_int_in_range(
    name: str,
    value: object,
    low: int,
    high: int,
    error: ErrorType = ConfigurationError,
) -> int:
    """Return ``value`` if it is an integer with ``low <= value <= high``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise error(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if not low <= result <= high:
        raise error(f"{name} must be between {low} and {high}, got {result}")
    return result


def require_text(name: str, value: str, error: ErrorType = ConfigurationError) -> str:
    """Return ``value`` stripped of whitespace, rejecting empty strings."""
    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a non-empty string")
    return value.strip()
