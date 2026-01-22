"""hydrosim.core.validation

Базовые проверки, чтобы ловить физически невозможные значения как можно раньше.
"""

from __future__ import annotations


def ensure_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def ensure_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def ensure_in_range(value: float, min_value: float, max_value: float, name: str) -> None:
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name} must be in [{min_value}, {max_value}], got {value}")
