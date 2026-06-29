import math
from hydrosim_v2.core.types import Vec2


def check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def check_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def check_vec2(name: str, v: Vec2) -> None:
    check_finite(f"{name}[0]", v[0])
    check_finite(f"{name}[1]", v[1])


def check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
