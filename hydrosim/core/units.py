"""hydrosim.core.units

Минимальный слой единиц измерения и удобных множителей.

Принцип: везде, где есть числа, должна быть явная единица (например, 210 * BAR).
"""

from __future__ import annotations

import math

# Base units (conceptual SI multipliers)
METER: float = 1.0
KILOGRAM: float = 1.0
SECOND: float = 1.0

# Derived units
NEWTON: float = KILOGRAM * METER / (SECOND**2)
PASCAL: float = NEWTON / (METER**2)

# Convenience multipliers
BAR: float = 1e5 * PASCAL
MPA: float = 1e6 * PASCAL
LITRE: float = 1e-3 * (METER**3)

RPM_TO_RPS: float = 1.0 / 60.0  # rpm -> rps
DEG_TO_RAD: float = math.pi / 180.0

# Useful constants
G: float = 9.81 * METER / (SECOND**2)

# Common nominal values (for sanity checks / defaults)
LS_NOMINAL_PRESSURE: float = 210.0 * BAR
