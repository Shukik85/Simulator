from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class CylinderState:
    name: str
    length_m: float
    extension_m: float       # rod extension (0..stroke)
    velocity_m_s: float = 0.0
    force_N: float = 0.0

    def update_length(self, new_length: float, stroke_m: float, l_min: float) -> None:
        self.length_m = new_length
        self.extension_m = max(0.0, min(stroke_m, new_length - l_min))


@dataclass
class ExcavatorKinematicState:
    boom_angle_rad: float
    arm_angle_rad: float
    bucket_angle_rad: float
    boom_tip_xy: Tuple[float, float]
    arm_tip_xy: Tuple[float, float]
    bucket_tip_xy: Tuple[float, float]


@dataclass
class ExcavatorState:
    cylinder_states: dict[str, CylinderState]
    kinematic_state: ExcavatorKinematicState | None = None
    external_forces: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        if self.external_forces is None:
            self.external_forces = {}

    def get_cylinder_lengths(self) -> dict[str, float]:
        return {name: cs.length_m for name, cs in self.cylinder_states.items()}
