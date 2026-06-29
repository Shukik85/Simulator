# hydrosim/mechanics/state.py
"""Состояние экскаватора: кинематика, гидравлика, динамика."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from hydrosim.mechanics.hydraulics import HydraulicCylinderSpec

Vec2 = np.ndarray


@dataclass
class CylinderState:
    """Состояние одного гидроцилиндра."""
    name: str
    length_m: float
    volume_piston_chamber_m3: float
    volume_rod_chamber_m3: float
    force_N: Optional[float] = None

    def update_length(self, new_length: float, spec) -> None:
        """Обновляет длину и объёмы полостей."""
        if abs(new_length - self.length_m) < 1e-8:
            return
        dL = new_length - self.length_m
        dV_piston = spec.area_piston * dL
        dV_rod = spec.area_rod * dL
        self.volume_piston_chamber_m3 += dV_piston
        self.volume_rod_chamber_m3 -= dV_rod
        self.length_m = new_length

    def set_force(self, force: float) -> None:
        self.force_N = force


@dataclass
class LinkPose2D:
    """2D поза звена."""
    joint_angle_rad: float
    pivot_xy: Tuple[float, float]
    tip_xy: Tuple[float, float]
    force_application_point_xy: Tuple[float, float]


@dataclass
class ExcavatorKinematicState:
    """Выход forward кинематики."""
    swing_angle_rad: float
    boom: LinkPose2D
    arm: LinkPose2D
    bucket: LinkPose2D
    bucket_tip_xyz: Tuple[float, float, float]


@dataclass
class SystemState:
    """Полное состояние системы."""
    cylinder_states: Dict[str, CylinderState]
    kinematic_state: Optional[ExcavatorKinematicState] = None
    external_forces: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        if self.external_forces is None:
            self.external_forces = {}

    def update_cylinder_lengths(self, lengths: Dict[str, float], specs: Dict[str, HydraulicCylinderSpec]) -> None:
        for name, new_length in lengths.items():
            self.cylinder_states[name].update_length(new_length, specs[name])

    def set_kinematics(self, state: ExcavatorKinematicState) -> None:
        self.kinematic_state = state

    def set_external_forces(self, forces: Dict[str, np.ndarray]) -> None:
        self.external_forces.update(forces)

    def set_cylinder_forces(self, forces_N: Dict[str, float]) -> None:
        for name, force in forces_N.items():
            if name in self.cylinder_states:
                self.cylinder_states[name].set_force(force)

    def get_cylinder_lengths(self) -> Dict[str, float]:
        return {name: cs.length_m for name, cs in self.cylinder_states.items()}