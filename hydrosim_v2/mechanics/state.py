"""Mechanical state representation."""

from __future__ import annotations

from typing import Dict
from dataclasses import dataclass, field


@dataclass
class ExcavatorPose:
    cyl_lengths: Dict[str, float] = field(default_factory=dict)
    theta_boom: float = 0.0
    theta_arm: float = 0.0
    theta_bucket: float = 0.0
    boom_tip: tuple[float, float] = (0.0, 0.0)
    arm_tip: tuple[float, float] = (0.0, 0.0)
    bucket_tip: tuple[float, float] = (0.0, 0.0)
    bucket_tip_cutting_edge: tuple[float, float] = (0.0, 0.0)
