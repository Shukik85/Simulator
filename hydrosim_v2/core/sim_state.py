from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SimState:
    t: float = 0.0

    cyl_pos: Dict[str, float] = field(default_factory=lambda: {
        "boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0,
    })
    cyl_vel: Dict[str, float] = field(default_factory=lambda: {
        "boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0,
    })
    p_a: Dict[str, float] = field(default_factory=lambda: {
        "boom_cyl": 1.0e5, "arm_cyl": 1.0e5, "bucket_cyl": 1.0e5,
    })
    p_b: Dict[str, float] = field(default_factory=lambda: {
        "boom_cyl": 1.0e5, "arm_cyl": 1.0e5, "bucket_cyl": 1.0e5,
    })

    p_pump: float = 1.0e5
    p_ls: float = 1.0e5
    p_tank: float = 1.0e5

    q_pump: float = 0.0
    q_relief: float = 0.0

    e_mech_in: float = 0.0
    e_cyl: float = 0.0
    e_valve_loss: float = 0.0
    e_relief: float = 0.0
    e_friction: float = 0.0
    e_kin: float = 0.0
    e_pot: float = 0.0

    def copy(self) -> SimState:
        return SimState(
            t=self.t,
            cyl_pos=dict(self.cyl_pos),
            cyl_vel=dict(self.cyl_vel),
            p_a=dict(self.p_a),
            p_b=dict(self.p_b),
            p_pump=self.p_pump,
            p_ls=self.p_ls,
            p_tank=self.p_tank,
            q_pump=self.q_pump,
            q_relief=self.q_relief,
            e_mech_in=self.e_mech_in,
            e_cyl=self.e_cyl,
            e_valve_loss=self.e_valve_loss,
            e_relief=self.e_relief,
            e_friction=self.e_friction,
            e_kin=self.e_kin,
            e_pot=self.e_pot,
        )
