"""Simple forward-Euler integrator for the excavator system."""

from __future__ import annotations

from typing import Dict, List, Callable
from dataclasses import dataclass, field

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.config.base import CylinderGeometry
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.hydraulics.system_rhs import SystemRHS
from hydrosim_v2.mechanics.inverse_dynamics import InverseDynamics


@dataclass
class IntegratorState:
    t: float = 0.0
    cyl_positions: Dict[str, float] = field(default_factory=dict)
    cyl_velocities: Dict[str, float] = field(default_factory=dict)
    p_a: Dict[str, float] = field(default_factory=dict)
    p_b: Dict[str, float] = field(default_factory=dict)
    p_pump: float = 0.0
    p_ls: float = 0.0
    log: List[Dict] = field(default_factory=list)


class ForwardEuler:
    def __init__(
        self,
        cfg: ExcavatorConfig,
        cylinders: Dict[str, Cylinder],
        valves: Dict[str, LSValveSection],
        pump: LSPump,
        relief: Dict[str, ReliefValve],
        spool_signal: Callable[[str, float], float],
    ) -> None:
        self.cfg = cfg
        self.cylinders = cylinders
        self.valves = valves
        self.pump = pump
        self.relief = relief
        self.spool_signal = spool_signal
        self.rhs = SystemRHS(cfg, pump, valves, cylinders, relief)
        self.inv_dyn = InverseDynamics(cfg)
        self.state = IntegratorState()

    def step(self, dt: float, max_dt: float = 0.001) -> IntegratorState:
        dt = min(dt, max_dt)
        t = self.state.t
        spools = {}
        for name in self.cylinders:
            spools[name] = self.spool_signal(name, t)
        geo_dict = {}
        for name, cyl in self.cylinders.items():
            geo_dict[name] = cyl.geo
        cyl_lengths = {}
        for name, cyl in self.cylinders.items():
            cyl_lengths[name] = cyl.length()
        cyl_velocities = {}
        for name, cyl in self.cylinders.items():
            cyl_velocities[name] = cyl.velocity
        forces = self.inv_dyn.compute_gravitational(cyl_lengths, geo_dict)
        for name, cyl in self.cylinders.items():
            u = forces.get(name, 0.0)
            a = cyl.step_mechanics(u)
            cyl.step_position(a, dt)
        self.rhs.step(spools, 2200.0, t, dt)
        self.state.t += dt
        self.state.p_pump = self.rhs.p_pump
        self.state.p_ls = self.rhs.p_ls
        for name, cyl in self.cylinders.items():
            self.state.cyl_positions[name] = cyl.position
            self.state.cyl_velocities[name] = cyl.velocity
            self.state.p_a[name] = cyl.p_a
            self.state.p_b[name] = cyl.p_b
        return self.state
