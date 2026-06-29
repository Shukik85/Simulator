"""Right-hand side of the coupled hydro-mechanical system."""

from __future__ import annotations

from typing import Dict

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.config.base import CylinderGeometry
from hydrosim_v2.config.hydraulics import LSConfig
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve


def _read_cyl_lengths(
    cylinders: Dict[str, Cylinder],
    geo: Dict[str, CylinderGeometry],
) -> Dict[str, float]:
    return {name: cylinders[name].length() for name in cylinders}


class SystemRHS:
    def __init__(
        self,
        cfg: ExcavatorConfig,
        pump: LSPump,
        valves: Dict[str, LSValveSection],
        cylinders: Dict[str, Cylinder],
        relief: Dict[str, ReliefValve],
    ) -> None:
        self.cfg = cfg
        self.pump = pump
        self.valves = valves
        self.cylinders = cylinders
        self.relief = relief
        self.p_pump = 0.0
        self.p_ls = 0.0

    def compute_ls_pressure(self) -> float:
        max_p = 0.0
        for name, cyl in self.cylinders.items():
            max_p = max(max_p, cyl.p_a, cyl.p_b)
        self.p_ls = max_p
        return self.p_ls

    def step(
        self,
        spools: Dict[str, float],
        pump_speed_rpm: float,
        t: float,
        dt: float,
    ) -> Dict[str, float]:
        cfg = self.cfg
        ls_cfg = cfg.hydraulics

        for name, spool in spools.items():
            if name in self.valves:
                self.valves[name].step(spool, self.p_pump, 0.0, 0.0, 0.0)

        for name, cyl in self.cylinders.items():
            valve = self.valves.get(name)
            if valve is not None:
                cyl.step_flow(valve.q_a, valve.q_b, dt)

        return {
            "p_pump": self.p_pump,
            "p_ls": self.p_ls,
        }
