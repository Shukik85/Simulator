from __future__ import annotations

from typing import Dict, Tuple
import math

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.loads import LoadModel


_RHO = 850.0
_PA_PER_BAR = 1e5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SimRHS:
    CYL_NAMES = ("boom_cyl", "arm_cyl", "bucket_cyl")

    def __init__(
        self,
        cfg: ExcavatorConfig,
        pump: LSPump,
        valves: Dict[str, LSValveSection],
        cylinders: Dict[str, Cylinder],
        relief: ReliefValve,
        loads: LoadModel,
    ) -> None:
        self.cfg = cfg
        self.pump = pump
        self.valves = valves
        self.cylinders = cylinders
        self.relief = relief
        self.loads = loads

    def _solve_p_pump(
        self,
        state: SimState,
        spools: Dict[str, float],
        q_pump: float,
    ) -> float:
        """Bisection: find P_pump where Q_pump = sum(Q_valves) + Q_relief."""
        ls_cfg = self.cfg.hydraulics
        p_tank = state.p_tank

        # Check if any valve is open
        total_demand = sum(abs(spools.get(n, 0.0)) for n in self.CYL_NAMES)
        if total_demand < 1e-6:
            p_standby = state.p_ls + ls_cfg.pump.margin_standby_bar * _PA_PER_BAR
            return max(p_standby, p_tank)

        def total_outflow(p: float) -> float:
            Qt = 0.0
            for name in self.CYL_NAMES:
                sp = spools.get(name, 0.0)
                vs = self.valves[name]
                vs.step(sp, p, state.p_a[name], state.p_b[name], _RHO)
                Qt += max(vs.q_a, 0.0) + max(vs.q_b, 0.0)
            Qt += self.relief.flow(p)
            return Qt

        p_lo = p_tank
        p_hi = ls_cfg.relief.max_bar * _PA_PER_BAR

        f_lo = q_pump - total_outflow(p_lo)
        f_hi = q_pump - total_outflow(p_hi)

        if f_lo <= 0.0:
            return p_lo
        if f_hi >= 0.0:
            return p_hi

        a, b = p_lo, p_hi
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = q_pump - total_outflow(m)
            if fm > 0:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def derivatives(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float) -> SimState:
        """Compute d(state)/dt. Returns a SimState where fields are derivatives."""
        cfg = self.cfg
        ls_cfg = cfg.hydraulics
        mech = cfg.mechanics
        fluid = ls_cfg.fluid

        ds = SimState()

        # LS pressure dynamics
        p_ls_target = max(state.p_tank, *(max(state.p_a[n], state.p_b[n]) for n in self.CYL_NAMES))
        tau_ls = 0.05
        ds.p_ls = (p_ls_target - state.p_ls) / max(tau_ls, 1e-6)

        # Pump flow
        q_pump = self.pump.step(pump_speed_rpm, state.p_pump, state.p_ls, 0.0)
        ds.q_pump = 0.0

        # Solve pump pressure (algebraic)
        p_pump = self._solve_p_pump(state, spools, q_pump)
        ds.p_pump = 0.0

        # Relief flow
        q_relief = self.relief.flow(p_pump)

        # External forces (gravity + soil)
        ext_forces = self.loads.external_cylinder_forces(state)

        for name in self.CYL_NAMES:
            sn = name.replace("_cyl", "")
            cyl = self.cylinders[name]
            geo = mech.cylinders()[name]
            dyn_cfg = ls_cfg.cylinder_dynamics[sn]
            sp = spools.get(name, 0.0)

            vs = self.valves[name]
            vs.step(sp, p_pump, state.p_a[name], state.p_b[name], _RHO)

            ahead = geo.area_piston_m2
            aann = geo.area_annulus_m2
            v = state.cyl_vel.get(name, 0.0)
            x = state.cyl_pos.get(name, 0.0)

            # Chamber volumes
            v_a = max(ahead * x + dyn_cfg.dead_volume_m3, 1e-8)
            v_b = max(aann * (geo.stroke_m - x) + dyn_cfg.dead_volume_m3, 1e-8)

            # Pressure derivatives: dP/dt = K/V * (Q - A*v - leak)
            leak = 1e-12
            q_a = vs.q_a
            q_b = vs.q_b

            ds.p_a[name] = (fluid.bulk_modulus / v_a) * (q_a - ahead * v - leak)
            ds.p_b[name] = (fluid.bulk_modulus / v_b) * (q_b + aann * v + leak)

            # Net hydraulic force
            f_hyd = state.p_a[name] * ahead - state.p_b[name] * aann
            f_ext = ext_forces.get(name, 0.0)

            # Friction
            kd = dyn_cfg.visc_damping
            fc = dyn_cfg.coulomb_friction
            f_fr = kd * v + fc * math.tanh(v / 0.01)

            # Acceleration
            m_eq = dyn_cfg.mass_equiv
            acc = (f_hyd - f_ext - f_fr) / max(m_eq, 1e-6)

            ds.cyl_pos[name] = v
            ds.cyl_vel[name] = acc

        return ds

    def euler_step(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float, dt: float) -> SimState:
        ds = self.derivatives(state, spools, pump_speed_rpm)
        ns = state.copy()
        ns.t = state.t + dt

        for name in self.CYL_NAMES:
            ns.cyl_pos[name] = state.cyl_pos[name] + dt * ds.cyl_pos[name]
            ns.cyl_vel[name] = state.cyl_vel[name] + dt * ds.cyl_vel[name]
            ns.p_a[name] = max(state.p_a[name] + dt * ds.p_a[name], 0.0)
            ns.p_b[name] = max(state.p_b[name] + dt * ds.p_b[name], 0.0)

            stroke = self.cfg.mechanics.cylinders()[name].stroke_m
            ns.cyl_pos[name] = _clamp(ns.cyl_pos[name], 0.0, stroke)
            if ns.cyl_pos[name] <= 0.0 or ns.cyl_pos[name] >= stroke:
                ns.cyl_vel[name] = 0.0
            ns.cyl_vel[name] = _clamp(ns.cyl_vel[name], -2.5, 2.5)

        ns.p_ls = max(state.p_ls + dt * ds.p_ls, 0.0)

        spools_here = {n: spools.get(n, 0.0) for n in self.CYL_NAMES}
        ns.p_pump = self._solve_p_pump(ns, spools_here, self.pump.step(pump_speed_rpm, ns.p_pump, ns.p_ls, 0.0))
        ns.q_pump = self.pump.step(pump_speed_rpm, ns.p_pump, ns.p_ls, 0.0)
        ns.q_relief = self.relief.flow(ns.p_pump)

        return ns

    def rk4_step(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float, dt: float) -> SimState:
        k1 = self.derivatives(state, spools, pump_speed_rpm)

        s2 = state.copy()
        for name in self.CYL_NAMES:
            s2.cyl_pos[name] = state.cyl_pos[name] + 0.5 * dt * k1.cyl_pos[name]
            s2.cyl_vel[name] = state.cyl_vel[name] + 0.5 * dt * k1.cyl_vel[name]
            s2.p_a[name] = state.p_a[name] + 0.5 * dt * k1.p_a[name]
            s2.p_b[name] = state.p_b[name] + 0.5 * dt * k1.p_b[name]
        s2.p_ls = state.p_ls + 0.5 * dt * k1.p_ls
        k2 = self.derivatives(s2, spools, pump_speed_rpm)

        s3 = state.copy()
        for name in self.CYL_NAMES:
            s3.cyl_pos[name] = state.cyl_pos[name] + 0.5 * dt * k2.cyl_pos[name]
            s3.cyl_vel[name] = state.cyl_vel[name] + 0.5 * dt * k2.cyl_vel[name]
            s3.p_a[name] = state.p_a[name] + 0.5 * dt * k2.p_a[name]
            s3.p_b[name] = state.p_b[name] + 0.5 * dt * k2.p_b[name]
        s3.p_ls = state.p_ls + 0.5 * dt * k2.p_ls
        k3 = self.derivatives(s3, spools, pump_speed_rpm)

        s4 = state.copy()
        for name in self.CYL_NAMES:
            s4.cyl_pos[name] = state.cyl_pos[name] + dt * k3.cyl_pos[name]
            s4.cyl_vel[name] = state.cyl_vel[name] + dt * k3.cyl_vel[name]
            s4.p_a[name] = state.p_a[name] + dt * k3.p_a[name]
            s4.p_b[name] = state.p_b[name] + dt * k3.p_b[name]
        s4.p_ls = state.p_ls + dt * k3.p_ls
        k4 = self.derivatives(s4, spools, pump_speed_rpm)

        ns = state.copy()
        ns.t = state.t + dt
        for name in self.CYL_NAMES:
            ns.cyl_pos[name] = state.cyl_pos[name] + (dt / 6.0) * (k1.cyl_pos[name] + 2.0 * k2.cyl_pos[name] + 2.0 * k3.cyl_pos[name] + k4.cyl_pos[name])
            ns.cyl_vel[name] = state.cyl_vel[name] + (dt / 6.0) * (k1.cyl_vel[name] + 2.0 * k2.cyl_vel[name] + 2.0 * k3.cyl_vel[name] + k4.cyl_vel[name])
            ns.p_a[name] = max(state.p_a[name] + (dt / 6.0) * (k1.p_a[name] + 2.0 * k2.p_a[name] + 2.0 * k3.p_a[name] + k4.p_a[name]), 0.0)
            ns.p_b[name] = max(state.p_b[name] + (dt / 6.0) * (k1.p_b[name] + 2.0 * k2.p_b[name] + 2.0 * k3.p_b[name] + k4.p_b[name]), 0.0)

            stroke = self.cfg.mechanics.cylinders()[name].stroke_m
            ns.cyl_pos[name] = _clamp(ns.cyl_pos[name], 0.0, stroke)
            if ns.cyl_pos[name] <= 0.0 or ns.cyl_pos[name] >= stroke:
                ns.cyl_vel[name] = 0.0
            ns.cyl_vel[name] = _clamp(ns.cyl_vel[name], -2.5, 2.5)

        ns.p_ls = max(state.p_ls + (dt / 6.0) * (k1.p_ls + 2.0 * k2.p_ls + 2.0 * k3.p_ls + k4.p_ls), 0.0)

        spools_here = {n: spools.get(n, 0.0) for n in self.CYL_NAMES}
        ns.p_pump = self._solve_p_pump(ns, spools_here, self.pump.step(pump_speed_rpm, ns.p_pump, ns.p_ls, 0.0))
        ns.q_pump = self.pump.step(pump_speed_rpm, ns.p_pump, ns.p_ls, 0.0)
        ns.q_relief = self.relief.flow(ns.p_pump)

        return ns
