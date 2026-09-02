from __future__ import annotations

from typing import Dict, Tuple
import math
import numpy as np

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.loads import LoadModel
from hydrosim_v2.mechanics.dynamics import MultibodyDynamics, extract_joint_angles


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
        self.dyn = MultibodyDynamics(cfg.mechanics)
        self._leakage: Dict[str, Tuple[float, float]] = {}
        self.payload_kg = 0.0
        self.spool_filtered: Dict[str, float] = {n: 0.0 for n in self.CYL_NAMES}
        self.clamp_state: Dict[str, bool] = {n: False for n in self.CYL_NAMES}

    def _filter_spools(self, spools: Dict[str, float], dt: float) -> Dict[str, float]:
        """First-order spool filter applied once per integrator step."""
        out: Dict[str, float] = {}
        for n in self.CYL_NAMES:
            target = float(np.clip(spools.get(n, 0.0), -1.0, 1.0))
            sn = n.replace("_cyl", "")
            tau = self.cfg.hydraulics.valve_sections[sn].spool_tau_s
            alpha = min(dt / max(tau, 1e-6), 1.0)
            self.spool_filtered[n] += alpha * (target - self.spool_filtered[n])
            out[n] = self.spool_filtered[n]
        return out

    def _hyst_clamp(self, name: str, sp: float, q_a: float, q_b: float,
                    ahead: float, aann: float, v: float) -> tuple[bool, float | None]:
        """Hysteresis-aware supply-flow velocity clamp.
        Returns (v_clamped, v_max_supply)."""
        sn = name.replace("_cyl", "")
        hyst = self.cfg.hydraulics.valve_sections[sn].clamp_hyst
        prev = self.clamp_state[name]
        if sp > 0 and q_a > 0:
            v_max_supply = q_a / max(ahead, 1e-6)
            v_lo = v_max_supply * (1.0 - hyst)
            v_hi = v_max_supply * (1.0 + hyst)
            clamped = (v > v_lo) if prev else (v > v_hi)
        elif sp < 0 and q_b > 0:
            v_max_supply = q_b / max(aann, 1e-6)
            v_lo = -v_max_supply * (1.0 + hyst)
            v_hi = -v_max_supply * (1.0 - hyst)
            clamped = (v < v_hi) if prev else (v < v_lo)
        else:
            v_max_supply = None
            clamped = False
        self.clamp_state[name] = clamped
        return clamped, v_max_supply

    def _valve_capacity_flow(self, name: str, sp: float, p_pump: float, p_a: float, p_b: float) -> float:
        """Max flow the valve can pass in the meter-out (reverse) direction at relief pressure."""
        vs = self.valves[name]
        rho = _RHO
        p_relief_max = self.cfg.hydraulics.relief.max_bar * _PA_PER_BAR
        A = vs._area(abs(sp))
        kv = vs.k_v
        if abs(A) < 1e-12:
            return 0.0
        q = kv * A * math.sqrt(p_relief_max / rho)
        return q

    def _meter_out_clamp(self, name: str, sp: float, p_pump: float, p_a: float, p_b: float,
                         ahead: float, aann: float, v: float, v_makeup: float) -> tuple[float, bool]:
        """Limit piston speed in the load-driven (meter-out) direction so the piston
        cannot outrun the fluid the valve can pass. Returns (v_limited, was_limited)."""
        if abs(sp) < 1e-6:
            return v, False
        q_cap = self._valve_capacity_flow(name, sp, p_pump, p_a, p_b)
        if sp > 0:
            # supply on A; load may drive piston to retract (v<0), fluid exits A through valve
            v_lim = q_cap / max(ahead, 1e-6) + v_makeup
            if v < -v_lim:
                return -v_lim, True
        else:
            # supply on B; load may drive piston to extend (v>0), fluid exits B through valve
            v_lim = q_cap / max(aann, 1e-6) + v_makeup
            if v > v_lim:
                return v_lim, True
        return v, False

    def _solve_p_pump(
        self,
        state: SimState,
        spools: Dict[str, float],
        q_pump: float,
    ) -> Tuple[float, float]:
        """Find p_pump from the flow balance: valve_flows(p_pump) + relief(p_pump) = q_pump.

        Uses bisection to find the pressure where total outflow equals pump supply.
        Lower bound is max(p_ls, max chamber pressure) to ensure the pump margin
        creates a positive error that drives the pump to produce flow.
        Applies engine power limiting (caps p_pump at P_max/q_pump) and proportional
        flow sharing when valve demand exceeds pump supply.
        Returns (p_pump, flow_scale).
        """
        ls_cfg = self.cfg.hydraulics
        p_tank = state.p_tank
        p_relief_max = ls_cfg.relief.max_bar * _PA_PER_BAR

        total_demand = sum(abs(spools.get(n, 0.0)) for n in self.CYL_NAMES)
        if total_demand < 1e-6:
            p_standby = state.p_ls + ls_cfg.pump.margin_standby_bar * _PA_PER_BAR
            return min(max(p_standby, p_tank), p_relief_max), 1.0

        def _total_outflow(p_pump: float, with_relief: bool = True) -> float:
            Qt = 0.0
            for name in self.CYL_NAMES:
                sp = spools.get(name, 0.0)
                vs = self.valves[name]
                vs.step(sp, p_pump, state.p_a[name], state.p_b[name], _RHO)
                Qt += max(vs.q_a, 0.0) + max(vs.q_b, 0.0)
            if with_relief:
                Qt += self.relief.flow(p_pump)
            return Qt

        active_max = max(max(state.p_a[n], state.p_b[n]) for n in self.CYL_NAMES
                         if abs(spools.get(n, 0.0)) > 1e-6)
        p_lo = max(state.p_ls, active_max, state.p_tank)
        p_hi = p_relief_max

        f_lo = _total_outflow(p_lo)
        if f_lo >= q_pump:
            valve_only = _total_outflow(p_lo, with_relief=False)
            flow_scale = 1.0
            if valve_only > q_pump and valve_only > 1e-12:
                flow_scale = q_pump / valve_only
            return p_lo, flow_scale

        f_hi = _total_outflow(p_hi)
        if f_hi <= q_pump:
            p_sol = p_hi
        else:
            for _ in range(40):
                p_mid = 0.5 * (p_lo + p_hi)
                f_mid = _total_outflow(p_mid)
                if f_mid < q_pump:
                    p_lo = p_mid
                else:
                    p_hi = p_mid
            p_sol = 0.5 * (p_lo + p_hi)

        # Engine power limit: p_pump * q_pump <= P_max
        p_max_power = self.pump.max_hyd_power_w / max(q_pump, 1e-12)
        if p_sol > p_max_power:
            p_sol = max(p_max_power, p_tank)

        valve_only = _total_outflow(p_sol, with_relief=False)
        flow_scale = 1.0
        if valve_only > q_pump and valve_only > 1e-12:
            flow_scale = q_pump / valve_only
        return p_sol, flow_scale

    def derivatives(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float,
                    p_pump_pre: float | None = None, flow_scale_pre: float | None = None) -> SimState:
        """Compute d(state)/dt. Returns a SimState where fields are derivatives.

        If p_pump_pre and flow_scale_pre are provided, skip internal pump pressure solve
        and use the pre-computed values (for consistency in integrators).
        """
        cfg = self.cfg
        ls_cfg = cfg.hydraulics
        mech = cfg.mechanics
        fluid = ls_cfg.fluid

        ds = SimState()

        # LS pressure dynamics — only track active sections (spool displaced from centre)
        # LS gallery connects to the supply-side port: P→A for sp>0, P→B for sp<0
        active_pressures = [state.p_tank]
        has_active = False
        for n in self.CYL_NAMES:
            sp = spools.get(n, 0.0)
            if abs(sp) > 1e-6:
                if sp > 0:
                    active_pressures.append(state.p_a[n])
                else:
                    active_pressures.append(state.p_b[n])
                has_active = True
        p_ls_target = max(active_pressures)
        p_ls_target = min(p_ls_target, ls_cfg.relief.crack_bar * _PA_PER_BAR)
        tau_ls = 0.005 if has_active else 0.05
        ds.p_ls = (p_ls_target - state.p_ls) / max(tau_ls, 1e-6)

        # Use pump's current flow (set by last pump.step in the integrator)
        q_pump = self.pump.q_out
        ds.q_pump = 0.0

        # Solve pump pressure (algebraic) using current state
        if p_pump_pre is not None and flow_scale_pre is not None:
            p_pump = p_pump_pre
            flow_scale = flow_scale_pre
        else:
            p_pump, flow_scale = self._solve_p_pump(state, spools, q_pump)
        ds.p_pump = 0.0

        # Relief flow
        q_relief = self.relief.flow(p_pump)

        # External forces (gravity + soil)
        ext_forces = self.loads.external_cylinder_forces(state, self.payload_kg)

        # Collect cylinder forces for coupled dynamics
        f_net = {}
        cyl_lengths = {}
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

            # Cylinder lengths (position + length_min)
            cyl_lengths[name] = x + geo.length_min_m

            # Chamber volumes (with safe minimum)
            v_a = max(ahead * x + dyn_cfg.dead_volume_m3, 1e-6)
            v_b = max(aann * (geo.stroke_m - x) + dyn_cfg.dead_volume_m3, 1e-6)

            # Net hydraulic force
            f_hyd = state.p_a[name] * ahead - state.p_b[name] * aann
            f_ext = ext_forces.get(name, 0.0)

            # Cross-port leakage when spool is centred
            if abs(sp) < 1e-6:
                q_leak_a, q_leak_b = self._leakage_flow(state.p_a[name], state.p_b[name])
            else:
                q_leak_a = q_leak_b = 0.0

            q_a = vs.q_a * flow_scale + q_leak_a
            q_b = vs.q_b * flow_scale + q_leak_b

            # Pressure derivatives - limit to ±5000 bar/s (5e8 Pa/s) to prevent overflow
            max_dp_dt = 5.0e8

            term_a = q_a - ahead * v
            if abs(term_a) > 1e-12:
                ds.p_a[name] = float(np.clip((fluid.bulk_modulus / v_a) * term_a, -max_dp_dt, max_dp_dt))
            else:
                ds.p_a[name] = 0.0

            term_b = q_b + aann * v
            if abs(term_b) > 1e-12:
                ds.p_b[name] = float(np.clip((fluid.bulk_modulus / v_b) * term_b, -max_dp_dt, max_dp_dt))
            else:
                ds.p_b[name] = 0.0

            # Friction
            kd = dyn_cfg.visc_damping
            fc = dyn_cfg.coulomb_friction
            f_fr = kd * v + fc * math.tanh(v / 0.01)

            # Accumulate net force (hydraulic - external - friction)
            f_net[name] = f_hyd - f_ext - f_fr

            ds.cyl_pos[name] = v

        # Coupled multibody dynamics: M(θ)·θ̈ = J^T·F_cyl - G(θ) - C(θ,θ̇)·θ̇
        theta = extract_joint_angles(mech, cyl_lengths)
        J = self.dyn.jacobian(cyl_lengths)
        F_vec = np.array([f_net[n] for n in self.CYL_NAMES])
        tau = J.T @ F_vec
        G = self.dyn.gravity_vector(theta[0], theta[1], theta[2])
        M = self.dyn.mass_matrix(theta[1], theta[2])
        L_vel = np.array([state.cyl_vel[n] for n in self.CYL_NAMES])
        try:
            theta_dot = np.linalg.solve(J, L_vel)
        except np.linalg.LinAlgError:
            theta_dot = np.zeros(3)
        C = self.dyn.coriolis_vector(theta[1], theta[2], theta_dot)
        theta_ddot = np.linalg.solve(M, tau - G - C)
        L_ddot = J @ theta_ddot
        for i, name in enumerate(self.CYL_NAMES):
            ds.cyl_vel[name] = float(L_ddot[i])

        return ds

    def _standby_pump(self) -> None:
        """Reset pump to zero-flow standby (all spools closed)."""
        self.pump.q_out = 0.0
        self.pump.swash = 0.0
        self.pump.dp_ls = 0.0

    def _leakage_flow(self, p_a: float, p_b: float) -> tuple[float, float]:
        """Cross-port leakage when spool is centred (m³/s per Pa)."""
        kleak = 1e-18
        q_leak = kleak * (p_a - p_b)
        return -q_leak, q_leak

    def euler_step(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float, dt: float) -> SimState:
        spools_eff = self._filter_spools(spools, dt)
        total_demand = sum(abs(spools_eff.get(n, 0.0)) for n in self.CYL_NAMES)

        # Compute p_ls target and new p_ls first
        # LS gallery connects to the supply-side port: P→A for sp>0, P→B for sp<0
        active_pressures = [state.p_tank]
        has_active = False
        for n in self.CYL_NAMES:
            sp = spools_eff.get(n, 0.0)
            if abs(sp) > 1e-6:
                if sp > 0:
                    active_pressures.append(state.p_a[n])
                else:
                    active_pressures.append(state.p_b[n])
                has_active = True
        p_ls_target = max(active_pressures)
        p_ls_target = min(p_ls_target, self.cfg.hydraulics.relief.crack_bar * _PA_PER_BAR)
        tau_ls = 0.005 if has_active else 0.05
        ds_p_ls = (p_ls_target - state.p_ls) / max(tau_ls, 1e-6)
        p_ls_new = _clamp(state.p_ls + dt * ds_p_ls, 0.0, self.cfg.hydraulics.relief.crack_bar * _PA_PER_BAR)

        # Step pump with OLD p_pump and updated p_ls
        self.pump.step(pump_speed_rpm, state.p_pump, p_ls_new, dt)

        if total_demand < 1e-6:
            self._standby_pump()

        q_pump = self.pump.q_out
        p_pump, flow_scale = self._solve_p_pump(state, spools_eff, q_pump)

        ns = state.copy()
        ns.t = state.t + dt
        ext_forces = self.loads.external_cylinder_forces(state, self.payload_kg)

        e_cyl = 0.0
        e_friction = 0.0
        e_kin = 0.0
        e_valve_loss = 0.0

        # --- Pass 1: collect forces and valve flows ---
        f_net = {}
        cyl_lengths = {}
        for name in self.CYL_NAMES:
            sp = spools_eff.get(name, 0.0)
            geo = self.cfg.mechanics.cylinders()[name]
            sn = name.replace("_cyl", "")
            dyn_cfg = self.cfg.hydraulics.cylinder_dynamics[sn]
            ahead = geo.area_piston_m2
            aann = geo.area_annulus_m2
            v_old = state.cyl_vel[name]
            x_old = state.cyl_pos[name]

            cyl_lengths[name] = x_old + geo.length_min_m

            vs = self.valves[name]
            vs.step(sp, p_pump, state.p_a[name], state.p_b[name], _RHO)

            if abs(sp) < 1e-6:
                q_leak_a, q_leak_b = self._leakage_flow(state.p_a[name], state.p_b[name])
            else:
                q_leak_a = q_leak_b = 0.0
            q_a = vs.q_a * flow_scale + q_leak_a
            q_b = vs.q_b * flow_scale + q_leak_b

            f_hyd = state.p_a[name] * ahead - state.p_b[name] * aann
            f_ext = ext_forces.get(name, 0.0)
            f_fr = dyn_cfg.visc_damping * v_old + dyn_cfg.coulomb_friction * math.tanh(v_old / 0.01)
            f_net[name] = f_hyd - f_ext - f_fr

            ns.cyl_pos[name] = x_old + dt * v_old

        # --- Coupled multibody dynamics: M(θ)·θ̈ = J^T·F_cyl - G(θ) - C(θ,θ̇)·θ̇ ---
        theta = extract_joint_angles(self.cfg.mechanics, cyl_lengths)
        J = self.dyn.jacobian(cyl_lengths)
        F_vec = np.array([f_net[n] for n in self.CYL_NAMES])
        tau = J.T @ F_vec
        G = self.dyn.gravity_vector(theta[0], theta[1], theta[2])
        M_mat = self.dyn.mass_matrix(theta[1], theta[2])
        L_vel_old = np.array([state.cyl_vel[n] for n in self.CYL_NAMES])
        try:
            theta_dot = np.linalg.solve(J, L_vel_old)
        except np.linalg.LinAlgError:
            theta_dot = np.zeros(3)
        C = self.dyn.coriolis_vector(theta[1], theta[2], theta_dot)
        theta_ddot = np.linalg.solve(M_mat, tau - G - C)
        L_ddot = J @ theta_ddot

        # --- Pass 2: update velocities, clamping, pressure ---
        for i, name in enumerate(self.CYL_NAMES):
            sp = spools_eff.get(name, 0.0)
            geo = self.cfg.mechanics.cylinders()[name]
            sn = name.replace("_cyl", "")
            dyn_cfg = self.cfg.hydraulics.cylinder_dynamics[sn]
            valve_cfg = self.cfg.hydraulics.valve_sections[sn]
            ahead = geo.area_piston_m2
            aann = geo.area_annulus_m2
            v_old = state.cyl_vel[name]
            x_old = state.cyl_pos[name]
            v_a = max(ahead * x_old + dyn_cfg.dead_volume_m3, 1e-6)
            v_b = max(aann * (geo.stroke_m - x_old) + dyn_cfg.dead_volume_m3, 1e-6)
            fluid = self.cfg.hydraulics.fluid
            max_dp_dt = 5.0e8

            vs = self.valves[name]
            if abs(sp) < 1e-6:
                q_leak_a, q_leak_b = self._leakage_flow(state.p_a[name], state.p_b[name])
            else:
                q_leak_a = q_leak_b = 0.0
            q_a = vs.q_a * flow_scale + q_leak_a
            q_b = vs.q_b * flow_scale + q_leak_b

            ns.cyl_vel[name] = v_old + dt * float(L_ddot[i])

            f_ext = ext_forces.get(name, 0.0)

            stroke = geo.stroke_m
            ns.cyl_pos[name] = _clamp(ns.cyl_pos[name], 0.0, stroke)
            if ns.cyl_pos[name] <= 0.0:
                ns.cyl_vel[name] = max(ns.cyl_vel[name], 0.0)
            elif ns.cyl_pos[name] >= stroke:
                ns.cyl_vel[name] = min(ns.cyl_vel[name], 0.0)

            v_clamped, v_max_supply = self._hyst_clamp(name, sp, q_a, q_b, ahead, aann, ns.cyl_vel[name])

            if v_clamped and v_max_supply is not None:
                ns.cyl_vel[name] = v_max_supply if sp > 0 else -v_max_supply

            ns.cyl_vel[name], _ = self._meter_out_clamp(
                name, sp, p_pump, state.p_a[name], state.p_b[name],
                ahead, aann, ns.cyl_vel[name], valve_cfg.anti_cav_v_makeup)

            v_eff = ns.cyl_vel[name]

            if v_clamped:
                f_fr_eff = dyn_cfg.visc_damping * v_eff + dyn_cfg.coulomb_friction * math.tanh(v_eff / 0.01)
                if sp > 0:
                    raw_a = (f_ext + f_fr_eff + state.p_b[name] * aann) / ahead
                    raw_b = state.p_b[name]
                    if raw_a < 0.0:
                        raw_a = 0.0
                        ns.cyl_vel[name] = min(v_max_supply + valve_cfg.anti_cav_v_makeup, ns.cyl_vel[name] + valve_cfg.anti_cav_v_makeup)
                        v_eff = ns.cyl_vel[name]
                else:
                    raw_b = (state.p_a[name] * ahead - f_ext - f_fr_eff) / aann
                    raw_a = state.p_a[name]
                    if raw_b < 0.0:
                        raw_b = 0.0
                        ns.cyl_vel[name] = max(-(v_max_supply + valve_cfg.anti_cav_v_makeup), ns.cyl_vel[name] - valve_cfg.anti_cav_v_makeup)
                        v_eff = ns.cyl_vel[name]
            else:
                term_a = q_a - ahead * v_eff
                term_b = q_b + aann * v_eff
                raw_a = state.p_a[name] + dt * float(np.clip((fluid.bulk_modulus / v_a) * term_a, -max_dp_dt, max_dp_dt))
                raw_b = state.p_b[name] + dt * float(np.clip((fluid.bulk_modulus / v_b) * term_b, -max_dp_dt, max_dp_dt))

            if abs(sp) > 1e-6:
                if sp > 0:
                    ns.p_a[name] = _clamp(raw_a, 0.0, p_pump)
                    ns.p_b[name] = _clamp(raw_b, state.p_tank, p_pump)
                else:
                    ns.p_a[name] = _clamp(raw_a, state.p_tank, p_pump)
                    ns.p_b[name] = _clamp(raw_b, 0.0, p_pump)
            else:
                ns.p_a[name] = max(raw_a, state.p_tank)
                ns.p_b[name] = max(raw_b, state.p_tank)

            e_cyl += (ns.p_a[name] * q_a + ns.p_b[name] * q_b) * dt
            f_fr_v = dyn_cfg.visc_damping * v_eff + dyn_cfg.coulomb_friction * math.tanh(v_eff / 0.01)
            e_friction += f_fr_v * v_eff * dt

        # Coupled kinetic energy: e_kin = 0.5 * θ̇ᵀ M(θ) θ̇
        cyl_lengths_new = {}
        for _name in self.CYL_NAMES:
            _geo = self.cfg.mechanics.cylinders()[_name]
            cyl_lengths_new[_name] = ns.cyl_pos[_name] + _geo.length_min_m
        _theta_new = extract_joint_angles(self.cfg.mechanics, cyl_lengths_new)
        _J_new = self.dyn.jacobian(cyl_lengths_new)
        _L_vel = np.array([ns.cyl_vel[_n] for _n in self.CYL_NAMES])
        try:
            _theta_dot = np.linalg.solve(_J_new, _L_vel)
        except np.linalg.LinAlgError:
            _theta_dot = np.zeros(3)
        _M_mat = self.dyn.mass_matrix(_theta_new[1], _theta_new[2])
        e_kin = 0.5 * float(_theta_dot @ _M_mat @ _theta_dot)

        q_relief = self.relief.flow(p_pump) if q_pump > 1e-12 else 0.0
        p_hyd = q_pump * p_pump
        mech_eff = max(self.pump.mech_eff, 0.5)
        e_mech_in = (p_hyd / mech_eff) * dt
        # Valve loss as residual: ensures exact energy balance.
        # Can be negative due to explicit time-stepping (pressures and flows
        # evaluated at different states), which is physically valid — it
        # represents load-induced pressure doing work on the fluid.
        e_valve_loss = e_mech_in - e_cyl - q_relief * p_pump * dt - self.pump.mech_loss * dt

        ns.p_ls = p_ls_new
        ns.p_pump = p_pump
        ns.q_pump = q_pump
        ns.q_relief = q_relief

        ns.e_mech_in = state.e_mech_in + e_mech_in
        ns.e_cyl = state.e_cyl + e_cyl
        ns.e_valve_loss = state.e_valve_loss + e_valve_loss
        ns.e_relief = state.e_relief + q_relief * p_pump * dt
        ns.e_friction = state.e_friction + e_friction
        ns.e_kin = e_kin
        ns.e_pot = self.loads.potential_energy(state, self.payload_kg)

        return ns

    def rk4_step(self, state: SimState, spools: Dict[str, float], pump_speed_rpm: float, dt: float) -> SimState:
        spools_eff = self._filter_spools(spools, dt)
        total_demand = sum(abs(spools_eff.get(n, 0.0)) for n in self.CYL_NAMES)

        # Compute p_ls target and new p_ls
        # LS gallery connects to the supply-side port: P→A for sp>0, P→B for sp<0
        active_pressures = [state.p_tank]
        has_active = False
        for n in self.CYL_NAMES:
            sp = spools_eff.get(n, 0.0)
            if abs(sp) > 1e-6:
                if sp > 0:
                    active_pressures.append(state.p_a[n])
                else:
                    active_pressures.append(state.p_b[n])
                has_active = True
        p_ls_target = max(active_pressures)
        p_ls_target = min(p_ls_target, self.cfg.hydraulics.relief.crack_bar * _PA_PER_BAR)
        tau_ls = 0.005 if has_active else 0.05
        ds_p_ls = (p_ls_target - state.p_ls) / max(tau_ls, 1e-6)
        p_ls_new = _clamp(state.p_ls + dt * ds_p_ls, 0.0, self.cfg.hydraulics.relief.crack_bar * _PA_PER_BAR)

        # Step pump with OLD p_pump and updated p_ls
        self.pump.step(pump_speed_rpm, state.p_pump, p_ls_new, dt)

        if total_demand < 1e-6:
            self._standby_pump()

        q_pump = self.pump.q_out
        p_pump, flow_scale = self._solve_p_pump(state, spools_eff, q_pump)

        k1 = self.derivatives(state, spools_eff, pump_speed_rpm, p_pump, flow_scale)

        s2 = state.copy()
        for name in self.CYL_NAMES:
            s2.cyl_pos[name] = state.cyl_pos[name] + 0.5 * dt * k1.cyl_pos[name]
            s2.cyl_vel[name] = state.cyl_vel[name] + 0.5 * dt * k1.cyl_vel[name]
            s2.p_a[name] = state.p_a[name] + 0.5 * dt * k1.p_a[name]
            s2.p_b[name] = state.p_b[name] + 0.5 * dt * k1.p_b[name]
        s2.p_ls = state.p_ls + 0.5 * dt * k1.p_ls
        k2 = self.derivatives(s2, spools_eff, pump_speed_rpm, p_pump, flow_scale)

        s3 = state.copy()
        for name in self.CYL_NAMES:
            s3.cyl_pos[name] = state.cyl_pos[name] + 0.5 * dt * k2.cyl_pos[name]
            s3.cyl_vel[name] = state.cyl_vel[name] + 0.5 * dt * k2.cyl_vel[name]
            s3.p_a[name] = state.p_a[name] + 0.5 * dt * k2.p_a[name]
            s3.p_b[name] = state.p_b[name] + 0.5 * dt * k2.p_b[name]
        s3.p_ls = state.p_ls + 0.5 * dt * k2.p_ls
        k3 = self.derivatives(s3, spools_eff, pump_speed_rpm, p_pump, flow_scale)

        s4 = state.copy()
        for name in self.CYL_NAMES:
            s4.cyl_pos[name] = state.cyl_pos[name] + dt * k3.cyl_pos[name]
            s4.cyl_vel[name] = state.cyl_vel[name] + dt * k3.cyl_vel[name]
            s4.p_a[name] = state.p_a[name] + dt * k3.p_a[name]
            s4.p_b[name] = state.p_b[name] + dt * k3.p_b[name]
        s4.p_ls = state.p_ls + dt * k3.p_ls
        k4 = self.derivatives(s4, spools_eff, pump_speed_rpm, p_pump, flow_scale)

        ns = state.copy()
        ns.t = state.t + dt
        ext_forces = self.loads.external_cylinder_forces(state, self.payload_kg)
        e_cyl = 0.0
        e_friction = 0.0
        e_kin = 0.0
        for name in self.CYL_NAMES:
            ns.cyl_pos[name] = state.cyl_pos[name] + (dt / 6.0) * (k1.cyl_pos[name] + 2.0 * k2.cyl_pos[name] + 2.0 * k3.cyl_pos[name] + k4.cyl_pos[name])
            ns.cyl_vel[name] = state.cyl_vel[name] + (dt / 6.0) * (k1.cyl_vel[name] + 2.0 * k2.cyl_vel[name] + 2.0 * k3.cyl_vel[name] + k4.cyl_vel[name])

            sp = spools_eff.get(name, 0.0)
            geo = self.cfg.mechanics.cylinders()[name]
            sn = name.replace("_cyl", "")
            dyn_cfg = self.cfg.hydraulics.cylinder_dynamics[sn]
            valve_cfg = self.cfg.hydraulics.valve_sections[sn]
            ahead = geo.area_piston_m2
            aann = geo.area_annulus_m2
            v_a = max(ahead * ns.cyl_pos[name] + dyn_cfg.dead_volume_m3, 1e-6)
            v_b = max(aann * (geo.stroke_m - ns.cyl_pos[name]) + dyn_cfg.dead_volume_m3, 1e-6)
            fluid = self.cfg.hydraulics.fluid
            max_dp_dt = 5.0e8

            stroke = geo.stroke_m
            ns.cyl_pos[name] = _clamp(ns.cyl_pos[name], 0.0, stroke)
            if ns.cyl_pos[name] <= 0.0:
                ns.cyl_vel[name] = max(ns.cyl_vel[name], 0.0)
            elif ns.cyl_pos[name] >= stroke:
                ns.cyl_vel[name] = min(ns.cyl_vel[name], 0.0)

            vs = self.valves[name]
            vs.step(sp, p_pump, state.p_a[name], state.p_b[name], _RHO)
            if abs(sp) < 1e-6:
                q_leak_a, q_leak_b = self._leakage_flow(state.p_a[name], state.p_b[name])
            else:
                q_leak_a = q_leak_b = 0.0
            q_a = vs.q_a * flow_scale + q_leak_a
            q_b = vs.q_b * flow_scale + q_leak_b

            v_clamped, v_max_supply = self._hyst_clamp(name, sp, q_a, q_b, ahead, aann, ns.cyl_vel[name])

            if v_clamped and v_max_supply is not None:
                ns.cyl_vel[name] = v_max_supply if sp > 0 else -v_max_supply

            ns.cyl_vel[name], _ = self._meter_out_clamp(
                name, sp, p_pump, state.p_a[name], state.p_b[name],
                ahead, aann, ns.cyl_vel[name], valve_cfg.anti_cav_v_makeup)

            v_eff = ns.cyl_vel[name]

            if v_clamped:
                f_ext = ext_forces.get(name, 0.0)
                f_fr_eff = dyn_cfg.visc_damping * v_eff + dyn_cfg.coulomb_friction * math.tanh(v_eff / 0.01)
                if sp > 0:
                    raw_a = (f_ext + f_fr_eff + state.p_b[name] * aann) / ahead
                    raw_b = state.p_b[name]
                    if raw_a < 0.0:
                        raw_a = 0.0
                        ns.cyl_vel[name] = min(v_max_supply + valve_cfg.anti_cav_v_makeup, ns.cyl_vel[name] + valve_cfg.anti_cav_v_makeup)
                        v_eff = ns.cyl_vel[name]
                else:
                    raw_b = (state.p_a[name] * ahead - f_ext - f_fr_eff) / aann
                    raw_a = state.p_a[name]
                    if raw_b < 0.0:
                        raw_b = 0.0
                        ns.cyl_vel[name] = max(-(v_max_supply + valve_cfg.anti_cav_v_makeup), ns.cyl_vel[name] - valve_cfg.anti_cav_v_makeup)
                        v_eff = ns.cyl_vel[name]
            else:
                term_a = q_a - ahead * v_eff
                term_b = q_b + aann * v_eff
                raw_a = state.p_a[name] + dt * float(np.clip((fluid.bulk_modulus / v_a) * term_a, -max_dp_dt, max_dp_dt))
                raw_b = state.p_b[name] + dt * float(np.clip((fluid.bulk_modulus / v_b) * term_b, -max_dp_dt, max_dp_dt))

            if abs(sp) > 1e-6:
                if sp > 0:
                    ns.p_a[name] = _clamp(raw_a, 0.0, p_pump)
                    ns.p_b[name] = _clamp(raw_b, state.p_tank, p_pump)
                else:
                    ns.p_a[name] = _clamp(raw_a, state.p_tank, p_pump)
                    ns.p_b[name] = _clamp(raw_b, 0.0, p_pump)
            else:
                ns.p_a[name] = max(raw_a, state.p_tank)
                ns.p_b[name] = max(raw_b, state.p_tank)

            e_cyl += (ns.p_a[name] * q_a + ns.p_b[name] * q_b) * dt
            f_fr_v = dyn_cfg.visc_damping * v_eff + dyn_cfg.coulomb_friction * math.tanh(v_eff / 0.01)
            e_friction += f_fr_v * v_eff * dt

        # Coupled kinetic energy: e_kin = 0.5 * θ̇ᵀ M(θ) θ̇
        cyl_lengths_new = {}
        for _name in self.CYL_NAMES:
            _geo = self.cfg.mechanics.cylinders()[_name]
            cyl_lengths_new[_name] = ns.cyl_pos[_name] + _geo.length_min_m
        _theta_new = extract_joint_angles(self.cfg.mechanics, cyl_lengths_new)
        _J_new = self.dyn.jacobian(cyl_lengths_new)
        _L_vel = np.array([ns.cyl_vel[_n] for _n in self.CYL_NAMES])
        try:
            _theta_dot = np.linalg.solve(_J_new, _L_vel)
        except np.linalg.LinAlgError:
            _theta_dot = np.zeros(3)
        _M_mat = self.dyn.mass_matrix(_theta_new[1], _theta_new[2])
        e_kin = 0.5 * float(_theta_dot @ _M_mat @ _theta_dot)

        q_relief = self.relief.flow(p_pump) if q_pump > 1e-12 else 0.0
        p_hyd = q_pump * p_pump
        mech_eff = max(self.pump.mech_eff, 0.5)
        e_mech_in = (p_hyd / mech_eff) * dt
        # Valve loss as residual: ensures exact energy balance.
        # Can be negative due to explicit time-stepping (pressures and flows
        # evaluated at different states), which is physically valid — it
        # represents load-induced pressure doing work on the fluid.
        e_valve_loss = e_mech_in - e_cyl - q_relief * p_pump * dt - self.pump.mech_loss * dt

        ns.p_ls = p_ls_new
        ns.p_pump = p_pump
        ns.q_pump = q_pump
        ns.q_relief = q_relief

        ns.e_mech_in = state.e_mech_in + e_mech_in
        ns.e_cyl = state.e_cyl + e_cyl
        ns.e_valve_loss = state.e_valve_loss + e_valve_loss
        ns.e_relief = state.e_relief + q_relief * p_pump * dt
        ns.e_friction = state.e_friction + e_friction
        ns.e_kin = e_kin
        ns.e_pot = self.loads.potential_energy(state, self.payload_kg)

        return ns
