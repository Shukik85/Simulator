from __future__ import annotations

from typing import Dict

import numpy as np

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.config.hydraulics import FluidConfig
from hydrosim_v2.faults.params import FaultConfig
from hydrosim_v2.hydraulics.pump import LSPumpModel
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import HydraulicCylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.inverse_dynamics import (
    joint_angles_from_cylinders,
    joint_space_inertia_matrix,
    gravity_torque,
    _cylinder_lengths_from_angles,
    _endeffector_jacobian,
)

_EPS = 1e-12

CYL_NAMES = ["boom_cyl", "arm_cyl", "bucket_cyl"]
N_STATES = 14

I_PPUMP = 0
I_EPS = 1
I_X0 = 2
I_V0 = 5
I_PA0 = 8
I_PB0 = 10


class SystemRHS:
    """RHS of coupled LS-excavator ODE, optionally with fault injection."""

    def __init__(
        self,
        config: ExcavatorConfig,
        V_hp_m3: float = 0.05,
        speed_rpm: float = 2200.0,
        faults: FaultConfig | None = None,
    ):
        self.cfg = config
        self.V_hp = V_hp_m3
        self.fluid = config.hydraulics.fluid
        self.faults = faults or FaultConfig()

        self.pump = LSPumpModel(config.hydraulics.pump, speed_rpm=speed_rpm)
        self.relief = ReliefValve(config.hydraulics.relief)

        self.valves: dict[str, LSValveSection] = {}
        self.cylinders: dict[str, HydraulicCylinder] = {}
        for name in CYL_NAMES:
            vc = config.hydraulics.valve_sections.get(name)
            cc = config.hydraulics.cylinder_dynamics.get(name)
            gc = config.mechanics.cylinders().get(name)
            if vc is None or cc is None or gc is None:
                continue
            self.valves[name] = LSValveSection(vc, FluidConfig())
            self.cylinders[name] = HydraulicCylinder(gc, cc, FluidConfig())

    def _pump_flow_m3s(self, epsilon: float) -> float:
        """Pump flow with fault-modified efficiency."""
        fp = self.faults.pump
        Q_th = epsilon * self.pump.max_theoretical_flow_m3s
        eff = self.pump.volumetric_efficiency(Q_th) * fp.vol_eff_multiplier
        return Q_th * float(max(0.0, min(1.0, eff)))

    def _pump_leak_m3s(self, p_pump: float) -> float:
        return self.pump.case_leak_flow_m3s(p_pump) * self.faults.pump.leakage_multiplier

    def _valve_flow_mult(self, name: str) -> float:
        return self.faults.valves.get(name, ValveFaultParams()).flow_multiplier

    def _valve_bias(self, name: str) -> float:
        return self.faults.valves.get(name, ValveFaultParams()).bias

    def _cyl_internal_leak(self, name: str) -> float:
        return self.faults.cylinders.get(name, CylinderFaultParams()).internal_leak_coeff

    def _cyl_external_leak(self, name: str) -> float:
        return self.faults.cylinders.get(name, CylinderFaultParams()).external_leak_coeff

    def _cyl_friction_mult(self, name: str) -> float:
        return self.faults.cylinders.get(name, CylinderFaultParams()).friction_multiplier

    def __call__(
        self,
        t: float,
        s: np.ndarray,
        u: Dict[str, float],
        ee_force: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        from hydrosim_v2.faults.params import CylinderFaultParams, ValveFaultParams

        ds = np.zeros(N_STATES, dtype=float)
        mc = self.cfg.mechanics

        pP = max(float(s[I_PPUMP]), _EPS)
        eps = float(np.clip(s[I_EPS], 0.0, 1.0))
        x = {n: float(s[I_X0 + i]) for i, n in enumerate(CYL_NAMES)}
        v = {n: float(s[I_V0 + i]) for i, n in enumerate(CYL_NAMES)}
        pA = {n: float(s[I_PA0 + i]) for i, n in enumerate(CYL_NAMES)}
        pB = {n: float(s[I_PB0 + i]) for i, n in enumerate(CYL_NAMES)}

        geoms = mc.cylinders()
        cyl_len = {n: geoms[n].length_min_m + x[n] for n in CYL_NAMES}

        # === Valve flows (with fault modifiers) ===
        Q_A, Q_B = {}, {}
        Q_sup = 0.0
        for name in CYL_NAMES:
            vl = self.valves.get(name)
            if vl is None:
                Q_A[name] = Q_B[name] = 0.0
                continue
            uv = u.get(name, 0.0) + self._valve_bias(name)
            fmult = self._valve_flow_mult(name)
            Q_A[name] = vl.net_flow_A(uv, pP, pA[name]) * fmult
            Q_B[name] = vl.net_flow_B(uv, pP, pB[name]) * fmult
            Q_sup += vl.supply_flow(uv, pP, pA[name], pB[name]) * fmult

        # === Cylinder forces (with fault) ===
        F_cyl = {}
        for name in CYL_NAMES:
            cy = self.cylinders.get(name)
            if cy is None:
                F_cyl[name] = 0.0
                continue
            fmult = self._cyl_friction_mult(name)
            F_cyl[name] = cy.force(pA[name], pB[name]) - cy.friction_force(v[name]) * fmult

        # === Mechanics ===
        q = joint_angles_from_cylinders(mc, cyl_len)
        H = joint_space_inertia_matrix(mc, q)
        g = gravity_torque(mc, q)
        J_ee = _endeffector_jacobian(mc, q)

        eps_fd = 1e-7
        J_cyl = np.zeros((3, 3))
        for j in range(3):
            qp = q.copy()
            qp[j] += eps_fd
            qm = q.copy()
            qm[j] -= eps_fd
            Lp = _cylinder_lengths_from_angles(mc, qp)
            Lm = _cylinder_lengths_from_angles(mc, qm)
            for i, n in enumerate(CYL_NAMES):
                J_cyl[i, j] = (Lp[n] - Lm[n]) / (2 * eps_fd)

        tau_cyl = J_cyl.T @ np.array([F_cyl[n] for n in CYL_NAMES])
        tau_ext = J_ee.T @ np.array(ee_force, dtype=float)
        ddq = np.linalg.solve(H, tau_cyl + tau_ext - g)
        a_cyl = J_cyl @ ddq

        # === Pressure dynamics (with leak faults) ===
        for i, name in enumerate(CYL_NAMES):
            cy = self.cylinders.get(name)
            if cy is None:
                ds[I_PA0 + 2 * i] = 0.0
                ds[I_PB0 + 2 * i] = 0.0
                continue
            Q_leak_int = self._cyl_internal_leak(name) * (pA[name] - pB[name])
            Q_leak_ext_A = self._cyl_external_leak(name) * pA[name]
            Q_leak_ext_B = self._cyl_external_leak(name) * pB[name]

            Q_A_eff = Q_A[name] + Q_leak_int - Q_leak_ext_A
            Q_B_eff = Q_B[name] - Q_leak_int - Q_leak_ext_B

            dpA, dpB = cy.rhs(pA[name], pB[name], x[name], v[name], Q_A_eff, Q_B_eff)
            ds[I_PA0 + 2 * i] = dpA
            ds[I_PB0 + 2 * i] = dpB

        # === Pump dynamics ===
        p_ls_max = max(pA[n] for n in CYL_NAMES)
        Q_relief = self.relief.flow_m3s(pP)
        Q_leak = self._pump_leak_m3s(pP)

        ds[I_PPUMP] = self.pump.flow_to_system_rhs(
            pP, eps, Q_sup + Q_relief + Q_leak, self.V_hp, self.fluid.bulk_modulus,
        )
        ds[I_EPS] = self.pump.swash_plate_rhs(eps, pP, p_ls_max)

        for i, name in enumerate(CYL_NAMES):
            ds[I_X0 + i] = v[name]
            ds[I_V0 + i] = a_cyl[i]

        return ds
