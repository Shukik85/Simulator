from __future__ import annotations

import math

from hydrosim_v2.config.hydraulics import CylinderDynamicsConfig, FluidConfig
from hydrosim_v2.config.base import CylinderGeometry

_EPS = 1e-12


class HydraulicCylinder:
    """Double-acting hydraulic cylinder with chamber pressure dynamics.

    State vector: [p_A (piston chamber), p_B (rod chamber)] in Pa.
    """

    def __init__(
        self,
        geometry: CylinderGeometry,
        dynamics: CylinderDynamicsConfig,
        fluid: FluidConfig,
    ):
        self.geom = geometry
        self.dyn = dynamics
        self.beta = fluid.bulk_modulus
        self.A_p = geometry.area_piston_m2
        self.A_r = geometry.area_rod_m2
        self.A_a = geometry.area_annulus_m2  # piston minus rod
        self.L_min = geometry.length_min_m
        self.stroke = geometry.stroke_m

    def chamber_volumes(self, x: float) -> tuple[float, float]:
        """Volumes of piston (A) and rod (B) chambers (m³).

        x = rod extension [0, stroke].
        """
        V_dead = self.dyn.line_volume_m3 + self.dyn.dead_volume_m3
        V_A = self.A_p * (self.L_min + x) + V_dead
        V_B = self.A_a * (self.L_min + self.stroke - x) + V_dead
        return max(V_A, _EPS), max(V_B, _EPS)

    def force(self, p_A: float, p_B: float) -> float:
        """Hydraulic force on the rod (N), positive = extending."""
        return p_A * self.A_p - p_B * self.A_a

    def rhs(
        self,
        p_A: float,
        p_B: float,
        x: float,
        v: float,
        Q_A: float,
        Q_B: float,
    ) -> tuple[float, float]:
        """Time derivatives of chamber pressures.

        dp_A/dt = β/V_A * (Q_A - A_p * v)
        dp_B/dt = β/V_B * (-Q_B + A_a * v)

        Returns (dp_A_dt, dp_B_dt).
        """
        V_A, V_B = self.chamber_volumes(x)
        dp_A = self.beta / V_A * (Q_A - self.A_p * v)
        dp_B = self.beta / V_B * (-Q_B + self.A_a * v)
        return dp_A, dp_B

    def friction_force(self, v: float) -> float:
        """Combined viscous + Coulomb friction (N)."""
        return self.dyn.visc_damping * v + self.dyn.coulomb_friction * math.copysign(1.0, v)

    def max_pressure_bar(self) -> float:
        """Maximum allowable pressure based on geometry (bar)."""
        return 350.0  # typical for this class
