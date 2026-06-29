"""Hydraulic cylinder model."""

from __future__ import annotations

import numpy as np

from hydrosim_v2.config.base import CylinderGeometry
from hydrosim_v2.config.hydraulics import CylinderDynamicsConfig, FluidConfig


def _sign_reg(x: float) -> float:
    return float(np.tanh(x * 1e4))


class Cylinder:
    def __init__(
        self,
        name: str,
        geo: CylinderGeometry,
        dyn: CylinderDynamicsConfig,
        fluid: FluidConfig,
    ) -> None:
        self.name = name
        self.geo = geo
        self.dyn = dyn
        self.fluid = fluid
        self.position = 0.0
        self.velocity = 0.0
        self.p_a = 0.0
        self.p_b = 0.0
        self.f_cyl = 0.0

    def length(self) -> float:
        return self.geo.length_min_m + self.position

    def _dp_dt(self, p: float, q_in: float, V: float, dV_dt: float) -> float:
        beta = self.fluid.bulk_modulus
        Vsafe = max(V, self.dyn.dead_volume_m3)
        return (beta / Vsafe) * (q_in - dV_dt)

    @property
    def V_a(self) -> float:
        A = self.geo.area_piston_m2
        return A * self.position + self.dyn.dead_volume_m3

    @property
    def V_b(self) -> float:
        A = self.geo.area_annulus_m2
        return A * (self.geo.stroke_m - self.position) + self.dyn.dead_volume_m3

    def step_mechanics(self, u: float) -> float:
        m = self.dyn.mass_equiv
        kd = self.dyn.visc_damping
        fc = self.dyn.coulomb_friction
        Aa = self.geo.area_piston_m2
        Ab = self.geo.area_annulus_m2
        Fa = self.p_a * Aa
        Fb = self.p_b * Ab
        Fhyd = Fa - Fb
        ffriction = float(kd * self.velocity + fc * _sign_reg(self.velocity))
        Fnet = Fhyd - ffriction + u
        a = Fnet / m
        return float(a)

    def step_flow(
        self,
        q_a: float,
        q_b: float,
        dt: float,
    ) -> tuple[float, float]:
        V_a = self.V_a
        V_b = self.V_b
        dVa_dt = self.geo.area_piston_m2 * self.velocity
        dVb_dt = -self.geo.area_annulus_m2 * self.velocity
        dp_a = self._dp_dt(self.p_a, q_a, V_a, dVa_dt)
        dp_b = self._dp_dt(self.p_b, q_b, V_b, dVb_dt)
        self.p_a += dp_a * dt
        self.p_b += dp_b * dt
        self.p_a = max(self.p_a, 0.0)
        self.p_b = max(self.p_b, 0.0)
        return self.p_a, self.p_b

    def step_position(self, a: float, dt: float) -> tuple[float, float]:
        self.velocity += a * dt
        v_max = 2.0
        self.velocity = float(np.clip(self.velocity, -v_max, v_max))
        self.position += self.velocity * dt
        Lmin = 0.0
        Lmax = self.geo.stroke_m
        if self.position < Lmin:
            self.position = Lmin
            self.velocity = 0.0
        elif self.position > Lmax:
            self.position = Lmax
            self.velocity = 0.0
        return self.position, self.velocity
