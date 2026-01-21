"""Динамика экскаватора (правая часть ОДУ).

Реализует минимально работоспособный end-to-end шаг:
- распаковка состояния;
- давление -> сила -> момент;
- внешние моменты (гравитация + грунт) через LoadModel;
- угловые ускорения по J;
- эволюция длин цилиндров и давлений.

Важно:
- Давления clamp'ятся в диапазон [0, P_max].
- В сингулярных точках механики dθ/dl может быть 0 (тогда dl/dt = 0 по текущей формуле).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from hydrosim.config.mechanics import MechanicsConfig
from hydrosim.mechanics.cylinder_link import CylinderLinkMechanism
from hydrosim.physics.hydraulics import HydraulicModel
from hydrosim.physics.load_model import LoadModel


@dataclass
class DynamicState:
    # Углы
    theta_swing: float
    theta_boom: float
    theta_arm: float
    theta_bucket: float

    # Угловые скорости
    omega_swing: float
    omega_boom: float
    omega_arm: float
    omega_bucket: float

    # Длины цилиндров
    cyl_boom_length: float
    cyl_arm_length: float
    cyl_bucket_length: float

    # Давления
    pressure_boom: float
    pressure_arm: float
    pressure_bucket: float

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.theta_swing,
                self.theta_boom,
                self.theta_arm,
                self.theta_bucket,
                self.omega_swing,
                self.omega_boom,
                self.omega_arm,
                self.omega_bucket,
                self.cyl_boom_length,
                self.cyl_arm_length,
                self.cyl_bucket_length,
                self.pressure_boom,
                self.pressure_arm,
                self.pressure_bucket,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, y: np.ndarray) -> "DynamicState":
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] != 14:
            raise ValueError(f"DynamicState vector must have length 14; got {y.shape[0]}")

        return cls(
            theta_swing=float(y[0]),
            theta_boom=float(y[1]),
            theta_arm=float(y[2]),
            theta_bucket=float(y[3]),
            omega_swing=float(y[4]),
            omega_boom=float(y[5]),
            omega_arm=float(y[6]),
            omega_bucket=float(y[7]),
            cyl_boom_length=float(y[8]),
            cyl_arm_length=float(y[9]),
            cyl_bucket_length=float(y[10]),
            pressure_boom=float(y[11]),
            pressure_arm=float(y[12]),
            pressure_bucket=float(y[13]),
        )


def _clamp(P: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(P)))


def _bucket_theta_and_dtheta_dl(bucket_lever: Any, cyl_bucket_length: float) -> Tuple[float, float]:
    res = bucket_lever.solve_angle(cyl_bucket_length)
    if isinstance(res, (tuple, list)):
        theta = float(res[0])
        dtheta_dl = float(res[1]) if len(res) > 1 else 0.0
        return theta, dtheta_dl
    return float(res), 0.0


class ExcavatorDynamics:
    def __init__(
        self,
        mech_cfg: MechanicsConfig,
        hyd_model: HydraulicModel,
        load_model: LoadModel,
        boom_mech: CylinderLinkMechanism,
        arm_mech: CylinderLinkMechanism,
        bucket_lever: Any,
        *,
        scenario_profile: Any | None = None,
    ) -> None:
        self._mech = mech_cfg
        self._hyd = hyd_model
        self._load = load_model
        self._boom_mech = boom_mech
        self._arm_mech = arm_mech
        self._bucket_lever = bucket_lever
        self._scenario_profile = scenario_profile

    def rhs(self, t: float, y: np.ndarray, inputs: Dict[str, float]) -> np.ndarray:
        _t = float(t)
        state = DynamicState.from_vector(y)

        Pmax = float(self._hyd.p_max_Pa)
        P_boom = _clamp(state.pressure_boom, 0.0, Pmax)
        P_arm = _clamp(state.pressure_arm, 0.0, Pmax)
        P_bucket = _clamp(state.pressure_bucket, 0.0, Pmax)

        F_boom = self._hyd.force_from_pressure("boom", P_boom)
        F_arm = self._hyd.force_from_pressure("arm", P_arm)
        F_bucket = self._hyd.force_from_pressure("bucket", P_bucket)

        M_boom_cyl = self._boom_mech.cylinder_force_to_moment(state.cyl_boom_length, F_boom)
        M_arm_cyl = self._arm_mech.cylinder_force_to_moment(state.cyl_arm_length, F_arm)

        if not hasattr(self._bucket_lever, "cylinder_force_to_moment"):
            raise ValueError("bucket_lever must implement cylinder_force_to_moment() for dynamics")
        M_bucket_cyl = float(
            self._bucket_lever.cylinder_force_to_moment(state.cyl_bucket_length, F_bucket)
        )

        ext = self._load.external_moments(state=state, scenario_profile=self._scenario_profile, inputs=inputs)

        M_boom_total = float(M_boom_cyl) + float(ext["boom"])
        M_arm_total = float(M_arm_cyl) + float(ext["arm"])
        M_bucket_total = float(M_bucket_cyl) + float(ext["bucket"])

        J_boom = float(self._mech.boom_link.moment_of_inertia_pivot)
        J_arm = float(self._mech.arm_link.moment_of_inertia_pivot)
        J_bucket = float(self._mech.bucket_link.moment_of_inertia_pivot)

        domega_swing = 0.0
        domega_boom = M_boom_total / J_boom
        domega_arm = M_arm_total / J_arm
        domega_bucket = M_bucket_total / J_bucket

        _theta_boom, dtheta_dl_boom = self._boom_mech.solve_angle(state.cyl_boom_length)
        _theta_arm, dtheta_dl_arm = self._arm_mech.solve_angle(state.cyl_arm_length)
        _theta_bucket, dtheta_dl_bucket = _bucket_theta_and_dtheta_dl(
            self._bucket_lever, state.cyl_bucket_length
        )

        dcyl_boom_dt = float(dtheta_dl_boom) * float(state.omega_boom)
        dcyl_arm_dt = float(dtheta_dl_arm) * float(state.omega_arm)
        dcyl_bucket_dt = float(dtheta_dl_bucket) * float(state.omega_bucket)

        dP_boom_dt = self._hyd.pressure_rate(
            axis="boom",
            P_Pa=P_boom,
            dcyl_length_dt=dcyl_boom_dt,
            spool_position=float(inputs.get("boom_spool", 0.0)),
        )
        dP_arm_dt = self._hyd.pressure_rate(
            axis="arm",
            P_Pa=P_arm,
            dcyl_length_dt=dcyl_arm_dt,
            spool_position=float(inputs.get("arm_spool", 0.0)),
        )
        dP_bucket_dt = self._hyd.pressure_rate(
            axis="bucket",
            P_Pa=P_bucket,
            dcyl_length_dt=dcyl_bucket_dt,
            spool_position=float(inputs.get("bucket_spool", 0.0)),
        )

        dy = np.array(
            [
                state.omega_swing,
                state.omega_boom,
                state.omega_arm,
                state.omega_bucket,
                domega_swing,
                domega_boom,
                domega_arm,
                domega_bucket,
                dcyl_boom_dt,
                dcyl_arm_dt,
                dcyl_bucket_dt,
                dP_boom_dt,
                dP_arm_dt,
                dP_bucket_dt,
            ],
            dtype=np.float64,
        )
        return dy

    def __repr__(self) -> str:
        return "ExcavatorDynamics()"
