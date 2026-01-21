"""Модель внешних нагрузок (моментов) на звеньях экскаватора.

Включает:
- гравитационные моменты звеньев;
- момент сопротивления грунта на ковше (упрощённо).

Соглашения:
- Z — вертикаль (вверх). Если bucket_tip_z < 0, считается что ковш в грунте.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import math

from hydrosim.config.mechanics import MechanicsConfig
from hydrosim.mechanics.kinematics import ExcavatorKinematics


@dataclass(frozen=True)
class SoilConfig:
    """Конфигурация модели сопротивления грунта."""

    base_resistance_N: float = 25000.0
    vel_gain_N_per_m_s: float = 8000.0
    penetration_gain_N_per_m: float = 60000.0
    penetration_exp: float = 1.5
    angle_factor_max: float = 2.0
    optimal_attack_angle_deg: float = 30.0
    randomness: float = 0.25


def _get_soil_factor(scenario_profile: Any) -> float:
    if scenario_profile is None:
        return 1.0
    if isinstance(scenario_profile, dict):
        return float(scenario_profile.get("soil_factor", 1.0))
    return float(getattr(scenario_profile, "soil_factor", 1.0))


class LoadModel:
    """Модель внешних нагрузок (моментов) на звеньях экскаватора."""

    def __init__(
        self,
        mech_cfg: MechanicsConfig,
        soil_cfg: SoilConfig,
        kin: ExcavatorKinematics | None,
    ) -> None:
        self._mech = mech_cfg
        self._soil = soil_cfg
        self._kin = kin

    def soil_resistance(
        self,
        theta_bucket: float,
        v_bucket_tip_m_s: float,
        penetration_depth_m: float,
        *,
        soil_factor: float = 1.0,
    ) -> float:
        """Сила сопротивления грунта на ковше (Н)."""

        cfg = self._soil
        depth = max(0.0, float(penetration_depth_m))

        F_base = float(cfg.base_resistance_N) * float(soil_factor)
        F_vel = float(cfg.vel_gain_N_per_m_s) * abs(float(v_bucket_tip_m_s))
        F_pen = float(cfg.penetration_gain_N_per_m) * (depth ** float(cfg.penetration_exp))

        angle_deg = math.degrees(float(theta_bucket))
        angle_deviation_deg = abs(angle_deg - float(cfg.optimal_attack_angle_deg))

        # cos ожидает радианы
        angle_factor = 1.0 + (float(cfg.angle_factor_max) - 1.0) * math.cos(
            math.radians(angle_deviation_deg)
        )

        angle_factor = max(0.5, min(float(cfg.angle_factor_max), float(angle_factor)))
        return (F_base + F_vel + F_pen) * angle_factor

    def gravity_moment_link(
        self,
        link_mass_kg: float,
        link_com_offset_m: float,
        link_theta_rad: float,
        *,
        g_m_s2: float = 9.81,
    ) -> float:
        """Гравитационный момент: M = m * g * d * sin(θ)."""

        return float(link_mass_kg) * float(g_m_s2) * float(link_com_offset_m) * math.sin(
            float(link_theta_rad)
        )

    def external_moments(
        self,
        state: Any,
        scenario_profile: Any,
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        """Вычислить внешние моменты на boom/arm/bucket (Н·м)."""

        if self._kin is None:
            raise ValueError("LoadModel requires ExcavatorKinematics to compute external moments")

        try:
            boom_len = float(getattr(state, "cyl_boom_length"))
            arm_len = float(getattr(state, "cyl_arm_length"))
            bucket_len = float(getattr(state, "cyl_bucket_length"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "state must provide cyl_boom_length, cyl_arm_length, cyl_bucket_length"
            ) from e

        swing = float(getattr(state, "theta_swing", 0.0))
        omega_bucket = float(getattr(state, "omega_bucket", 0.0))

        kin_state = self._kin.forward(
            boom_cyl_length_m=boom_len,
            arm_cyl_length_m=arm_len,
            bucket_cyl_length_m=bucket_len,
            swing_angle_rad=swing,
        )

        M_gravity_boom = self.gravity_moment_link(
            link_mass_kg=self._mech.boom_link.mass_kg,
            link_com_offset_m=self._mech.boom_link.com_offset_m,
            link_theta_rad=kin_state.boom.theta_rad,
        )
        M_gravity_arm = self.gravity_moment_link(
            link_mass_kg=self._mech.arm_link.mass_kg,
            link_com_offset_m=self._mech.arm_link.com_offset_m,
            link_theta_rad=kin_state.arm.theta_rad,
        )
        M_gravity_bucket = self.gravity_moment_link(
            link_mass_kg=self._mech.bucket_link.mass_kg,
            link_com_offset_m=self._mech.bucket_link.com_offset_m,
            link_theta_rad=kin_state.bucket.theta_rad,
        )

        # Грунт (только на ковш).
        _x, _y, z = kin_state.bucket_tip_xyz
        penetration_depth = max(0.0, -float(z))

        v_bucket_tip = abs(omega_bucket) * float(self._mech.bucket_link.length_m)
        soil_factor = _get_soil_factor(scenario_profile)

        if penetration_depth > 0.0:
            F_soil = self.soil_resistance(
                theta_bucket=kin_state.bucket.theta_rad,
                v_bucket_tip_m_s=v_bucket_tip,
                penetration_depth_m=penetration_depth,
                soil_factor=soil_factor,
            )
            M_soil_bucket = F_soil * float(self._mech.bucket_link.length_m) * math.sin(
                float(kin_state.bucket.theta_rad)
            )
        else:
            M_soil_bucket = 0.0

        return {
            "boom": float(M_gravity_boom),
            "arm": float(M_gravity_arm),
            "bucket": float(M_gravity_bucket + M_soil_bucket),
        }

    def __repr__(self) -> str:
        return f"LoadModel(soil={self._soil})"
