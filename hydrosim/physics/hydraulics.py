"""Простейшая гидравлическая модель (контракт для динамики).

На этом этапе нужна не «идеальная гидравлика», а стабильный и детерминированный
контракт для `ExcavatorDynamics.rhs()`:
- давление -> сила цилиндра;
- `pressure_rate()` -> dP/dt как функция spool и скорости изменения длины цилиндра.

Единицы:
- Давление: Па
- Площадь: м²
- Сила: Н
- Скорость изменения длины: м/с
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HydraulicConfig:
    """Конфигурация гидравлической модели.

    p_max_bar:
        Максимально допустимое давление (bar) для safety-clamp.

    response_rate_per_s:
        Скорость (1/с) первого порядка, с которой давление стремится к целевому
        значению, заданному spool_position.

    cyl_motion_gain_Pa_per_m:
        Насколько движение цилиндра разгружает/нагружает давление.
        Это упрощённый термин, чтобы давление реагировало на изменение объёма.
    """

    area_boom_m2: float = 0.005
    area_arm_m2: float = 0.004
    area_bucket_m2: float = 0.003

    p_max_bar: float = 350.0

    response_rate_per_s: float = 15.0
    cyl_motion_gain_Pa_per_m: float = 2.0e7


class HydraulicModel:
    def __init__(self, cfg: HydraulicConfig | None = None) -> None:
        self.cfg = cfg or HydraulicConfig()

    @property
    def p_max_Pa(self) -> float:
        return float(self.cfg.p_max_bar) * 1e5

    def piston_area_m2(self, axis: str) -> float:
        axis_norm = axis.strip().lower()
        if axis_norm == "boom":
            return float(self.cfg.area_boom_m2)
        if axis_norm == "arm":
            return float(self.cfg.area_arm_m2)
        if axis_norm == "bucket":
            return float(self.cfg.area_bucket_m2)
        raise ValueError(f"Unknown hydraulic axis: {axis}")

    def force_from_pressure(self, axis: str, P_Pa: float) -> float:
        return float(P_Pa) * self.piston_area_m2(axis)

    def pressure_rate(
        self,
        *,
        axis: str,
        P_Pa: float,
        dcyl_length_dt: float,
        spool_position: float,
    ) -> float:
        """Упрощённая динамика давления dP/dt.

        Модель:
        - Целевое давление зависит от |spool_position|: 0..Pmax.
        - Давление стремится к цели с first-order response_rate_per_s.
        - Движение цилиндра снижает давление (термин cyl_motion_gain_Pa_per_m).
        """

        u = float(spool_position)
        u = max(-1.0, min(1.0, u))

        P = float(P_Pa)
        P = max(0.0, min(self.p_max_Pa, P))

        P_target = abs(u) * self.p_max_Pa

        dP_dt = float(self.cfg.response_rate_per_s) * (P_target - P)
        dP_dt -= float(self.cfg.cyl_motion_gain_Pa_per_m) * float(dcyl_length_dt)
        return dP_dt
