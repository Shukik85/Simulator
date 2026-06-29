# hydrosim/mechanics/rotary_actuator.py
# --------------------------------------------------------------
#  ИСПРАВЛЕННЫЙ ФАЙЛ – тонкая оболочка вокруг kinematics.py
# --------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from hydrosim.config.mechanics import MechanicsConfig
from hydrosim.mechanics.kinematics import forward_kinematics, backward_static

Vec2 = np.ndarray


class RotaryActuator:
    """
    Универсальный расчёт для одного звена (стрела, рукоять или ковш).
    Предоставляет два метода:
        * solve_angle(cylinder_lengths) → угол звена (рад)
        * solve_force(cylinder_lengths, ee_force) → сила в штоке (Н)
    Всё построено на чисто‑геометрических функциях из kinematics.py,
    поэтому объект не хранит внутреннего состояния и полностью
    потокобезопасен.
    """

    def __init__(self, cfg: MechanicsConfig, link_name: str):
        """
        Parameters
        ----------
        cfg : MechanicsConfig
            Полный механический конфиг экскаватора.
        link_name : str
            Имя звена, которое будем обрабатывать:
                "boom", "arm" или "bucket".
        """
        if link_name not in ("boom", "arm", "bucket"):
            raise ValueError(f"Неподдерживаемое звено: {link_name}")
        self.cfg = cfg
        self.link_name = link_name
        self._cyl_name = f"{link_name}_cyl"   # например, "boom_cyl"

    # ------------------------------------------------------------------
    # Прямая задача: длина ГЦ → угол звена
    # ------------------------------------------------------------------
    def solve_angle(self, cylinder_lengths: Dict[str, float]) -> float:
        """
        Возвращает угол звена (рад) при заданных pin‑to‑pin длинах
        всех трёх гидроцилиндров.

        Parameters
        ----------
        cylinder_lengths : dict
            Ключи – имена цилиндров из cfg.cylinders()
            (например, "boom_cyl", "arm_cyl", "bucket_cyl").
            Значения – текущие pin‑to‑pin длины (м).

        Returns
        -------
        float
            Угол звена в радианах.
        """
        pts = forward_kinematics(self.cfg, cylinder_lengths)

        # Угол берём из разности векторов pivot → tip
        if self.link_name == "boom":
            pivot = pts["base"]
            tip = pts["boom_tip"]
        elif self.link_name == "arm":
            pivot = pts["boom_joint"]
            tip = pts["arm_tip"]
        else:  # bucket
            pivot = pts["arm_joint"]
            tip = pts["bucket_tip"]

        angle = np.arctan2(tip[1] - pivot[1], tip[0] - pivot[0])
        return float(angle)

    # ------------------------------------------------------------------
    # Обратная задача: внешняя сила на конце ковша → сила в штоке данного звена
    # ------------------------------------------------------------------
    def solve_force(
        self,
        cylinder_lengths: Dict[str, float],
        ee_force: Vec2,
        ee_point: str = "bucket_tip",
    ) -> float:
        """
        По известным длинам ГЦ и внешней силе, приложенной к точке
        ee_point (по‑умолчанию кончик ковша), возвращает требуемую
        силу в штоке данного звена (Н).

        Parameters
        ----------
        cylinder_lengths : dict
            Текущие длины всех трёх гидроцилиндров (см. ``solve_angle``).
        ee_force : np.ndarray shape (2,)
            Внешняя сила, приложенная к точке ee_point (в мировой СК).
        ee_point : str, optional
            Ключ из словаря, возвращаемого forward_kinematics,
            указывающий точку приложения силы.
            По‑умолчанию "bucket_tip" (конец ковша).

        Returns
        -------
        float
            Требуемая сила в штоке данного звена (Н). Положительная –
            вытягивает шток (увеличивает длину цилиндра).
        """
        forces = backward_static(
            cfg=self.cfg,
            cyl_lengths=cylinder_lengths,
            ee_force=ee_force,
            ee_point=ee_point,
        )
        return float(forces[self._cyl_name])
