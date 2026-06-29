# hydrosim/mechanics/dynamics.py
# --------------------------------------------------------------
#  ИСПРАВЛЕННЫЙ ФАЙЛ
# --------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

# --- Импортируем только то, что уже есть в проекте -----------------
from hydrosim.config.mechanics import MechanicsConfig, LinkGeometry, CylinderGeometry
from hydrosim.mechanics.bucket_lever import BucketLeverParams, BucketLeverSolver
from hydrosim.mechanics.kinematics import forward_kinematics, backward_static
from hydrosim.mechanics.hydraulics import HydraulicCylinderSpec
from hydrosim.mechanics.state import SystemState, ExcavatorKinematicState

Vec2 = np.ndarray


@dataclass
class ExcavatorDynamics:
    """
    Объект, который соединяет гидравлические усилия (состояние)
    с механической моделью экскаватора.
    """
    cfg: MechanicsConfig                     # неизменяемая геометрия
    cylinder_specs: Dict[str, HydraulicCylinderSpec]   # массы, диаметры и т.д.

    # ------------------------------------------------------------------
    # Прямой проход: из текущих длин ГЦ получаем полную кинематическую
    # информацию (координаты, углы и т.д.).
    # ------------------------------------------------------------------
    def forward(self, state: SystemState) -> ExcavatorKinematicState:
        """
        Преобразует текущие длины гидроцилиндров в полное kinematic state.
        """
        cyl_lengths: Dict[str, float] = state.get_cylinder_lengths()

        # Прямая kinematics (чистая геометрия, без состояния)
        pts = forward_kinematics(self.cfg, cyl_lengths)

        # Из полученных точек формируем объекты KinematicLinkState.
        boom_state = self._make_link_state(
            name="boom",
            cyl_len=cyl_lengths["boom_cyl"],
            pts=pts,
            cyl_lengths=cyl_lengths,
        )
        arm_state = self._make_link_state(
            name="arm",
            cyl_len=cyl_lengths["arm_cyl"],
            pts=pts,
            cyl_lengths=cyl_lengths,
        )
        bucket_state = self._make_link_state(
            name="bucket",
            cyl_len=cyl_lengths["bucket_cyl"],
            pts=pts,
            cyl_lengths=cyl_lengths,
        )

        # Swing – отдельная ось, пока просто передаём угол из состояния.
        swing_angle = (
            state.kinematic_state.swing_angle_rad
            if state.kinematic_state
            else 0.0
        )

        kin_state = ExcavatorKinematicState(
            boom=boom_state,
            arm=arm_state,
            bucket=bucket_state,
            swing_angle_rad=swing_angle,
            bucket_tip_xyz=pts["bucket_tip"],
            gravity_global=np.array([0.0, 0.0, -9.81]),
        )
        state.set_kinematics(kin_state)
        return kin_state

    # ------------------------------------------------------------------
    # Обратный проход: из внешней силы/момента считаем требуемые
    # усилия в штоках гидроцилиндров.
    # ------------------------------------------------------------------
    def backward(self, state: SystemState) -> Dict[str, float]:
        """
        Вычисляет силы, которые должны создаваться в штоках ГЦ,
        чтобы уравновесить внешнюю силу, приложенную к кончику ковша
        (или к любой другой точке, указанной в state.external_forces).
        """
        if state.kinematic_state is None:
            raise RuntimeError("Сначала выполните forward().")

        cyl_lengths: Dict[str, float] = state.get_cylinder_lengths()

        # По‑умолчанию внешняя сила приложена к кончику ковша.
        ee_point = "bucket_tip"
        ee_force = np.array(
            state.external_forces.get(ee_point, [0.0, 0.0]),
            dtype=float,
        )

        # Обратная статика – уже реализована в kinematics.py
        cyl_forces: Dict[str, float] = backward_static(
            cfg=self.cfg,
            cyl_lengths=cyl_lengths,
            ee_force=ee_force,
            ee_point=ee_point,
        )
        state.set_cylinder_forces(cyl_forces)
        return cyl_forces

    # ------------------------------------------------------------------
    # Вспомогательная функция: построение KinematicLinkState из
    # результата forward_kinematics.
    # ------------------------------------------------------------------
    def _make_link_state(
        self,
        *,
        name: str,
        cyl_len: float,
        pts: Dict[str, Vec2],
        cyl_lengths: Dict[str, float],
    ) -> ExcavatorKinematicState.__annotations__["boom"]:  # тип KinematicLinkState
        """
        Преобразует геометрические данные точки в структуру,
        которую ожидает ExcavatorKinematicState.
        """
        geom: LinkGeometry = getattr(self.cfg, f"{name}_link")
        cyl: CylinderGeometry = getattr(self.cfg, f"{name}_cyl")

        # Угол звена – получаем из точек, которые уже посчитаны в forward_kinematics.
        if name == "boom":
            pivot = pts["base"]
            tip = pts["boom_tip"]
        elif name == "arm":
            pivot = pts["boom_joint"]
            tip = pts["arm_tip"]
        else:  # bucket
            pivot = pts["arm_joint"]
            tip = pts["bucket_tip"]

        # угол относительно глобальной оси X
        angle = np.arctan2(tip[1] - pivot[1], tip[0] - pivot[0])

        # Точка приложения силы от ГЦ – это место, где шток соединяется со звёном.
        # В нашей схеме это точка крепления штока цилиндра к звёну.
        force_app = pts.get(
            f"{name}_cyl_P",  # например, bucket_cyl_P, boom_cyl_P и т.д.
            tip,  # fallback – конец звена
        )

        # Центр масс звена (в глобальной СК)
        com_local = np.array(geom.com_local)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        com_g = pivot + R @ com_local

        # Масса и инерция звена (уже посчитаны в LinkGeometry)
        mass = geom.mass_kg
        inertia = geom.inertia_kgm2

        # При необходимости добавляем массы штока и поршня (из spec)
        spec = self.cylinder_specs[name]
        # Приближённо считаем, что масса штока и поршня сосредоточена в середине цилиндра.
        base_mount = pts.get(f"{name}_base_mount", pivot)
        rod_mount = pts.get(f"{name}_rod_mount", tip)
        cyl_mid = (base_mount + rod_mount) * 0.5

        # Добавляем к инерции и массе (параллельная ось theorem)
        r_cyl = cyl_mid - pivot
        inertia += spec.cylinder_mass_kg * (r_cyl @ r_cyl)
        mass += spec.cylinder_mass_kg + spec.piston_mass_kg

        # Центр масс всей assemblies (звено + шток + поршень)
        total_mass = mass
        com_total = com_g * mass
        if spec.cylinder_mass_kg:
            cyl_com = (base_mount + rod_mount) * 0.5  # грубая аппроксимация
            com_total += spec.cylinder_mass_kg * cyl_com
            total_mass += spec.cylinder_mass_kg
        if spec.piston_mass_kg:
            pis_com = cyl_com  # та же аппроксимация
            com_total += spec.piston_mass_kg * pis_com
            total_mass += spec.piston_mass_kg
        com_g = com_total / total_mass if total_mass > 0 else com_g

        # Производная угла по длине ГЦ (для обратного прохода, если понадобится)
        # Считаем её численно здесь же – центральная разность.
        delta = 1e-6
        plus_lengths = cyl_lengths.copy()
        minus_lengths = cyl_lengths.copy()
        plus_lengths[name] = cyl_len + delta
        minus_lengths[name] = cyl_len - delta

        pts_plus = forward_kinematics(self.cfg, plus_lengths)
        pts_minus = forward_kinematics(self.cfg, minus_lengths)

        ang_plus = np.arctan2(
            pts_plus[f"{name}_tip"][1] - pts_plus[f"{name}_joint"][1],
            pts_plus[f"{name}_tip"][0] - pts_plus[f"{name}_joint"][0],
        )
        ang_minus = np.arctan2(
            pts_minus[f"{name}_tip"][1] - pts_minus[f"{name}_joint"][1],
            pts_minus[f"{name}_tip"][0] - pts_minus[f"{name}_joint"][0],
        )
        dangle_dl = (ang_plus - ang_minus) / (2.0 * delta)

        return ExcavatorKinematicState.__annotations__["boom"](
            name=name,
            joint_angle_rad=float(angle),
            pivot=pivot,
            tip=tip,
            force_application_point=force_app,
            com=com_g,
            mass_kg=float(mass),
            inertia_kgm2=float(inertia),
            dangle_dl=float(dangle_dl),
            cylinder_com_global=None,   # если нужно – можно добавить
            piston_com_global=None,
        )
