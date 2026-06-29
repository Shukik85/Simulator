"""kinematics.py

Чистая геометрическая кинематика и обратная статика для экскаватора
(плоская 2‑D модель).

* ``forward_kinematics`` – из известных длин гидроцилиндров
  вычисляет мировые координаты всех шарниров и характерных точек.
* ``backward_static``   – по известной внешней силе, приложенной
  к точке интереса (по‑умолчанию кончик ковша), возвращает
  требуемые силы в штоках гидроцилиндров (принцип виртуальной работы).

Все функции являются **stateless** и полностью потокобезопасны.
Они используют только данные из ``mechanics.py`` и ``bucket_lever.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from hydrosim.config.mechanics import (
    MechanicsConfig,
    LinkGeometry,
    CylinderGeometry,
    BucketLeverMechanismParams,  # алиас для обратной совместимости
)
from hydrosim.mechanics.bucket_lever import (
    BucketLeverParams,
    BucketLeverSolver,
    _DEFAULT_THETA_MIN,   # <-- импортируем значения по‑умолчанию
    _DEFAULT_THETA_MAX,
)

# ----------------------------------------------------------------------
# Типы и небольшие константы
# ----------------------------------------------------------------------
Vec2 = np.ndarray          # shape (2,), dtype=float64
_EPS = 1e-9                # tolerance for geometric checks


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def _rot2d(angle: float) -> np.ndarray:
    """2‑D матрица поворота вокруг начала координат."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s), (s, c)), dtype=float)


def _solve_link_angle(
    A: Vec2,
    P_local: Vec2,
    L_cyl: float,
) -> float:
    """
    Находит угол звена (рад) при известной длине гидроцилиндра,
    соединяющего фиксированную точку A (в базе) и точку P,
    заданную в локальной СК звена.

    Решение основано на законе косинусов для треугольника O‑A‑P,
    где O – начало локальной СК звена (шарнир), A – фиксированная
    точка на базе, P – точка крепления штока на звене.
    """
    OA = np.linalg.norm(A)               # |OA|
    OP = np.linalg.norm(P_local)         # |OP| (не зависит от угла)
    # Закон косинусов: cos(θ) = (OA² + OP² – L²) / (2·OA·OP)
    cos_theta = (OA * OA + OP * OP - L_cyl * L_cyl) / (2.0 * OA * OP)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # защита от численных шумов
    return np.arccos(cos_theta)              # угол от +X к OA


def _solve_link_angle_general(
    joint: Vec2,
    fixed_point_global: Vec2,
    rod_local: Vec2,
    cyl_len: float,
) -> float:
    """
    Угол звена (рад) при условии, что фиксированная точка крепления
    ГЦ находится в ``fixed_point_global``, а точка крепления штока
    задаётся в локальной СК звена ``rod_local``, а шарнир звена — в
    ``joint`` (все координаты – глобальные).

    Решение сводится к переводу системы координат в шарнир:
        A' = fixed_point_global - joint
    после чего задача идентична ``_solve_link_angle``.
    """
    A_prime = fixed_point_global - joint
    return _solve_link_angle(A_prime, rod_local, cyl_len)


# ----------------------------------------------------------------------
# Прямая kinematics
# ----------------------------------------------------------------------
def forward_kinematics(
    cfg: MechanicsConfig,
    cyl_lengths: Dict[str, float],
) -> Dict[str, Vec2]:
    """
    Прямая kinematics: из известных длин гидроцилиндров
    вычисляет мировые координаты всех шарниров и характерных точек.

    Parameters
    ----------
    cfg : MechanicsConfig
        Неизменяемый механический конфиг (длины звеньев,
        точки крепления и т.д.).
    cyl_lengths : dict
        Ключи – имена цилиндров из ``cfg.cylinders()``
        (например, ``"boom_cyl"``, ``"arm_cyl"``, ``"bucket_cyl"``).
        Значения – текущие pin‑to‑pin длины (м).

    Returns
    -------
    points : dict
        Словарь с ключами:
            ``"base"``               – origine мировой СК (0,0)
            ``"boom_joint"``         – шарнир база‑стрела
            ``"boom_tip"``           – конец стрелы (шарнир стрела‑рукоять)
            ``"arm_joint"``          – шарнир рукоять‑ковш
            ``"arm_tip"``            – конец рукояти (точка крепления ГЦ ковша
                                    на рукояти)
            ``"bucket_joint"``       – ось ковша D (точка крепления ковша к рукояти)
            ``"bucket_tip"``         – конец ковша (точка E)
            ``"bucket_com"``         – центр масс ковша (в глобальной СК)
            ``"bucket_cyl_P"``       – шток цилиндра ковша (точка P)
            ``"boom_base_mount"``    – точка крепления корпуса ГЦ стрелы на базе
            ``"boom_rod_mount"``     – точка крепления штока ГЦ стрелы на стреле
            ``"arm_base_mount"``     – точка крепления корпуса ГЦ рукояти на базе
            ``"arm_rod_mount"``      – точка крепления штока ГЦ рукояти на рукояти
            ``"bucket_base_mount"``  – точка крепления корпуса ГЦ ковша на базе
            ``"bucket_rod_mount"``   – точка крепления штока ГЦ ковша на ковше
        Все координаты – ``np.ndarray`` shape ``(2,)`` в мировой СК.
    """
    # ------------------------------------------------------------------
    # 0. База – origine мировой СК
    # ------------------------------------------------------------------
    base = np.array([0.0, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # 1. Стрела (boom)
    # ------------------------------------------------------------------
    boom_link: LinkGeometry = cfg.boom_link
    boom_cyl: CylinderGeometry = cfg.boom_cyl

    A_boom = np.array(boom_cyl.base_mount.point_local, dtype=float)   # на базе
    P_boom = np.array(boom_cyl.rod_mount.point_local, dtype=float)    # на стреле

    theta_boom = _solve_link_angle(A_boom, P_boom, float(cyl_lengths["boom_cyl"]))
    R_boom = _rot2d(theta_boom)

    boom_joint = base.copy()                     # шарнир база‑стрела (в нашей модели – origine)
    boom_tip = boom_joint + R_boom @ np.array(
        [boom_link.length_m, 0.0], dtype=float
    )                                            # конец стрелы

    boom_base_mount = base + A_boom               # крепление корпуса ГЦ стрелы на базе
    boom_rod_mount = boom_joint + R_boom @ P_boom  # крепление штока ГЦ стрелы на стреле

    # ------------------------------------------------------------------
    # 2. Рукоять (arm)
    # ------------------------------------------------------------------
    arm_link: LinkGeometry = cfg.arm_link
    arm_cyl: CylinderGeometry = cfg.arm_cyl

    A_arm = np.array(arm_cyl.base_mount.point_local, dtype=float)   # на базе
    P_arm = np.array(arm_cyl.rod_mount.point_local, dtype=float)    # на рукояти

    # Шарнир рукоять‑ковш находится на конце стрелы
    arm_joint = boom_tip.copy()

    # Для решения угла рукояти нужно перейти в систему координат шарнира:
    #   фиксированная точка крепления ГЦ (на базе) → вектор от шарнира к этой точке
    A_arm_global = base + A_arm               # позиция точки крепления ГЦ к рукояти (глобальная)
    A_prime = A_arm_global - arm_joint        # вектор от шарнира рукояти к фиксированной точке

    theta_arm = _solve_link_angle(A_prime, P_arm, float(cyl_lengths["arm_cyl"]))
    R_arm = _rot2d(theta_arm)

    arm_tip = arm_joint + R_arm @ np.array(
        [arm_link.length_m, 0.0], dtype=float
    )                                           # конец рукояти

    arm_base_mount = base + A_arm               # крепление корпуса ГЦ рукояти на базе
    arm_rod_mount = arm_joint + R_arm @ P_arm    # крепление штока ГЦ рукояти на рукояти

    # ------------------------------------------------------------------
    # 3. Ковш (bucket) – используем готовый решатель рычажного механизма
    # ------------------------------------------------------------------
    bucket_link: LinkGeometry = cfg.bucket_link
    bucket_cyl: CylinderGeometry = cfg.bucket_cyl

    # Преобразуем параметры ковша из mechanics к типу BucketLeverParams
    bucket_params = BucketLeverParams(
        L_lever=cfg.bucket_lever.lever_length_m,
        L_rod=cfg.bucket_lever.rod_length_m,
        L_bucket=bucket_link.length_m,
        mass=bucket_link.mass_kg,
        A=np.array(cfg.bucket_lever.anchor_A, dtype=float),   # фиксированная точка крепления ГЦ ковша (на базе)
        C=np.array(cfg.bucket_lever.pivot_C, dtype=float),   # ось рычага (на рукояти – будет учтена в решателе)
        D=np.array(cfg.bucket_lever.pivot_D, dtype=float),   # ось ковша (на рукояти – будет учтена в решателе)
        com_local=np.array(bucket_link.com_local, dtype=float),  # центр масс ковша в его локальной СК
        inertia=None,                # позволим решателю вычислить её автоматически
        # угловые пределы берём из механизма, если они заданы,
        # иначе используем значения по‑умолчанию из BucketLeverParams
        theta_min=getattr(cfg.bucket_lever, "theta_min", _DEFAULT_THETA_MIN),
        theta_max=getattr(cfg.bucket_lever, "theta_max", _DEFAULT_THETA_MAX),
    )

    L_bucket_cyl = float(cyl_lengths["bucket_cyl"])

    # Прямая кинематика ковша
    theta_bucket, points_bucket = BucketLeverSolver.solve(
        params=bucket_params,
        L_cyl=L_bucket_cyl,
        theta_guess=0.0,
        prev_P=None,
    )
    if theta_bucket is None:
        raise RuntimeError("Не удалось решить кинематику ковша для заданной длины ГЦ")

    # Из points_bucket берём нужные глобальные точки
    bucket_joint = points_bucket["D"]          # ось ковша (должна совпадать с arm_tip)
    bucket_tip = points_bucket["E"]            # конец ковша
    bucket_com = points_bucket["com"]          # центр масс ковша
    bucket_cyl_P = points_bucket["P"]          # шток цилиндра ковша

    # Точки крепления цилиндра ковша в мировой СК
    bucket_base_mount = points_bucket["A"]     # фиксированная точка крепления ГЦ ковша (на базе)
    bucket_rod_mount = bucket_cyl_P            # крепление штока ГЦ ковша на ковше

    # ------------------------------------------------------------------
    # 4. Формируем итоговый словарь точек
    # ------------------------------------------------------------------
    points: Dict[str, Vec2] = {
        "base": base,
        "boom_joint": boom_joint,
        "boom_tip": boom_tip,
        "arm_joint": arm_joint,
        "arm_tip": arm_tip,
        "bucket_joint": bucket_joint,
        "bucket_tip": bucket_tip,
        "bucket_com": bucket_com,
        "bucket_cyl_P": bucket_cyl_P,
        "boom_base_mount": boom_base_mount,
        "boom_rod_mount": boom_rod_mount,
        "arm_base_mount": arm_base_mount,
        "arm_rod_mount": arm_rod_mount,
        "bucket_base_mount": bucket_base_mount,
        "bucket_rod_mount": bucket_rod_mount,
    }
    return points


# ----------------------------------------------------------------------
# Обратная статика (принцип виртуальной работы)
# ----------------------------------------------------------------------
def backward_static(
    cfg: MechanicsConfig,
    cyl_lengths: Dict[str, float],
    ee_force: Vec2,
    ee_point: str = "bucket_tip",
    delta_L: float = 1e-6,
) -> Dict[str, float]:
    """
    Обратная статика: по известной внешней силе, приложенной к точке
    ``ee_point`` (по‑умолчанию кончик ковша), вычисляет требуемые
    силы в штоках гидроцилиндров.

    Parameters
    ----------
    cfg : MechanicsConfig
        Механический конфиг (тот же, что использовался в forward_kinematics).
    cyl_lengths : dict
        Текущие длины всех трёх гидроцилиндров
        (ключи такие же, как в ``forward_kinematics``).
    ee_force : np.ndarray shape (2,)
        Внешняя сила, приложенная к точке ``ee_point`` (в мировой СК).
    ee_point : str, optional
        Ключ из словаря, возвращаемого ``forward_kinematics``,
        указывающий точку приложения силы.
        По‑умолчанию ``"bucket_tip"`` (конец ковша).
    delta_L : float, optional
        Шаг для численного дифференцирования Якобиана (метры).
        По‑умолчанию 1e‑6 м – достаточен для точности ~1e‑9 в силе.

    Returns
    -------
    cyl_forces : dict
        Словарь с теми же ключами, что и ``cyl_lengths``.
        Значения – требуемая сила в штоке соответствующего цилиндра (Н).
        Положительная сила – вытягивает шток (увеличивает длину цилиндра).
    """
    # ------------------------------------------------------------------
    # 1. Текущая конфигурация (прямая kinematics)
    # ------------------------------------------------------------------
    pts = forward_kinematics(cfg, cyl_lengths)
    if ee_point not in pts:
        raise KeyError(f"Точка '{ee_point}' отсутствует в результате forward_kinematics")
    # x0 не используется дальше – оставляем для совместимости с предыдущим кодом
    _ = pts[ee_point].copy()

    # ------------------------------------------------------------------
    # 2. Строим Якобиан J (2 x n_cyl) конечными разностями
    # ------------------------------------------------------------------
    cyl_names = list(cyl_lengths.keys())
    n_cyl = len(cyl_names)
    J = np.zeros((2, n_cyl), dtype=float)

    for i, name in enumerate(cyl_names):
        L_plus = cyl_lengths.copy()
        L_minus = cyl_lengths.copy()
        L_plus[name] += delta_L
        L_minus[name] -= delta_L

        pts_plus = forward_kinematics(cfg, L_plus)
        pts_minus = forward_kinematics(cfg, L_minus)

        dx = (pts_plus[ee_point] - pts_minus[ee_point]) / (2.0 * delta_L)
        J[:, i] = dx

    # ------------------------------------------------------------------
    # 3. Обобщённые усилия в приводах: τ = J^T * F_ee
    # ------------------------------------------------------------------
    # Поскольку наши обобщённые координаты – это длины цилиндров
    # (prismatic joints), τ уже является требуемой силой в штоке.
    tau = J.T @ ee_force   # shape (n_cyl,)

    cyl_forces: Dict[str, float] = {name: float(tau[i]) for i, name in enumerate(cyl_names)}
    return cyl_forces
