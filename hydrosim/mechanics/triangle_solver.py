# hydrosim/mechanics/triangle_solver.py
"""Решатель треугольника для определения угла звена по длине гидроцилиндра."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

Vec2 = np.ndarray


def solve_triangle_angle(
    cylinder_length: float,
    parent_pivot_local: Vec2,
    parent_cylinder_anchor_local: Vec2,
    link_force_point_local: Vec2,
    parent_base_angle: float = 0.0  # Базовый угол родителя относительно оси X
) -> Tuple[float, float]:
    """
    Решает треугольник для определения угла звена и производной dθ/dl.
    
    Параметры:
    cylinder_length: длина гидроцилиндра
    parent_pivot_local: точка шарнира родителя в локальной системе родителя
    parent_cylinder_anchor_local: точка крепления цилиндра на родителе в локальной системе родителя
    link_force_point_local: точка приложения силы на звене в локальной системе звена
    parent_base_angle: базовый угол родителя относительно оси X
    
    Returns:
    (угол_звена_рад, dθ/dl)
    """
    
    # Преобразуем в numpy массивы
    A = np.array(parent_cylinder_anchor_local)  # Точка крепления цилиндра на родителе
    B = np.array(parent_pivot_local)            # Шарнир родителя (совпадает с pivot звена)
    C_local = np.array(link_force_point_local)  # Точка приложения силы на звене
    
    # Длины сторон (фиксированные геометрические параметры)
    AB = np.linalg.norm(A - B)                   # Расстояние от шарнира до точки крепления на родителе
    BC = np.linalg.norm(C_local)                 # Расстояние от шарнира до точки приложения силы на звене
    AC = cylinder_length                         # Длина цилиндра (изменяемый параметр)
    
    # Проверка на существование треугольника
    if not _is_triangle_possible(AB, BC, AC):
        raise ValueError(f"Треугольник невозможен с длинами: AB={AB}, BC={BC}, AC={AC}")
    
    # Решаем треугольник ABC используя теорему косинусов
    # Угол при точке B (угол звена относительно родителя)
    cos_angle_B = (AB**2 + BC**2 - AC**2) / (2 * AB * BC)
    cos_angle_B = np.clip(cos_angle_B, -1.0, 1.0)
    angle_B = np.arccos(cos_angle_B)
    
    # Определяем знак угла на основе геометрии
    # Учитываем базовый угол родителя
    vec_BA = A - B
    base_angle_BA = np.arctan2(vec_BA[1], vec_BA[0])
    
    # Корректируем угол с учётом ориентации родителя
    angle_B = base_angle_BA + parent_base_angle + angle_B
    
    # Вычисляем производную dθ/dl аналитически
    sin_angle_B = np.sin(angle_B - base_angle_BA - parent_base_angle)
    if abs(sin_angle_B) < 1e-8:
        dtheta_dl = 0.0
    else:
        dtheta_dl = -1.0 / (AB * BC * sin_angle_B)
    
    return angle_B, dtheta_dl


def _is_triangle_possible(a: float, b: float, c: float) -> bool:
    """Проверяет, возможен ли треугольник с данными сторонами."""
    return (a + b > c and 
            a + c > b and 
            b + c > a and
            abs(a - b) < c and
            abs(a - c) < b and
            abs(b - c) < a)


def solve_triangle_with_branch(
    cylinder_length: float,
    parent_pivot_local: Vec2,
    parent_cylinder_anchor_local: Vec2,
    link_force_point_local: Vec2,
    parent_base_angle: float = 0.0,
    branch_prev: Optional[int] = None
) -> Tuple[float, float, int]:
    """
    Решает треугольник с поддержкой ветвей (два возможных решения).
    
    Returns:
    (угол_звена_рад, dθ/dl, выбранная_ветвь)
    """
    
    # Получаем оба возможных решения (положительный и отрицательный углы)
    try:
        angle_pos, dtheta_dl_pos = solve_triangle_angle(
            cylinder_length, parent_pivot_local, 
            parent_cylinder_anchor_local, link_force_point_local,
            parent_base_angle
        )
    except ValueError:
        angle_pos, dtheta_dl_pos = parent_base_angle, 0.0
    
    # Второе решение (отрицательный угол)
    angle_neg = 2 * parent_base_angle - angle_pos if angle_pos != parent_base_angle else parent_base_angle
    dtheta_dl_neg = -dtheta_dl_pos if dtheta_dl_pos != 0.0 else 0.0
    
    # Выбираем ветвь
    if branch_prev is not None:
        # Используем предыдущую ветвь
        if branch_prev > 0:
            return angle_pos, dtheta_dl_pos, 1
        else:
            return angle_neg, dtheta_dl_neg, -1
    else:
        # Выбираем ветвь по умолчанию (положительную)
        return angle_pos, dtheta_dl_pos, 1


def get_cylinder_length_from_angle(
    angle_rad: float,
    parent_p极local: Vec2,
    parent_cylinder_anchor_local: Vec2,
    link_force_point_local: Vec2,
    parent_base_angle: float = 0.0
) -> float:
    """
    Обратная задача: вычисляет длину цилиндра по углу звена.
    """
    A = np.array(parent_cylinder_anchor_local)
    B = np.array(parent_pivot_local)
    C_local = np.array(link_force_point_local)
    
    # Корректируем угол с учётом базового угла родителя
    effective_angle = angle_rad - parent_base_angle
    
    # Поворачиваем точку приложения силы на эффективный угол
    c, s = np.cos(effective_angle), np.sin(effective_angle)
    R = np.array([[c, -s], [s, c]])
    C_global = B + R @ C_local
    
    # Длина цилиндра - расстояние между точками A и C_global
    return np.linalg.norm(A - C_global)