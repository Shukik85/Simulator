# synth_bucket_mechanism.py
"""
Синтез 4-звенного механизма ковша.
Находит точку P (крепление штока) как пересечение:
  |P - A| = L_cyl
  |P - C| = L_lever
и проверяет, что |P - E| ≈ L_rod.
"""

import numpy as np
from typing import Optional

Vec2 = np.ndarray


def v2(x: float, y: float) -> Vec2:
    return np.array([x, y])


def circle_intersection(A: Vec2, r1: float, B: Vec2, r2: float) -> Optional[list[Vec2]]:
    """Возвращает точки пересечения двух окружностей."""
    d = np.linalg.norm(B - A)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h2 = r1**2 - a**2
    if h2 < 0:
        return None
    h = np.sqrt(h2)

    P = A + a * (B - A) / d
    n = np.array([-(B - A)[1], (B - A)[0]]) / d

    return [P + h * n, P - h * n]


def main():
    print("🔧 Синтез механизма ковша\n")

    # === ПАРАМЕТРЫ ИЗ settings.py ===
    arm_data = {
        "cylinder_anchoring_on_arm": v2(1.5, 0.2),   # A
        "lever_pivot_on_arm": v2(2.8, -0.05),        # C
        "bucket_pivot_on_arm": v2(3, 0),       # D
    }
    bucket_data = {
        "lever": {"length_m": 0.55},
        "linkage": {"rod_length_m": 0.55},
        "link_attachment_to_bucket": v2(0.1, 0.25),
    }
    specs = {"bucket": {"l_min": 1.1, "l_max": 1.7}}

    A = arm_data["cylinder_anchoring_on_arm"]
    C = arm_data["lever_pivot_on_arm"]
    D = arm_data["bucket_pivot_on_arm"]
    E_loc = bucket_data["link_attachment_to_bucket"]
    L_lever = bucket_data["lever"]["length_m"]
    L_rod = bucket_data["linkage"]["rod_length_m"]
    L_cyl = specs["bucket"]["l_min"]  # начальная длина

    # Предполагаем: bucket_angle = 0 → E_global = D + E_loc
    E_global = D + E_loc

    print("Параметры:")
    print(f"  A = {A}")
    print(f"  C = {C}")
    print(f"  D = {D}")
    print(f"  E_global = {E_global}")
    print(f"  L_cyl = {L_cyl}, L_lever = {L_lever}, L_rod = {L_rod}\n")

    # === ШАГ 1: найти P как пересечение |P-A| = L_cyl и |P-C| = L_lever ===
    points_P = circle_intersection(A, L_cyl, C, L_lever)
    if not points_P:
        print("❌ Окружности не пересекаются. Проверьте длины.")
        return

    print(f"✅ Найдены точки P:")
    candidates = []
    for i, P in enumerate(points_P):
        d_rod = np.linalg.norm(P - E_global)
        error = abs(d_rod - L_rod)
        candidates.append((P, d_rod, error))
        print(f"  P{i+1} = [{P[0]:.6f}, {P[1]:.6f}] → |P-E| = {d_rod:.6f} (ошибка = {error:.6f})")

    # === ШАГ 2: выбрать лучшую ===
    best_P, d_rod, err = min(candidates, key=lambda x: x[2])
    tol = 0.02

    if err < tol:
        print(f"\n✅ Подходящая точка P найдена!")
        print(f"  P_global = [{best_P[0]:.6f}, {best_P[1]:.6f}]")

        # === ШАГ 3: выразить P в ЛСК ковша ===
        P_in_bucket_frame = best_P - D
        print(f"\n📌 Для settings.py:")
        print(f'  "rod_end_attachment_on_lever": [{P_in_bucket_frame[0]:.6f}, {P_in_bucket_frame[1]:.6f}]')

        # Проверка
        print("\n🧪 Финальная проверка:")
        print(f"  |P-A| = {np.linalg.norm(best_P - A):.6f} ≈ {L_cyl}")
        print(f"  |P-C| = {np.linalg.norm(best_P - C):.6f} ≈ {L_lever}")
        print(f"  |P-E| = {np.linalg.norm(best_P - E_global):.6f} ≈ {L_rod}")
    else:
        print(f"\n❌ Ни одна точка P не даёт хорошей длины тяги.")
        print(f"   Мин. ошибка = {err:.6f} > {tol}")
        print("\n💡 Советы:")
        print("  - Скорректируйте E_loc или L_rod")
        print("  - Измените L_cyl (например, взять среднюю)")
        print("  - Пересмотрите положение A или C")


if __name__ == "__main__":
    main()