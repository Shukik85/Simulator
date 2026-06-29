# hydrosim/main.py
from hydrosim.endpoint import build_model_and_initial_state
import numpy as np

from settings import EXCAVATOR_CONFIG

# === Сборка модели и начального состояния ===
dynamics, specs, system_state = build_model_and_initial_state()

print("✅ Модель построена. Начальные длины ГЦ:")
for name, cyl in system_state.cylinder_states.items():
    print(f"  {name}: {cyl.length_m:.3f} м")

# === Задаём изменение объёма масла (м³) ===
delta_volumes = [
    {"cylinder": "boom", "chamber": "piston", "dV": +0.001},
    {"cylinder": "arm", "chamber": "rod", "dV": +0.001},
    {"cylinder": "bucket", "chamber": "piston", "dV": +0.001},
]

# === Применяем изменения объёмов ===
for item in delta_volumes:
    name = item["cylinder"]
    chamber = item["chamber"]
    dV = item["dV"]
    spec = specs[name]
    state_cyl = system_state.cylinder_states[name]

    # Проверка на предел хода
    is_limited, direction = spec.is_motion_limited(state_cyl.length_m, dV, chamber)
    if is_limited:
        force_N = spec.get_force_at_max_pressure(chamber)
        print(f"  {name} достиг предела хода. Усилие = {force_N:.1f} Н")
        system_state.cylinder_states[name].set_force(force_N)
        continue

    # Вычисляем изменение длины
    dl = spec.get_length_change(dV, chamber)
    delta_l = dl if chamber == "piston" else -dl
    new_length = state_cyl.length_m + delta_l

    # Обновляем состояние
    system_state.cylinder_states[name].update_length(new_length, spec)
    print(f"  {name}: длина изменена на {dl:+.3f} м → {new_length:.3f} м")

# === Обновляем кинематику ===
print("\n🔄 Обновляем кинематику...")
kinematic_state = dynamics.forward(system_state)

# === Полная диагностика системы ===
print("\n🔧 ПОЛНАЯ ДИАГНОСТИКА ПОЛОЖЕНИЯ СИСТЕМЫ\n" + "="*50)

# База
print("📍 БАЗА:")
print("  Опора: (0, 0)")

# Boom
boom = kinematic_state.boom
print("\n🡺 СТРЕЛА (Boom):")
pivot = boom.pivot_xy
tip = boom.tip_xy
print(f"  pivot: ({pivot[0]:.2f}, {pivot[1]:.2f})")
print(f"  tip:   ({tip[0]:.2f}, {tip[1]:.2f})")
print(f"  angle: {np.degrees(boom.joint_angle_rad):.1f}°")

# Arm
arm = kinematic_state.arm
print("\n🡺 РУКОЯТЬ (Arm):")
pivot = arm.pivot_xy
tip = arm.tip_xy
print(f"  pivot: ({pivot[0]:.2f}, {pivot[1]:.2f})")
print(f"  tip:   ({tip[0]:.2f}, {tip[1]:.2f})")
print(f"  angle: {np.degrees(arm.joint_angle_rad):.1f}°")

# Bucket
bucket = kinematic_state.bucket
print("\n🡺 КОВШ (Bucket):")
pivot = bucket.pivot_xy
tip = bucket.tip_xy
print(f"  pivot: ({pivot[0]:.2f}, {pivot[1]:.2f})")
print(f"  tip:   ({tip[0]:.2f}, {tip[1]:.2f})")
print(f"  angle: {np.degrees(bucket.joint_angle_rad):.1f}°")
print(f"  bucket tip (xyz): ({kinematic_state.bucket_tip_xyz[0]:.2f}, {kinematic_state.bucket_tip_xyz[1]:.2f}, {kinematic_state.bucket_tip_xyz[2]:.2f})")

# 4-звенный механизм ковша
print("\n 4-ЗВЕННЫЙ МЕХАНИЗМ КОВША:")

# Точки на рукояти (в глобальной СК)
arm_data = {
    "cylinder_anchoring_on_arm": np.array(EXCAVATOR_CONFIG["links"]["arm"]["cylinder_anchoring_on_arm"]),
    "lever_pivot_on_arm": np.array(EXCAVATOR_CONFIG["links"]["arm"]["lever_pivot_on_arm"]),
    "bucket_pivot_on_arm": np.array(EXCAVATOR_CONFIG["links"]["arm"]["bucket_pivot_on_arm"]),
}

def rotate_and_translate(pose, local):
    c, s = np.cos(pose[1]), np.sin(pose[1])
    R = np.array([[c, -s], [s, c]])
    return pose[0] + R @ local

# arm_pose = (pivot, angle)
arm_pose = (np.array(arm.pivot_xy), arm.joint_angle_rad)
A = rotate_and_translate(arm_pose, arm_data["cylinder_anchoring_on_arm"])  # крепление корпуса ГЦ
C = rotate_and_translate(arm_pose, arm_data["lever_pivot_on_arm"])         # ось рычага
D = rotate_and_translate(arm_pose, arm_data["bucket_pivot_on_arm"])        # опора ковша

print(f"  A (крепление ГЦ):     ({A[0]:.3f}, {A[1]:.3f})")
print(f"  C (ось рычага):       ({C[0]:.3f}, {C[1]:.3f})")
print(f"  D (опора ковша):      ({D[0]:.3f}, {D[1]:.3f})")

# Точка E — крепление тяги на ковше
bucket_data = {
    "link_attachment_to_bucket": np.array(EXCAVATOR_CONFIG["links"]["bucket"]["link_attachment_to_bucket"])
}
bucket_pose = (np.array(bucket.pivot_xy), bucket.joint_angle_rad)
E = rotate_and_translate(bucket_pose, bucket_data["link_attachment_to_bucket"])
print(f"  E (тяга→ковш):        ({E[0]:.3f}, {E[1]:.3f})")

# === Рассчитываем усилия в ГЦ ===
print("\n📊 РАСЧЁТ УСИЛИЙ")
forces_N = dynamics.backward(system_state)

print("\n🎯 УСИЛИЯ В ГИДРОЦИЛИНДРАХ:")
for name, force in forces_N.items():
    print(f"  {name}_cyl: {force:+.1f} Н ({'выдвижение' if force > 0 else 'втягивание'})")

# === Итог ===
print("\n✅ Симуляция завершена.")