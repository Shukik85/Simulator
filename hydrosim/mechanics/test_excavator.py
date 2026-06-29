#!/usr/bin/env python3
# test_excavator.py
# Минимальная проверка работы всей цепочки:
#   mechanics → bucket_lever → kinematics → dynamics → rotary_actuator

import numpy as np
from types import SimpleNamespace   # простой объект с атрибутами

from hydrosim.config.mechanics import DEFAULT_MECHANICS_CONFIG as cfg
from hydrosim.mechanics.state import SystemState
from hydrosim.mechanics.dynamics import ExcavatorDynamics
from hydrosim.mechanics.kinematics import forward_kinematics, backward_static
from hydrosim.mechanics.rotary_actuator import RotaryActuator

# ----------------------------------------------------------------------
# 1. Заглушка для спецификаций гидроцилиндров
# ----------------------------------------------------------------------
def make_cylinder_state(name: str, length_m: float) -> SimpleNamespace:
    """
    Возвращает объект, который одновременно:
      * имеет атрибут length_m – нужен SystemState.get_cylinder_lengths()
      * содержит все атрибуты, которые читает ExcavatorDynamics
        (cylinder_mass_kg, piston_mass_kg, bore_diameter_m,
         rod_diameter_m, stroke_m, l_min, max_pressure_*_bar).
    """
    return SimpleNamespace(
        length_m=length_m,                     # для SystemState
        cylinder_mass_kg=5.0,
        piston_mass_kg=2.0,
        bore_diameter_m=0.1,
        rod_diameter_m=0.05,
        stroke_m=0.5,
        l_min=0.2,
        max_pressure_piston_bar=200,
        max_pressure_rod_bar=150,
    )


# ----------------------------------------------------------------------
# 2. Инициализация состояния системы
# ----------------------------------------------------------------------
# Начальные длины цилиндров (м) – типичные рабочие значения
initial_lengths = {
    "boom_cyl": 2.30,
    "arm_cyl":  1.80,
    "bucket_cyl": 1.45,
}

# Формируем словарь cylinder_states, где каждое значение – объект с нужными атрибутами
cylinder_states = {
    name: make_cylinder_state(name, length_m)
    for name, length_m in initial_lengths.items()
}

state = SystemState(cylinder_states=cylinder_states)

# Внешняя сила, приложенная к кончику ковша (например, сопротивление грунта)
state.set_external_forces({
    "bucket_tip": np.array([-500.0, -1200.0])   # N, направлено назад и вниз
})

# ----------------------------------------------------------------------
# 3. Прямая kinematics (через низкоуровневую функцию)
# ----------------------------------------------------------------------
pts = forward_kinematics(cfg, state.get_cylinder_lengths())
print("=== Прямая kinematics (forward_kinematics) ===")
for k, v in pts.items():
    print(f"{k:20}: {v}")

# ----------------------------------------------------------------------
# 4. Обратная статика (через низкоуровневую функцию)
# ----------------------------------------------------------------------
cyl_forces = backward_static(
    cfg=cfg,
    cyl_lengths=state.get_cylinder_lengths(),
    ee_force=state.external_forces["bucket_tip"],
    ee_point="bucket_tip",
)
print("\n=== Обратная статика (backward_static) ===")
for name, f in cyl_forces.items():
    print(f"{name:12}: {f:8.2f} Н")

# ----------------------------------------------------------------------
# 5. Проверка принципа виртуальной работы (должно выполняться с хорошей точностью)
# ----------------------------------------------------------------------
delta = 1e-6
work_external = 0.0
work_internal = 0.0
for name in ["boom_cyl", "arm_cyl", "bucket_cyl"]:
    L_plus = state.get_cylinder_lengths().copy()
    L_minus = state.get_cylinder_lengths().copy()
    L_plus[name]  += delta
    L_minus[name] -= delta

    pts_plus  = forward_kinematics(cfg, L_plus)
    pts_minus = forward_kinematics(cfg, L_minus)

    dx = (pts_plus["bucket_tip"] - pts_minus["bucket_tip"]) / (2.0 * delta)
    work_external += np.dot(state.external_forces["bucket_tip"], dx) * delta
    work_internal += cyl_forces[name] * delta   # сила * dL

print(f"\nПринцип виртуальной работы:")
print(f"  Внешняя работа  = {work_external:.6e} Дж")
print(f"  Внутренняя работа = {work_internal:.6e} Дж")
print(f"  Разница          = {abs(work_external - work_internal):.2e} Дж")
assert abs(work_external - work_internal) < 1e-6, "Нарушение принципа виртуальной работы!"

# ----------------------------------------------------------------------
# 6. То же самое через высокоуровневые классы
# ----------------------------------------------------------------------
# Для ExcavatorDynamics нам всё ещё нужен словарь со спецификациями
# (масса штока/поршня, диаметры и т.д.). Создаём простой объект,
# содержащий только те атрибуты, которые действительно читаются.
def dummy_spec(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        cylinder_mass_kg=5.0,
        piston_mass_kg=2.0,
        bore_diameter_m=0.1,
        rod_diameter_m=0.05,
        stroke_m=0.5,
        l_min=0.2,
        max_pressure_piston_bar=200,
        max_pressure_rod_bar=150,
    )

cylinder_specs = {
    "boom": dummy_spec("boom"),
    "arm":  dummy_spec("arm"),
    "bucket": dummy_spec("bucket"),
}

dyn = ExcavatorDynamics(cfg=cfg, cylinder_specs=cylinder_specs)

# forward
kin_state = dyn.forward(state)
print("\n=== ExcavatorDynamics.forward() ===")
print(f"Угол стрелы (рад): {kin_state.boom.joint_angle_rad:.4f}")
print(f"Угол рукояти (рад): {kin_state.arm.joint_angle_rad:.4f}")
print(f"Угол ковша (рад):   {kin_state.bucket.joint_angle_rad:.4f}")

# backward
cyl_forces_dyn = dyn.backward(state)
print("\n=== ExcavatorDynamics.backward() ===")
for name, f in cyl_forces_dyn.items():
    print(f"{name:12}: {f:8.2f} Н")

# ----------------------------------------------------------------------
# 7. RotaryActuator – быстрый доступ к одному звену
# ----------------------------------------------------------------------
act_boom = RotaryActuator(cfg, link_name="boom")
theta_boom = act_boom.solve_angle(state.get_cylinder_lengths())
print("\n=== RotaryActuator (boom) ===")
print(f"Угол стрелы через RotaryActuator: {theta_boom:.4f} rad "
      f"({np.degrees(theta_boom):.2f} deg)")

force_boom = act_boom.solve_force(
    cylinder_lengths=state.get_cylinder_lengths(),
    ee_force=state.external_forces["bucket_tip"],
    ee_point="bucket_tip",
)
print(f"Сила в штоке стрелы: {force_boom:.2f} Н")

print("\nТест завершён без ошибок.")
