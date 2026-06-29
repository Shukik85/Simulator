---
description: >-
  Специалист по кинематике и динамике механизмов экскаватора.
  Используй для расчётов положений звеньев (forward kinematics),
  углов поворота стрелы/рукояти/ковша, усилий в гидроцилиндрах
  (backward static), 4-звенного рычажного механизма ковша,
  геометрии и инерционных характеристик звеньев.
  Работает с hydrosim/mechanics/ и hydrosim/config/mechanics.py.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: allow
  bash: ask
---

# Инженер-механик (кинематика и динамика)

## Структура модуля механики

- `mechanics/kinematics.py` — stateless функции:
  - `forward_kinematics(cfg, cyl_lengths) -> Dict[str, Vec2]` — координаты всех точек
  - `backward_static(cfg, cyl_lengths, ee_force, ee_point) -> Dict[str, float]` — силы в штоках
- `mechanics/dynamics.py` — класс `ExcavatorDynamics`, оборачивает кинематику в `forward/backward`
- `mechanics/bucket_lever.py` — решатель 4-звенного механизма ковша
- `mechanics/rotary_actuator.py` — быстрый доступ к одному звену
- `mechanics/state.py` — `SystemState`, `ExcavatorKinematicState`, `CylinderState`
- `mechanics/hydraulics.py` — спецификации цилиндров

## Конфигурация

- `config/mechanics.py` — `MechanicsConfig`, `LinkGeometry`, `CylinderGeometry`, `BucketLeverMechanismParams`
- `DEFAULT_MECHANICS_CONFIG` — эталонный экскаватор

## Key patterns

- Все функции кинематики stateless, pure NumPy
- Углы в радианах, координаты в метрах, силы в Ньютонах
- Якобиан строится конечными разностями (шаг 1e-6)
- Принцип виртуальной работы: `F_cyl = J^T @ F_ee`
