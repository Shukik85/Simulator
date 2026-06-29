---
name: excavator-simulation
description: >-
  Use when working on excavator hydraulic simulation code.
  Covers kinematic chains, bucket linkage (4-bar mechanism),
  hydraulic pump/valve/cylinder modelling, flow-sharing,
  fault simulation (pump wear, valve leakage), RK4 integration,
  and digging load modelling.
  Use for any code in hydrosim/ or projects with similar
  mechanical-hydraulic co-simulation.
---

# Моделирование гидравлического экскаватора (hydrosim)

## Архитектура

Проект симулирует гидравлический экскаватор для генерации синтетических данных диагностики.

### Уровни моделирования

1. **Кинематика** (`mechanics/kinematics.py`) — чистая геометрия 2D
2. **Статика** — принцип виртуальной работы (Jacobian)
3. **Гидравлика** (`physics.py`) — давления, расходы, RK4
4. **Тепло** — нагрев масла от потерь насоса

### Структура проекта

```
hydrosim/
├── main.py                    # точка входа
├── config.py                  # общие конфиги (SystemConfig и т.д.)
├── config/
│   ├── mechanics.py           # механическая геометрия
│   └── models.py              # системные конфиги
├── mechanics/
│   ├── kinematics.py          # forward/backward кинематика
│   ├── dynamics.py            # ExcavatorDynamics
│   ├── bucket_lever.py        # 4-звенный механизм ковша
│   ├── rotary_actuator.py     # упрощённый доступ к звену
│   ├── hydraulics.py          # спецификации цилиндров
│   ├── state.py               # структуры состояния
│   └── test_excavator.py      # тест
├── physics.py                 # HydraulicModel (основная)
├── physics/
│   ├── dynamics.py, hydraulics.py, load_model.py, flow_sharing.py
├── faults.py                  # модель отказов
├── sensors.py                 # сенсоры с шумом
├── endpoint.py                # фабрика моделей (устаревшая)
├── scenarios.py               # сценарии
├── generator.py               # генератор датасетов
└── loads.py                   # нагрузки
```

## Единицы измерения

- Давление: Па (но бар в конфигах: 1 бар = 1e5 Па)
- Расход: м³/с (LPM = литр/мин)
- Длина: м
- Усилие: Н
- Температура: °C
- Масса: кг
- Инерция: кг·м²

## Запуск теста

```bash
cd hydrosim && python -m hydrosim.mechanics.test_excavator
```

## Формулы

- Давление: `P = K/V * (Q - A*v - leak)`
- Orifice: `Q = Cd*A*sqrt(2*dp/rho)`
- Виртуальная работа: `F_cyl = J^T @ F_ee`
- RK4: классический для ODE
