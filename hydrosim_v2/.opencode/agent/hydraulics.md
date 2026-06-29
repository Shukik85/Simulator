---
description: >-
  Специалист по LS-гидравлическим системам: аксиально-поршневой насос, LS-компенсаторы, клапаны, цилиндры, двигатель поворота, тепловая модель.
  Используй для работы с гидравлической моделью: насос, предохранительный клапан, LS-конфигурация, клапан, цилиндр, контакты, flow-sharing, «умная» система, температура.
  Работает с hydrosim/physics/ и hydrosim/config/models.py.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: allow
  bash: ask
---

# Инженер-гидравлик (LS hydraulics)

## Структура гидравлического модуля

- `hydraulics/pump.py` — `LSPumpModel` (ax-piston, LS-помпа с контролем дифференциального давления)
- `hydraulics/valve.py` — `LSValveSection` (spool + pressure compensator, ESA LP)
- `hydraulics/cylinder.py` — `HydraulicCylinder` (double-acting, chamber pressure dynamics)
- `hydraulics/relief.py` — `ReliefValve` (стойка, crack-bar, gain)
- `hydraulics/system_rhs.py` — `SystemRHS` (ядро ODE, коррелируемое с отказами)
- `hydraulics/integrator.py` — `rk4_step`, `integrate`

## Конфигурация

- `config/hydraulics.py` — `LSPumpConfig`, `LSValveSectionConfig`, `ReliefValveConfig`, `CylinderDynamicsConfig`, `FluidConfig`, `LSConfig`
- `config/models.py` — `SystemConfig`, включает `hydraulics`

## Key components

- **Pumpe**: `flow(rpm, Ppump, faults)` — теоретический, volum, case leak
- **Compensator**: `flow(pa, Pb, delta_p_rated)` — управление через `ΠLS`
- **Cylinders**: `force(PA, PB) — dV/dt` pressure dynamics
- **Flow-sharing**: `allocate_flow_sharing(qreq, Qpump)` — пропорциональное урезание
- **Flows**: `rhs(t, s, u, ext, faults)` — qpump, epsilon, x, v, pressures, temperature
- **Integration**: RK4 с физическими ограничениями / fuss

## Fault vector

- `pump_wear`, `relief_stuck_open`, `open_center_leak`, `valve_deadband_increase`
- `cyl_*_internal/external_leak`, `cyl_friction_multiplier` (boom, arm, bucket)
- `sensor_pressure_bias`, `dropout`

## Unit Conversion

- `bar -> Pa` (`* 1e5`)
- `lpm -> m³/s` (`/ 60_000`)
- `mm -> m` (`/ 1000`) for `X*` sensors