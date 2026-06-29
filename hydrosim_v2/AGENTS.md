# hydrosim_v2 — симулятор гидравлического экскаватора с LS-системой

## Цель
Генерация синтетических данных (временные ряды датчиков + multi-label отказы) для GNN-диагностики.

## Архитектура
- `config/` — датаклассы + YAML (ExcavatorConfig, LSConfig, ExcavatorMechanicsConfig)
- `core/` — Vec2, Vec3, единицы, валидаторы
- `kinematics/` — forward/backward кинематика, bucket-lever solver
- `hydraulics/` — LS pump, valve, cylinder, system RHS + RK4
- `faults/` — непрерывные fault-параметры, бинаризация для multi-label
- `data/` — генерация датасета, сенсоры, экспорт HDF5

## Отличия от v1
- Только 3 звена (boom/arm/bucket), без swing и travel
- LS вместо open-center
- Код с нуля, без обратной совместимости

## Конвенции
- Импорты: `from hydrosim_v2.xxx import yyy`
- Stateless функции для кинематики
- Frozen dataclasses для конфигов
- YAML для конфигурации машины

## Тесты
```bash
python -m pytest tests/
```
