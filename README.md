# Hydrosim v2 — LS Excavator Simulator

Симулятор гидравлического экскаватора с Load-Sensing (LS) гидравликой.
Реализация на Python 3.14+.

## Архитектура

```
hydrosim_v2/
├── config/          — Dataclass-конфигурация (механика, гидравлика)
├── core/            — Примитивы: типы, единицы, состояние симуляции
├── kinematics/      — Forward/backward кинематика (3 звена)
├── hydraulics/      — LS насос, золотник, цилиндр, клапан, RHS ОДУ
├── mechanics/       — Обратная динамика, модель нагрузок
├── faults/          — Параметры неисправностей
├── scenarios/       — Сценарии движения (digging, boom, combined)
├── logger/          — HDF5-логгер для датасетов
├── data/            — Генератор датасета
├── tests/           — 52 теста (unit + integration)
└── .opencode/       — Агенты opencode
```

## Быстрый старт

```bash
pip install -r requirements_simulator.txt
python -m pytest hydrosim_v2/tests/ -q
```

## Генерация датасета

```python
from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.data.generator import DatasetGenerator

cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
gen = DatasetGenerator(cfg, out_dir="out_dataset", n_cycles=200)
gen.run()
```

## Дизайн-спецификации

Ссылки на математику и архитектуру: [LINKS.md](./LINKS.md)
