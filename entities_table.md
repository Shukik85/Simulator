### Таблица сущностей проекта и их назначение

| Модуль | Сущность | Назначение |
|--------|----------|-----------|
| `config.py` / `models.py` | `SystemConfig` | Глобальная конфигурация системы: гидравлика, насосы, клапаны, цилиндры, массы, тепловая модель и т.д. |
| `config.py` / `models.py` | `FluidConfig`, `PumpConfig`, `CylinderConfig` и др. | Конкретные компоненты системы, используемые внутри `SystemConfig`. |
| `config/mechanics.py` | `MechanicsConfig` | Механическая конфигурация экскаватора: геометрия звеньев, точки крепления цилиндров. |
| `state.py` | `State` | Состояние системы: давления, позиции цилиндров, углы, температура и т.д. |
| `physics.py` / `physics/hydraulic_model.py` | `HydraulicModel`, `FlowDiagnostics` | Модель гидравлической системы: потоки, давления, тепловая модель, динамика. |
| `physics/hydraulics.py` | `HydraulicModel` | Упрощённая гидравлическая модель для использования в динамике. |
| `physics/load_model.py` | `LoadModel` | Модель внешних нагрузок: гравитация, сопротивление грунта. |
| `physics/dynamics.py` | `ExcavatorDynamics` | Интегрируемая система динамики, объединяющая гидравлику, механику и нагрузки. |
| `mechanics/cylinder_link.py` | `CylinderLinkMechanism` | Механическая связь между цилиндром и звеном: по длине цилиндра вычисляет угол звена. |
| `mechanics/bucket_lever.py` | `BucketLeverMechanism` | Специализированная модель для ковшевого рычажного механизма. |
| `mechanics/kinematics.py` | `ExcavatorKinematics` | Прямая кинематика: из длин цилиндров вычисляет позиции звеньев. |
| `scenarios.py` | `ScenarioGenerator` | Генератор сценариев операций: копание, поворот и т.д. |
| `loads.py` | `LoadModel` | Модель внешних нагрузок, специфичная для сценария. |
| `sensors.py` | `SensorModel` | Модель датчиков с шумом, дрейфом и отказами. |
| `faults.py` | `FaultConfig` | Описание неисправностей (износ, утечки, дрейф датчиков и т.д.). |
| `logger.py` | `H5Logger` | Логирование результатов в формате HDF5 и JSON. |
| `generator.py` | `DatasetGenerator` | Генератор датасета: orchestrator, который объединяет все компоненты для генерации данных. |
| `generate_dataset.py` | `main` | Точка входа для запуска генерации датасета. |


### Граф зависимостей (импортов)

Граф зависимостей показывает, как модули импортируют друг друга. Стрелка `A -> B` означает, что `A` импортирует `B`.

```
hydrosim/__init__.py
    └── physics.HydraulicModel

hydrosim/generate_dataset.py
    └── hydrosim.config.SystemConfig

hydrosim/generator.py
    ├── hydrosim.config.SystemConfig
    ├── hydrosim.physics.HydraulicModel
    ├── hydrosim.scenarios.ScenarioGenerator
    ├── hydrosim.loads.LoadModel
    ├── hydrosim.sensors.SensorModel
    └── hydrosim.logger.H5Logger

hydrosim/physics.py
    ├── hydrosim.config.SystemConfig
    └── hydrosim.faults.FaultConfig

hydrosim/physics/hydraulic_model.py
    ├── hydrosim.config.models.SystemConfig
    ├── hydrosim.state.State
    └── hydrosim.faults.FaultConfig

hydrosim/physics/dynamics.py
    ├── hydrosim.config.mechanics.MechanicsConfig
    ├── hydrosim.physics.hydraulics.HydraulicModel
    └── hydrosim.mechanics.kinematics.ExcavatorKinematics

hydrosim/physics/load_model.py
    ├── hydrosim.config.mechanics.MechanicsConfig
    └── hydrosim.mechanics.kinematics.ExcavatorKinematics

hydrosim/mechanics/kinematics.py
    └── hydrosim.config.mechanics.MechanicsConfig

hydrosim/config/__init__.py
    └── hydrosim.config.models.SystemConfig

hydrosim/config/models.py
    ├── hydrosim.config.mechanics.MechanicsConfig
    └── hydrosim.config.models.MassPropertiesConfig

hydrosim/sensors.py
    ├── hydrosim.config.SystemConfig
    └── hydrosim.physics.FlowDiagnostics

hydrosim/loads.py
    ├── hydrosim.config.SystemConfig
    └── hydrosim.scenarios.ScenarioProfile

hydrosim/scenarios.py
    └── hydrosim.config.SystemConfig

hydrosim/mechanics/__init__.py
    └── hydrosim.mechanics.cylinder_link.CylinderAttachment

hydrosim/logger.py
    └── hydrosim.faults.FaultConfig

hydrosim/faults.py
    └── (самодостаточен)

hydrosim/state.py
    └── (самодостаточен)
```
