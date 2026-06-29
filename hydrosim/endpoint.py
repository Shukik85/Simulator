"""Фабрика для создания модели экскаватора и начального состояния."""

from typing import Tuple, Dict

from hydrosim.mechanics.dynamics import ExcavatorDynamics, LinkGeometry
from hydrosim.mechanics.kinematics import ExcavatorKinematics, ExcavatorKinematicsStepper
from hydrosim.mechanics.hydraulics import HydraulicCylinderSpec
from hydrosim.mechanics.state import SystemState, CylinderState
from settings import EXCAVATOR_CONFIG

def build_model_and_initial_state() -> Tuple[ExcavatorDynamics, Dict[str, HydraulicCylinderSpec], SystemState]:
    """
    Создаёт и инициализирует модель экскаватора.
    Returns:
        Tuple с динамикой, спецификациями цилиндров и начальным состоянием
    """
    config = EXCAVATOR_CONFIG

    # Создаём спецификации гидроцилиндров
    cylinder_specs = _create_cylinder_specs(config)

    # Создаём геометрии звеньев
    link_geoms = _create_link_geometries(config)

    # Создаём кинематику
    kinematics = ExcavatorKinematics(config)
    stepper = ExcavatorKinematicsStepper(kinematics)

    # Создаём динамику
    dynamics = ExcavatorDynamics(stepper, link_geoms, cylinder_specs)

    # Создаём начальное состояние
    state = _create_initial_state(cylinder_specs, config)

    return dynamics, cylinder_specs, state
def _create_cylinder_specs(config: dict) -> Dict[str, HydraulicCylinderSpec]:
    """Создаёт спецификации гидроцилиндров из конфигурации."""
    cylinders = {name: data["hydraulic_cylinder"] for name, data in config.items()}
    specs = {}
    for name, cylinder_config in cylinders.items():
        specs[name] = HydraulicCylinderSpec(**cylinder_config)
    return specs

def _create_link_geometries(config: dict) -> Dict[str, LinkGeometry]:
    """Создаёт геометрии звеньев из конфигурации."""
    link_geoms = {}
    for name, data in config.items():
        if name == "base":
            continue  # База не является звеном в смысле динамики
            
        link_geoms[name] = LinkGeometry(
            mass_kg=data["mass_kg"],
            com_local=data["com_local"],
            child_pivot_local=data["child_pivot_local"],
            child_cylinder_anchoring_local=data["child_cylinder_anchoring_local"],
            force_application_local=data["force_application_local"]
        )

    return link_geoms
def _create_initial_state(specs: Dict[str, HydraulicCylinderSpec], config: dict) -> SystemState:
    """Создаёт начальное состояние системы."""
    cylinder_states = {}
    for name, spec in specs.items():
        # Начальная длина - середина хода
        initial_length = (spec.l_min + spec.l_max) / 2
        
        # Начальные объёмы полостей
        stroke_used = initial_length - spec.l_min
        volume_piston = spec.area_piston * stroke_used
        volume_rod = spec.area_rod * (spec.stroke - stroke_used)
        
        cylinder_states[name] = CylinderState(
            name=name,
            length_m=initial_length,
            volume_piston_chamber_m3=volume_piston,
            volume_rod_chamber_m3=volume_rod
        )

    return SystemState(
        cylinder_states=cylinder_states
    )
def get_initial_cylinder_lengths(specs: Dict[str, HydraulicCylinderSpec]) -> Dict[str, float]:
    """Возвращает начальные длины цилиндров (середина хода)."""
    return {
    name: (spec.l_min + spec.l_max) / 2
    for name, spec in specs.items()
    }