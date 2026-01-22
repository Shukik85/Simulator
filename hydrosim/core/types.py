"""hydrosim.core.types

Минимальные типы данных для Phase 0.

Важно: сейчас это skeleton, чтобы стабилизировать интерфейсы (state, config) и тесты.
В следующих фазах будем расширять и/или адаптировать под существующие модули hydrosim/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


PumpMode = Literal["off", "pressure_control", "flow_control"]


class OilProperties(TypedDict):
    """Свойства масла (точка измерения/модельная переменная)."""

    temperature_C: float
    density: float
    bulk_modulus: float
    viscosity_Pa_s: float


@dataclass(frozen=True, slots=True)
class CylinderState:
    x_m: float
    v_m_s: float
    p_a_pa: float
    p_b_pa: float


@dataclass(frozen=True, slots=True)
class PumpState:
    speed_rps: float
    p_out_pa: float
    q_out_m3_s: float
    mode: PumpMode


@dataclass(frozen=True, slots=True)
class ExcavatorState:
    """Упрощённый state-vector (Phase 0), без полной 40-мерной версии."""

    pump: PumpState
    boom: CylinderState
    arm: CylinderState
    bucket: CylinderState
    oil: OilProperties
