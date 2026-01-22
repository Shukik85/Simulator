"""Пакет физики (нагрузки, динамика и т.п.)."""

from __future__ import annotations

from .hydraulic_model import FlowDiagnostics, HydraulicModel

__all__ = [
    "HydraulicModel",
    "FlowDiagnostics",
]
