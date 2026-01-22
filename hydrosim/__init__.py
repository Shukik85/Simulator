"""hydrosim package.

Важно: пакет не должен иметь побочных эффектов при импорте.
Поэтому здесь нет eager-import'ов тяжёлых модулей (генератор/физика/конфиги).

Импортируй нужное напрямую:
- from hydrosim.physics import HydraulicModel
- from hydrosim.config import SystemConfig
"""

from __future__ import annotations

__all__: list[str] = []
