from __future__ import annotations
from typing import Dict
import numpy as np
import math

from .config import SystemConfig
from .state import State
from .scenarios import ScenarioProfile


class LoadModel:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def external_loads(self, s: State, prof: ScenarioProfile, u: Dict[str, float]) -> Dict[str, float]:
        mech = self.cfg.mech
        soil = self.cfg.soil
        g = mech.g

        loads = {"boom": 0.0, "arm": 0.0, "bucket": 0.0, "swing": 0.0}

        # гравитация (упрощённая, но зависит от положения)
        xb = s.xboom / max(self.cfg.boom_cyl.stroke_m, 1e-6)
        xa = s.xarm / max(self.cfg.arm_cyl.stroke_m, 1e-6)
        xk = s.xbucket / max(self.cfg.bucket_cyl.stroke_m, 1e-6)

        loads["boom"] += mech.boom_mass * g * (0.4 + 0.6 * (1.0 - xb))
        loads["arm"] += mech.arm_mass * g * (0.2 + 0.8 * xa)
        loads["bucket"] += mech.bucket_mass * g * (0.2 + 0.8 * xk)

        # digging нагрузка на bucket: нелинейная по заглублению и скорости
        if prof.mode.startswith("digging") or prof.mode == "combined":
            penetration = float(np.clip(xk, 0.0, 1.0))
            rnd = float(1.0 + self.rng.uniform(-soil.randomness, soil.randomness))
            Fsoil = rnd * prof.soil_factor * (soil.base_resistance_N + soil.penetration_gain_N * penetration)
            Fvel = soil.vel_gain_N_per_m_s * abs(s.vbucket)
            Fpayload = prof.payload_kg * g
            loads["bucket"] += (Fsoil + Fvel + 0.25 * Fpayload)

            # arm/boom получают реакцию от копания
            loads["arm"] += 0.35 * loads["bucket"]
            loads["boom"] += 0.20 * loads["bucket"]

        # swing сопротивление (вязкое + «масса»)
        loads["swing"] += 1200.0 * math.tanh(s.omega / 0.5) + 250.0 * s.omega

        # режим boom_down — отрицательная нагрузка (помогает движению, даёт регенерацию)
        if prof.mode == "boom_down":
            loads["boom"] *= -0.65

        return loads
