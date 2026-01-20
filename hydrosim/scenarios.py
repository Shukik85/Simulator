from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import math

from .config import SystemConfig
from .state import State


@dataclass(frozen=True)
class ScenarioProfile:
    mode: str
    duration_s: float
    payload_kg: float
    soil_factor: float
    aggressiveness: float
    description: str


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


class ScenarioGenerator:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample_profile(self, mode: str) -> ScenarioProfile:
        sim = self.cfg.sim
        mech = self.cfg.mech
        soil = self.cfg.soil

        payload = float(self.rng.uniform(0.0, mech.payload_max)) if sim.randomize_payload else 0.0
        aggress = float(self.rng.uniform(0.35, 0.95))
        soil_factor = float(1.0 + self.rng.uniform(-soil.randomness, soil.randomness)) if sim.randomize_soil else 1.0

        if mode == "digging_light":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.25 * payload, 0.8 * soil_factor, aggress, "Light digging")
        if mode == "digging_medium":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.6 * payload, 1.0 * soil_factor, aggress, "Medium digging")
        if mode == "digging_heavy":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.9 * payload, 1.3 * soil_factor, aggress, "Heavy digging")
        if mode == "swing_only":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.3 * payload, 0.2 * soil_factor, aggress, "Swing only")
        if mode == "boom_up":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.5 * payload, 0.2 * soil_factor, aggress, "Boom up")
        if mode == "boom_down":
            return ScenarioProfile(mode, sim.cycle_duration_s, 0.5 * payload, 0.2 * soil_factor, aggress, "Boom down")
        if mode == "combined":
            return ScenarioProfile(mode, sim.cycle_duration_s, payload, 1.1 * soil_factor, aggress, "Combined cycle")
        return ScenarioProfile("idle", sim.cycle_duration_s, 0.0, 0.0, 0.0, "Idle")

    def _pumpspeed(self, base: float, aggress: float, tnorm: float) -> float:
        # «оператор» поднимает обороты в активных фазах
        bump = 1.0 + 0.20 * aggress * math.sin(2.0 * math.pi * tnorm)
        return float(base * bump)

    def command(self, prof: ScenarioProfile, t: float) -> Dict[str, float]:
        T = max(prof.duration_s, 1e-9)
        tnorm = float(np.clip(t / T, 0.0, 1.0))
        ag = prof.aggressiveness

        u = {"boom": 0.0, "arm": 0.0, "bucket": 0.0, "swing": 0.0, "pumpspeed": 1800.0}

        if prof.mode.startswith("digging"):
            # фазы: заглубление -> черпание -> подъём/вынос -> поворот -> разгрузка -> возврат
            if tnorm < 0.18:
                k = smoothstep(tnorm / 0.18)
                u["boom"] = -lerp(0.15, 0.45, k) - 0.15 * ag
                u["arm"] = -lerp(0.10, 0.55, k) - 0.10 * ag
                u["bucket"] = +lerp(0.10, 0.35, k) + 0.10 * ag
                u["pumpspeed"] = self._pumpspeed(2100.0, ag, tnorm)
            elif tnorm < 0.42:
                k = smoothstep((tnorm - 0.18) / (0.42 - 0.18))
                u["boom"] = +lerp(0.10, 0.55, k) + 0.10 * ag
                u["arm"] = +lerp(0.15, 0.60, k) + 0.10 * ag
                u["bucket"] = -lerp(0.15, 0.70, k) - 0.10 * ag
                u["pumpspeed"] = self._pumpspeed(2200.0, ag, tnorm)
            elif tnorm < 0.60:
                k = smoothstep((tnorm - 0.42) / (0.60 - 0.42))
                u["swing"] = +0.55 * math.sin(2.0 * math.pi * k)
                u["boom"] = +0.15
                u["pumpspeed"] = self._pumpspeed(1900.0, ag, tnorm)
            elif tnorm < 0.72:
                k = smoothstep((tnorm - 0.60) / (0.72 - 0.60))
                u["bucket"] = +lerp(0.10, 0.85, k)
                u["pumpspeed"] = self._pumpspeed(1700.0, ag, tnorm)
            else:
                k = smoothstep((tnorm - 0.72) / (1.0 - 0.72))
                u["boom"] = -lerp(0.05, 0.35, k)
                u["arm"] = -lerp(0.05, 0.35, k)
                u["bucket"] = -lerp(0.10, 0.45, k)
                u["swing"] = -0.35 * math.sin(2.0 * math.pi * k)
                u["pumpspeed"] = self._pumpspeed(1800.0, ag, tnorm)

        elif prof.mode == "swing_only":
            u["swing"] = 0.75 * math.sin(2.0 * math.pi * tnorm)
            u["pumpspeed"] = self._pumpspeed(1600.0, ag, tnorm)

        elif prof.mode == "boom_up":
            u["boom"] = +0.70 if tnorm < 0.45 else -0.35
            u["pumpspeed"] = self._pumpspeed(2300.0, ag, tnorm)

        elif prof.mode == "boom_down":
            u["boom"] = -0.65 if tnorm < 0.45 else +0.25
            u["pumpspeed"] = self._pumpspeed(1900.0, ag, tnorm)

        elif prof.mode == "combined":
            # комбинируем dig + swing + dump + return
            if tnorm < 0.35:
                u["boom"] = -0.35 - 0.15 * ag
                u["arm"] = -0.45 - 0.10 * ag
                u["bucket"] = +0.25 + 0.10 * ag
                u["pumpspeed"] = self._pumpspeed(2200.0, ag, tnorm)
            elif tnorm < 0.55:
                k = smoothstep((tnorm - 0.35) / 0.20)
                u["boom"] = +0.35
                u["bucket"] = -0.55
                u["swing"] = +0.70 * math.sin(2.0 * math.pi * k)
                u["pumpspeed"] = self._pumpspeed(2100.0, ag, tnorm)
            elif tnorm < 0.68:
                u["bucket"] = +0.85
                u["pumpspeed"] = self._pumpspeed(1750.0, ag, tnorm)
            else:
                k = smoothstep((tnorm - 0.68) / 0.32)
                u["boom"] = -0.25 - 0.10 * ag
                u["arm"] = -0.25 - 0.10 * ag
                u["bucket"] = -0.35
                u["swing"] = -0.45 * math.sin(2.0 * math.pi * k)
                u["pumpspeed"] = self._pumpspeed(1800.0, ag, tnorm)

        else:
            u["pumpspeed"] = 1100.0

        # финальный клип
        for k in ("boom", "arm", "bucket", "swing"):
            u[k] = float(np.clip(u[k], -1.0, 1.0))
        u["pumpspeed"] = float(np.clip(u["pumpspeed"], 800.0, self.cfg.pump.max_speed_rpm))
        return u
