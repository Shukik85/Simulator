from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import math

from .config import SystemConfig
from .state import State
from .physics import PA_PER_BAR, FlowDiagnostics
from .faults import FaultConfig


class SensorModel:
    def __init__(self, cfg: SystemConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._drift_state = {k: 0.0 for k in cfg.sensor.sensors.keys()}

    def _range_span(self, name: str) -> float:
        lo, hi = self.cfg.sensor.sensors[name]["range"]
        return float(hi - lo)

    def _add_noise(self, name: str, value: float) -> float:
        sc = self.cfg.sensor
        if name.startswith("P"):
            std = (self._range_span(name) * sc.pressure_noise_pct / 100.0)
            return float(value + self.rng.normal(0.0, std))
        if name.startswith("X") or name.startswith("Theta"):
            std = (self._range_span(name) * sc.position_noise_pct / 100.0)
            return float(value + self.rng.normal(0.0, std))
        if name.startswith("T"):
            return float(value + self.rng.normal(0.0, sc.temp_noise_abs_C))
        if name == "pumpspeed":
            return float(value + self.rng.normal(0.0, sc.rpm_noise_abs))
        if name.startswith("Q"):
            # пусть будет небольшой относительный шум
            std = 0.01 * max(1.0, abs(value))
            return float(value + self.rng.normal(0.0, std))
        return float(value)

    def _apply_drift(self, name: str, value: float, faults: FaultConfig) -> float:
        if not name.startswith("P") or faults.sensor_pressure_drift <= 0:
            return value
        # дрейф как медленный случайный процесс (random walk)
        step = (0.002 * faults.sensor_pressure_drift) * self._range_span(name)
        self._drift_state[name] += float(self.rng.normal(0.0, step))
        return float(value + self._drift_state[name])

    def _maybe_dropout(self, name: str, value: float, faults: FaultConfig) -> float:
        if faults.sensor_dropout <= 0:
            return value
        # вероятность «провала» измерения на этом тике
        p = 0.02 * faults.sensor_dropout
        if self.rng.random() < p:
            return float("nan")
        return value

    def observe(self, s: State, rpm: float, diag: FlowDiagnostics, faults: FaultConfig) -> Dict[str, float]:
        out = {}

        # давления Pa -> bar
        out["Ppump"] = s.Ppump / PA_PER_BAR
        out["PLS"] = s.PLS / PA_PER_BAR
        out["PboomA"] = s.PboomA / PA_PER_BAR
        out["PboomB"] = s.PboomB / PA_PER_BAR
        out["ParmA"] = s.ParmA / PA_PER_BAR
        out["ParmB"] = s.ParmB / PA_PER_BAR
        out["PbucketA"] = s.PbucketA / PA_PER_BAR
        out["PbucketB"] = s.PbucketB / PA_PER_BAR
        out["PswingA"] = s.PswingA / PA_PER_BAR
        out["PswingB"] = s.PswingB / PA_PER_BAR

        # позиции m -> mm
        out["Xboom"] = s.xboom * 1000.0
        out["Xarm"] = s.xarm * 1000.0
        out["Xbucket"] = s.xbucket * 1000.0
        out["Thetaswing"] = float(np.degrees(s.theta))
        out["Thydraulic"] = float(s.Thydraulic)

        out["pumpspeed"] = float(rpm)

        # расходы m3/s -> lpm
        out["Qpump"] = diag.Qpump * 60.0 * 1000.0
        out["Qrelief"] = diag.Qrelief * 60.0 * 1000.0
        out["Qopen_center"] = diag.Qopen_center * 60.0 * 1000.0

        # шум/дрейф/дропаут
        for k in list(out.keys()):
            v = out[k]
            v = self._apply_drift(k, v, faults)
            v = self._add_noise(k, v)
            v = self._maybe_dropout(k, v, faults)
            out[k] = float(v)

        return out
