from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class FaultConfig:
    # интенсивности 0..1
    # --- pump ---
    pump_wear: float = 0.0
    relief_stuck_open: float = 0.0
    open_center_leak: float = 0.0

    # --- cylinders / motor internal leakage ---
    boom_internal_leak: float = 0.0
    arm_internal_leak: float = 0.0
    bucket_internal_leak: float = 0.0
    swing_internal_leak: float = 0.0

    # --- mechanical / thermal degradations (component-scoped) ---
    boom_seal_friction_increase: float = 0.0
    arm_seal_friction_increase: float = 0.0
    bucket_seal_friction_increase: float = 0.0
    swing_bearing_friction_increase: float = 0.0
    cooling_degraded: float = 0.0

    # --- sensors (можно позже вынести в отдельный "sensor" node, но сейчас держим как глобальные) ---
    sensor_pressure_drift: float = 0.0
    sensor_dropout: float = 0.0

    # --- valve bank (если захотите — можно сделать отдельный node "valve_bank") ---
    valve_deadband_increase: float = 0.0

    def labels_by_component(self, thr: float = 0.25) -> Dict[str, Dict[str, int]]:
        b = lambda x: int(float(x) >= thr)
        return {
            "pump": {
                "wear": b(self.pump_wear),
                "relief_stuck_open": b(self.relief_stuck_open),
                "open_center_leak": b(self.open_center_leak),
            },
            "boom_cyl": {
                "internal_leak": b(self.boom_internal_leak),
                "seal_friction_increase": b(self.boom_seal_friction_increase),
            },
            "arm_cyl": {
                "internal_leak": b(self.arm_internal_leak),
                "seal_friction_increase": b(self.arm_seal_friction_increase),
            },
            "bucket_cyl": {
                "internal_leak": b(self.bucket_internal_leak),
                "seal_friction_increase": b(self.bucket_seal_friction_increase),
            },
            "swing_motor": {
                "internal_leak": b(self.swing_internal_leak),
                "bearing_friction_increase": b(self.swing_bearing_friction_increase),
            },
            "oil": {
                "cooling_degraded": b(self.cooling_degraded),
            },
        }

    def flat_labels(self, thr: float = 0.25, prefix: str = "fault__") -> Dict[str, int]:
        out = {}
        byc = self.labels_by_component(thr=thr)
        for comp, d in byc.items():
            for k, v in d.items():
                out[f"{prefix}{comp}__{k}"] = int(v)
        return out

    @staticmethod
    def sample(rng: np.random.Generator, p_any: float = 0.65) -> "FaultConfig":
        if rng.random() > p_any:
            return FaultConfig()

        # pump-related
        pump_wear = float(rng.beta(2, 6)) if rng.random() < 0.45 else 0.0
        relief = float(rng.beta(2, 5)) if rng.random() < 0.25 else 0.0
        oc_leak = float(rng.beta(2, 6)) if rng.random() < 0.20 else 0.0

        # internal leakage
        boom_leak = float(rng.beta(2, 7)) if rng.random() < 0.30 else 0.0
        arm_leak = float(rng.beta(2, 7)) if rng.random() < 0.30 else 0.0
        bucket_leak = float(rng.beta(2, 7)) if rng.random() < 0.30 else 0.0
        swing_leak = float(rng.beta(2, 7)) if rng.random() < 0.25 else 0.0

        # friction/thermal
        boom_fr = float(rng.beta(2, 8)) if rng.random() < 0.18 else 0.0
        arm_fr = float(rng.beta(2, 8)) if rng.random() < 0.18 else 0.0
        bucket_fr = float(rng.beta(2, 8)) if rng.random() < 0.18 else 0.0
        swing_fr = float(rng.beta(2, 8)) if rng.random() < 0.18 else 0.0
        cooling_deg = float(rng.beta(2, 8)) if rng.random() < 0.15 else 0.0

        # sensors / valve
        drift = float(rng.beta(2, 6)) if rng.random() < 0.20 else 0.0
        dropout = float(rng.beta(2, 8)) if rng.random() < 0.15 else 0.0
        valve_db = float(rng.beta(2, 7)) if rng.random() < 0.15 else 0.0

        return FaultConfig(
            pump_wear=pump_wear,
            relief_stuck_open=relief,
            open_center_leak=oc_leak,
            boom_internal_leak=boom_leak,
            arm_internal_leak=arm_leak,
            bucket_internal_leak=bucket_leak,
            swing_internal_leak=swing_leak,
            boom_seal_friction_increase=boom_fr,
            arm_seal_friction_increase=arm_fr,
            bucket_seal_friction_increase=bucket_fr,
            swing_bearing_friction_increase=swing_fr,
            cooling_degraded=cooling_deg,
            sensor_pressure_drift=drift,
            sensor_dropout=dropout,
            valve_deadband_increase=valve_db,
        )
