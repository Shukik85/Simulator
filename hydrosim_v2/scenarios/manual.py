from __future__ import annotations

from typing import Dict, Set, Tuple
import numpy as np


class ManualController:
    """Ручное управление через numpad/стрелки в окне matplotlib.

    NumLock ON (цифры):
      8/2 — Boom (±)
      4/6 — Arm (±)
      7/9 — Bucket (±)

    NumLock OFF (стрелки + PgUp/PgDn):
      Up/Down       — Boom (±)
      Left/Right    — Arm (±)
      PageUp/Down   — Bucket (±)

    Space — Toggle auto/manual
    R — Reset spools to zero
    """

    KEY_MAP: Dict[str, Tuple[str, int]] = {
        # numpad (NumLock ON)
        # Cylinder retract = boom UP, arm IN, bucket curl
        "8": ("boom", -1),  "2": ("boom", 1),
        "4": ("arm", -1),   "6": ("arm", 1),
        "7": ("bucket", -1), "9": ("bucket", 1),
        # стрелки (NumLock OFF)
        "up": ("boom", -1),    "down": ("boom", 1),
        "left": ("arm", -1),  "right": ("arm", 1),
        "pageup": ("bucket", -1), "pagedown": ("bucket", 1),
    }

    def __init__(self, ramp_rate: float = 2.0) -> None:
        self.ramp_rate = ramp_rate
        self.auto_mode = True
        self.keys_held: Set[str] = set()
        self.targets: Dict[str, float] = {"boom": 0.0, "arm": 0.0, "bucket": 0.0}
        self.current: Dict[str, float] = {"boom": 0.0, "arm": 0.0, "bucket": 0.0}

    def on_key_press(self, event) -> None:
        if event.key == " ":
            self.auto_mode = not self.auto_mode
            if self.auto_mode:
                self.targets = {"boom": 0.0, "arm": 0.0, "bucket": 0.0}
            return
        if event.key == "r" or event.key == "R":
            self.targets = {"boom": 0.0, "arm": 0.0, "bucket": 0.0}
            self.current = {"boom": 0.0, "arm": 0.0, "bucket": 0.0}
            return
        if event.key in self.KEY_MAP:
            self.keys_held.add(event.key)
            axis, sign = self.KEY_MAP[event.key]
            self.targets[axis] = float(np.clip(sign * 0.7, -1.0, 1.0))

    def on_key_release(self, event) -> None:
        if event.key in self.KEY_MAP:
            self.keys_held.discard(event.key)
            axis, sign = self.KEY_MAP[event.key]
            opposite_sign = -sign
            opposite_key = None
            for k in self.keys_held:
                if k in self.KEY_MAP:
                    a, s = self.KEY_MAP[k]
                    if a == axis and s == opposite_sign:
                        opposite_key = k
                        break
            if opposite_key is None:
                self.targets[axis] = 0.0

    def step(self, dt: float) -> Dict[str, float]:
        if self.auto_mode:
            return self.current
        for axis in ("boom", "arm", "bucket"):
            diff = self.targets[axis] - self.current[axis]
            step = self.ramp_rate * dt
            if abs(diff) < step:
                self.current[axis] = self.targets[axis]
            else:
                self.current[axis] += step * (1.0 if diff > 0 else -1.0)
            self.current[axis] = float(np.clip(self.current[axis], -1.0, 1.0))
        return dict(self.current)

    def state_str(self) -> str:
        mode = "MANUAL" if not self.auto_mode else "AUTO"
        b, a, bk = self.current["boom"], self.current["arm"], self.current["bucket"]
        return f"[{mode}] B:{b:+.2f}  A:{a:+.2f}  BK:{bk:+.2f}"
