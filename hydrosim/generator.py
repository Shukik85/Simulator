from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import SystemConfig
from .state import State
from .physics import HydraulicModel
from .scenarios import ScenarioGenerator
from .loads import LoadModel
from .faults import FaultConfig
from .sensors import SensorModel
from .logger import H5Logger, CycleMeta


@dataclass
class GeneratorSettings:
    out_dir: str = "out_dataset"
    n_cycles: int = 200
    p_faulty: float = 0.65


class DatasetGenerator:
    def __init__(self, cfg: SystemConfig, settings: GeneratorSettings):
        self.cfg = cfg
        self.settings = settings
        self.rng = np.random.default_rng(cfg.sim.seed)

        self.model = HydraulicModel(cfg)
        self.scenarios = ScenarioGenerator(cfg, self.rng)
        self.loads = LoadModel(cfg, self.rng)
        self.sensors = SensorModel(cfg, self.rng)

        self.logger = H5Logger(settings.out_dir)
        self.logger.write_graph()

    def _sample_mode(self) -> str:
        modes = list(self.cfg.sim.modes)
        w = np.array([self.cfg.sim.mode_distribution.get(m, 0.0) for m in modes], dtype=np.float64)
        if w.sum() <= 0:
            return modes[int(self.rng.integers(0, len(modes)))]
        w = w / w.sum()
        return str(self.rng.choice(modes, p=w))

    def run(self):
        dt = self.cfg.sim.dt
        steps = int(self.cfg.sim.cycle_duration_s / dt)

        s = State()
        if self.cfg.sim.randomize_oil_temp:
            s.Thydraulic = float(self.rng.uniform(30.0, 65.0))

        for cid in range(self.settings.n_cycles):
            mode = self._sample_mode()
            prof = self.scenarios.sample_profile(mode)
            faults = FaultConfig.sample(self.rng, p_any=self.settings.p_faulty)

            time = np.zeros((steps,), dtype=np.float32)
            obs_buf = {k: np.zeros((steps,), dtype=np.float32) for k in self.cfg.sensor.sensors.keys()}

            for i in range(steps):
                t = i * dt
                time[i] = t

                u = self.scenarios.command(prof, t)
                ext = self.loads.external_loads(s, prof, u)

                s, diag = self.model.rk4_step(s, u, ext, dt, faults)

                rpm = float(u.get("pumpspeed", 0.0))
                obs = self.sensors.observe(s, rpm=rpm, diag=diag, faults=faults)

                for k in obs_buf.keys():
                    obs_buf[k][i] = obs.get(k, np.nan)

            timeline = {"time": time, **obs_buf}

            faults_by_component = faults.labels_by_component(thr=0.25)
            faults_flat = faults.flat_labels(thr=0.25, prefix="fault__")

            meta = CycleMeta(
                cycle_id=cid,
                mode=prof.mode,
                duration_s=prof.duration_s,
                payload_kg=prof.payload_kg,
                soil_factor=prof.soil_factor,
                aggressiveness=prof.aggressiveness,
                faults_by_component=faults_by_component,
                faults_flat=faults_flat,
            )
            self.logger.log_cycle(meta, timeline)

            if (cid + 1) % 20 == 0:
                print(f"[{cid+1}/{self.settings.n_cycles}] written")

        self.logger.close()
        print("Done. Output:", Path(self.settings.out_dir).resolve())
