from __future__ import annotations

from typing import Dict
import numpy as np

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve, SimRHS
from hydrosim_v2.mechanics.loads import LoadModel
from hydrosim_v2.scenarios import ScenarioGenerator
from hydrosim_v2.logger.h5_logger import H5Logger, CycleMeta

CYL_NAMES = ("boom_cyl", "arm_cyl", "bucket_cyl")
SEC_NAMES = ("boom", "arm", "bucket")


class DatasetGenerator:
    def __init__(
        self,
        cfg: ExcavatorConfig,
        out_dir: str = "out_dataset",
        n_cycles: int = 200,
        live_plot: bool = False,
        plot_interval: int = 100,
    ) -> None:
        self.cfg = cfg
        self.n_cycles = n_cycles
        self.live_plot = live_plot
        self.plot_interval = plot_interval
        self.plotter = None
        if live_plot:
            from hydrosim_v2.visualization import LivePlotter
            self.plotter = LivePlotter()

        self.rng = np.random.default_rng(42)
        self.scenarios = ScenarioGenerator(self.rng)

        mech = cfg.mechanics
        ls_cfg = cfg.hydraulics

        pump = LSPump(ls_cfg.pump)
        relief = ReliefValve("main", ls_cfg.relief)
        valves = {}
        cylinders = {}
        for cn, sn in zip(CYL_NAMES, SEC_NAMES):
            geo = mech.cylinders()[cn]
            valves[cn] = LSValveSection(cn, ls_cfg.valve_sections[sn])
            cylinders[cn] = Cylinder(cn, geo, ls_cfg.cylinder_dynamics[sn], ls_cfg.fluid)

        loads = LoadModel(cfg)
        self.rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)
        self.logger = H5Logger(out_dir)
        self.logger.write_graph()

    def _sample_mode(self) -> str:
        modes = ["digging_light", "digging_medium", "combined", "boom_up", "boom_down"]
        return str(self.rng.choice(modes))

    def _build_timeline(self, states: list[SimState]) -> Dict[str, np.ndarray]:
        n = len(states)
        arrs: Dict[str, list] = {
            "time": [],
            "p_pump": [], "p_ls": [],
            "p_boom_a": [], "p_boom_b": [], "x_boom": [],
            "p_arm_a": [], "p_arm_b": [], "x_arm": [],
            "p_bucket_a": [], "p_bucket_b": [], "x_bucket": [],
        }
        for s in states:
            arrs["time"].append(s.t)
            arrs["p_pump"].append(s.p_pump)
            arrs["p_ls"].append(s.p_ls)
            arrs["p_boom_a"].append(s.p_a["boom_cyl"])
            arrs["p_boom_b"].append(s.p_b["boom_cyl"])
            arrs["x_boom"].append(s.cyl_pos["boom_cyl"])
            arrs["p_arm_a"].append(s.p_a["arm_cyl"])
            arrs["p_arm_b"].append(s.p_b["arm_cyl"])
            arrs["x_arm"].append(s.cyl_pos["arm_cyl"])
            arrs["p_bucket_a"].append(s.p_a["bucket_cyl"])
            arrs["p_bucket_b"].append(s.p_b["bucket_cyl"])
            arrs["x_bucket"].append(s.cyl_pos["bucket_cyl"])
        return {k: np.array(v, dtype=np.float32) for k, v in arrs.items()}

    def run(self) -> None:
        dt = 0.002
        steps_per_cycle = int(60.0 / dt)

        state = SimState()
        state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}

        for cid in range(self.n_cycles):
            mode = self._sample_mode()
            prof = self.scenarios.sample_profile(mode)

            if self.plotter:
                self.plotter.set_cycle_info(cid, prof.mode)

            states: list[SimState] = []
            state_buf = state.copy()

            for i in range(steps_per_cycle):
                t = i * dt
                u_sec = self.scenarios.command(prof, t)
                spools = {
                    "boom_cyl": u_sec.get("boom", 0.0),
                    "arm_cyl": u_sec.get("arm", 0.0),
                    "bucket_cyl": u_sec.get("bucket", 0.0),
                }
                pump_speed = u_sec.get("pumpspeed", 1800.0)
                state_buf = self.rhs.euler_step(state_buf, spools, pump_speed, dt)
                states.append(state_buf)

                if self.plotter and i % self.plot_interval == 0:
                    self.plotter.push(t, state_buf, spools)
                    self.plotter.refresh()

            timeline = self._build_timeline(states)
            meta = CycleMeta(
                cycle_id=cid,
                mode=prof.mode,
                duration_s=prof.duration_s,
                payload_kg=prof.payload_kg,
                soil_factor=prof.soil_factor,
                aggressiveness=prof.aggressiveness,
            )
            self.logger.log_cycle(meta, timeline)

            if (cid + 1) % 20 == 0:
                print(f"[{cid+1}/{self.n_cycles}] done")

        self.logger.close()
        if self.plotter:
            self.plotter.close()
        print(f"Dataset written to {self.logger.out_dir}")
