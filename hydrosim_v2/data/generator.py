from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve, SimRHS
from hydrosim_v2.mechanics.loads import LoadModel
from hydrosim_v2.scenarios import ScenarioGenerator
from hydrosim_v2.scenarios.manual import ManualController
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
        live_control: bool = False,
        plot_interval: int = 100,
    ) -> None:
        self.cfg = cfg
        self.n_cycles = n_cycles
        self.live_plot = live_plot
        self.live_control = live_control
        self.plot_interval = plot_interval
        self.plotter = None
        self.controller: Optional[ManualController] = None
        if live_plot:
            from hydrosim_v2.visualization import LivePlotter
            self.plotter = LivePlotter()
            if live_control:
                self.controller = ManualController()
                self.plotter.fig.canvas.mpl_connect("key_press_event", self.controller.on_key_press)
                self.plotter.fig.canvas.mpl_connect("key_release_event", self.controller.on_key_release)

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

        self.cfg = cfg
        self.out_dir = out_dir
        loads = LoadModel(cfg)
        self.rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)
        self.logger = H5Logger(out_dir)
        self.logger.write_graph()
        self.pump_speed_cur = 1800.0

    def _ramp_pump_speed(self, target: float, dt: float, tau: float = 0.2) -> float:
        alpha = min(dt / max(tau, 1e-6), 1.0)
        self.pump_speed_cur += alpha * (target - self.pump_speed_cur)
        return self.pump_speed_cur

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

    def _set_gravity_pressures(self, state: SimState) -> None:
        """Set p_a/p_b for gravity balance on differential cylinders.

        Force balance: f_ext = p_a * A_p - p_b * A_a
        For f_ext < 0 (gravity pulls down): set p_a = tank, solve for p_b.
        For f_ext > 0 (gravity pushes up): set p_b = tank, solve for p_a.
        """
        mech = self.cfg.mechanics
        ext0 = self.rhs.loads.external_cylinder_forces(state)
        for name in CYL_NAMES:
            geo = mech.cylinders()[name]
            ahead = geo.area_piston_m2
            aann = geo.area_annulus_m2
            f_ext = ext0.get(name, 0.0)
            p_tank = 1.5e5
            if f_ext < 0:
                p_a_eq = p_tank
                p_b_eq = max(p_tank, (p_tank * ahead - f_ext) / max(aann, 1e-10))
            else:
                p_b_eq = p_tank
                p_a_eq = max(p_tank, (f_ext + p_tank * aann) / max(ahead, 1e-10))
            state.p_a[name] = p_a_eq
            state.p_b[name] = p_b_eq

    def run(self) -> None:
        dt = 0.002
        steps_per_cycle = int(60.0 / dt)

        state = SimState()
        state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}
        self._set_gravity_pressures(state)

        for cid in range(self.n_cycles):
            mode = self._sample_mode()
            prof = self.scenarios.sample_profile(mode)

            if self.plotter:
                self.plotter.set_cycle_info(cid, prof.mode)

            states: list[SimState] = []
            state_buf = state.copy()
            self.rhs.payload_kg = float(prof.payload_kg)

            for i in range(steps_per_cycle):
                t = i * dt

                if self.controller and not self.controller.auto_mode:
                    u_sec = self.controller.step(dt)
                    u_sec["pumpspeed"] = 1800.0
                else:
                    u_sec = self.scenarios.command(prof, t)

                spools = {
                    "boom_cyl": u_sec.get("boom", 0.0),
                    "arm_cyl": u_sec.get("arm", 0.0),
                    "bucket_cyl": u_sec.get("bucket", 0.0),
                }
                target_speed = u_sec.get("pumpspeed", 1800.0)
                pump_speed = self._ramp_pump_speed(target_speed, dt)
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

    def run_manual(self) -> None:
        """Интерактивное ручное управление: Space — переключить AUTO/MANUAL, закрыть окно — выход."""
        dt = 0.002

        state = SimState()
        state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}
        self._set_gravity_pressures(state)

        if not self.plotter or not self.controller:
            raise RuntimeError("run_manual requires live_plot=True and live_control=True")

        # Short settling to let LS pressure stabilize (20 steps)
        for _ in range(20):
            spools_0 = {"boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0}
            state = self.rhs.euler_step(state, spools_0, 1500.0, dt)

        step = 0
        all_states = []
        print("Manual control — Space: AUTO <-> MANUAL, R: сброс, закрыть окно — выход.")
        self.plotter.fig.suptitle(
            "Hydrosim v2 — Manual Control  |  [AUTO]  Space=toggle  R=reset",
            fontsize=14, fontweight="bold",
        )
        self.plotter.fig.canvas.draw_idle()

        try:
            while plt.fignum_exists(self.plotter.fig.number):
                t = step * dt

                if self.controller.auto_mode:
                    u_sec = {"boom": 0.0, "arm": 0.0, "bucket": 0.0, "pumpspeed": 1100.0}
                else:
                    u_sec = self.controller.step(dt)
                    u_sec["pumpspeed"] = 1800.0

                spools = {
                    "boom_cyl": u_sec["boom"],
                    "arm_cyl": u_sec["arm"],
                    "bucket_cyl": u_sec["bucket"],
                }
                pump_speed = self._ramp_pump_speed(u_sec["pumpspeed"], dt)
                state = self.rhs.euler_step(state, spools, pump_speed, dt)
                all_states.append(state)

                if step % self.plot_interval == 0:
                    self.plotter.push(t, state, spools)
                    mode_str = "AUTO" if self.controller.auto_mode else "MANUAL"
                    self.plotter.fig.suptitle(
                        f"Hydrosim v2 — Manual Control  |  {mode_str}  {self.controller.state_str()}",
                        fontsize=14, fontweight="bold",
                    )
                    self.plotter.refresh()

                step += 1
        finally:
            # Запись в HDF5 при выходе
            if all_states:
                timeline = self._build_timeline(all_states)
                from hydrosim_v2.logger.h5_logger import CycleMeta
                meta = CycleMeta(
                    cycle_id=0,
                    mode="manual_interactive",
                    duration_s=step * dt,
                    payload_kg=0.0,
                    soil_factor=0.0,
                    aggressiveness=0.0,
                )
                self.logger.log_cycle(meta, timeline)
                print(f"\nЗаписано: {len(all_states)} кадров в {self.logger.out_dir}/dataset.h5")

            if self.plotter:
                self.plotter.close()
            self.logger.close()
            print("Manual session ended.")
