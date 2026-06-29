from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydrosim_v2.core.sim_state import SimState

CYL_NAMES = ("boom_cyl", "arm_cyl", "bucket_cyl")
CYL_LABELS = ("Boom", "Arm", "Bucket")
CYL_COLORS = ("#e41a1c", "#377eb8", "#4daf4a")


class LivePlotter:
    def __init__(self, max_points: int = 2000) -> None:
        self.max_points = max_points
        self.fig, axes = plt.subplots(3, 2, figsize=(16, 10))
        self.fig.suptitle("Hydrosim v2 — Live Simulation", fontsize=14, fontweight="bold")
        self.axes = axes.flatten()

        self.lines: Dict[str, list[Line2D]] = {}
        self._setup_plots()

        self.buf_t: np.ndarray = np.zeros(max_points, dtype=np.float64)
        self.buf: Dict[str, np.ndarray] = {
            k: np.zeros(max_points, dtype=np.float64) for k in (
                "x_boom", "x_arm", "x_bucket",
                "v_boom", "v_arm", "v_bucket",
                "pa_boom", "pa_arm", "pa_bucket",
                "pb_boom", "pb_arm", "pb_bucket",
                "p_pump", "p_ls",
                "sp_boom", "sp_arm", "sp_bucket",
            )
        }
        self.n = 0

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.ion()
        plt.show(block=False)

    def _setup_plots(self) -> None:
        titles = [
            "Cylinder Positions (m)",
            "Cylinder Velocities (m/s)",
            "Chamber A Pressure (bar)",
            "Chamber B Pressure (bar)",
            "Pump & LS Pressure (bar)",
            "Spool Commands",
        ]
        ylabels = ["Position (m)", "Velocity (m/s)", "Pressure (bar)", "Pressure (bar)", "Pressure (bar)", "Spool [-1,1]"]

        for idx, (title, ylbl) in enumerate(zip(titles, ylabels)):
            ax = self.axes[idx]
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylbl)
            ax.grid(True, alpha=0.3)

        self.lines["pos"] = []
        for i, (lbl, clr) in enumerate(zip(CYL_LABELS, CYL_COLORS)):
            (ln,) = self.axes[0].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["pos"].append(ln)
        self.axes[0].legend(fontsize=8)

        self.lines["vel"] = []
        for i, (lbl, clr) in enumerate(zip(CYL_LABELS, CYL_COLORS)):
            (ln,) = self.axes[1].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["vel"].append(ln)
        self.axes[1].legend(fontsize=8)

        self.lines["pa"] = []
        for i, (lbl, clr) in enumerate(zip(CYL_LABELS, CYL_COLORS)):
            (ln,) = self.axes[2].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["pa"].append(ln)
        self.axes[2].legend(fontsize=8)

        self.lines["pb"] = []
        for i, (lbl, clr) in enumerate(zip(CYL_LABELS, CYL_COLORS)):
            (ln,) = self.axes[3].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["pb"].append(ln)
        self.axes[3].legend(fontsize=8)

        self.lines["pump"] = []
        for lbl, clr in [("Pump", "#000000"), ("LS", "#ff7f00")]:
            (ln,) = self.axes[4].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["pump"].append(ln)
        self.axes[4].legend(fontsize=8)

        self.lines["sp"] = []
        for i, (lbl, clr) in enumerate(zip(CYL_LABELS, CYL_COLORS)):
            (ln,) = self.axes[5].plot([], [], label=lbl, color=clr, lw=1.5)
            self.lines["sp"].append(ln)
        self.axes[5].legend(fontsize=8)

    def push(self, t: float, state: SimState, spools: Dict[str, float]) -> None:
        idx = self.n % self.max_points
        self.buf_t[idx] = t
        for name, key in [("boom_cyl", "x_boom"), ("arm_cyl", "x_arm"), ("bucket_cyl", "x_bucket")]:
            self.buf[key][idx] = state.cyl_pos[name]
        for name, key in [("boom_cyl", "v_boom"), ("arm_cyl", "v_arm"), ("bucket_cyl", "v_bucket")]:
            self.buf[key][idx] = state.cyl_vel[name]
        for name, key in [("boom_cyl", "pa_boom"), ("arm_cyl", "pa_arm"), ("bucket_cyl", "pa_bucket")]:
            self.buf[key][idx] = state.p_a[name] / 1e5
        for name, key in [("boom_cyl", "pb_boom"), ("arm_cyl", "pb_arm"), ("bucket_cyl", "pb_bucket")]:
            self.buf[key][idx] = state.p_b[name] / 1e5
        self.buf["p_pump"][idx] = state.p_pump / 1e5
        self.buf["p_ls"][idx] = state.p_ls / 1e5
        for name, key in [("boom_cyl", "sp_boom"), ("arm_cyl", "sp_arm"), ("bucket_cyl", "sp_bucket")]:
            self.buf[key][idx] = spools.get(name, 0.0)
        self.n += 1

    def refresh(self) -> None:
        n_show = min(self.n, self.max_points)
        if n_show < 2:
            return

        t = self.buf_t[:n_show]
        idx_map = [
            ("pos", ["x_boom", "x_arm", "x_bucket"]),
            ("vel", ["v_boom", "v_arm", "v_bucket"]),
            ("pa", ["pa_boom", "pa_arm", "pa_bucket"]),
            ("pb", ["pb_boom", "pb_arm", "pb_bucket"]),
            ("pump", ["p_pump", "p_ls"]),
            ("sp", ["sp_boom", "sp_arm", "sp_bucket"]),
        ]
        for grp, keys in idx_map:
            for ln, key in zip(self.lines[grp], keys):
                ln.set_data(t, self.buf[key][:n_show])
            ax = self.axes[0 if grp == "pos" else 1 if grp == "vel" else 2 if grp == "pa" else 3 if grp == "pb" else 4 if grp == "pump" else 5]
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_cycle_info(self, cid: int, mode: str) -> None:
        self.fig.suptitle(f"Hydrosim v2 — Cycle {cid} — {mode}", fontsize=14, fontweight="bold")
        self.fig.canvas.draw_idle()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)
