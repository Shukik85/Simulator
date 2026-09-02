"""
Простейшая 2D-анимация экскаватора по данным HDF5.
Читает dataset, вычисляет forward kinematics и рисует звенья.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from hydrosim_v2.config import DEFAULT_MECHANICS_CONFIG
from hydrosim_v2.kinematics import forward_kinematics

CYL_PARAMS = {
    "boom_cyl":   (1.500, 1.000),
    "arm_cyl":    (1.685, 1.185),
    "bucket_cyl": (1.400, 0.900),
}


def x_to_length(name: str, x: float) -> float:
    lo, stroke = CYL_PARAMS[name]
    return lo + float(np.clip(x, 0.0, 1.0)) * stroke


def draw_frame(ax, pts, alpha=1.0, color="steelblue", lw=3):
    boom = np.array([pts["base"], pts["boom_tip"]])
    arm = np.array([pts["boom_tip"], pts["arm_tip"]])
    bk = np.array([pts["arm_tip"], pts["bucket_tip_cutting_edge"]])

    ax.plot(boom[:, 0], boom[:, 1], color=color, lw=lw + 1, alpha=alpha, solid_capstyle="round")
    ax.plot(arm[:, 0], arm[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(bk[:, 0], bk[:, 1], color="saddlebrown", lw=lw, alpha=alpha, solid_capstyle="round")

    for name in ("base", "boom_tip", "arm_tip", "bucket_tip_cutting_edge"):
        ax.plot(*pts[name], "o", color="black", ms=5, alpha=alpha, zorder=5)

    if "P_boom" in pts:
        ax.plot(*pts["P_boom"], "s", color="gray", ms=4, alpha=alpha * 0.6, zorder=4)
    if "A_arm" in pts:
        ax.plot(*pts["A_arm"], "s", color="gray", ms=4, alpha=alpha * 0.6, zorder=4)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Excavator 2D animation")
    p.add_argument("h5", help="Path to dataset.h5")
    p.add_argument("--step", type=int, default=4, help="Frame skip (default=4)")
    p.add_argument("--interval", type=int, default=20, help="ms between frames (default=20)")
    p.add_argument("--trail", action="store_true", help="Show bucket tip trail")
    args = p.parse_args()

    f = h5py.File(args.h5, "r")
    cycle = f["cycles/cycle_000000"]

    time = cycle["time"][:]
    x_boom = cycle["x_boom"][:]
    x_arm = cycle["x_arm"][:]
    x_bucket = cycle["x_bucket"][:]
    n_frames = len(time)
    step = max(1, args.step)
    indices = list(range(0, n_frames, step))

    cfg = DEFAULT_MECHANICS_CONFIG

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#f0f0f0")
    ax.set_facecolor("#e8e8e8")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")

    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], "-", color="red", lw=1, alpha=0.5)

    link_line, = ax.plot([], [], "-", color="steelblue", lw=3)
    bucket_line, = ax.plot([], [], "-", color="saddlebrown", lw=3)
    joints, = ax.plot([], [], "o", color="black", ms=5, zorder=5)
    title = ax.set_title("")

    prev_bucket_theta = None
    x_all, y_all = [], []
    for i in indices:
        cyl = {
            "boom_cyl": x_to_length("boom_cyl", x_boom[i]),
            "arm_cyl": x_to_length("arm_cyl", x_arm[i]),
            "bucket_cyl": x_to_length("bucket_cyl", x_bucket[i]),
        }
        try:
            pts = forward_kinematics(cfg, cyl)
            for v in pts.values():
                x_all.append(v[0])
                y_all.append(v[1])
        except Exception:
            pass

    if x_all:
        margin = 1.0
        ax.set_xlim(min(x_all) - margin, max(x_all) + margin)
        ax.set_ylim(min(y_all) - margin, max(y_all) + margin)

    def init():
        link_line.set_data([], [])
        bucket_line.set_data([], [])
        joints.set_data([], [])
        trail_line.set_data([], [])
        return link_line, bucket_line, joints, trail_line, title

    def update(frame_idx):
        nonlocal prev_bucket_theta
        i = indices[frame_idx]
        cyl = {
            "boom_cyl": x_to_length("boom_cyl", x_boom[i]),
            "arm_cyl": x_to_length("arm_cyl", x_arm[i]),
            "bucket_cyl": x_to_length("bucket_cyl", x_bucket[i]),
        }
        try:
            pts = forward_kinematics(cfg, cyl, prev_bucket_theta=prev_bucket_theta)
            prev_bucket_theta = pts.get("bucket_theta_rad", None)
        except Exception:
            return link_line, bucket_line, joints, trail_line, title

        boom = np.array([pts["base"], pts["boom_tip"], pts["arm_tip"]])
        bk = np.array([pts["arm_tip"], pts["bucket_tip_cutting_edge"]])

        link_line.set_data(boom[:, 0], boom[:, 1])
        bucket_line.set_data(bk[:, 0], bk[:, 1])

        jx = [pts[k][0] for k in ("base", "boom_tip", "arm_tip", "bucket_tip_cutting_edge")]
        jy = [pts[k][1] for k in ("base", "boom_tip", "arm_tip", "bucket_tip_cutting_edge")]
        joints.set_data(jx, jy)

        if args.trail:
            trail_x.append(pts["bucket_tip_cutting_edge"][0])
            trail_y.append(pts["bucket_tip_cutting_edge"][1])
            trail_line.set_data(trail_x, trail_y)

        title.set_text(f"t = {time[i]:.1f}s  |  Boom: {x_boom[i]:.2f}  Arm: {x_arm[i]:.2f}  Bucket: {x_bucket[i]:.2f}")
        return link_line, bucket_line, joints, trail_line, title

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(indices), interval=args.interval, blit=True,
    )

    plt.tight_layout()
    plt.show()
    f.close()


if __name__ == "__main__":
    main()
