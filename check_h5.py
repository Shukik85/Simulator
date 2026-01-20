from pathlib import Path
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) найти последний h5
h5_files = sorted(glob.glob("data/raw/simulator/simulation_*.h5"))
if not h5_files:
    raise FileNotFoundError("Не найдено data/raw/simulator/simulation_*.h5")
h5_path = h5_files[-1]
print("Using:", h5_path)

def pick_cycle_by_mode(h5, wanted_mode=None):
    cycles = h5["cycles"]
    if wanted_mode is None:
        # первый цикл
        return list(cycles.keys())[0]
    for k in cycles.keys():
        attrs = dict(cycles[k].attrs)
        if attrs.get("mode", None) == wanted_mode:
            return k
    raise ValueError(f"Цикл с mode={wanted_mode} не найден")

with h5py.File(h5_path, "r") as f:
    cycle_key = pick_cycle_by_mode(f, wanted_mode=None)  # например wanted_mode="swing_only"
    g = f["cycles"][cycle_key]
    t = g["time"][:]

    # список сенсоров (как ожидается по формату датасета)
    sensor_names = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in f["sensor_names"][:]]

    # оставляем только те, которые реально есть в выбранном cycle (на случай обратной совместимости)
    sensor_names = [s for s in sensor_names if s in g]
    def infer_unit(name: str) -> str:
        if name.startswith("P_"):
            return "bar"
        if name.startswith("X_"):
            return "mm"
        if name.startswith("Theta_"):
            return "deg"
        if name.startswith("T_"):
            return "C"
        if name == "pump_speed":
            return "rpm"
        return ""

    # 2) по одному PNG на сенсор
    for name in sensor_names:
        y = g[name][:]
        unit = infer_unit(name)
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, linewidth=1)
        plt.title(f"{cycle_key} / {name}" + (f" [{unit}]" if unit else ""))
        plt.xlabel("t, s")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{cycle_key}_{name}.png", dpi=150)
        plt.close()

    # 3) мультиплот (несколько графиков на странице)
    n = len(sensor_names)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.2 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(sensor_names):
        ax = axes[i]
        y = g[name][:]
        ax.plot(t, y, linewidth=0.8)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{cycle_key} sensors", y=1.002)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{cycle_key}_ALL.png", dpi=150)
    plt.close(fig)

print("Saved to:", OUT_DIR)
