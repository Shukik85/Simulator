import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

META_SENSORS = ['P_pump', 'P_LS', 'P_boom_A', 'P_boom_B', 'P_arm_A', 'P_arm_B',
                'P_bucket_A', 'P_bucket_B', 'P_swing', 'X_boom', 'X_arm',
                'X_bucket', 'Theta_swing', 'T_hydraulic']

def load_windows(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        # попробуем типичные ключи
        for k in ["X", "windows", "data"]:
            if k in obj.files:
                return obj[k]
        raise KeyError(f"Не найдено X/windows/data в {path}, есть только: {obj.files}")
    return obj  # .npy

def load_labels(path):
    df = pd.read_csv(path)
    # если парсинг “сломался” и всё попало в 1 колонку — попробуем альтернативы
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    # приведём имена к ожидаемым
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ["cycleid", "cycle_id"]: rename[c] = "cycle_id"
        if cl == "mode": rename[c] = "mode"
        if cl in ["payloadkg", "payload_kg"]: rename[c] = "payload_kg"
    df = df.rename(columns=rename)
    return df

def summarize_cycle(Xc):
    # Xc: (n_w, n_s, w)
    mean = Xc.mean(axis=2)
    p95  = np.quantile(Xc, 0.95, axis=2)
    mx   = Xc.max(axis=2)
    return mean, p95, mx

def plot_cycle(cycle_id, X, idxs, labels_row, out_dir="cycle_plots"):
    os.makedirs(out_dir, exist_ok=True)
    Xc = X[idxs]  # (n_w, n_s, w)
    mean, p95, mx = summarize_cycle(Xc)

    title = f"cycle={cycle_id}"
    if labels_row is not None:
        title += f" | {labels_row.get('mode', '')} | {labels_row.get('payload_kg', '')}"

    # 1) Тренды по окнам
    fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
    axes = axes.ravel()
    for s, name in enumerate(META_SENSORS):
        ax = axes[s]
        ax.plot(mean[:, s], label="mean", lw=1, alpha=0.8)
        ax.plot(p95[:, s],  label="p95",  lw=1)
        ax.plot(mx[:, s],   label="max",  lw=1, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if s == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} (window stats)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cycle_{cycle_id:03d}_stats.png"), dpi=160)
    plt.close(fig)

    # 2) Несколько “сырых” окон внутри цикла (начало/середина/конец)
    picks = sorted(set([0, len(idxs)//2, max(0, len(idxs)-1)]))
    fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
    axes = axes.ravel()
    t = np.arange(Xc.shape[2])
    for s, name in enumerate(META_SENSORS):
        ax = axes[s]
        for pi in picks:
            ax.plot(t, Xc[pi, s, :], lw=1, alpha=0.9, label=f"win#{pi}" if s == 0 else None)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)
        if s == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} (raw windows: begin/mid/end)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"cycle_{cycle_id:03d}_raw_windows.png"), dpi=160)
    plt.close(fig)

def main():
    X_PATH = "windows.npz"      # <-- поправь: .npy или .npz
    LABELS_PATH = "labels.csv"    # <-- твой labels.csv

    X = load_windows(X_PATH)
    df = load_labels(LABELS_PATH)

    assert X.ndim == 3, f"Ожидалось (n_windows,n_sensors,window_size), получено {X.shape}"
    assert X.shape[1] == 14 and X.shape[2] == 100, f"Ожидалось 14 сенсоров и 100 точек, получено {X.shape}"

    # индексы окон по циклам
    cycle_to_idxs = df.groupby("cycle_id").indices

    for cycle_id, idxs in sorted(cycle_to_idxs.items()):
        # возьмем первую строку метаданных цикла для заголовка
        row = df[df["cycle_id"] == cycle_id].iloc[0].to_dict()
        plot_cycle(cycle_id, X, idxs, row)

if __name__ == "__main__":
    main()
