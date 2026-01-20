#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

SENSORS = [
    "P_pump", "P_LS", "P_boom_A", "P_boom_B", "P_arm_A", "P_arm_B",
    "P_bucket_A", "P_bucket_B", "P_swing", "X_boom", "X_arm",
    "X_bucket", "Theta_swing", "T_hydraulic"
]

def load_X(npz_path: str, key: str | None):
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.files)
    if not keys:
        raise RuntimeError(f"No arrays found in {npz_path}")

    if key is None:
        key = "X" if "X" in keys else keys[0]

    if key not in keys:
        raise KeyError(f"Key '{key}' not found. Available: {keys}")

    X = npz[key]
    return X, keys, key

def main():
    ap = argparse.ArgumentParser(description="Sanity-check windows.npz + labels.csv")
    ap.add_argument("--windows", default="windows.npz", help="Path to windows.npz")
    ap.add_argument("--labels", default="labels.csv", help="Path to labels.csv")
    ap.add_argument("--key", default=None, help="Key in npz (default: X or first key)")
    ap.add_argument("--eps", type=float, default=1e-4, help="Flat-window std threshold")
    args = ap.parse_args()

    X, keys, used_key = load_X(args.windows, args.key)
    print(f"[OK] Loaded: {args.windows}")
    print(f"     npz keys: {keys}")
    print(f"     using key: {used_key}")
    print(f"     X.shape: {getattr(X, 'shape', None)} dtype={getattr(X, 'dtype', None)}")

    if not (isinstance(X, np.ndarray) and X.ndim == 3):
        raise RuntimeError("Expected X to be a 3D numpy array: (n_windows, n_sensors, window_size)")

    n_windows, n_sensors, window_size = X.shape
    print(f"\n[INFO] n_windows={n_windows} n_sensors={n_sensors} window_size={window_size}")

    if n_sensors != len(SENSORS):
        print(f"[WARN] Expected {len(SENSORS)} sensors, got {n_sensors}. Names may not match metadata.")

    # labels check
    try:
        df = pd.read_csv(args.labels)
        if df.shape[1] == 1:
            df = pd.read_csv(args.labels, sep=r"\s+", engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"cycleid": "cycle_id", "payloadkg": "payload_kg"})
        if len(df) != n_windows:
            print(f"[WARN] labels rows={len(df)} != n_windows={n_windows}")
        else:
            print(f"[OK] labels rows match windows: {len(df)}")
            print(f"     cycles: {df['cycle_id'].nunique() if 'cycle_id' in df.columns else 'unknown'}")
    except Exception as e:
        print(f"[WARN] Could not load labels.csv: {e}")
        df = None

    # per-sensor stats
    print("\n[STATS] Per-sensor min/max/std over ALL points:")
    for s in range(n_sensors):
        name = SENSORS[s] if s < len(SENSORS) else f"s{s}"
        mn = float(X[:, s, :].min())
        mx = float(X[:, s, :].max())
        sd = float(X[:, s, :].std())
        print(f"  {s:02d} {name:12s} min={mn:.6g} max={mx:.6g} std={sd:.6g}")

    # flat windows fraction
    std_per_window = X.std(axis=2)  # (n_windows, n_sensors)
    flat_frac = (std_per_window < args.eps).mean(axis=0)

    print(f"\n[CHECK] Flat-window fraction per sensor (std < {args.eps}):")
    for s in range(n_sensors):
        name = SENSORS[s] if s < len(SENSORS) else f"s{s}"
        print(f"  {s:02d} {name:12s} flat_frac={flat_frac[s]:.4f}")

    # Theta_swing check (if present)
    if "Theta_swing" in SENSORS and n_sensors >= (SENSORS.index("Theta_swing") + 1):
        si = SENSORS.index("Theta_swing")
        th = X[:, si, :]
        print("\n[CHECK] Theta_swing overall:")
        print(f"  min={float(th.min()):.6g} max={float(th.max()):.6g} std={float(th.std()):.6g}")
    else:
        print("\n[INFO] Theta_swing index not available for this X.")

if __name__ == "__main__":
    main()
