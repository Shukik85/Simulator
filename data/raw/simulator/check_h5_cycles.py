#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_SENSORS = [
    "P_pump", "P_LS", "P_boom_A", "P_boom_B", "P_arm_A", "P_arm_B",
    "P_bucket_A", "P_bucket_B", "P_swing", "X_boom", "X_arm",
    "X_bucket", "Theta_swing", "T_hydraulic",
]

def decode_names(raw):
    out = []
    for x in raw:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

def main():
    ap = argparse.ArgumentParser(description="Inspect simulator .h5: per-sensor datasets per cycle")
    ap.add_argument("--h5", required=True, help="Path to simulation_*.h5")
    ap.add_argument("--cycle", default="cycle_00000", help="Cycle group name under /cycles/")
    ap.add_argument("--list_cycles", action="store_true", help="List cycle groups and exit")
    ap.add_argument("--out_csv", default=None, help="Optional: write per-sensor stats CSV")
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--tail", type=int, default=5)
    args = ap.parse_args()

    import h5py

    path = Path(args.h5)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with h5py.File(path, "r") as h5:
        if "/cycles" not in h5:
            raise SystemExit("No /cycles group in H5.")

        cycles_grp = h5["/cycles"]

        if args.list_cycles:
            print("[CYCLES]")
            for k in cycles_grp.keys():
                print(" ", k)
            return

        cyc_path = f"/cycles/{args.cycle}"
        if cyc_path not in h5:
            raise SystemExit(f"Cycle not found: {cyc_path}")

        g = h5[cyc_path]
        keys = list(g.keys())
        print(f"[OK] Opened: {path}")
        print(f"[OK] Using cycle: {cyc_path}")
        print(f"[INFO] Datasets: {keys}")

        # sensor names
        names = None
        if "/sensor_names" in h5:
            names = decode_names(h5["/sensor_names"][...])
        elif "/metadata/sensor_names" in h5:
            names = decode_names(h5["/metadata/sensor_names"][...])

        if not names:
            names = [k for k in keys if k != "time"]

        # time
        if "time" not in g:
            raise SystemExit("No 'time' dataset under cycle group.")
        t = g["time"][...].astype(np.float64)
        n = len(t)

        if n > 1:
            dt = float(np.median(np.diff(t)))
            fs = 1.0 / dt if dt > 0 else float("inf")
            print(f"[INFO] time: n={n} dt~{dt:.6g} sec fs~{fs:.6g} Hz")
        else:
            print(f"[INFO] time: n={n}")

        # decide which sensors to inspect
        present = [s for s in names if s in g.keys()]
        if not present:
            # fallback: anything except time
            present = [k for k in keys if k != "time"]

        rows = []
        for s in present:
            x = g[s][...]
            if len(x) != n:
                print(f"[WARN] {s}: len={len(x)} != len(time)={n}")

            x = x.astype(np.float64)
            rows.append({
                "sensor": s,
                "len": int(len(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "std": float(np.std(x)),
            })

        df = pd.DataFrame(rows).sort_values("sensor")
        print("\n[STATS] per-sensor min/max/std:")
        for r in df.itertuples(index=False):
            print(f"  {r.sensor:12s} len={r.len:6d} min={r.min:.6g} max={r.max:.6g} std={r.std:.6g}")

        # focus channels
        def focus(name):
            if name in g:
                x = g[name][...].astype(np.float64)
                print(f"\n[FOCUS] {name}:")
                print("  head:", np.array2string(x[:args.head], precision=6))
                print("  tail:", np.array2string(x[-args.tail:], precision=6))
                print("  unique~:", int(np.unique(np.round(x, 8)).shape[0]))

        for k in ["Theta_swing", "P_swing", "P_pump", "P_LS", "T_hydraulic", "X_boom", "X_arm", "X_bucket"]:
            focus(k)

        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"\n[OK] wrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()
