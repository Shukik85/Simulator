import h5py
import numpy as np

f = h5py.File("out_manual/dataset.h5", "r")
cycle = f["cycles/cycle_000000"]

time = cycle["time"][:]

channels = {
    "p_boom_a": cycle["p_boom_a"][:],
    "p_boom_b": cycle["p_boom_b"][:],
    "p_arm_a": cycle["p_arm_a"][:],
    "p_arm_b": cycle["p_arm_b"][:],
    "p_bucket_a": cycle["p_bucket_a"][:],
    "p_bucket_b": cycle["p_bucket_b"][:],
    "p_pump": cycle["p_pump"][:],
    "p_ls": cycle["p_ls"][:],
    "x_boom": cycle["x_boom"][:],
    "x_arm": cycle["x_arm"][:],
    "x_bucket": cycle["x_bucket"][:],
}

# Overview
print(f"Duration: {time[-1]:.1f} s, frames: {len(time)}")
for k, v in channels.items():
    print(f"  {k:15s}: [{v.min()/1e5:7.1f}, {v.max()/1e5:7.1f}] bar" if "p_" in k else f"  {k:15s}: [{v.min():.4f}, {v.max():.4f}] m")

# Focus on t=38-57.6s (bucket anomaly)
print("\n=== t=38-57.6s: bucket anomaly region ===")
print(f"{'t':>6s}  {'pa_bk':>7s}  {'pb_bk':>7s}  {'x_bk':>7s}  {'pa_bm':>7s}  {'pb_bm':>7s}  {'x_bm':>7s}  {'p_pump':>7s}")
for sec in range(38, 58):
    idx = np.searchsorted(time, float(sec))
    if idx < len(time):
        print(f"{time[idx]:6.1f}  "
              f"{channels['p_bucket_a'][idx]/1e5:7.1f}  "
              f"{channels['p_bucket_b'][idx]/1e5:7.1f}  "
              f"{channels['x_bucket'][idx]:7.4f}  "
              f"{channels['p_boom_a'][idx]/1e5:7.1f}  "
              f"{channels['p_boom_b'][idx]/1e5:7.1f}  "
              f"{channels['x_boom'][idx]:7.4f}  "
              f"{channels['p_pump'][idx]/1e5:7.1f}")

# Last frame
idx = len(time) - 1
print(f"\n=== Final state (t={time[idx]:.1f}s) ===")
for k, v in channels.items():
    unit = "bar" if "p_" in k else "m"
    val = v[idx]/1e5 if "p_" in k else v[idx]
    print(f"  {k:15s}: {val:.2f} {unit}")

f.close()
