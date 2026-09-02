"""Test arm extension dynamics without velocity clamp."""
import sys
sys.path.insert(0, ".")
import math
from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG
from hydrosim_v2.config.hydraulics import LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics.system_rhs import SimRHS
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.loads import LoadModel

CYL = ("boom_cyl", "arm_cyl", "bucket_cyl")
SEC = ("boom", "arm", "bucket")
cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
ls_cfg = cfg.hydraulics
pump = LSPump(ls_cfg.pump)
valves = {}
cylinders = {}
for cn, sn in zip(CYL, SEC):
    geo = cfg.mechanics.cylinders()[cn]
    valves[cn] = LSValveSection(cn, ls_cfg.valve_sections[sn])
    cylinders[cn] = Cylinder(cn, geo, ls_cfg.cylinder_dynamics[sn], ls_cfg.fluid)
relief = ReliefValve("main", ls_cfg.relief)
loads = LoadModel(cfg)
rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)

geo_a = cfg.mechanics.cylinders()["arm_cyl"]
Ap = geo_a.area_piston_m2
Aa = geo_a.area_annulus_m2
dt = 0.002
state = SimState()

for name in CYL:
    state.cyl_pos[name] = 0.5
state.cyl_pos["arm_cyl"] = 0.8

ext0 = loads.external_cylinder_forces(state)
for name in CYL:
    geo = cfg.mechanics.cylinders()[name]
    ap = geo.area_piston_m2
    aa = geo.area_annulus_m2
    f = ext0.get(name, 0.0)
    pt = 1.5e5
    if f < 0:
        state.p_a[name] = pt
        state.p_b[name] = max(pt, (pt * ap - f) / aa)
    else:
        state.p_b[name] = pt
        state.p_a[name] = max(pt, (f + pt * aa) / ap)

print("=== Arm extension (boom+bucket held, arm extending) ===")
print("Initial positions:", {k: round(v, 4) for k, v in state.cyl_pos.items()})
print("Initial f_ext:", {k: round(v, 1) for k, v in ext0.items()})
print()

print("t       pa(bar) pb(bar) F_pa(N)  F_pb(N)  F_hyd(N) F_ext(N) v(m/s)   x(m)    vel_check")
print("-" * 90)
for i in range(500):
    sp = {n: 0.0 for n in CYL}
    sp["arm_cyl"] = 0.5
    state = rhs.euler_step(state, sp, 1800.0, dt)
    if i % 25 == 0:
        t = i * dt
        pa = state.p_a["arm_cyl"]
        pb = state.p_b["arm_cyl"]
        F_pa = pa * Ap
        F_pb = pb * Aa
        F_hyd = F_pa - F_pb
        ext = loads.external_cylinder_forces(state)
        f_ext = ext.get("arm_cyl", 0.0)
        v = state.cyl_vel["arm_cyl"]
        x = state.cyl_pos["arm_cyl"]
        q_a = valves["arm_cyl"].q_a
        vel_theory = q_a / Ap if q_a > 0 else 0
        print("t=%5.2f  pa=%6.1f pb=%6.1f F_hyd=%7.0f F_ext=%7.0f v=%+.4f  x=%.4f  Q/A=%.4f" % (
            t, pa/1e5, pb/1e5, F_hyd, f_ext, v, x, vel_theory))
