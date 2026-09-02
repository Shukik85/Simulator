import sys
sys.path.insert(0, ".")
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

dt = 0.002
geo_b = cfg.mechanics.cylinders()["bucket_cyl"]
Ap = geo_b.area_piston_m2
Aa = geo_b.area_annulus_m2
print("bucket Ap=%.6f m2  Aa=%.6f m2  ratio=%.3f" % (Ap, Aa, Aa/Ap))

state = SimState()
state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}
ext0 = loads.external_cylinder_forces(state)
f_ext_b = ext0.get("bucket_cyl", 0.0)
print("f_ext_bucket = %.1f N" % f_ext_b)

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

print("\n=== Detailed Phase 1: Extend bucket - first 0.5s, every 0.02s ===")
print("  t      pa(bar)  pb(bar)  F_pa(N)   F_pb(N)   F_net(N)  vel(m/s)  xb")
for i in range(250):
    s = {n: 0.0 for n in CYL}
    s["bucket_cyl"] = 0.5
    state = rhs.euler_step(state, s, 1800.0, dt)
    if i % 10 == 0:
        t = i * dt
        pa = state.p_a["bucket_cyl"]
        pb = state.p_b["bucket_cyl"]
        F_pa = pa * Ap
        F_pb = pb * Aa
        F_net = F_pa - F_pb - f_ext_b
        v = state.cyl_vel["bucket_cyl"]
        xb = state.cyl_pos["bucket_cyl"]
        print("t=%5.3f  pa=%6.1f  pb=%6.1f  F_pa=%8.0f  F_pb=%8.0f  F_net=%8.0f  v=%+.4f  xb=%.4f" % (
            t, pa/1e5, pb/1e5, F_pa, F_pb, F_net, v, xb))

print("\n=== Phase 2: Center spool at high velocity - every 0.002s ===")
for i in range(300):
    state = rhs.euler_step(state, {n: 0.0 for n in CYL}, 1800.0, dt)
    t = i * dt
    pa = state.p_a["bucket_cyl"]
    pb = state.p_b["bucket_cyl"]
    F_pa = pa * Ap
    F_pb = pb * Aa
    F_net = F_pa - F_pb - f_ext_b
    v = state.cyl_vel["bucket_cyl"]
    xb = state.cyl_pos["bucket_cyl"]
    if i % 5 == 0:
        print("t=%5.3f  pa=%6.1f  pb=%6.1f  F_pa=%8.0f  F_pb=%8.0f  F_net=%8.0f  v=%+.4f  xb=%.4f" % (
            t, pa/1e5, pb/1e5, F_pa, F_pb, F_net, v, xb))
