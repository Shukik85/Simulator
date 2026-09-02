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
P_RPM = 1800.0

state = SimState()
state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}

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


def sp_zero():
    return {n: 0.0 for n in CYL}


def dump(t, label=""):
    pa = state.p_a["bucket_cyl"] / 1e5
    pb = state.p_b["bucket_cyl"] / 1e5
    xb = state.cyl_pos["bucket_cyl"]
    v = state.cyl_vel["bucket_cyl"]
    print("t=%5.2f  pa=%7.1f  pb=%7.1f  xb=%.4f  v=%+.4f  %s" % (t, pa, pb, xb, v, label))


print("=== Initial state (gravity hold) ===")
dump(0.0)

for _ in range(20):
    state = rhs.euler_step(state, sp_zero(), P_RPM, dt)

print("\n=== After settle ===")
dump(0.04)

print("\n=== Phase 1: Extend bucket 3s (sp=+0.5) ===")
for i in range(1500):
    s = sp_zero()
    s["bucket_cyl"] = 0.5
    state = rhs.euler_step(state, s, P_RPM, dt)
    if i % 300 == 0:
        dump(i * dt)

print("\n=== Phase 2: Center spool - observe pb ===")
for i in range(2000):
    state = rhs.euler_step(state, sp_zero(), P_RPM, dt)
    if i % 200 == 0:
        dump(i * dt)

print("\n=== Phase 3: Retract bucket 3s (sp=-0.5) ===")
for i in range(1500):
    s = sp_zero()
    s["bucket_cyl"] = -0.5
    state = rhs.euler_step(state, s, P_RPM, dt)
    if i % 300 == 0:
        dump(i * dt)

print("\n=== Phase 4: Center spool again - observe pb ===")
for i in range(2000):
    state = rhs.euler_step(state, sp_zero(), P_RPM, dt)
    if i % 200 == 0:
        dump(i * dt)
