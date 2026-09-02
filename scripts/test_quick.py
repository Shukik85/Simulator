import sys
sys.path.insert(0, ".")
print("imports...")
from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG
from hydrosim_v2.config.hydraulics import LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics.system_rhs import SimRHS
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.loads import LoadModel
print("done imports")

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
print("sim built")

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

print("initial pb_b =", state.p_b["bucket_cyl"] / 1e5, "bar")
sp = {n: 0.0 for n in CYL}
print("running 1 step...")
state = rhs.euler_step(state, sp, 1800.0, 0.002)
print("step OK, pb_b =", state.p_b["bucket_cyl"] / 1e5, "bar")
print("running 50 steps...")
for i in range(50):
    state = rhs.euler_step(state, sp, 1800.0, 0.002)
print("50 steps OK, pb_b =", state.p_b["bucket_cyl"] / 1e5, "bar")
