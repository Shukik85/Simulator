"""Direct check: print pressures and forces for bucket cylinder each step."""
import sys
sys.path.insert(0, ".")
import math
from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
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
state = SimState()
for name in CYL:
    state.cyl_pos[name] = 0.5
sp = {n: 0.0 for n in CYL}
sp["bucket_cyl"] = 0.5

name = "bucket_cyl"
geo = cfg.mechanics.cylinders()[name]
Aa = geo.area_piston_m2
Ab = geo.area_annulus_m2

print("t      pa(bar) pb(bar) v(m/s)   F_hyd   F_ext   F_net    pa*pump?")
for i in range(100):
    p_a = state.p_a[name]
    p_b = state.p_b[name]
    v = state.cyl_vel[name]
    F_hyd = p_a * Aa - p_b * Ab
    ext = loads.external_cylinder_forces(state)
    F_ext = ext.get(name, 0.0)
    F_net = F_hyd - F_ext

    state_new = rhs.euler_step(state, sp, 1800.0, dt)
    pa2 = state_new.p_a[name]
    v2 = state_new.cyl_vel[name]

    if i % 5 == 0:
        print("t=%4.2f  pa=%6.1f pb=%6.1f v=%+.4f F_hyd=%7.0f F_ext=%7.0f F_net=%+7.0f  pa2=%6.1f  v2=%+.4f" % (
            i*dt, p_a/1e5, p_b/1e5, v, F_hyd, F_ext, F_net, pa2/1e5, v2))

    state = state_new
