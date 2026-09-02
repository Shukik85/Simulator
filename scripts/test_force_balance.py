"""Test force balance closure: when velocity is clamped, does pressure match the load?"""
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
    state.p_a[name] = 10e5
    state.p_b[name] = 10e5

geo_boom = cfg.mechanics.cylinders()["boom_cyl"]
geo_arm = cfg.mechanics.cylinders()["arm_cyl"]
geo_bucket = cfg.mechanics.cylinders()["bucket_cyl"]

sp = {"boom_cyl": 0.5, "arm_cyl": 0.0, "bucket_cyl": 0.0}

print("t      cyl        pa(bar) pb(bar) v(m/s) F_hyd   F_ext   F_net  v_max   clamped?")
print("-" * 90)

for i in range(300):
    state_new = rhs.euler_step(state, sp, 1800.0, dt)
    
    if i % 30 == 0:
        ext = loads.external_cylinder_forces(state)
        for name in CYL:
            geo = cfg.mechanics.cylinders()[name]
            Aa = geo.area_piston_m2
            Ab = geo.area_annulus_m2
            pa = state.p_a[name]
            pb = state.p_b[name]
            v = state.cyl_vel[name]
            f_hyd = pa * Aa - pb * Ab
            f_ext = ext.get(name, 0.0)
            f_net = f_hyd - f_ext
            
            sn = name.replace("_cyl", "")
            vs = rhs.valves[name]
            sp_val = sp[name]
            if sp_val > 0 and vs.q_a > 0:
                v_max = vs.q_a / max(Aa, 1e-6)
            elif sp_val < 0 and vs.q_b > 0:
                v_max = vs.q_b / max(Ab, 1e-6)
            else:
                v_max = 999
            clamped = (sp_val > 0 and v >= v_max - 1e-6) or (sp_val < 0 and v <= -v_max + 1e-6)
            
            print("t=%4.2f  %-10s pa=%6.1f pb=%6.1f v=%+.4f F_hyd=%7.0f F_ext=%7.0f F_net=%+7.0f v_max=%6.4f %s" % (
                i*dt, name, pa/1e5, pb/1e5, v, f_hyd, f_ext, f_net, v_max if v_max < 999 else 0, "YES" if clamped else ""))
    
    state = state_new
