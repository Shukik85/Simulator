import sys, time
sys.path.insert(0, ".")
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

state = SimState()
for name in CYL:
    state.cyl_pos[name] = 0.5

dt = 0.002
t0 = time.time()
for i in range(200):
    sp = {"boom_cyl": 0.3, "arm_cyl": -0.3, "bucket_cyl": 0.3}
    state = rhs.euler_step(state, sp, 1800.0, dt)
    if i % 20 == 0:
        bx = state.cyl_pos["boom_cyl"]
        ax = state.cyl_pos["arm_cyl"]
        kx = state.cyl_pos["bucket_cyl"]
        pba = state.p_a["boom_cyl"] / 1e5
        pbb = state.p_b["boom_cyl"] / 1e5
        paa = state.p_a["arm_cyl"] / 1e5
        pka = state.p_a["bucket_cyl"] / 1e5
        vbx = state.cyl_vel["boom_cyl"]
        vax = state.cyl_vel["arm_cyl"]
        vkx = state.cyl_vel["bucket_cyl"]
        ext = loads.external_cylinder_forces(state)
        fe_b = ext.get("boom_cyl", 0)
        fe_a = ext.get("arm_cyl", 0)
        fe_k = ext.get("bucket_cyl", 0)
        print("t=%4.2f boom=%.4f(%.3f) arm=%.4f(%.3f) bucket=%.4f(%.3f)  pba=%.1f pbb=%.1f paa=%.1f pka=%.1f  fe_a=%7.0f fe_k=%7.0f" % (
            i*dt, bx, vbx, ax, vax, kx, vkx, pba, pbb, paa, pka, fe_a, fe_k))

dt_elapsed = time.time() - t0
print("Done. %d steps in %.2fs (%.0f steps/s)" % (200, dt_elapsed, 200/dt_elapsed))
bx = state.cyl_pos["boom_cyl"]
ax = state.cyl_pos["arm_cyl"]
kx = state.cyl_pos["bucket_cyl"]
print("Final: boom=%.4f arm=%.4f bucket=%.4f" % (bx, ax, kx))
print("No crash - simulation stable")
