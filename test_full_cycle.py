"""Full cycle test: settle → retract → stop → extend → stop."""
import sys
sys.path.insert(0, ".")

from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics.pump import LSPump
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import Cylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.mechanics.loads import LoadModel
from hydrosim_v2.hydraulics.system_rhs import SimRHS

cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
pump = LSPump(cfg.hydraulics.pump)
valves = {n: LSValveSection(n, cfg.hydraulics.valve_sections[n.replace("_cyl", "")]) for n in ("boom_cyl", "arm_cyl", "bucket_cyl")}
cylinders = {n: Cylinder(n, cfg.mechanics.cylinders()[n], cfg.hydraulics.cylinder_dynamics[n.replace("_cyl", "")], cfg.hydraulics.fluid) for n in ("boom_cyl", "arm_cyl", "bucket_cyl")}
relief = ReliefValve("main", cfg.hydraulics.relief)
loads = LoadModel(cfg)
rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)

state = SimState()
state.p_tank = 1e5
state.p_ls = 1e5
state.p_pump = 1e5 + 18e5
state.cyl_pos["boom_cyl"] = 0.5
state.cyl_vel["boom_cyl"] = 0.0
state.p_a["boom_cyl"] = 1.5e5
boom_geo = cfg.mechanics.cylinders()["boom_cyl"]
state.p_b["boom_cyl"] = (35746 + 1.5e5 * boom_geo.area_piston_m2) / boom_geo.area_annulus_m2
spools = {"boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0}

# Settle 20 steps
for _ in range(20):
    state = rhs.euler_step(state, spools, 2200.0, 0.002)
print(f"SETTLE: x={state.cyl_pos['boom_cyl']:.4f} v={state.cyl_vel['boom_cyl']:.4f} "
      f"pa={state.p_a['boom_cyl']/1e5:.2f} pb={state.p_b['boom_cyl']/1e5:.2f} "
      f"p_pump={state.p_pump/1e5:.2f} p_ls={state.p_ls/1e5:.2f} q={state.q_pump*60000:.1f}")

# Retract 200 steps
spools["boom_cyl"] = -0.75
print("\n--- RETRACT (spool=-0.75) ---")
for i in range(200):
    state = rhs.euler_step(state, spools, 2200.0, 0.002)
    if i % 40 == 0 or i == 199:
        print(f"  t={state.t:.3f} x={state.cyl_pos['boom_cyl']:.4f} v={state.cyl_vel['boom_cyl']:.4f} "
              f"pa={state.p_a['boom_cyl']/1e5:.2f} pb={state.p_b['boom_cyl']/1e5:.2f} "
              f"p_pump={state.p_pump/1e5:.2f} p_ls={state.p_ls/1e5:.2f} q={state.q_pump*60000:.1f}")

retract_v = state.cyl_vel["boom_cyl"]
retract_pa = state.p_a["boom_cyl"] / 1e5
retract_pb = state.p_b["boom_cyl"] / 1e5

# Stop 50 steps
spools["boom_cyl"] = 0.0
print("\n--- STOP (spool=0) ---")
for i in range(50):
    state = rhs.euler_step(state, spools, 2200.0, 0.002)
    if i % 10 == 0 or i == 49:
        print(f"  t={state.t:.3f} x={state.cyl_pos['boom_cyl']:.4f} v={state.cyl_vel['boom_cyl']:.4f} "
              f"pa={state.p_a['boom_cyl']/1e5:.2f} pb={state.p_b['boom_cyl']/1e5:.2f} "
              f"p_pump={state.p_pump/1e5:.2f} p_ls={state.p_ls/1e5:.2f} q={state.q_pump*60000:.1f}")

stop_v = state.cyl_vel["boom_cyl"]

# Extend 200 steps
spools["boom_cyl"] = 0.75
print("\n--- EXTEND (spool=+0.75) ---")
for i in range(200):
    state = rhs.euler_step(state, spools, 2200.0, 0.002)
    if i % 40 == 0 or i == 199:
        print(f"  t={state.t:.3f} x={state.cyl_pos['boom_cyl']:.4f} v={state.cyl_vel['boom_cyl']:.4f} "
              f"pa={state.p_a['boom_cyl']/1e5:.2f} pb={state.p_b['boom_cyl']/1e5:.2f} "
              f"p_pump={state.p_pump/1e5:.2f} p_ls={state.p_ls/1e5:.2f} q={state.q_pump*60000:.1f}")

extend_v = state.cyl_vel["boom_cyl"]
extend_pa = state.p_a["boom_cyl"] / 1e5
extend_pb = state.p_b["boom_cyl"] / 1e5

# Stop 500 steps (1 second)
spools["boom_cyl"] = 0.0
print("\n--- FINAL STOP ---")
for i in range(500):
    state = rhs.euler_step(state, spools, 2200.0, 0.002)
    if i % 100 == 0 or i == 499:
        print(f"  t={state.t:.3f} x={state.cyl_pos['boom_cyl']:.4f} v={state.cyl_vel['boom_cyl']:.4f} "
              f"pa={state.p_a['boom_cyl']/1e5:.2f} pb={state.p_b['boom_cyl']/1e5:.2f} "
              f"p_pump={state.p_pump/1e5:.2f} p_ls={state.p_ls/1e5:.2f} q={state.q_pump*60000:.1f}")

final_v = state.cyl_vel["boom_cyl"]
final_pa = state.p_a["boom_cyl"] / 1e5
final_pb = state.p_b["boom_cyl"] / 1e5

print(f"\n=== SUMMARY ===")
print(f"Retract: v={retract_v:.4f} pa={retract_pa:.2f} pb={retract_pb:.2f}")
print(f"Extend:  v={extend_v:.4f} pa={extend_pa:.2f} pb={extend_pb:.2f}")
print(f"Stop:    v={final_v:.4f} pa={final_pa:.2f} pb={final_pb:.2f}")

ok = True
if retract_v > -0.05:
    print("FAIL: retract velocity too slow"); ok = False
if retract_pa > 5.0:
    print(f"FAIL: retract pa={retract_pa:.1f} bar (should be <5)"); ok = False
if extend_v < 0.05:
    print("FAIL: extend velocity too slow"); ok = False
if extend_pb > 5.0:
    print(f"FAIL: extend pb={extend_pb:.1f} bar (should be <5)"); ok = False
if abs(final_v) > 0.01:
    print(f"FAIL: final velocity={final_v:.4f} (should be ~0)"); ok = False
if ok:
    print("\nALL CHECKS PASSED")
