"""End-to-end smoke test: generate 1 cycle and verify HDF5 output."""

from pathlib import Path
import tempfile

import numpy as np

from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve, SimRHS
from hydrosim_v2.mechanics.loads import LoadModel
from hydrosim_v2.logger.h5_logger import H5Logger, CycleMeta
from hydrosim_v2.scenarios import ScenarioGenerator

CYL_NAMES = ("boom_cyl", "arm_cyl", "bucket_cyl")
SEC_NAMES = ("boom", "arm", "bucket")


def test_single_step_logged():
    """Verify 100-step simulation produces finite HDF5 output."""
    cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
    mech = cfg.mechanics
    ls_cfg = cfg.hydraulics

    pump = LSPump(ls_cfg.pump)
    relief = ReliefValve("main", ls_cfg.relief)
    valves = {}
    for cn, sn in zip(CYL_NAMES, SEC_NAMES):
        valves[cn] = LSValveSection(cn, ls_cfg.valve_sections[sn])
    cylinders = {}
    for cn, sn in zip(CYL_NAMES, SEC_NAMES):
        cylinders[cn] = Cylinder(cn, mech.cylinders()[cn], ls_cfg.cylinder_dynamics[sn], ls_cfg.fluid)
    loads = LoadModel(cfg)
    rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)

    rng = np.random.default_rng(0)
    scenarios = ScenarioGenerator(rng)
    prof = scenarios.sample_profile("combined")

    with tempfile.TemporaryDirectory() as tmp:
        logger = H5Logger(tmp)
        logger.write_graph()

        state = SimState()
        state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}

        dt = 0.002
        n_steps = 100

        times = np.zeros(n_steps, dtype=np.float32)
        p_pumps = np.zeros(n_steps, dtype=np.float32)
        x_booms = np.zeros(n_steps, dtype=np.float32)

        for i in range(n_steps):
            t = i * dt
            times[i] = t
            u_sec = scenarios.command(prof, t)
            spools = {
                "boom_cyl": u_sec.get("boom", 0.0),
                "arm_cyl": u_sec.get("arm", 0.0),
                "bucket_cyl": u_sec.get("bucket", 0.0),
            }
            state = rhs.euler_step(state, spools, u_sec.get("pumpspeed", 1800.0), dt)
            p_pumps[i] = state.p_pump
            x_booms[i] = state.cyl_pos["boom_cyl"]

        assert all(np.isfinite(p_pumps))
        assert all(np.isfinite(x_booms))

        timeline = {
            "time": times,
            "p_pump": p_pumps,
            "x_boom": x_booms,
        }
        meta = CycleMeta(cycle_id=0, mode=prof.mode, duration_s=prof.duration_s)
        logger.log_cycle(meta, timeline)
        logger.close()

        out = Path(tmp)
        assert (out / "dataset.h5").exists()
        import h5py
        with h5py.File(out / "dataset.h5", "r") as f:
            assert "cycles" in f
            g = f["cycles"]["cycle_000000"]
            assert "p_pump" in g
            assert len(g["p_pump"]) == n_steps
