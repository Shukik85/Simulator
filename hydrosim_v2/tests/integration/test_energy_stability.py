"""Tests for energy balance, power limiting, payload, and stability fixes."""

import math

import numpy as np
import pytest

from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve, SimRHS
from hydrosim_v2.mechanics.loads import LoadModel

CYL = ("boom_cyl", "arm_cyl", "bucket_cyl")
SEC = ("boom", "arm", "bucket")


def _make_sim(cfg: ExcavatorConfig | None = None):
    cfg = cfg or ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
    ls_cfg = cfg.hydraulics
    pump = LSPump(ls_cfg.pump)
    relief = ReliefValve("main", ls_cfg.relief)
    valves = {}
    cylinders = {}
    for cn, sn in zip(CYL, SEC):
        geo = cfg.mechanics.cylinders()[cn]
        valves[cn] = LSValveSection(cn, ls_cfg.valve_sections[sn])
        cylinders[cn] = Cylinder(cn, geo, ls_cfg.cylinder_dynamics[sn], ls_cfg.fluid)
    loads = LoadModel(cfg)
    rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)
    return rhs, cfg


def _init_state(rhs: SimRHS) -> SimState:
    state = SimState()
    for name in CYL:
        state.cyl_pos[name] = 0.5
    return state


class TestEnergyBalance:
    def test_energy_closure_with_mech_loss(self):
        rhs, _ = _make_sim()
        state = _init_state(rhs)
        rhs.payload_kg = 1000.0
        dt = 0.002
        seq = [
            {"boom_cyl": 0.5, "arm_cyl": -0.3, "bucket_cyl": 0.4},
            {"boom_cyl": 0.7, "arm_cyl": 0.3, "bucket_cyl": -0.4},
        ]
        mech_loss_acc = 0.0
        for i in range(400):
            sp = seq[(i // 200) % 2]
            state = rhs.euler_step(state, sp, 1800.0, dt)
            mech_loss_acc += rhs.pump.mech_loss * dt

        e_in = state.e_mech_in
        e_out = state.e_cyl + state.e_valve_loss + state.e_relief + mech_loss_acc
        rel_err = abs(e_in - e_out) / max(e_in, 1e-6)
        assert rel_err < 0.05, f"energy balance off by {rel_err*100:.1f}%"

    def test_power_limit_caps_pressure(self):
        cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
        rhs, _ = _make_sim(cfg)
        state = _init_state(rhs)
        dt = 0.002
        p_max = rhs.pump.max_hyd_power_w
        sp = {"boom_cyl": 0.9, "arm_cyl": 0.8, "bucket_cyl": 0.8}
        max_power_seen = 0.0
        for _ in range(200):
            state = rhs.euler_step(state, sp, 1800.0, dt)
            power = state.q_pump * state.p_pump
            max_power_seen = max(max_power_seen, power)
        assert max_power_seen <= p_max * 1.02, f"power exceeded limit: {max_power_seen:.0f} > {p_max:.0f}"

    def test_power_limit_value(self):
        pump = LSPump(LSConfig().pump)
        # 120 hp = 120*745.7 W; *0.92 mech_eff
        expected = 120.0 * 745.699872 * 0.92
        assert abs(pump.max_hyd_power_w - expected) < 1e-6


class TestPayload:
    def test_payload_increases_bucket_force(self):
        cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
        loads = LoadModel(cfg)
        state = SimState()
        for name in CYL:
            state.cyl_pos[name] = 0.5
        f0 = loads.external_cylinder_forces(state, payload_kg=0.0)
        f1 = loads.external_cylinder_forces(state, payload_kg=2000.0)
        # bucket force must grow with payload
        assert abs(f1["bucket_cyl"]) > abs(f0["bucket_cyl"])
        # boom/arm should also change (payload raises COM of the linkage)
        assert not math.isclose(f0["boom_cyl"], f1["boom_cyl"], abs_tol=1.0)

    def test_rhs_payload_passes_to_loads(self):
        rhs, _ = _make_sim()
        state = _init_state(rhs)
        rhs.payload_kg = 1500.0
        ns = rhs.euler_step(state, {"boom_cyl": 0.4, "arm_cyl": -0.2, "bucket_cyl": 0.1}, 1800.0, 0.002)
        assert np.isfinite(ns.cyl_pos["boom_cyl"])


class TestStability:
    def test_no_runaway_velocity_under_gravity(self):
        """Boom with sp>0 under heavy gravity must not free-fall (meter-out clamp)."""
        rhs, _ = _make_sim()
        state = _init_state(rhs)
        dt = 0.002
        sp = {"boom_cyl": 0.3, "arm_cyl": 0.0, "bucket_cyl": 0.0}
        max_speed = 0.0
        for _ in range(300):
            state = rhs.euler_step(state, sp, 1800.0, dt)
            max_speed = max(max_speed, abs(state.cyl_vel["boom_cyl"]))
        assert max_speed < 1.0, f"boom runaway velocity {max_speed:.2f} m/s"

    def test_spool_filter_smooths_steps(self):
        rhs, _ = _make_sim()
        rhs._filter_spools({"boom_cyl": 1.0, "arm_cyl": 0.0, "bucket_cyl": 0.0}, 0.002)
        # after a small dt the filtered spool must not reach the full step immediately
        assert rhs.spool_filtered["boom_cyl"] < 1.0
        assert rhs.spool_filtered["boom_cyl"] > 0.0

    def test_rk4_stable(self):
        rhs, _ = _make_sim()
        state = _init_state(rhs)
        dt = 0.002
        sp = {"boom_cyl": 0.4, "arm_cyl": -0.3, "bucket_cyl": 0.2}
        for _ in range(100):
            state = rhs.rk4_step(state, sp, 1800.0, dt)
        for name in CYL:
            assert np.isfinite(state.cyl_pos[name])
            assert 0.0 <= state.cyl_pos[name] <= rhs.cfg.mechanics.cylinders()[name].stroke_m


class TestFlowSharing:
    def test_flow_scale_limits_total(self):
        rhs, _ = _make_sim()
        state = _init_state(rhs)
        # force low pump flow and high demand -> flow_scale should shrink
        sp = {"boom_cyl": 0.9, "arm_cyl": 0.9, "bucket_cyl": 0.9}
        p, fs = rhs._solve_p_pump(state, sp, q_pump=0.0001)
        assert p > 0.0
        assert fs <= 1.0