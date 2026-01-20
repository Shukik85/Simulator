from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np

from .config import SystemConfig
from .state import State
from .faults import FaultConfig


PA_PER_BAR = 1e5


def sign0(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def smooth_abs(x: float, eps: float = 1e-9) -> float:
    return math.sqrt(x * x + eps)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def orifice_flow(cd: float, area: float, dp: float, rho: float) -> float:
    # турбулентный режим (классика): Q = Cd*A*sqrt(2*dp/rho)
    if area <= 0.0:
        return 0.0
    dp_eff = max(dp, 0.0)
    return float(cd * area * math.sqrt(2.0 * dp_eff / max(rho, 1e-9)))


@dataclass
class FlowDiagnostics:
    Qpump: float = 0.0        # m3/s
    Qrelief: float = 0.0      # m3/s
    Qopen_center: float = 0.0 # m3/s


class HydraulicModel:
    SECTIONS = ("boom", "arm", "bucket", "swing")

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

        self.rho = cfg.fluid.rho
        self.K = cfg.fluid.bulk_modulus

        # цилиндры
        self.cyl_cfg = {
            "boom": cfg.boom_cyl,
            "arm": cfg.arm_cyl,
            "bucket": cfg.bucket_cyl,
        }
        self.swing_cfg = cfg.swing

        # эквивалентные «объёмы камер»: dead + line + A*x
        self._A = {
            "boom": (cfg.boom_cyl.area_head, cfg.boom_cyl.area_annulus),
            "arm": (cfg.arm_cyl.area_head, cfg.arm_cyl.area_annulus),
            "bucket": (cfg.bucket_cyl.area_head, cfg.bucket_cyl.area_annulus),
        }

    # ---------- Pump / Open center / Relief ----------

    def pump_flow(self, rpm: float, Ppump: float, faults: FaultConfig) -> float:
        p = self.cfg.pump
        rpm = p.speed_clip(rpm)

        # базовый теоретический расход
        Qth = (p.displacement_cc_rev * 1e-6) * (rpm / 60.0)

        # volumetric efficiency падает с давлением + износом
        Pbar = Ppump / PA_PER_BAR
        wear = faults.pump_wear
        vol_eff = p.vol_eff_0 * (1.0 - p.vol_eff_kp * (Pbar / 350.0)) * (1.0 - 0.35 * wear)
        vol_eff = clamp(vol_eff, 0.60, 0.98)

        Q = Qth * vol_eff

        # утечка в картер растёт с давлением и износом
        Qcase = p.case_leak_k * Ppump * (1.0 + 4.0 * wear)
        Q = max(0.0, Q - Qcase)
        return float(Q)

    def relief_flow(self, Ppump: float, faults: FaultConfig) -> float:
        r = self.cfg.relief
        Pbar = Ppump / PA_PER_BAR

        crack = r.crack_bar
        # stuck_open => «как будто crack ниже/или gain выше»
        if faults.relief_stuck_open > 0:
            crack = crack * (1.0 - 0.35 * faults.relief_stuck_open)

        over = max(0.0, Pbar - crack)
        Q = r.gain_m3_s_per_bar * over
        return float(Q)

    def open_center_flow(self, Ppump: float, Ptank: float, Qpump: float, PLS: float, faults: FaultConfig) -> float:
        oc = self.cfg.open_center
        dp = max(0.0, Ppump - Ptank)
        if dp <= 0.0 or Qpump <= 0.0:
            return 0.0

        dp_standby = oc.standby_bar * PA_PER_BAR
        dp_band = max(oc.band_bar * PA_PER_BAR, oc.min_dp_pa)

        # байпас «управляется» относительным превышением Ppump над (PLS + standby)
        Ptarget = max(PLS, Ptank) + dp_standby
        alpha = clamp((Ppump - Ptarget) / dp_band, 0.0, 1.0)

        # выбираем эквивалентную площадь так, чтобы при dp_standby байпас мог «пропустить» Qpump
        Aeq = Qpump / (oc.cd * math.sqrt(2.0 * max(dp_standby, oc.min_dp_pa) / self.rho) + 1e-12)
        Qoc = orifice_flow(oc.cd, Aeq * alpha, dp, self.rho)

        # дополнительная утечка open-center (fault)
        if faults.open_center_leak > 0:
            Qoc = Qoc * (1.0 + 2.0 * faults.open_center_leak)

        return float(Qoc)

    # ---------- Valve sections ----------

    def _section_config(self, sec: str):
        vb = self.cfg.valve_bank
        return getattr(vb, sec)

    def spool_opening(self, u: float, sec: str, faults: FaultConfig) -> float:
        sc = self._section_config(sec)
        uabs = abs(float(u))

        db = sc.deadband
        if faults.valve_deadband_increase > 0:
            db = clamp(db + 0.10 * faults.valve_deadband_increase, 0.0, 0.5)

        if uabs <= db:
            return 0.0
        ue = (uabs - db) / max(1.0 - db, 1e-12)
        ue = clamp(ue, 0.0, 1.0)
        return float(ue ** sc.flow_exp)

    def req_flow_at_dp(self, sec: str, u: float, dp_ref_bar: float = 20.0, faults: FaultConfig | None = None) -> float:
        # «паспортная» аппроксимация: Qnom@20bar * opening * sign
        if faults is None:
            faults = FaultConfig()
        sc = self._section_config(sec)
        opening = self.spool_opening(u, sec, faults)
        Qnom = sc.q_nom_lpm_at_dp20 / 1000.0 / 60.0
        return float(sign0(u) * Qnom * opening)

    def allocate_flow_sharing(self, qreq: Dict[str, float], Qpump: float) -> Dict[str, float]:
        # идеальный flow-sharing: пропорционально урезаем, если суммарно не хватает
        qsum = sum(abs(q) for q in qreq.values())
        if qsum <= Qpump + 1e-18:
            return dict(qreq)
        alpha = Qpump / max(qsum, 1e-18)
        return {k: float(alpha * v) for k, v in qreq.items()}

    def valve_flows(self, sec: str, u: float, Ppump: float, PA: float, PB: float, Ptank: float, faults: FaultConfig) -> Tuple[float, float, float]:
        """
        Возвращает (QA, QB, Qp_in), где:
        QA>0 означает P->A, QA<0 означает A->T.
        QB>0 означает P->B, QB<0 означает B->T.
        Qp_in — сколько из насоса реально потребляет секция (>=0).
        """
        vb = self.cfg.valve_bank
        opening = self.spool_opening(u, sec, faults)
        if opening <= 0.0:
            return 0.0, 0.0, 0.0

        # калибруем площадь по Qnom при dp_ref=20 бар
        sc = self._section_config(sec)
        Qnom = sc.q_nom_lpm_at_dp20 / 1000.0 / 60.0
        dp_ref = 20.0 * PA_PER_BAR
        Aeq = Qnom / (vb.cd * math.sqrt(2.0 * dp_ref / self.rho) + 1e-12)
        A = Aeq * opening

        if u >= 0.0:
            # P->A, B->T
            QA = orifice_flow(vb.cd, A, max(Ppump - PA, 0.0), self.rho)
            QB = -orifice_flow(vb.cd, A, max(PB - Ptank, 0.0), self.rho)
            Qp_in = max(QA, 0.0)
        else:
            # P->B, A->T
            QB = orifice_flow(vb.cd, A, max(Ppump - PB, 0.0), self.rho)
            QA = -orifice_flow(vb.cd, A, max(PA - Ptank, 0.0), self.rho)
            Qp_in = max(QB, 0.0)

        return float(QA), float(QB), float(Qp_in)

    # ---------- Dynamics RHS ----------

    def _cyl_volumes(self, sec: str, x: float) -> Tuple[float, float]:
        c = self.cyl_cfg[sec]
        Ahead, Aann = self._A[sec]
        x = clamp(x, 0.0, c.stroke_m)

        VA = c.dead_volume_m3 + c.line_volume_m3 + Ahead * x
        VB = c.dead_volume_m3 + c.line_volume_m3 + Aann * (c.stroke_m - x)
        return max(VA, 1e-8), max(VB, 1e-8)

    def rhs(self, s: State, u: Dict[str, float], ext: Dict[str, float], faults: FaultConfig) -> Tuple[State, FlowDiagnostics]:
        cfg = self.cfg

        # LS target = max(PA/PB) по активным секциям
        active = []
        for sec in self.SECTIONS:
            if self.spool_opening(u.get(sec, 0.0), sec, faults) > 0:
                active.append(max(getattr(s, f"P{sec}A"), getattr(s, f"P{sec}B")) if sec != "swing" else max(s.PswingA, s.PswingB))
        PLS_target = max(active) if active else s.Ptank + cfg.open_center.standby_bar * PA_PER_BAR
        tau_ls = 0.05
        dPLS = (PLS_target - s.PLS) / max(tau_ls, 1e-6)

        # 1) насосный расход
        rpm = float(u.get("pumpspeed", cfg.pump.max_speed_rpm * 0.8))
        Qpump = self.pump_flow(rpm, s.Ppump, faults)

        # 2) предварительный расчёт запросов расхода (flow-sharing)
        qreq = {sec: self.req_flow_at_dp(sec, u.get(sec, 0.0), faults=faults) for sec in self.SECTIONS}
        qalloc = self.allocate_flow_sharing(qreq, Qpump)

        # 3) прикидываем Ppump (решением баланса расходов) через бискекцию
        def Qout(P: float) -> float:
            Qt = 0.0
            # секции потребляют от насоса только входной расход (P->порт)
            for sec in self.SECTIONS:
                uu = float(u.get(sec, 0.0))
                # в этой модели мы не подменяем u на qalloc напрямую, но qalloc влияет через «эффективное u»
                # чтобы сохранить физический смысл, сжимаем команду пропорционально выделенному расходу.
                q0 = qreq[sec]
                qa = qalloc[sec]
                scale = 1.0 if abs(q0) < 1e-12 else min(1.0, abs(qa) / max(abs(q0), 1e-12))
                u_eff = float(sign0(uu) * min(1.0, abs(uu) * scale))

                if sec == "swing":
                    QA, QB, Qpin = self.valve_flows("swing", u_eff, P, s.PswingA, s.PswingB, s.Ptank, faults)
                else:
                    QA, QB, Qpin = self.valve_flows(sec, u_eff, P, getattr(s, f"P{sec}A"), getattr(s, f"P{sec}B"), s.Ptank, faults)
                Qt += max(Qpin, 0.0)

            Qt += self.open_center_flow(P, s.Ptank, Qpump, s.PLS, faults)
            Qt += self.relief_flow(P, faults)
            return float(Qt)

        Plo = s.Ptank
        Phi = cfg.relief.max_bar * PA_PER_BAR
        flo = Qpump - Qout(Plo)
        fhi = Qpump - Qout(Phi)
        if flo <= 0.0:
            Ppump = Plo
        elif fhi >= 0.0:
            Ppump = Phi
        else:
            a, b = Plo, Phi
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = Qpump - Qout(m)
                if fm > 0:
                    a = m
                else:
                    b = m
            Ppump = 0.5 * (a + b)

        # 4) теперь, имея Ppump, считаем реальные расходы по секциям
        diag = FlowDiagnostics(Qpump=Qpump)
        diag.Qrelief = self.relief_flow(Ppump, faults)
        diag.Qopen_center = self.open_center_flow(Ppump, s.Ptank, Qpump, s.PLS, faults)

        # секционные расходы
        flows = {}
        for sec in self.SECTIONS:
            uu = float(u.get(sec, 0.0))
            q0 = qreq[sec]
            qa = qalloc[sec]
            scale = 1.0 if abs(q0) < 1e-12 else min(1.0, abs(qa) / max(abs(q0), 1e-12))
            u_eff = float(sign0(uu) * min(1.0, abs(uu) * scale))

            if sec == "swing":
                QA, QB, Qpin = self.valve_flows("swing", u_eff, Ppump, s.PswingA, s.PswingB, s.Ptank, faults)
            else:
                QA, QB, Qpin = self.valve_flows(sec, u_eff, Ppump, getattr(s, f"P{sec}A"), getattr(s, f"P{sec}B"), s.Ptank, faults)
            flows[sec] = (QA, QB)

        # 5) цилиндры: dP = K/V*(Q - A*v - leak), dv = (Fhyd - Fext - friction)/m
        ds = State(**s.__dict__)
        ds.Ppump = 0.0  # алгебраическое, не интегрируем
        ds.PLS = dPLS

        for sec in ("boom", "arm", "bucket"):
            c = self.cyl_cfg[sec]
            Ahead, Aann = self._A[sec]

            x = getattr(s, f"x{sec}")
            v = getattr(s, f"v{sec}")
            PA = getattr(s, f"P{sec}A")
            PB = getattr(s, f"P{sec}B")

            VA, VB = self._cyl_volumes(sec, x)
            QA, QB = flows[sec]

            # утечки (faults + базовые)
            leak_int = 1.0e-12 * (1.0 + 30.0 * float(getattr(faults, f"{sec}_internal_leak", 0.0)))
            Qleak = leak_int * (PA - PB)

            # давление
            dPA = (self.K / VA) * (QA - Ahead * v - Qleak)
            dPB = (self.K / VB) * (QB + Aann * v + Qleak)

            # силы
            Fhyd = PA * Ahead - PB * Aann
            Fext = float(ext.get(sec, 0.0))

            # трение (вязкое + кулоновское)
            Ffr = c.visc_damping * v + c.coulomb_friction * math.tanh(v / 0.01)

            dv = (Fhyd - Fext - Ffr) / max(c.mass_equiv, 1e-6)
            dx = v

            setattr(ds, f"x{sec}", dx)
            setattr(ds, f"v{sec}", dv)
            setattr(ds, f"P{sec}A", dPA)
            setattr(ds, f"P{sec}B", dPB)

        # 6) swing motor: Qm = D*omega; torque = (PswingA - PswingB)*D
        sm = self.swing_cfg
        QA, QB = flows["swing"]
        VswA = max(sm.dead_volume_m3 + sm.line_volume_m3, 1e-8)
        VswB = max(sm.dead_volume_m3 + sm.line_volume_m3, 1e-8)

        Psa = s.PswingA
        Psb = s.PswingB
        omega = s.omega

        leak_sw = 1.0e-12 * (1.0 + 30.0 * faults.swing_internal_leak)
        Qleak_sw = leak_sw * (Psa - Psb)

        Qm = sm.disp_m3_rad * omega
        dPsA = (self.K / VswA) * (QA - Qm - Qleak_sw)
        dPsB = (self.K / VswB) * (QB + Qm + Qleak_sw)

        Thyd = (Psa - Psb) * sm.disp_m3_rad
        Text = float(ext.get("swing", 0.0))
        Tfr = sm.visc_friction * omega + sm.coulomb_friction * math.tanh(omega / 0.02)

        domega = (Thyd - Text - Tfr) / max(sm.inertia, 1e-6)
        dtheta = omega

        ds.PswingA = dPsA
        ds.PswingB = dPsB
        ds.omega = domega
        ds.theta = dtheta

        # 7) тепловая модель: нагрев от гидравлических потерь + охлаждение
        # мощность на насосе ~ Ppump * Qpump / mech_eff; полезная ~ Ppump*Qpump; потери = разница
        p = cfg.pump
        Pbar = Ppump / PA_PER_BAR
        wear = faults.pump_wear
        mech_eff = p.mech_eff_0 * (1.0 - p.mech_eff_kp * (Pbar / 350.0)) * (1.0 - 0.25 * wear)
        mech_eff = clamp(mech_eff, 0.55, 0.98)

        P_hyd = Ppump * Qpump
        P_shaft = P_hyd / max(mech_eff, 1e-6)
        P_loss = max(0.0, P_shaft - P_hyd)

        th = cfg.thermal
        oil_mass = max(th.oil_mass_kg, 1e-6)
        Cth = oil_mass * cfg.fluid.cp
        cooling = th.cooling_W_per_K * (s.Thydraulic - th.ambient_C)
        dT = (th.pump_loss_to_oil * P_loss - cooling) / max(Cth, 1e-6)

        ds.Thydraulic = dT

        # алгебраический Ppump возвращаем отдельным полем (в интеграторе пересчитаем/клипнем)
        ds.Ppump = 0.0
        return ds, diag

    # ---------- Integrator ----------

    def _add(self, s: State, ds: State, h: float) -> State:
        out = s.copy()
        for k, v in ds.__dict__.items():
            if k == "Ppump":
                continue
            setattr(out, k, getattr(s, k) + h * v)
        return out

    def rk4_step(self, s: State, u: Dict[str, float], ext: Dict[str, float], dt: float, faults: FaultConfig) -> Tuple[State, FlowDiagnostics]:
        k1, d1 = self.rhs(s, u, ext, faults)
        s2 = self._add(s, k1, 0.5 * dt)
        k2, _ = self.rhs(s2, u, ext, faults)
        s3 = self._add(s, k2, 0.5 * dt)
        k3, _ = self.rhs(s3, u, ext, faults)
        s4 = self._add(s, k3, dt)
        k4, _ = self.rhs(s4, u, ext, faults)

        ns = s.copy()
        for k in ns.__dict__.keys():
            if k == "Ppump":
                continue
            ns_val = getattr(s, k) + (dt / 6.0) * (
                getattr(k1, k) + 2.0 * getattr(k2, k) + 2.0 * getattr(k3, k) + getattr(k4, k)
            )
            setattr(ns, k, float(ns_val))

        # ограничения по физике
        Pmin = 1.0e5
        Pmax = self.cfg.relief.max_bar * PA_PER_BAR
        for key in ("PLS", "PboomA", "PboomB", "ParmA", "ParmB", "PbucketA", "PbucketB", "PswingA", "PswingB"):
            setattr(ns, key, clamp(getattr(ns, key), Pmin, Pmax))

        # ограничения по ходам и скоростям
        ns.xboom = clamp(ns.xboom, 0.0, self.cfg.boom_cyl.stroke_m)
        ns.xarm = clamp(ns.xarm, 0.0, self.cfg.arm_cyl.stroke_m)
        ns.xbucket = clamp(ns.xbucket, 0.0, self.cfg.bucket_cyl.stroke_m)

        ns.vboom = clamp(ns.vboom, -2.5, 2.5)
        ns.varm = clamp(ns.varm, -2.5, 2.5)
        ns.vbucket = clamp(ns.vbucket, -2.5, 2.5)

        # theta в [-pi, pi]
        ns.theta = float((ns.theta + math.pi) % (2.0 * math.pi) - math.pi)
        ns.omega = clamp(ns.omega, -math.pi, math.pi)

        ns.Thydraulic = clamp(ns.Thydraulic, 0.0, 90.0)

        # пересчёт Ppump после интеграции (алгебраика в rhs)
        # для согласованности просто ещё раз решаем баланс в rhs, но уже без интегрирования:
        _, diag = self.rhs(ns, u, ext, faults)

        # восстановим Ppump через баланс (повторим бискекцию из rhs компактно)
        rpm = float(u.get("pumpspeed", self.cfg.pump.max_speed_rpm * 0.8))
        Qpump = self.pump_flow(rpm, ns.Ppump, faults)

        def Qout(P: float) -> float:
            Qt = 0.0
            for sec in self.SECTIONS:
                uu = float(u.get(sec, 0.0))
                if sec == "swing":
                    QA, QB, Qpin = self.valve_flows("swing", uu, P, ns.PswingA, ns.PswingB, ns.Ptank, faults)
                else:
                    QA, QB, Qpin = self.valve_flows(sec, uu, P, getattr(ns, f"P{sec}A"), getattr(ns, f"P{sec}B"), ns.Ptank, faults)
                Qt += max(Qpin, 0.0)
            Qt += self.open_center_flow(P, ns.Ptank, Qpump, ns.PLS, faults)
            Qt += self.relief_flow(P, faults)
            return float(Qt)

        Plo = ns.Ptank
        Phi = self.cfg.relief.max_bar * PA_PER_BAR
        flo = Qpump - Qout(Plo)
        fhi = Qpump - Qout(Phi)
        if flo <= 0.0:
            ns.Ppump = Plo
        elif fhi >= 0.0:
            ns.Ppump = Phi
        else:
            a, b = Plo, Phi
            for _ in range(50):
                m = 0.5 * (a + b)
                fm = Qpump - Qout(m)
                if fm > 0:
                    a = m
                else:
                    b = m
            ns.Ppump = 0.5 * (a + b)

        diag.Qpump = self.pump_flow(rpm, ns.Ppump, faults)
        diag.Qrelief = self.relief_flow(ns.Ppump, faults)
        diag.Qopen_center = self.open_center_flow(ns.Ppump, ns.Ptank, diag.Qpump, ns.PLS, faults)

        return ns, diag
