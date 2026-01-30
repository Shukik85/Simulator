from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple
import math


@dataclass(frozen=True)
class FluidConfig:
    rho: float = 850.0              # kg/m^3
    bulk_modulus: float = 1.7e9     # Pa
    nu_cSt: float = 46.0            # cSt (для справки/оценок)
    cp: float = 1900.0              # J/(kg*K)
    t_ref_C: float = 40.0


@dataclass(frozen=True)
class PumpConfig:
    displacement_cc_rev: float = 45.0
    max_speed_rpm: float = 2500.0
    vol_eff_0: float = 0.95
    vol_eff_kp: float = 0.10        # падение volumetric eff с ростом давления (безразм.)
    mech_eff_0: float = 0.92
    mech_eff_kp: float = 0.08
    case_leak_k: float = 2.0e-12    # m^3/(s*Pa) — утечка в картер ~ k*P

    def speed_clip(self, rpm: float) -> float:
        return max(0.0, min(self.max_speed_rpm, float(rpm)))


@dataclass(frozen=True)
class OpenCenterConfig:
    standby_bar: float = 10.0     # на нейтрали держим небольшое давление
    band_bar: float = 5.0         # насколько быстро «прикрывается/открывается» байпас
    cd: float = 0.65              # коэффициент расхода
    min_dp_pa: float = 1e3        # нижняя отсечка dp для стабильности


@dataclass(frozen=True)
class ReliefValveConfig:
    crack_bar: float = 300.0
    gain_m3_s_per_bar: float = 1.0e-5
    max_bar: float = 350.0


@dataclass(frozen=True)
class ValveSectionConfig:
    q_nom_lpm_at_dp20: float
    deadband: float               # 0..1 по |u|
    flow_exp: float               # n (обычно 1..2) нелинейность opening^n
    spool_hysteresis: float = 0.0 # 0..1 (опционально)


@dataclass(frozen=True)
class ValveBankConfig:
    cd: float = 0.65
    # Номинальные расходы при ΔP=20 бар (как «паспортные»)
    boom: ValveSectionConfig = ValveSectionConfig(q_nom_lpm_at_dp20=60.0, deadband=0.02, flow_exp=1.0)
    arm: ValveSectionConfig = ValveSectionConfig(q_nom_lpm_at_dp20=60.0, deadband=0.02, flow_exp=1.0)
    bucket: ValveSectionConfig = ValveSectionConfig(q_nom_lpm_at_dp20=40.0, deadband=0.02, flow_exp=1.0)
    swing: ValveSectionConfig = ValveSectionConfig(q_nom_lpm_at_dp20=40.0, deadband=0.03, flow_exp=1.0)


@dataclass(frozen=True)
class CylinderConfig:
    bore_mm: float
    rod_mm: float
    stroke_mm: float
    dead_volume_m3: float = 1.0e-4  # «мертвый» объём камеры для избежания сингулярностей
    line_volume_m3: float = 5.0e-4  # добавочный объём линий/полостей на камеру
    visc_damping: float = 120.0     # N*s/m
    coulomb_friction: float = 600.0 # N
    mass_equiv: float = 800.0       # kg эквивалентная приведённая масса

    @property
    def stroke_m(self) -> float:
        return self.stroke_mm / 1000.0

    @property
    def area_head(self) -> float:
        r = (self.bore_mm / 1000.0) / 2.0
        return math.pi * r * r

    @property
    def area_rod(self) -> float:
        r = (self.rod_mm / 1000.0) / 2.0
        return math.pi * r * r

    @property
    def area_annulus(self) -> float:
        return max(1e-9, self.area_head - self.area_rod)


@dataclass(frozen=True)
class SwingMotorConfig:
    displacement_cc_rev: float = 35.0
    inertia: float = 8000.0        # kg*m^2
    visc_friction: float = 80.0    # N*m*s/rad
    coulomb_friction: float = 200.0# N*m
    dead_volume_m3: float = 2.0e-4
    line_volume_m3: float = 5.0e-4

    @property
    def disp_m3_rad(self) -> float:
        # cc/rev -> m3/rev, rev -> 2*pi rad
        return (self.displacement_cc_rev * 1e-6) / (2.0 * math.pi)


@dataclass(frozen=True)
class MassPropertiesConfig:
    g: float = 9.81
    boom_mass: float = 800.0
    arm_mass: float = 500.0
    bucket_mass: float = 200.0
    payload_max: float = 2000.0
    superstructure_mass: float = 5000.0


@dataclass(frozen=True)
class SoilConfig:
    # очень грубо, но нелинейно и с доменной рандомизацией
    base_resistance_N: float = 25000.0
    vel_gain_N_per_m_s: float = 8000.0
    penetration_gain_N: float = 60000.0
    randomness: float = 0.25


@dataclass(frozen=True)
class ThermalConfig:
    # Потери превращаем в тепло в массе жидкости «эквивалентного бака»
    oil_mass_kg: float = 40.0
    ambient_C: float = 25.0
    cooling_W_per_K: float = 220.0  # чем больше, тем быстрее остывает
    pump_loss_to_oil: float = 0.7   # доля потерь насоса, уходящая в масло


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.001
    cycle_duration_s: float = 60.0
    seed: int = 42

    # доменная рандомизация
    randomize_payload: bool = True
    randomize_soil: bool = True
    randomize_oil_temp: bool = True

    # режимы
    modes: Tuple[str, ...] = ("digging_light", "digging_medium", "digging_heavy", "swing_only", "boom_up", "boom_down", "combined", "idle")
    mode_distribution: Dict[str, float] = field(default_factory=lambda: {
        "digging_light": 0.15,
        "digging_medium": 0.25,
        "digging_heavy": 0.20,
        "swing_only": 0.15,
        "boom_up": 0.10,
        "boom_down": 0.10,
        "combined": 0.05,
        "idle": 0.00,
    })


@dataclass(frozen=True)
class SensorConfig:
    # шум как доля диапазона (std)
    pressure_noise_pct: float = 0.5
    position_noise_pct: float = 0.2
    temp_noise_abs_C: float = 0.8
    rpm_noise_abs: float = 10.0

    # частоты можно потом использовать для ресэмплинга, но в симуляторе всё на dt
    sensors: Dict[str, Dict] = field(default_factory=lambda: {
        "Ppump": {"unit": "bar", "range": (0.0, 350.0)},
        "PLS": {"unit": "bar", "range": (0.0, 80.0)},
        "PboomA": {"unit": "bar", "range": (0.0, 350.0)},
        "PboomB": {"unit": "bar", "range": (0.0, 350.0)},
        "ParmA": {"unit": "bar", "range": (0.0, 350.0)},
        "ParmB": {"unit": "bar", "range": (0.0, 350.0)},
        "PbucketA": {"unit": "bar", "range": (0.0, 350.0)},
        "PbucketB": {"unit": "bar", "range": (0.0, 350.0)},
        "PswingA": {"unit": "bar", "range": (0.0, 350.0)},
        "PswingB": {"unit": "bar", "range": (0.0, 350.0)},
        "Xboom": {"unit": "mm", "range": (0.0, 1500.0)},
        "Xarm": {"unit": "mm", "range": (0.0, 1200.0)},
        "Xbucket": {"unit": "mm", "range": (0.0, 800.0)},
        "Thetaswing": {"unit": "deg", "range": (-180.0, 180.0)},
        "Thydraulic": {"unit": "C", "range": (0.0, 90.0)},
        "pumpspeed": {"unit": "rpm", "range": (0.0, 2500.0)},
        "Qpump": {"unit": "lpm", "range": (0.0, 250.0)},
        "Qrelief": {"unit": "lpm", "range": (0.0, 250.0)},
        "Qopen_center": {"unit": "lpm", "range": (0.0, 250.0)},
    })


@dataclass(frozen=True)
class SystemConfig:
    # How limited pump flow is allocated across simultaneous operations.
    # - "proportional": current behavior (uniform scaling)
    # - "priority": weighted allocation (swing gets more share by default)
    flow_sharing_mode: Literal["proportional", "priority"] = "proportional"

    fluid: FluidConfig = FluidConfig()
    pump: PumpConfig = PumpConfig()
    open_center: OpenCenterConfig = OpenCenterConfig()
    relief: ReliefValveConfig = ReliefValveConfig()
    valve_bank: ValveBankConfig = ValveBankConfig()
    boom_cyl: CylinderConfig = CylinderConfig(bore_mm=90.0, rod_mm=50.0, stroke_mm=1500.0)
    arm_cyl: CylinderConfig = CylinderConfig(bore_mm=80.0, rod_mm=45.0, stroke_mm=1200.0)
    bucket_cyl: CylinderConfig = CylinderConfig(bore_mm=70.0, rod_mm=40.0, stroke_mm=800.0)
    swing: SwingMotorConfig = SwingMotorConfig()
    mech: MassPropertiesConfig = MassPropertiesConfig()
    soil: SoilConfig = SoilConfig()
    thermal: ThermalConfig = ThermalConfig()
    sim: SimulationConfig = SimulationConfig()
    sensor: SensorConfig = SensorConfig()
