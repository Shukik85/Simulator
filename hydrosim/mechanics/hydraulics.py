# hydrosim/mechanics/hydraulics.py
"""Модель гидроцилиндров."""

from dataclasses import dataclass
import numpy as np

@dataclass
class HydraulicCylinderSpec:
    l_min: float
    l_max: float
    piston_d: float
    rod_d: float
    max_pressure_piston_bar: float
    max_pressure_rod_bar: float
    density: float = 7850.0

    def __post_init__(self):
        self.stroke = self.l_max - self.l_min
        self.area_piston = 0.25 * np.pi * self.piston_d**2
        self.area_rod = 0.25 * np.pi * self.rod_d**2  # ✅ Площадь штока
        self.effective_area_retraction = self.area_piston - self.area_rod  # ✅ Полезная площадь при втягивании

        self.max_pressure_piston_Pa = self.max_pressure_piston_bar * 1e5
        self.max_pressure_rod_Pa = self.max_pressure_rod_bar * 1e5

        # Расчёт масс - ИСПРАВЛЕНО: используем stroke вместо l_min
        wall_t = 0.1 * self.piston_d
        inner_d = max(self.piston_d - 2 * wall_t, self.rod_d + 0.01)
        
        # Масса цилиндра (корпуса)
        vol_cyl = 0.25 * np.pi * (self.piston_d**2 - inner_d**2) * self.stroke
        mass_cyl = vol_cyl * self.density

        # Масса штока (длина = минимальная длина + 80% хода)
        vol_rod = 0.25 * np.pi * self.rod_d**2 * (self.l_min + self.stroke * 0.8)
        mass_rod = vol_rod * self.density

        # Масса поршня
        piston_thick = 0.04 * self.piston_d
        vol_piston = 0.25 * np.pi * (self.piston_d**2 - self.rod_d**2) * piston_thick
        mass_piston = vol_piston * self.density

        self.cylinder_mass_kg = mass_cyl
        self.piston_mass_kg = mass_rod + mass_piston

    def get_length_change(self, dV: float, chamber: str) -> float:
        area = self.area_piston if chamber == "piston" else self.effective_area_retraction
        return dV / area

    def is_motion_limited(self, current_length: float, dV: float, chamber: str) -> tuple[bool, str]:
        dl = self.get_length_change(dV, chamber)
        delta_l = dl if chamber == "piston" else -dl
        new_length = current_length + delta_l
        if new_length >= self.l_max:
            return True, "extension_limited"
        if new_length <= self.l_min:
            return True, "retraction_limited"
        return False, "ok"

    def get_force_at_max_pressure(self, chamber: str) -> float:
        if chamber == "piston":
            return self.area_piston * self.max_pressure_piston_Pa
        else:
            return -self.area_rod * self.max_pressure_rod_Pa