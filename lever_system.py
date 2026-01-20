from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

Vec2 = NDArray[np.float64]
Vec4 = NDArray[np.float64]

# =============================================================================
# Вспомогательные функции
# =============================================================================
def v2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float64)

def norm2(v: Vec2) -> float:
    return float(np.hypot(v[0], v[1]))

def wrap_angle_pi(a: float) -> float:
    """Приведение угла к диапазону (-π, π]"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)

def cross2_z(a: Vec2, b: Vec2) -> float:
    """2D векторное произведение (z-компонента)"""
    return float(a[0] * b[1] - a[1] * b[0])

def mm_to_m(mm: float) -> float:
    return mm / 1000.0

def m_to_mm(m: float) -> float:
    return m * 1000.0


# =============================================================================
# Модели данных
# =============================================================================
@dataclass(frozen=True, slots=True)
class FixedPoints:
    O: Vec2  # шарнир гидроцилиндра
    V: Vec2  # шарнир нагрузки
    G: Vec2  # шарнир балки

@dataclass(frozen=True, slots=True)
class LinkLengths:
    AB: float  # длина тяги AB
    BV: float  # длина рычага BV
    GA: float  # длина тяги GA
    OA_min: float  # мин. длина цилиндра
    OA_max: float  # макс. длина цилиндра

@dataclass(frozen=True, slots=True)
class KinematicState:
    l_OA: float
    A: Vec2
    B: Vec2
    fixed: FixedPoints
    lengths: LinkLengths
    residual_norm: float  # невязка кинематики
    J_constraint: Optional[NDArray[np.float64]] = None  # якобиан ограничений (4x4)

@dataclass(frozen=True, slots=True)
class Transmission:
    theta_OA: float         # угол OA, рад
    theta_BV: float         # угол BV, рад
    dtheta_BV_dl_OA: float  # передаточное отношение, рад/м (АНАЛИТИЧЕСКОЕ)
    is_singular: bool       # истинная мертвая точка?
    condition_number: float # обусловленность якобиана

@dataclass(frozen=True, slots=True)
class MomentResult:
    F_OA: float
    M_V_developed: float
    dead_point: bool

@dataclass(frozen=True, slots=True)
class ForceResult:
    M_V: float
    F_OA_required: float
    dead_point: bool

@dataclass(frozen=True, slots=True)
class AnalysisResult:
    state: KinematicState
    transmission: Transmission
    direct: MomentResult
    inverse: ForceResult


# =============================================================================
# Кинематика: решение геометрических ограничений
# =============================================================================
class KinematicsError(RuntimeError):
    pass

class LeverSystemGeometry:
    """
    Решает кинематику механизма с учётом:
      - постоянных длин звеньев
      - фиксированных точек O, V, G
      - переменной длины OA (гидроцилиндр)
      - требование положения точки A "выше OG"
    
    Новое: вычисляет якобиан ограничений для аналитической производной.
    """

    def __init__(
        self,
        fixed_points_mm: dict[str, tuple[float, float]],
        A0_mm: tuple[float, float],
        B0_mm: tuple[float, float],
        OA_min_mm: float = 1210.0,
        OA_max_mm: float = 1920.0,
        desired_a_side: int = +1,
    ) -> None:
        self.fixed = FixedPoints(
            O=v2(mm_to_m(fixed_points_mm["O"][0]), mm_to_m(fixed_points_mm["O"][1])),
            V=v2(mm_to_m(fixed_points_mm["V"][0]), mm_to_m(fixed_points_mm["V"][1])),
            G=v2(mm_to_m(fixed_points_mm["G"][0]), mm_to_m(fixed_points_mm["G"][1])),
        )
        A0 = v2(mm_to_m(A0_mm[0]), mm_to_m(A0_mm[1]))
        B0 = v2(mm_to_m(B0_mm[0]), mm_to_m(B0_mm[1]))

        self.lengths = LinkLengths(
            AB=norm2(B0 - A0),
            BV=norm2(B0 - self.fixed.V),
            GA=norm2(A0 - self.fixed.G),
            OA_min=mm_to_m(OA_min_mm),
            OA_max=mm_to_m(OA_max_mm),
        )

        self.desired_a_side = 1 if desired_a_side >= 0 else -1
        self._last_x: Optional[Vec4] = None

    def _residuals(self, x: Vec4, l_OA: float) -> Vec4:
        A, B = x[:2], x[2:]
        O, V, G = self.fixed.O, self.fixed.V, self.fixed.G
        L = self.lengths
        return np.array([
            norm2(A - O) - l_OA,
            norm2(B - A) - L.AB,
            norm2(B - V) - L.BV,
            norm2(A - G) - L.GA,
        ])

    def _jacobian_constraints(self, x: Vec4, l_OA: float) -> NDArray[np.float64]:
        """
        Вычисляет матрицу Якобиана ограничений: J[i,j] = ∂f_i/∂x_j
        
        Ограничения:
        f_0 = |A-O|² - l_OA²
        f_1 = |B-A|² - L_AB²
        f_2 = |B-V|² - L_BV²
        f_3 = |A-G|² - L_GA²
        
        Переменные: x = [A_x, A_y, B_x, B_y]
        """
        A = x[:2]
        B = x[2:]
        O = self.fixed.O
        V = self.fixed.V
        G = self.fixed.G
        
        J = np.zeros((4, 4), dtype=np.float64)
        
        # ∂(|A-O|²)/∂A = 2(A-O)
        J[0, :2] = 2.0 * (A - O)
        
        # ∂(|B-A|²)/∂A = -2(B-A),  ∂(|B-A|²)/∂B = 2(B-A)
        J[1, :2] = -2.0 * (B - A)
        J[1, 2:] = 2.0 * (B - A)
        
        # ∂(|B-V|²)/∂B = 2(B-V)
        J[2, 2:] = 2.0 * (B - V)
        
        # ∂(|A-G|²)/∂A = 2(A-G)
        J[3, :2] = 2.0 * (A - G)
        
        return J

    def _initial_guesses(self, l_OA: float) -> Iterable[Vec4]:
        """Генерация начальных приближений — улучшает сходимость."""
        O, V = self.fixed.O, self.fixed.V
        dir_ov = (V - O)
        if norm2(dir_ov) > 1e-14:
            dir_ov /= norm2(dir_ov)
            rot90 = np.array([[0, -1], [1, 0]])

            # 1. Последнее решение (warm-start)
            if self._last_x is not None:
                yield self._last_x

            # 2. Вдоль OV
            A1 = O + l_OA * dir_ov
            B1 = V - self.lengths.BV * dir_ov
            yield np.array([A1[0], A1[1], B1[0], B1[1]])

            # 3. Перпендикулярно OV
            dir_perp = rot90 @ dir_ov
            A2 = O + l_OA * dir_perp
            B2 = V - self.lengths.BV * dir_perp
            yield np.array([A2[0], A2[1], B2[0], B2[1]])

    def solve(self, l_OA_mm: float, max_residual_mm: float = 0.1) -> KinematicState:
        l_OA = mm_to_m(l_OA_mm)
        if not (self.lengths.OA_min <= l_OA <= self.lengths.OA_max):
            raise ValueError(f"l_OA={l_OA_mm} мм вне допустимого диапазона.")

        best_solution: Optional[Vec4] = None
        best_norm = np.inf
        best_jacobian: Optional[NDArray[np.float64]] = None

        for x0 in self._initial_guesses(l_OA):
            res = least_squares(
                fun=lambda x: self._residuals(x, l_OA),
                x0=x0,
                method='trf',
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=200,
            )
            if not res.success:
                continue

            x = res.x.astype(np.float64)
            r = self._residuals(x, l_OA)
            r_norm = float(np.linalg.norm(r))
            A = x[:2]

            # Проверка позиции A относительно OG
            if self.a_side_sign(A) != self.desired_a_side:
                continue

            if r_norm < best_norm:
                best_norm = r_norm
                best_solution = x
                best_jacobian = self._jacobian_constraints(x, l_OA)

            if r_norm <= mm_to_m(max_residual_mm):
                break

        if best_solution is None:
            raise KinematicsError("Решение не найдено на заданной ветви (A выше OG).")
        if best_norm > mm_to_m(max_residual_mm):
            raise KinematicsError(f"Слишком большая невязка: {best_norm:.3e} м.")

        A = best_solution[:2].copy()
        B = best_solution[2:].copy()
        self._last_x = best_solution

        return KinematicState(
            l_OA=l_OA,
            A=A,
            B=B,
            fixed=self.fixed,
            lengths=self.lengths,
            residual_norm=best_norm,
            J_constraint=best_jacobian,
        )

    def a_side_sign(self, A: Vec2) -> int:
        """Определяет, с какой стороны от OG находится точка A."""
        og = self.fixed.G - self.fixed.O
        oa = A - self.fixed.O
        cz = cross2_z(og, oa)
        return 1 if cz > 1e-14 else (-1 if cz < -1e-14 else 0)

    def theta_OA(self, state: KinematicState) -> float:
        return np.arctan2(state.A[1] - state.fixed.O[1], state.A[0] - state.fixed.O[0])

    def theta_BV(self, state: KinematicState) -> float:
        v = state.fixed.V - state.B
        return np.arctan2(v[1], v[0])


# =============================================================================
# Динамика: передаточное отношение через аналитический якобиан
# =============================================================================
class LeverSystemDynamics:
    """
    Расчёт передаточного отношения dθ_BV / dl_OA через АНАЛИТИЧЕСКИЙ якобиан.
    
    Метод:
    - Неявные функции: F(x(l), l) = 0 (4 ограничения кинематики)
    - Нужна производная: dθ_BV / dl
    - Вычисляем через цепное правило + обратная матрица якобиана
    """

    def __init__(self, geom: LeverSystemGeometry) -> None:
        self.geom = geom

    def transmission(self, state: KinematicState) -> Transmission:
        """
        Вычисляет передаточное отношение АНАЛИТИЧЕСКИ через якобиан.
        
        Принцип: F(x(l), l) = 0 ⟹ ∂F/∂x · dx/dl + ∂F/∂l = 0 ⟹ dx/dl = -(J_x)⁻¹ · J_l
        
        где:
        - J_x: якобиан по переменным (4x4)
        - J_l: якобиан по параметру l (4x1)
        """
        theta_0 = self.geom.theta_BV(state)
        
        # Якобиан по переменным x = [A_x, A_y, B_x, B_y]
        J_x = state.J_constraint
        if J_x is None:
            raise RuntimeError("J_constraint не вычислен. Требуется пересолвить.")
        
        # Якобиан по параметру l (производная остатков по l_OA):
        # f_0 = |A-O|² - l_OA² ⟹ ∂f_0/∂l = -2·l_OA
        # f_1, f_2, f_3 не зависят от l ⟹ ∂f_i/∂l = 0
        J_l = np.array([-2.0 * state.l_OA, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Решаем: J_x · dx_dl = -J_l
        try:
            cond = np.linalg.cond(J_x)
            dx_dl = np.linalg.solve(J_x, -J_l)
            is_singular = cond > 1e10
        except np.linalg.LinAlgError:
            # Якобиан вырожден (мертвая точка)
            is_singular = True
            cond = np.inf
            dx_dl = np.zeros(4, dtype=np.float64)
        
        # Теперь нужна производная θ_BV по компонентам x
        # θ_BV = atan2(V_y - B_y, V_x - B_x)
        # dθ/dB = (-ry/(rx²+ry²), rx/(rx²+ry²)) где (rx, ry) = (V_x - B_x, V_y - B_y)
        
        V = state.fixed.V
        B = state.B
        r = V - B
        r_sq = r[0]**2 + r[1]**2
        
        if r_sq < 1e-12:
            # Вырожденная позиция
            dtheta_dl = 0.0
            is_singular = True
        else:
            dtheta_dB = np.array([-r[1] / r_sq, r[0] / r_sq], dtype=np.float64)
            # dx_dl[2:4] — это dB/dl
            dtheta_dl = float(np.dot(dtheta_dB, dx_dl[2:4]))
        
        return Transmission(
            theta_OA=self.geom.theta_OA(state),
            theta_BV=theta_0,
            dtheta_BV_dl_OA=dtheta_dl,
            is_singular=is_singular,
            condition_number=float(cond) if not np.isinf(cond) else 1e100,
        )

    def developed_moment(self, state: KinematicState, F_OA_N: float, eps: float = 1e-10) -> MomentResult:
        tr = self.transmission(state)
        k = tr.dtheta_BV_dl_OA

        if tr.is_singular or abs(k) < eps:
            return MomentResult(F_OA=F_OA_N, M_V_developed=0.0, dead_point=True)

        M_V = F_OA_N / k
        return MomentResult(F_OA=F_OA_N, M_V_developed=M_V, dead_point=False)

    def required_force(self, state: KinematicState, M_V_Nm: float, eps: float = 1e-10) -> ForceResult:
        tr = self.transmission(state)
        k = tr.dtheta_BV_dl_OA

        if tr.is_singular or abs(k) < eps:
            return ForceResult(M_V=M_V_Nm, F_OA_required=float('inf'), dead_point=True)

        F_OA = M_V_Nm * k
        return ForceResult(M_V=M_V_Nm, F_OA_required=F_OA, dead_point=False)


# =============================================================================
# Основной интерфейс
# =============================================================================
class LeverSystem:
    """
    Упрощённый интерфейс для анализа рычажной системы.
    """

    def __init__(self) -> None:
        self.geom = LeverSystemGeometry(
            fixed_points_mm={
                "O": (5182.414, 1840.724),
                "V": (6372.718, 448.550),
                "G": (6170.904, 596.101),
            },
            A0_mm=(6058.310, 1005.915),
            B0_mm=(6410.900, 751.150),
            OA_min_mm=1210.0,
            OA_max_mm=1920.0,
            desired_a_side=+1,
        )
        self.dyn = LeverSystemDynamics(self.geom)

    def analyze(self, l_OA_mm: float, F_OA_N: float, M_V_Nm: Optional[float] = None) -> AnalysisResult:
        state = self.geom.solve(l_OA_mm)
        tr = self.dyn.transmission(state)
        direct = self.dyn.developed_moment(state, F_OA_N)

        M_V_used = M_V_Nm if M_V_Nm is not None else direct.M_V_developed
        inverse = self.dyn.required_force(state, M_V_used)

        return AnalysisResult(
            state=state,
            transmission=tr,
            direct=direct,
            inverse=inverse,
        )

    def sweep(self, F_OA_N: float, n: int = 37) -> list[AnalysisResult]:
        l_min = m_to_mm(self.geom.lengths.OA_min)
        l_max = m_to_mm(self.geom.lengths.OA_max)
        return [self.analyze(l, F_OA_N) for l in np.linspace(l_min, l_max, n)]


# =============================================================================
# Тестирование
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys = LeverSystem()
    res = sys.analyze(l_OA_mm=1565.0, F_OA_N=1000.0)

    logger.info("Результаты анализа при l_OA = 1565 мм:")
    logger.info("  Невязка: %.3e м", res.state.residual_norm)
    logger.info("  θ_BV: %.3f°", np.degrees(res.transmission.theta_BV))
    logger.info("  dθ/dl: %.3e рад/м (АНАЛИТИЧЕСКОЕ)", res.transmission.dtheta_BV_dl_OA)
    logger.info("  Обусловленность якобиана: %.2e", res.transmission.condition_number)
    logger.info("  Мертвая точка: %s", res.transmission.is_singular)
    logger.info("  Развиваемый момент: %.3f Н·м", res.direct.M_V_developed)
    logger.info("  Требуемое усилие: %.1f Н", res.inverse.F_OA_required)
