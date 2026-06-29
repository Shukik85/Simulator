"""bucket_lever.py

Кинематика и кинетостатика рычажного механизма ковша экскаватора.
Реализует чисто‑геометрические расчёты без внутреннего состояния.
Все функции потокобезопасны и могут быть использованы напрямую
из kinematics.py или dynamics.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal, overload

import numpy as np
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Types and small constants
# ----------------------------------------------------------------------
Vec2 = np.ndarray          # shape (2,), dtype=float64
_EPS = 1e-9                # tolerance for geometric checks
_DEFAULT_THETA_MIN = -2 * np.pi / 3   # -120°
_DEFAULT_THETA_MAX =  5 * np.pi / 6   # +150°


# ----------------------------------------------------------------------
# Immutable parameters of the mechanism
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BucketLeverParams:
    """Полное, неизменяемое описание рычажного механизма ковша.

    Все длины – в метрах, все точки – в глобальной СК excavator‑base,
    за исключением ``com_local``, которое задаётся в СК ковша
    (от точки D к центру масс).
    """
    L_lever: float               # длина рычага CP
    L_rod: float                 # длина тяги PE
    L_bucket: float              # радиус ковша (D → E)
    mass: float                  # масса ковша
    A: Vec2                      # крепление ГЦ в глобальной СК
    C: Vec2                      # ось рычага в глобальной СК
    D: Vec2                      # ось ковша в глобальной СК
    com_local: Vec2              # вектор D → centre‑of‑mass в СК ковша
    inertia: Optional[float] = None   # момент инерции относительно оси D
    theta_min: float = _DEFAULT_THETA_MIN
    theta_max: float = _DEFAULT_THETA_MAX

    def __post_init__(self) -> None:
        if any(l <= 0 for l in (self.L_lever, self.L_rod, self.L_bucket)):
            raise ValueError("All lengths must be positive")
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if np.linalg.norm(self.A - self.C) < _EPS or np.linalg.norm(self.C - self.D) < _EPS:
            raise ValueError("Points A, C, D must be distinct")
        if self.theta_min >= self.theta_max:
            raise ValueError("theta_min must be < theta_max")
        if np.linalg.norm(self.com_local) < _EPS:
            raise ValueError("com_local must be a non‑zero vector")
        # Если инерция не задана – считаем её точечной массой в com_local
        if self.inertia is None:
            object.__setattr__(
                self,
                "inertia",
                self.mass * float(np.dot(self.com_local, self.com_local)),
            )
        if self.inertia <= 0:
            raise ValueError("Inertia must be positive")


# ----------------------------------------------------------------------
# Stateless solver – pure geometry
# ----------------------------------------------------------------------
class BucketLeverSolver:
    """Набор статических методов, решающих геометрические подзадачи."""

    @staticmethod
    def _solve_triangle_CPE(
        E: Vec2,
        C: Vec2,
        L_lever: float,
        L_rod: float,
    ) -> Tuple[Optional[Vec2], Optional[Vec2]]:
        """
        Находит обе возможные позиции точки P при известном E.

        Возвращает (P1, P2) – либо None, если треугольник невозможен.
        """
        CE_vec = E - C
        CE = np.linalg.norm(CE_vec)
        if CE < _EPS:
            return None, None

        # проверка неравенства треугольника
        if CE > L_lever + L_rod + _EPS or \
           CE < abs(L_lever - L_rod) - _EPS:
            return None, None

        # теорема косинусов → угол при C
        cos_phi = (L_lever ** 2 + CE ** 2 - L_rod ** 2) / (2 * L_lever * CE)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)

        angle_CE = np.arctan2(CE_vec[1], CE_vec[0])
        P1 = C + L_lever * np.array(
            [np.cos(angle_CE + phi), np.sin(angle_CE + phi)]
        )
        P2 = C + L_lever * np.array(
            [np.cos(angle_CE - phi), np.sin(angle_CE - phi)]
        )
        return P1, P2

    # ------------------------------------------------------------------
    # Прямая кинематика: L_cyl → θ (+ точки)
    # ------------------------------------------------------------------
    @staticmethod
    def solve(
        params: BucketLeverParams,
        L_cyl: float,
        theta_guess: Optional[float] = None,
        prev_P: Optional[Vec2] = None,
    ) -> Tuple[Optional[float], Dict[str, Vec2]]:
        """
        Находит угол ковша θ (рад) по известной pin‑to‑pin длине гидроцилиндра.

        Parameters
        ----------
        params : BucketLeverParams
            Геометрические и inertial‑параметры механизма.
        L_cyl : float
            Текущая длина гидроцилиндра (м).
        theta_guess : float | None, optional
            Начальное приближение для поиска. Если None – берётся середина
            допустимого диапазона.
        prev_P : Vec2 | None, optional
            Точка P с предыдущего шага (необходима для выбора правильной
            ветви решения треугольника CPE). Если None – будет выбран первый
            найденный кандидат.

        Returns
        -------
        theta : float | None
            Угол ковша в радианах, либо ``None`` если решений нет в
            допустимом диапазоне.
        points : dict
            Словарь с ключами:
                ``"A"``, ``"C"``, ``"D"`` – фиксированные точки,
                ``"E"``  – конец ковша,
                ``"P"``  – шарнир соединения рычага и тяги (может быть None),
                ``"com"``– центр масс ковша (в глобальной СК).
        """
        L_cyl = float(L_cyl)
        theta_min, theta_max = params.theta_min, params.theta_max
        if theta_guess is None:
            theta_guess = 0.5 * (theta_min + theta_max)

        # ---- замыкание, хранящее последнюю валидную точку P ------------
        last_valid_P: list[Optional[Vec2]] = [prev_P]

        def residual(theta: float) -> float:
            """Разница между текущей и целевой длиной гидроцилиндра."""
            E = params.D + params.L_bucket * np.array(
                [np.cos(theta), np.sin(theta)]
            )
            P1, P2 = BucketLeverSolver._solve_triangle_CPE(
                E, params.C, params.L_lever, params.L_rod
            )
            candidates = [p for p in (P1, P2) if p is not None]
            if not candidates:
                return 1e9  # большое число → никакого пересечения с 0

            # Выбор ветви: если известна предыдущая точка – берём ближайшую к ней
            if last_valid_P[0] is not None:
                Pchosen = min(
                    candidates,
                    key=lambda p: np.linalg.norm(p - last_valid_P[0]),
                )
            else:
                Pchosen = candidates[0]

            last_valid_P[0] = Pchosen.copy()
            return np.linalg.norm(Pchosen - params.A) - L_cyl

        # ---- поиск интервала со сменой знака ---------------------------
        n_samples = 200
        thetas = np.linspace(theta_min, theta_max, n_samples)
        residuals = np.empty_like(thetas)
        for i, th in enumerate(thetas):
            r = residual(th)
            residuals[i] = r if np.isfinite(r) else 1e9

        bracket: Optional[Tuple[float, float]] = None
        for i in range(len(residuals) - 1):
            r1, r2 = residuals[i], residuals[i + 1]
            if r1 * r2 < 0:
                bracket = (thetas[i], thetas[i + 1])
                break

        # ---- если скобка не найдена – берём ближайшее значение ----------
        if bracket is None:
            idx = int(np.argmin(np.abs(residuals)))
            theta_best = thetas[idx]
            logger.warning(
                f"No sign change for L_cyl={L_cyl:.3f}; "
                f"using closest θ={np.degrees(theta_best):.1f}°"
            )
            P_best = last_valid_P[0]
            if P_best is None:
                return None, {}
            points = BucketLeverSolver._build_points(params, theta_best, P_best)
            return theta_best, points

        # ---- основной поиск корня (Брент) -----------------------------
        try:
            sol = root_scalar(
                residual,
                bracket=bracket,
                method="brentq",
                xtol=1e-12,
                maxiter=100,
            )
            if not sol.converged:
                logger.warning("Root finding did not converge.")
                theta_best = (bracket[0] + bracket[1]) * 0.5
            else:
                theta_best = sol.root
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Root finding failed: {exc}")
            theta_best = (bracket[0] + bracket[1]) * 0.5

        P_best = last_valid_P[0]
        if P_best is None:
            return None, {}

        points = BucketLeverSolver._build_points(params, theta_best, P_best)
        return theta_best, points

    # ------------------------------------------------------------------
    # Обратная задача: θ → L_cyl
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def cylinder_length(
        params: BucketLeverParams,
        theta: float,
        *,
        return_points: Literal[False] = False,
    ) -> Optional[float]: ...
    @overload
    @staticmethod
    def cylinder_length(
        params: BucketLeverParams,
        theta: float,
        *,
        return_points: Literal[True],
    ) -> Tuple[Optional[float], Optional[Dict[str, Vec2]]]: ...

    @staticmethod
    def cylinder_length(
        params: BucketLeverParams,
        theta: float,
        *,
        return_points: bool = False,
    ) -> ...:
        """
        Вычисляет длину гидроцилиндра по заданному углу ковша.

        Parameters
        ----------
        params : BucketLeverParams
        theta : float
            Угол ковша (рад).
        return_points : bool, default False
            Если True – возвращает additionally dict с точками.

        Returns
        -------
        L_cyl : float | None
            Длина ГЦ (м) или None, если геометрически невозможно.
        points : dict | None (только если ``return_points=True``)
        """
        theta = float(theta)
        E = params.D + params.L_bucket * np.array(
            [np.cos(theta), np.sin(theta)]
        )
        P1, P2 = BucketLeverSolver._solve_triangle_CPE(
            E, params.C, params.L_lever, params.L_rod
        )
        candidates = [p for p in (P1, P2) if p is not None]
        if not candidates:
            if return_points:
                return None, None
            return None

        # Выбираем ту же ветвь, что и в прямом решении (ближайшая к предыдущей P)
        # Здесь previous P неизвестна – просто берём первый кандидат.
        Pchosen = candidates[0]
        L_cyl = float(np.linalg.norm(Pchosen - params.A))

        if return_points:
            points = BucketLeverSolver._build_points(params, theta, Pchosen)
            return L_cyl, points
        return L_cyl

    # ------------------------------------------------------------------
    # Статические утилиты, не зависящие от состояния
    # ------------------------------------------------------------------
    @staticmethod
    def cylinder_force_to_moment(
        params: BucketLeverParams,
        force: float,
        theta: float,
        dtheta: float = 1e-6,
    ) -> float:
        """
        Вычисляет момент, создаваемый силой гидроцилиндра относительно оси D,
        учитывая переменное передаточное число механизма (принцип виртуальной работы).

        M = F * dL_cyl / dθ

        Parameters
        ----------
        force : float
            Сила, действующая вдоль оси цилиндра (Н). Положительная –
            вытягивает шток (если ваша модель обратна, поменяйте знак).
        theta : float
            Текущий угол ковша (рад).
        dtheta : float, optional
            Шаг для численного дифференцирования (по‑умолчанию 1e-6 рад).

        Returns
        -------
        float
            Момент (Н·м), положительный – увеличивает угол ковша.
        """
        # длины цилиндра в смежных точках
        L_plus = BucketLeverSolver.cylinder_length(params, theta + dtheta)
        L_minus = BucketLeverSolver.cylinder_length(params, theta - dtheta)

        if L_plus is None or L_minus is None:
            return 0.0   # конфигурация недоступна → момент нулевой

        dL_dtheta = (L_plus - L_minus) / (2.0 * dtheta)
        return float(force * dL_dtheta)

    @staticmethod
    def get_inertia(params: BucketLeverParams) -> float:
        """Возвращает момент инерции ковша относительно оси D."""
        return params.inertia

    @staticmethod
    def angular_velocity(
        theta_prev: float,
        theta_curr: float,
        dt: float,
    ) -> float:
        """
        Оценка угловой скорости через конечную разность.
        Угол нормализуется в интервал (-π, π] перед дифференцированием.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        dtheta = theta_curr - theta_prev
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        return dtheta / dt

    # ------------------------------------------------------------------
    # Вспомогательная функция – формирует словарь всех точек
    # ------------------------------------------------------------------
    @staticmethod
    def _build_points(
        params: BucketLeverParams,
        theta: float,
        P: Vec2,
    ) -> Dict[str, Vec2]:
        """Собирает словарь всех характерных точек для данного угла."""
        E = params.D + params.L_bucket * np.array(
            [np.cos(theta), np.sin(theta)]
        )
        # вращаем com_local из СК ковша в глобальную СК
        com = params.D + np.array(
            [
                np.cos(theta) * params.com_local[0]
                - np.sin(theta) * params.com_local[1],
                np.sin(theta) * params.com_local[0]
                + np.cos(theta) * params.com_local[1],
            ]
        )
        return {
            "A": params.A.copy(),
            "C": params.C.copy(),
            "D": params.D.copy(),
            "E": E,
            "P": P.copy(),
            "com": com,
        }
