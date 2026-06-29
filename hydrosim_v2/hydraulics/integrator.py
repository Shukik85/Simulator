from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    s: np.ndarray,
    dt: float,
    *args,
    **kwargs,
) -> np.ndarray:
    """Single RK4 step.

    f(t, s, *args, **kwargs) → ds/dt
    """
    k1 = f(t, s, *args, **kwargs)
    k2 = f(t + 0.5 * dt, s + 0.5 * dt * k1, *args, **kwargs)
    k3 = f(t + 0.5 * dt, s + 0.5 * dt * k2, *args, **kwargs)
    k4 = f(t + dt, s + dt * k3, *args, **kwargs)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    s0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ODE using fixed-step RK4.

    Args:
        f: RHS function f(t, s) → ds/dt.
        s0: Initial state.
        t_span: (t_start, t_end).
        dt: Time step.
        *args, **kwargs: Extra arguments to f.

    Returns:
        (times, states) where times is shape (N,) and states is (N, len(s0)).
    """
    t0, tf = t_span
    N = max(2, int(np.ceil((tf - t0) / dt)) + 1)
    dt_actual = (tf - t0) / (N - 1)

    ts = np.linspace(t0, tf, N)
    ss = np.zeros((N, len(s0)))
    ss[0] = s0

    s = s0.copy()
    for i in range(1, N):
        s = rk4_step(f, ts[i - 1], s, dt_actual, *args, **kwargs)
        ss[i] = s

    return ts, ss
