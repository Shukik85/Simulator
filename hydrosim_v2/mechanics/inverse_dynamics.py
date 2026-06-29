from __future__ import annotations

from typing import Dict

import numpy as np

from hydrosim_v2.config import ExcavatorMechanicsConfig, DEFAULT_MECHANICS_CONFIG
from hydrosim_v2.kinematics import forward_kinematics, solve_link_angle
from hydrosim_v2.core.types import Vec2

_EPS = 1e-12
_G = 9.81


def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rot2d_deriv(theta: float) -> np.ndarray:
    """Derivative of 2D rotation matrix wrt theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


def joint_angles_from_cylinders(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
) -> np.ndarray:
    """Compute joint angles [θ_boom, θ_arm, θ_bucket] from cylinder lengths."""
    # Boom
    A = np.array(cfg.boom_cyl.base_mount.point_local)
    P = np.array(cfg.boom_cyl.rod_mount.point_local)
    q1 = solve_link_angle(
        (float(A[0]), float(A[1])),
        (float(P[0]), float(P[1])),
        float(cyl_lengths["boom_cyl"]),
        sign=1,
    )

    # Arm
    A_arm = (boom_joint := np.zeros(2)) + _rot2d(q1) @ np.array(
        cfg.arm_cyl.base_mount.point_local
    )
    arm_joint = _rot2d(q1) @ np.array([cfg.boom_link.length_m, 0.0])
    A_prime = A_arm - arm_joint
    P_arm = np.array(cfg.arm_cyl.rod_mount.point_local)
    q2 = solve_link_angle(
        (float(A_prime[0]), float(A_prime[1])),
        (float(P_arm[0]), float(P_arm[1])),
        float(cyl_lengths["arm_cyl"]),
        sign=1,
    )

    # Bucket — use the lever solver
    from hydrosim_v2.kinematics.bucket_lever import solve_bucket
    sol = solve_bucket(
        cfg.bucket_lever,
        float(cyl_lengths["bucket_cyl"]),
        q2,
        (float(arm_joint[0]), float(arm_joint[1])),
    )
    q3 = sol.theta_rad if sol is not None else 0.0

    return np.array([q1, q2, q3])


def _com_positions(
    cfg: ExcavatorMechanicsConfig, q: np.ndarray
) -> list[np.ndarray]:
    """Global COM positions of the 3 links: [p1, p2, p3]."""
    l1 = cfg.boom_link.length_m
    l2 = cfg.arm_link.length_m

    R1 = _rot2d(q[0])
    R12 = _rot2d(q[0] + q[1])
    R123 = _rot2d(q[0] + q[1] + q[2])

    com1 = np.array(cfg.boom_link.com_local, dtype=float)
    com2 = np.array(cfg.arm_link.com_local, dtype=float)
    com3 = np.array(cfg.bucket_link.com_local, dtype=float)

    p1 = R1 @ com1

    p2 = R1 @ np.array([l1, 0.0]) + R12 @ com2

    p3 = R1 @ np.array([l1, 0.0]) + R12 @ np.array([l2, 0.0]) + R123 @ com3

    return [p1, p2, p3]


def joint_space_inertia_matrix(
    cfg: ExcavatorMechanicsConfig, q: np.ndarray
) -> np.ndarray:
    """3×3 joint-space inertia matrix H(q) for a planar 3-DOF arm."""
    l1 = cfg.boom_link.length_m
    l2 = cfg.arm_link.length_m
    m = [cfg.boom_link.mass_kg, cfg.arm_link.mass_kg, cfg.bucket_link.mass_kg]
    I = [cfg.boom_link.inertia_kgm2, cfg.arm_link.inertia_kgm2, cfg.bucket_link.inertia_kgm2]

    com1 = np.array(cfg.boom_link.com_local)
    com2 = np.array(cfg.arm_link.com_local)
    com3 = np.array(cfg.bucket_link.com_local)

    # Rotation matrices
    c1, s1 = np.cos(q[0]), np.sin(q[0])
    c12, s12 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
    c123, s123 = np.cos(q[0] + q[1] + q[2]), np.sin(q[0] + q[1] + q[2])

    # Translational Jacobians [2×3 for each link]
    # Column i = ∂p_k / ∂θ_i

    # Link 1
    Jv1 = np.zeros((2, 3))
    Jv1[:, 0] = _rot2d_deriv(q[0]) @ com1  # ∂p1/∂θ1

    # Link 2
    Jv2 = np.zeros((2, 3))
    Jv2[:, 0] = _rot2d_deriv(q[0]) @ np.array([l1, 0.0]) + _rot2d_deriv(q[0] + q[1]) @ com2
    Jv2[:, 1] = _rot2d_deriv(q[0] + q[1]) @ com2

    # Link 3
    Jv3 = np.zeros((2, 3))
    Jv3[:, 0] = (
        _rot2d_deriv(q[0]) @ np.array([l1, 0.0])
        + _rot2d_deriv(q[0] + q[1]) @ np.array([l2, 0.0])
        + _rot2d_deriv(q[0] + q[1] + q[2]) @ com3
    )
    Jv3[:, 1] = (
        _rot2d_deriv(q[0] + q[1]) @ np.array([l2, 0.0])
        + _rot2d_deriv(q[0] + q[1] + q[2]) @ com3
    )
    Jv3[:, 2] = _rot2d_deriv(q[0] + q[1] + q[2]) @ com3

    # Angular Jacobians (planar: scalar angular velocity)
    Jw = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)

    H = np.zeros((3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                H[i, j] += (
                    m[k] * np.dot(Jv_k := [Jv1, Jv2, Jv3][k][:, i],
                                  [Jv1, Jv2, Jv3][k][:, j])
                    + I[k] * Jw[k, i] * Jw[k, j]
                )
    return H


def gravity_torque(cfg: ExcavatorMechanicsConfig, q: np.ndarray) -> np.ndarray:
    """3×1 gravity torque vector g(q) for a planar 3-DOF arm."""
    m = [cfg.boom_link.mass_kg, cfg.arm_link.mass_kg, cfg.bucket_link.mass_kg]
    g = _G

    pe = 0.0  # total potential energy = Σ m_k * g * y_k
    # We'll compute τ_i = ∂PE/∂θ_i

    coms = _com_positions(cfg, q)

    tau = np.zeros(3)
    eps = 1e-8
    for i in range(3):
        q_plus = q.copy()
        q_plus[i] += eps
        coms_plus = _com_positions(cfg, q_plus)
        pe_plus = sum(m[k] * g * coms_plus[k][1] for k in range(3))
        q_minus = q.copy()
        q_minus[i] -= eps
        coms_minus = _com_positions(cfg, q_minus)
        pe_minus = sum(m[k] * g * coms_minus[k][1] for k in range(3))
        tau[i] = (pe_plus - pe_minus) / (2 * eps)

    return tau


def cylinder_jacobian(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
) -> np.ndarray:
    """3×3 cylinder Jacobian J_cyl: L̇_cyl = J_cyl @ q̇.

    Computed via finite differences of forward kinematics.
    """
    names = ["boom_cyl", "arm_cyl", "bucket_cyl"]
    eps = 1e-7
    J = np.zeros((3, 3))

    q = joint_angles_from_cylinders(cfg, cyl_lengths)

    for j in range(3):
        for i, name in enumerate(names):
            q_plus = q.copy()
            q_plus[j] += eps
            L_plus = _cylinder_lengths_from_angles(cfg, q_plus)

            q_minus = q.copy()
            q_minus[j] -= eps
            L_minus = _cylinder_lengths_from_angles(cfg, q_minus)

            J[i, j] = (L_plus[name] - L_minus[name]) / (2 * eps)

    return J


def _cylinder_lengths_from_angles(
    cfg: ExcavatorMechanicsConfig, q: np.ndarray
) -> Dict[str, float]:
    """Compute cylinder lengths from joint angles (inverse of forward)."""
    l1 = cfg.boom_link.length_m
    R1 = _rot2d(q[0])
    R12 = _rot2d(q[0] + q[1])
    arm_pivot = R1 @ np.array([l1, 0.0])

    # Boom cylinder: |A_base - R1 @ P_boom_local|
    A_boom = np.array(cfg.boom_cyl.base_mount.point_local)
    P_boom_global = R1 @ np.array(cfg.boom_cyl.rod_mount.point_local)
    L_boom = float(np.linalg.norm(P_boom_global - A_boom))

    # Arm cylinder
    A_arm_global = R1 @ np.array(cfg.arm_cyl.base_mount.point_local)
    P_arm_global = arm_pivot + R12 @ np.array(cfg.arm_cyl.rod_mount.point_local)
    L_arm = float(np.linalg.norm(P_arm_global - A_arm_global))

    # Bucket cylinder via 4-bar solver (forward: angle → length)
    from hydrosim_v2.kinematics.bucket_lever import bucket_cylinder_length

    L_bucket = bucket_cylinder_length(
        cfg.bucket_lever,
        theta_bucket=q[2],
        arm_angle_rad=q[1],
        arm_pivot_xy=(float(arm_pivot[0]), float(arm_pivot[1])),
    ) or 0.0

    return {"boom_cyl": L_boom, "arm_cyl": L_arm, "bucket_cyl": L_bucket}


def inverse_dynamics(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
    dcyl_lengths: Dict[str, float],
    ddcyl_lengths: Dict[str, float],
    ee_force: Vec2 = (0.0, 0.0),
) -> Dict[str, float]:
    """Compute required cylinder forces for a given trajectory point.

    The trajectory is specified in cylinder space (L, L̇, L̈), avoiding
    the need to invert the cylinder → joint mapping.

    Returns dict mapping cylinder name → force (N).
    """
    names = ["boom_cyl", "arm_cyl", "bucket_cyl"]
    n = 3
    q = joint_angles_from_cylinders(cfg, cyl_lengths)

    # Map cylinder velocities/accelerations to joint space
    J_cyl = cylinder_jacobian(cfg, cyl_lengths)
    J_cyl_inv = np.linalg.inv(J_cyl)
    dq = J_cyl_inv @ np.array([dcyl_lengths[n] for n in names])
    ddq = J_cyl_inv @ (
        np.array([ddcyl_lengths[n] for n in names])
        - _jacobian_dot_times_dq(cfg, cyl_lengths, J_cyl, dq)
    )

    # Inverse dynamics: H(q) @ ddq + C(q, dq) @ dq + g(q) = τ + J_ee^T @ F_ee
    H = joint_space_inertia_matrix(cfg, q)
    g = gravity_torque(cfg, q)

    # Neglect Coriolis for low-speed operation
    tau = H @ ddq + g

    # External force at bucket tip
    ee_jacobian = _endeffector_jacobian(cfg, q)
    tau -= ee_jacobian.T @ np.array(ee_force)

    # Cylinder forces from joint torques
    F_cyl = J_cyl_inv.T @ tau

    return {name: float(F_cyl[i]) for i, name in enumerate(names)}


def _jacobian_dot_times_dq(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
    J_cyl: np.ndarray,
    dq: np.ndarray,
) -> np.ndarray:
    """Approximate J̇_cyl @ dq via finite differences."""
    eps = 1e-7
    names = ["boom_cyl", "arm_cyl", "bucket_cyl"]

    q = joint_angles_from_cylinders(cfg, cyl_lengths)
    qd = q + eps * dq
    Ld = _cylinder_lengths_from_angles(cfg, qd)
    Jd = cylinder_jacobian(cfg, Ld)

    return (Jd - J_cyl) / eps @ dq


def _endeffector_jacobian(cfg: ExcavatorMechanicsConfig, q: np.ndarray) -> np.ndarray:
    """2×3 end-effector Jacobian J_ee: ẋ_ee = J_ee @ q̇."""
    l1 = cfg.boom_link.length_m
    l2 = cfg.arm_link.length_m
    from hydrosim_v2.kinematics.bucket_lever import solve_bucket

    R1 = _rot2d(q[0])
    R12 = _rot2d(q[0] + q[1])
    arm_pivot = R1 @ np.array([l1, 0.0])
    D = arm_pivot + R12 @ np.array(cfg.bucket_lever.pivot_D)

    R123 = _rot2d(q[0] + q[1] + q[2])
    ee = D + R123 @ np.array(cfg.bucket_lever.bucket_tip_local)

    J = np.zeros((2, 3))
    eps = 1e-8
    for i in range(3):
        qp = q.copy()
        qp[i] += eps
        ee_p = _ee_pos(cfg, qp)
        qm = q.copy()
        qm[i] -= eps
        ee_m = _ee_pos(cfg, qm)
        J[:, i] = (ee_p - ee_m) / (2 * eps)
    return J


def _ee_pos(cfg: ExcavatorMechanicsConfig, q: np.ndarray) -> np.ndarray:
    l1 = cfg.boom_link.length_m
    R1 = _rot2d(q[0])
    R12 = _rot2d(q[0] + q[1])
    R123 = _rot2d(q[0] + q[1] + q[2])
    arm_pivot = R1 @ np.array([l1, 0.0])
    D = arm_pivot + R12 @ np.array(cfg.bucket_lever.pivot_D)
    return D + R123 @ np.array(cfg.bucket_lever.bucket_tip_local)
