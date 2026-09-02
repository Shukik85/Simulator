"""Full multibody dynamics for 3-DOF planar serial chain (boom-arm-bucket).

Equation of motion:  M(theta) * theta_ddot + C(theta, theta_dot) * theta_dot + G(theta) = tau
where tau = J^T * F_cyl (cylinder forces mapped to joint torques).
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from hydrosim_v2.config import ExcavatorMechanicsConfig
from hydrosim_v2.kinematics.forward import forward_kinematics


def extract_joint_angles(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
    prev_bucket_theta: float | None = None,
) -> Tuple[float, float, float]:
    """Extract joint angles (theta_boom, theta_arm, theta_bucket) from cylinder lengths.

    Returns angles in radians. theta_arm and theta_bucket are RELATIVE to the
    preceding link (not absolute).
    """
    pts = forward_kinematics(cfg, cyl_lengths, prev_bucket_theta=prev_bucket_theta)

    boom_dx = pts["boom_tip"][0] - pts["boom_joint"][0]
    boom_dy = pts["boom_tip"][1] - pts["boom_joint"][1]
    theta_boom = float(np.arctan2(boom_dy, boom_dx))

    arm_dx = pts["arm_tip"][0] - pts["arm_joint"][0]
    arm_dy = pts["arm_tip"][1] - pts["arm_joint"][1]
    theta_arm_abs = float(np.arctan2(arm_dy, arm_dx))
    theta_arm = theta_arm_abs - theta_boom

    theta_bucket = float(pts["bucket_theta_rad"])

    return theta_boom, theta_arm, theta_bucket


class MultibodyDynamics:
    """Compute M(theta), C(theta,theta_dot), G(theta) for a 3-link planar serial chain.

    Coordinate convention:
      theta[0] = theta_boom  (absolute angle of boom from horizontal)
      theta[1] = theta_arm   (RELATIVE angle of arm w.r.t. boom)
      theta[2] = theta_bucket (RELATIVE angle of bucket w.r.t. arm)

    Link geometry (from config):
      Link i: mass m_i, length L_i, COM at local (com_x_i, com_y_i),
              inertia I_i about COM.
    """

    def __init__(self, cfg: ExcavatorMechanicsConfig) -> None:
        self.cfg = cfg
        bl = cfg.boom_link
        al = cfg.arm_link
        bk = cfg.bucket_link

        self.m1 = bl.mass_kg
        self.m2 = al.mass_kg
        self.m3 = bk.mass_kg

        self.L1 = bl.length_m
        self.L2 = al.length_m
        self.L3 = bk.length_m

        self.r1x = bl.com_local[0]
        self.r1y = bl.com_local[1]
        self.r2x = al.com_local[0]
        self.r2y = al.com_local[1]
        self.r3x = bk.com_local[0]
        self.r3y = bk.com_local[1]

        self.r1 = float(np.hypot(self.r1x, self.r1y))
        self.r2 = float(np.hypot(self.r2x, self.r2y))
        self.r3 = float(np.hypot(self.r3x, self.r3y))

        self.I1 = bl.inertia_kgm2
        self.I2 = al.inertia_kgm2
        self.I3 = bk.inertia_kgm2

    def mass_matrix(self, theta_arm: float, theta_bucket: float) -> np.ndarray:
        """3x3 symmetric positive-definite mass matrix M(theta).

        Only depends on relative angles theta_arm and theta_bucket (not theta_boom).
        """
        m1, m2, m3 = self.m1, self.m2, self.m3
        L1, L2 = self.L1, self.L2
        r1, r2, r3 = self.r1, self.r2, self.r3
        I1, I2, I3 = self.I1, self.I2, self.I3

        c2 = np.cos(theta_arm)
        c3 = np.cos(theta_bucket)
        c23 = np.cos(theta_arm + theta_bucket)

        M = np.zeros((3, 3))

        M[0, 0] = (
            m1 * r1**2 + I1
            + m2 * (L1**2 + r2**2 + 2.0 * L1 * r2 * c2) + I2
            + m3 * (L1**2 + L2**2 + r3**2
                    + 2.0 * L1 * L2 * c2
                    + 2.0 * L1 * r3 * c23
                    + 2.0 * L2 * r3 * c3) + I3
        )

        M[0, 1] = (
            m2 * (r2**2 + L1 * r2 * c2) + I2
            + m3 * (L2**2 + r3**2
                    + L1 * L2 * c2
                    + L1 * r3 * c23
                    + 2.0 * L2 * r3 * c3) + I3
        )

        M[0, 2] = m3 * (r3**2 + L1 * r3 * c23 + L2 * r3 * c3) + I3

        M[1, 0] = M[0, 1]
        M[1, 1] = m2 * r2**2 + I2 + m3 * (L2**2 + r3**2 + 2.0 * L2 * r3 * c3) + I3
        M[1, 2] = m3 * (r3**2 + L2 * r3 * c3) + I3

        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = m3 * r3**2 + I3

        return M

    def gravity_vector(
        self, theta_boom: float, theta_arm: float, theta_bucket: float
    ) -> np.ndarray:
        """3x1 gravity vector G(theta) = dU/d(theta).

        G[i] is the torque at joint i due to gravity (positive = CCW).
        """
        m1, m2, m3 = self.m1, self.m2, self.m3
        L1, L2 = self.L1, self.L2
        r1, r2, r3 = self.r1, self.r2, self.r3
        g = 9.81

        phi1 = theta_boom
        phi2 = theta_boom + theta_arm
        phi3 = theta_boom + theta_arm + theta_bucket

        G = np.zeros(3)
        G[0] = g * (
            m1 * r1 * np.cos(phi1)
            + m2 * (L1 * np.cos(phi1) + r2 * np.cos(phi2))
            + m3 * (L1 * np.cos(phi1) + L2 * np.cos(phi2) + r3 * np.cos(phi3))
        )
        G[1] = g * (
            m2 * r2 * np.cos(phi2)
            + m3 * (L2 * np.cos(phi2) + r3 * np.cos(phi3))
        )
        G[2] = g * m3 * r3 * np.cos(phi3)

        return G

    def coriolis_vector(
        self, theta_arm: float, theta_bucket: float, theta_dot: np.ndarray
    ) -> np.ndarray:
        """3x1 Coriolis/centrifugal vector C(theta, theta_dot).

        Uses Christoffel symbols of the first kind.
        """
        m2, m3 = self.m2, self.m3
        L1, L2 = self.L1, self.L2
        r2, r3 = self.r2, self.r3

        th1d, th2d, th3d = float(theta_dot[0]), float(theta_dot[1]), float(theta_dot[2])

        s2 = np.sin(theta_arm)
        s3 = np.sin(theta_bucket)
        s23 = np.sin(theta_arm + theta_bucket)

        h12 = -m2 * L1 * r2 * s2 - m3 * L1 * L2 * s2 - m3 * L1 * r3 * s23
        h13 = -m3 * L1 * r3 * s23 - m3 * L2 * r3 * s3
        h23 = -m3 * L2 * r3 * s3

        C = np.zeros(3)
        C[0] = h12 * th2d + h13 * th3d
        C[1] = h12 * th1d + h23 * th3d
        C[2] = h13 * th1d + h23 * th2d

        return C

    def jacobian(
        self, cyl_lengths: Dict[str, float], delta: float = 1e-7
    ) -> np.ndarray:
        """3x3 Jacobian J = d(L_cyl)/d(theta).

        J[i,j] = d(L_cyl_i)/d(theta_j).

        Maps joint velocities to cylinder velocities: L_dot = J * theta_dot.
        Maps joint torques to cylinder forces: F_cyl = J^{-T} * tau
        (equivalently: tau = J^T * F_cyl).
        """
        cyl_names = ["boom_cyl", "arm_cyl", "bucket_cyl"]
        J = np.zeros((3, 3))

        theta0 = extract_joint_angles(self.cfg, cyl_lengths)

        for j in range(3):
            theta_plus = list(theta0)
            theta_minus = list(theta0)
            theta_plus[j] += delta
            theta_minus[j] -= delta

            L_plus = self._angles_to_cyl(*theta_plus)
            L_minus = self._angles_to_cyl(*theta_minus)

            for i, name in enumerate(cyl_names):
                J[i, j] = (L_plus[name] - L_minus[name]) / (2.0 * delta)

        return J

    def _angles_to_cyl(
        self, theta_boom: float, theta_arm: float, theta_bucket: float
    ) -> Dict[str, float]:
        """Convert joint angles to cylinder lengths (inverse kinematics).

        Uses the triangle geometry of each cylinder attachment.
        """
        cfg = self.cfg
        R_boom = _rot2d(theta_boom)
        R_arm = _rot2d(theta_boom + theta_arm)

        A_boom = np.array(cfg.boom_cyl.base_mount.point_local, dtype=float)
        P_boom_local = np.array(cfg.boom_cyl.rod_mount.point_local, dtype=float)

        P_boom_global = R_boom @ P_boom_local
        L_boom_cyl = float(np.linalg.norm(P_boom_global - A_boom))

        A_arm_global = R_boom @ np.array(cfg.arm_cyl.base_mount.point_local, dtype=float)
        boom_tip = R_boom @ np.array([cfg.boom_link.length_m, 0.0])
        A_prime = A_arm_global - boom_tip
        P_arm_local = np.array(cfg.arm_cyl.rod_mount.point_local, dtype=float)
        P_arm_global = R_arm @ P_arm_local
        L_arm_cyl = float(np.linalg.norm(P_arm_global - A_prime))

        from hydrosim_v2.kinematics.bucket_lever import bucket_cylinder_length
        L_bk_cyl = bucket_cylinder_length(
            cfg.bucket_lever,
            theta_bucket,
            theta_boom + theta_arm,
            _vec2(float(boom_tip[0]), float(boom_tip[1])),
        )

        return {"boom_cyl": L_boom_cyl, "arm_cyl": L_arm_cyl, "bucket_cyl": L_bk_cyl}

    def cylinder_forces_to_joint_torques(
        self, J: np.ndarray, F_cyl: np.ndarray
    ) -> np.ndarray:
        """Map cylinder forces to joint torques: tau = J^T * F_cyl."""
        return J.T @ F_cyl

    def joint_cylinder_accelerations(
        self, J: np.ndarray, theta_ddot: np.ndarray,
        theta_dot: np.ndarray, cyl_lengths: Dict[str, float],
        delta: float = 1e-5,
    ) -> Dict[str, float]:
        """Compute cylinder accelerations from joint accelerations.

        L_ddot = J * theta_ddot + dJ/dt * theta_dot
        dJ/dt is approximated numerically.
        """
        cyl_names = ["boom_cyl", "arm_cyl", "bucket_cyl"]
        theta0 = extract_joint_angles(self.cfg, cyl_lengths)

        dJdt_theta = np.zeros(3)
        for j in range(3):
            theta_p = list(theta0)
            theta_m = list(theta0)
            theta_p[j] += delta * float(theta_dot[j])
            theta_m[j] -= delta * float(theta_dot[j])

            J_p = self.jacobian(self._cyl_from_angles(*theta_p))
            J_m = self.jacobian(self._cyl_from_angles(*theta_m))

            dJdt_theta += (J_p - J_m) @ theta_dot / (2.0 * delta)

        L_ddot = J @ theta_ddot + dJdt_theta
        return {name: float(L_ddot[i]) for i, name in enumerate(cyl_names)}

    def _cyl_from_angles(self, t1: float, t2: float, t3: float) -> Dict[str, float]:
        return self._angles_to_cyl(t1, t2, t3)


def _rot2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _vec2(x: float, y: float) -> tuple[float, float]:
    return (float(x), float(y))
