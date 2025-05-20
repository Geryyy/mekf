import numpy as np
from scipy.spatial.transform import Rotation as R


# ========== Quaternion Utilities ==========

def normalize_quat(q):
    """Ensure quaternion has unit norm"""
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise ValueError("Quaternion norm is near zero")
    return q / norm


def quat_inverse(q):
    """Return the inverse (conjugate) of a unit quaternion"""
    q = normalize_quat(q)
    q_inv = q.copy()
    q_inv[1:] *= -1
    return q_inv


def quat_left_matrix(q):
    """Left quaternion multiplication matrix: q ⊗ p = L(q) @ p"""
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w]
    ])


def quat_mult(q1, q2):
    """Quaternion multiplication q1 ⊗ q2 using matrix form"""
    return quat_left_matrix(q1) @ q2


def quat_exp(v):
    """Quaternion exponential for small angle vector v"""
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v_unit = v / theta
    return np.concatenate([[np.cos(theta)], v_unit * np.sin(theta)])


def skew(v):
    """Skew-symmetric matrix of a 3D vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def omega_matrix(omega):
    """Omega matrix for quaternion propagation"""
    wx, wy, wz = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])


class MEKF:
    def __init__(self, r_init, omega_init, P_init, Q, R, outlier_threshold=7.81):
        """
        r_init: initial quaternion [w, x, y, z]
        omega_init: initial angular velocity [3x1]
        P_init: initial 6x6 covariance matrix
        Q: process noise covariance [6x6] (continuous-time)
        R: quaternion measurement noise covariance [3x3]
        outlier_threshold: Chi-squared threshold for outlier rejection (3 DoF, 95% confidence)
        """
        if len(r_init) != 4 or len(omega_init) != 3:
            raise ValueError("Invalid dimensions for r_init or omega_init")
        if P_init.shape != (6, 6) or Q.shape != (6, 6) or R.shape != (3, 3):
            raise ValueError("Invalid dimensions for P_init, Q, or R")

        self.r = normalize_quat(np.array(r_init))
        self.omega = np.array(omega_init)
        self.P = np.array(P_init)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.outlier_threshold = outlier_threshold

    def propagate(self, dt):
        """Propagate state and covariance forward in time"""
        w_dt = self.omega * dt
        dq = quat_exp(0.5 * w_dt)
        self.r = normalize_quat(quat_mult(dq, self.r))

        F = np.zeros((6, 6))
        F[0:3, 0:3] = -skew(self.omega)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 3:6] = -0.01 * np.eye(3)  # Reduced damping for faster omega changes
        Phi = np.eye(6) + F * dt
        self.P = Phi @ self.P @ Phi.T + self.Q * dt
        self.P = (self.P + self.P.T) / 2

    def update(self, r_meas):
        """Update step with quaternion measurement"""
        r_meas = normalize_quat(np.array(r_meas))
        r_inv = quat_inverse(self.r)
        delta_r = quat_mult(r_meas, r_inv)
        delta_theta = 2 * delta_r[1:]

        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = H @ self.P @ H.T + self.R
        mahalanobis = delta_theta.T @ np.linalg.inv(S) @ delta_theta
        if mahalanobis > self.outlier_threshold:
            print(f"Outlier detected: Mahalanobis distance = {mahalanobis:.2f}")
            return

        K = self.P @ H.T @ np.linalg.inv(S)
        print(f"Kalman gain for omega: {K[3:6, :]}")
        delta_x = K @ delta_theta
        delta_theta = delta_x[:3]
        delta_omega = delta_x[3:]

        dq = normalize_quat(np.concatenate([[1.0], 0.5 * delta_theta]))
        self.r = normalize_quat(quat_mult(dq, self.r))
        self.omega += delta_omega

        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2

    def step(self, dt, q_meas):
        """Full propagation + update step"""
        self.propagate(dt)
        self.update(q_meas)
        return self.get_state()

    def get_state(self):
        """Return current state estimate"""
        return self.r.copy(), self.omega.copy(), self.P.copy()