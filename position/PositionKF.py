import numpy as np

class PositionKF:
    def __init__(self, p_init, v_init, P_init, Q, R, dt=0.1):
        """
        Linear Kalman Filter for 3D position tracking with random walk velocity model.
        
        Parameters:
        p_init: initial position [x, y, z] (3x1)
        v_init: initial velocity [vx, vy, vz] (3x1)
        P_init: initial 6x6 covariance matrix for [position, velocity]
        Q: process noise covariance [6x6] (discrete-time, affects velocity)
        R: measurement noise covariance [3x3] (position measurements)
        dt: time step for prediction (seconds)
        """
        if len(p_init) != 3 or len(v_init) != 3:
            raise ValueError("Invalid dimensions for p_init or v_init")
        if P_init.shape != (6, 6) or Q.shape != (6, 6) or R.shape != (3, 3):
            raise ValueError("Invalid dimensions for P_init, Q, or R")

        self.x = np.concatenate([p_init, v_init])  # State: [x, y, z, vx, vy, vz]
        self.P = np.array(P_init)  # State covariance
        self.Q = np.array(Q)       # Process noise covariance
        self.R = np.array(R)       # Measurement noise covariance
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt  # Position += velocity * dt

        # Measurement matrix (observe position only)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)

    def predict(self):
        """Predict step: Propagate state and covariance"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry

    def update(self, z):
        """Update step with position measurement [x, y, z]"""
        z = np.array(z)
        if len(z) != 3:
            raise ValueError("Invalid dimension for measurement")

        # Measurement residual
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry

    def step(self, z):
        """Full predict + update step"""
        self.predict()
        self.update(z)
        return self.get_state()

    def get_state(self):
        """Return current state estimate: position, velocity, covariance"""
        return self.x[:3].copy(), self.x[3:].copy(), self.P.copy()