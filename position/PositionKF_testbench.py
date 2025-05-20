import numpy as np
import matplotlib.pyplot as plt
from PositionKF import PositionKF  # Assumes the PositionKF class is in a separate file

# Simulation parameters
dt = 0.05  # Time step (seconds)
t_max = 10.0  # Total simulation time (seconds)
n_steps = int(t_max / dt) + 1
t = np.linspace(0, t_max, n_steps)

# True trajectory (linear motion with constant velocity)
v_true = np.array([1.0, 0.5, 0.2])  # Constant velocity [vx, vy, vz]
p0_true = np.array([0.0, 0.0, 0.0])  # Initial position
p_true = p0_true + v_true * t.reshape(-1, 1)  # True position over time

# Generate noisy measurements
measurement_noise_std = 0.05  # Standard deviation of measurement noise
noise = np.random.normal(0, measurement_noise_std, (n_steps, 3))
z_meas = p_true + noise  # Noisy measurements

# Initialize Kalman Filter
p_init = z_meas[0]  # Use first measurement as initial position
v_init = np.array([0.0, 0.0, 0.0])  # Initial velocity guess
P_init = np.eye(6) * 0.1  # Initial covariance
Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])  # Process noise (velocity)
R = np.eye(3) * (measurement_noise_std ** 2*10)  # Measurement noise covariance
kf = PositionKF(p_init, v_init, P_init, Q, R, dt)

# Run Kalman Filter
p_est = np.zeros((n_steps, 3))  # Estimated positions
v_est = np.zeros((n_steps, 3))  # Estimated velocities
p_est[0] = p_init
v_est[0] = v_init

for i in range(1, n_steps):
    p_est[i], v_est[i], _ = kf.step(z_meas[i])

# Plot results
plt.figure(figsize=(12, 8))

# X-coordinate
plt.subplot(3, 1, 1)
plt.plot(t, p_true[:, 0], 'b-', label='True Position (X)')
plt.plot(t, z_meas[:, 0], 'r.', alpha=0.5, label='Noisy Measurements (X)')
plt.plot(t, p_est[:, 0], 'g-', label='Filtered Estimate (X)')
plt.ylabel('X Position (m)')
plt.legend()
plt.grid(True)

# Y-coordinate
plt.subplot(3, 1, 2)
plt.plot(t, p_true[:, 1], 'b-', label='True Position (Y)')
plt.plot(t, z_meas[:, 1], 'r.', alpha=0.5, label='Noisy Measurements (Y)')
plt.plot(t, p_est[:, 1], 'g-', label='Filtered Estimate (Y)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)

# Z-coordinate
plt.subplot(3, 1, 3)
plt.plot(t, p_true[:, 2], 'b-', label='True Position (Z)')
plt.plot(t, z_meas[:, 2], 'r.', alpha=0.5, label='Noisy Measurements (Z)')
plt.plot(t, p_est[:, 2], 'g-', label='Filtered Estimate (Z)')
plt.xlabel('Time (s)')
plt.ylabel('Z Position (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('position_kf_results.png')