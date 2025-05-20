import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mekf import *
import os


# -------------------------------
# Helper Functions
# -------------------------------

def normalize_quat(q):
    """Ensure quaternion has unit norm"""
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise ValueError("Quaternion norm is near zero")
    return q / norm


def add_noise_to_quaternion(q, noise_level=0.05, outlier_prob=0.05, outlier_scale=5.0):
    """Add small rotational noise to a quaternion, with occasional outliers"""
    if np.random.rand() < outlier_prob:
        noise_level *= outlier_scale
    angle_noise = np.random.normal(0, noise_level)
    axis_noise = np.random.normal(0, 1, 3)
    axis_noise /= np.linalg.norm(axis_noise)
    noise_rot = R.from_rotvec(angle_noise * axis_noise)
    q_noisy = (noise_rot * R.from_quat(q)).as_quat()
    return normalize_quat(q_noisy)


def quaternion_distance(q1, q2):
    """Compute angular distance (in degrees) between two quaternions"""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    relative = r1.inv() * r2
    return np.degrees(np.abs(relative.magnitude()))


def estimate_initial_omega(measurements, dt, num_samples=5):
    """Estimate initial angular velocity with weighted average"""
    omegas = []
    weights = np.linspace(0.5, 1.0, min(num_samples, len(measurements) - 1))
    weights /= np.sum(weights)
    for i in range(min(num_samples, len(measurements) - 1)):
        r1 = R.from_quat(measurements[i])
        r2 = R.from_quat(measurements[i + 1])
        relative = r1.inv() * r2
        omega = relative.as_rotvec() / dt
        omegas.append(omega)
    return np.average(omegas, axis=0, weights=weights) if omegas else np.zeros(3)


def generate_trajectory(num_steps, dt):
    """Generate a synthetic trajectory with fading angular velocity"""
    times = np.arange(num_steps) * dt
    quaternions = []
    angular_velocities = []
    q = np.array([1.0, 0.0, 0.0, 0.0])
    lambda_decay = 0.05  # Decay rate (e^(-lambda * 100) ≈ 0.0067)

    for t in times:
        omega_base = np.array([
            0.5 * np.sin(0.2 * t),
            0.3 * np.cos(0.15 * t + 0.5),
            0.2 * np.sin(0.3 * t + 1.0)
        ])
        omega = omega_base * np.exp(-lambda_decay * t)
        w_dt = omega * dt
        theta = np.linalg.norm(w_dt)
        if theta < 1e-6:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            v_unit = w_dt / theta
            dq = np.concatenate([[np.cos(theta)], v_unit * np.sin(theta)])
        q = normalize_quat(np.dot(np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]]
        ]), dq))

        quaternions.append(q.copy())
        angular_velocities.append(omega.copy())

    return np.array(quaternions), np.array(angular_velocities), times


# -------------------------------
# Convergence Test with Fading Angular Velocity
# -------------------------------

def run_convergence_test(num_steps=10000, dt=0.01, noise_level=0.01):
    # Simulation parameters
    outlier_prob = 0.05
    outlier_scale = 5.0
    window_size = 100  # For running mean

    # Create output directory
    output_dir = "mekf_convergence_fading_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate ground truth trajectory
    ground_truth_quat, ground_truth_omega, times = generate_trajectory(num_steps, dt)
    noisy_signal = np.array([
        add_noise_to_quaternion(q, noise_level=noise_level,
                                outlier_prob=outlier_prob, outlier_scale=outlier_scale)
        for q in ground_truth_quat
    ])

    # Initialize filter
    r_init = noisy_signal[0].copy()
    omega_init = estimate_initial_omega(noisy_signal, dt, num_samples=5)
    P_init = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0])
    Q = np.diag([1e-2, 1e-2, 1e-2, 1e0, 1e0, 1e0])
    R = np.eye(3) * (noise_level ** 2)
    filt = MEKF(r_init, omega_init, P_init, Q, R, outlier_threshold=7.81)

    # Run filter
    filter_quat = []
    filter_omega = []
    filter_cov = []
    mahalanobis_distances = []
    p_theta_omega = []
    kalman_gain_norms = []

    for q_meas in noisy_signal:
        r_filt, omega_filt, P_filt = filt.step(dt, q_meas)
        filter_quat.append(r_filt.copy())
        filter_omega.append(omega_filt.copy())
        filter_cov.append(np.trace(P_filt))
        p_theta_omega.append(np.linalg.norm(P_filt[3:6, 0:3]))

        r_inv = quat_inverse(r_filt)
        delta_r = quat_mult(q_meas, r_inv)
        delta_theta = 2 * delta_r[1:]
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = H @ P_filt @ H.T + R
        mahalanobis = delta_theta.T @ np.linalg.inv(S) @ delta_theta
        mahalanobis_distances.append(mahalanobis)

        K = P_filt @ H.T @ np.linalg.inv(S)
        kalman_gain_norms.append(np.linalg.norm(K[3:6, :]))

    filter_quat = np.array(filter_quat)
    filter_omega = np.array(filter_omega)
    filter_cov = np.array(filter_cov)
    mahalanobis_distances = np.array(mahalanobis_distances)
    p_theta_omega = np.array(p_theta_omega)
    kalman_gain_norms = np.array(kalman_gain_norms)

    # Compute errors
    noisy_error = np.array([quaternion_distance(gt, nq) for gt, nq in zip(ground_truth_quat, noisy_signal)])
    filter_error = np.array([quaternion_distance(gt, fq) for gt, fq in zip(ground_truth_quat, filter_quat)])
    omega_error = np.linalg.norm(ground_truth_omega - filter_omega, axis=1)
    per_axis_errors = ground_truth_omega - filter_omega

    # Compute running mean errors
    running_means = np.zeros_like(per_axis_errors)
    for i in range(num_steps):
        start = max(0, i - window_size + 1)
        running_means[i] = np.mean(per_axis_errors[start:i + 1], axis=0)

    # Compute error slopes (last 50% of data)
    start_idx = num_steps // 2
    t_fit = times[start_idx:]
    slopes = []
    for i in range(3):
        coeffs = np.polyfit(t_fit, running_means[start_idx:, i], 1)
        slopes.append(coeffs[0])  # Slope in rad/s per second

    # Compute correlations over time windows
    window_steps = 1000
    num_windows = num_steps // window_steps
    correlations = np.zeros((num_windows, 3))
    window_times = []
    for w in range(num_windows):
        start = w * window_steps
        end = min((w + 1) * window_steps, num_steps)
        window_times.append(times[start])
        for i in range(3):
            corr = np.corrcoef(
                ground_truth_omega[start:end, i],
                filter_omega[start:end, i]
            )[0, 1]
            correlations[w, i] = corr if not np.isnan(corr) else 0.0

    # Log results
    log_file = open(f"{output_dir}/convergence_fading_log.txt", "w")
    log_file.write("MEKF Convergence Test with Fading Angular Velocity\n")
    log_file.write(f"Duration: {num_steps * dt:.1f} seconds\n")
    log_file.write(f"Mean noisy error: {np.mean(noisy_error):.2f} deg\n")
    log_file.write(f"Mean filter error: {np.mean(filter_error):.2f} deg\n")
    log_file.write(f"Max filter error: {np.max(filter_error):.2f} deg\n")
    log_file.write(f"Mean omega error: {np.mean(omega_error):.2f} rad/s\n")
    log_file.write(f"Max omega error: {np.max(omega_error):.2f} rad/s\n")
    log_file.write(
        f"Final per-axis mean errors (wx, wy, wz): {[np.mean(np.abs(per_axis_errors[-window_steps:, i])) for i in range(3)]}\n")
    log_file.write(
        f"Final per-axis max errors (wx, wy, wz): {[np.max(np.abs(per_axis_errors[-window_steps:, i])) for i in range(3)]}\n")
    log_file.write(f"Per-axis error slopes (wx, wy, wz): {slopes}\n")
    log_file.write(f"Final correlations (wx, wy, wz): {correlations[-1]}\n")
    log_file.write(f"Outliers detected: {np.sum(mahalanobis_distances > 7.81)}\n")
    log_file.write(f"Mean P_theta_omega norm: {np.mean(p_theta_omega):.2e}\n")
    log_file.write(f"Mean Kalman gain norm for omega: {np.mean(kalman_gain_norms):.2e}\n")
    log_file.close()

    # Plot results
    plt.figure(figsize=(15, 18))

    plt.subplot(5, 2, 1)
    plt.plot(times, noisy_error, label='Noisy Error', alpha=0.5)
    plt.plot(times, filter_error, label='Filter Error', linewidth=2)
    plt.ylabel('Angular Error (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Angular Error (Noise = {noise_level:.2f} rad)')

    plt.subplot(5, 2, 2)
    plt.plot(times, omega_error, label='Omega Error', color='purple')
    plt.ylabel('Angular Velocity Error (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Angular Velocity Error')

    plt.subplot(5, 2, 3)
    for i, label in enumerate(['wx', 'wy', 'wz']):
        plt.plot(times, per_axis_errors[:, i], label=f'Error {label}', alpha=0.5)
        plt.plot(times, running_means[:, i], label=f'Running Mean {label}', linewidth=2)
    plt.ylabel('Per-Axis Omega Error (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Per-Axis Omega Errors and Running Means')

    plt.subplot(5, 2, 4)
    for i, label in enumerate(['wx', 'wy', 'wz']):
        plt.plot(times, ground_truth_omega[:, i], label=f'True {label}', alpha=0.5)
        plt.plot(times, filter_omega[:, i], label=f'Est. {label}', linestyle='--')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Angular Velocity Components (Final Corr: {correlations[-1]})')

    plt.subplot(5, 2, 5)
    plt.plot(times, filter_cov, label='Covariance Trace', color='green')
    plt.ylabel('Trace of Covariance')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Covariance Trace')

    plt.subplot(5, 2, 6)
    plt.plot(times, np.abs(np.linalg.norm(filter_quat, axis=1) - 1.0), label='Norm Deviation', color='red')
    plt.ylabel('Quaternion Norm Deviation')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Quaternion Norm Deviation')

    plt.subplot(5, 2, 7)
    plt.plot(times, mahalanobis_distances, label='Mahalanobis Distance')
    plt.axhline(y=7.81, color='r', linestyle='--', label='Outlier Threshold')
    plt.ylabel('Mahalanobis Distance')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Measurement Residuals')

    plt.subplot(5, 2, 8)
    plt.plot(times, kalman_gain_norms, label='Kalman Gain Norm (Omega)', color='orange')
    plt.ylabel('Kalman Gain Norm')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Kalman Gain Norm for Omega')

    plt.subplot(5, 2, 9)
    for i, label in enumerate(['wx', 'wy', 'wz']):
        plt.plot(window_times, correlations[:, i], label=f'Corr {label}')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.title('Omega Correlations Over Time')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence_fading_noise_{noise_level:.2f}.png")
    plt.close()

    np.savetxt(f"{output_dir}/convergence_fading_data_noise_{noise_level:.2f}.txt",
               np.column_stack((times, noisy_error, filter_error, omega_error,
                                per_axis_errors[:, 0], per_axis_errors[:, 1], per_axis_errors[:, 2],
                                filter_cov, p_theta_omega, kalman_gain_norms)),
               header="Time Noisy_Error_deg Filter_Error_deg Omega_Error_rad/s Omega_Error_wx Omega_Error_wy Omega_Error_wz Cov_Trace P_theta_omega_norm Kalman_Gain_Norm",
               fmt="%.6f")


if __name__ == "__main__":
    run_convergence_test(num_steps=10000, dt=0.01, noise_level=0.05)