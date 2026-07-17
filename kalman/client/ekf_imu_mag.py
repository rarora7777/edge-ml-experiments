import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix (cross-product matrix: np.cross(v, x) = skew_symmetric(v) @ x) of a vector.
    # Eq (11)
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class MagGyroCalibrationEKF:
    """
    Extended Kalman Filter to calibrate magnetometer using calibrated gyroscope measurements.
    Based on Han et al (2017) https://doi.org/10.1109/JSEN.2016.2624821
    """

    def __init__(self, dt, h_p0, sigma_m=0.5, sigma_g=np.radians(0.1), phi=10):
        self.dt = dt
        self.phi = phi
        self.sigma_m = sigma_m
        self.sigma_g = sigma_g

        # State vector x = [h_cal(3), W11, W22, W33, W12, W13, W23, b(3)]^T
        # Initializing W to Identity, b to 0
        # Eq (14)
        self.x = np.zeros(12)
        self.x[0:3] = h_p0  # h_cal_k initial guess
        self.x[3:6] = 1.0  # W11, W22, W33
        self.x[6:9] = 0.0  # W12, W13, W23
        self.x[9:12] = 0.0  # b_x, b_y, b_z

        # Initial Covariance Matrix P0
        self.P = np.zeros((12, 12))
        self.P[0:3, 0:3] = 500 * np.eye(3)
        self.P[3:9, 3:9] = 1e-4 * np.eye(6)
        self.P[9:12, 9:12] = 500 * np.eye(3)

    def predict(self, gyro):
        """
        EKF Predict Step using the Simplified Gyroscope Model.
        # Eq (7) is implicitly used by treating 'gyro' as the true/calibrated rotation rate.
        """
        h_cal_prev = self.x[0:3]

        # Eq (15a) - State transition for h_cal
        omega_skew = skew_symmetric(gyro)
        h_cal_pred = (np.eye(3) + omega_skew * self.dt) @ h_cal_prev

        # Eq (15b) & Eq (15c) - W and b are constants in transition
        self.x[0:3] = h_cal_pred

        # Eq (18) - State Transition Jacobian F_{k-1}
        # Note: The paper writes the bottom right 9x9 as 0_{9x9}.
        # This is a typo in the paper; it must be Identity for constant states W and b.
        F = np.zeros((12, 12))
        F[0:3, 0:3] = np.eye(3) + omega_skew * self.dt
        F[3:12, 3:12] = np.eye(9)

        # Eq (17) - State Transition Covariance Matrix Q_{k-1}
        # Scaled dynamically based on the current magnitude of h_cal
        Q = np.zeros((12, 12))
        h_cal_mag_sq = np.dot(h_cal_prev, h_cal_prev)
        q_scalar = (self.phi**2) * (self.sigma_g**2) * h_cal_mag_sq * (self.dt**2)
        Q[0:3, 0:3] = q_scalar * np.eye(3)

        # Eq (21a) is implicitly handled by the self.x update above
        # Eq (21b) - Predict Covariance
        self.P = F @ self.P @ F.T + Q

    def update(self, mag):
        """
        EKF Update Step using the un-calibrated magnetometer measurements.
        """
        h_cal_pred = self.x[0:3]
        W11, W22, W33, W12, W13, W23 = self.x[3:9]
        b = self.x[9:12]

        # Construct symmetric matrix W
        # Eq (3) representation
        W_matrix = np.array([[W11, W12, W13], [W12, W22, W23], [W13, W23, W33]])

        # Observation Model h(x_k)
        # Eq (19)
        z_pred = W_matrix @ h_cal_pred + b

        # Observation Jacobian h_k (Labelled as Eq 9 structurally in IV.B)
        x1, x2, x3 = h_cal_pred
        h_Wk = np.array(
            [[x1, 0, 0, x2, x3, 0], [0, x2, 0, x1, 0, x3], [0, 0, x3, 0, x1, x2]]
        )

        H = np.hstack([W_matrix, h_Wk, np.eye(3)])

        # Measurement Covariance R_k
        # Eq (20)
        R = (self.sigma_m**2) * np.eye(3)

        # EKF Update Equations
        # Eq (22a) - Innovation
        y = mag - z_pred

        # Eq (22b) - Innovation Covariance
        S = H @ self.P @ H.T + R

        # Eq (22c) - Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Eq (22d) - Update State
        self.x = self.x + K @ y

        # Eq (22e) - Update Covariance
        # Note: The paper writes P_k|k = (1 - K*h)*P*Q. This is mathematically invalid
        # and non-standard. Applied the correct standard EKF covariance update below.
        self.P = (np.eye(12) - K @ H) @ self.P


def plot_calibrated_field(time_s, calibrated_fields, true_magnitude=None):
    """Plot the estimated calibrated magnetic-field vector and its magnitude."""
    calibrated_fields = np.asarray(calibrated_fields)
    magnitude = np.linalg.norm(calibrated_fields, axis=1)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for index, axis in enumerate("xyz"):
        plt.plot(time_s, calibrated_fields[:, index], label=f"h_cal {axis}")
    plt.title("Calibrated Magnetic Field")
    plt.xlabel("Time (s)")
    plt.ylabel("Field")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_s, magnitude, label="|h_cal|")
    if true_magnitude is not None:
        plt.axhline(true_magnitude, color="r", linestyle="--", label="True magnitude")
    plt.title("Calibrated Magnetic Field Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Field magnitude")
    plt.legend()
    plt.tight_layout()


def run_simulation(sigma_m=0.5, sigma_g_deg=0.1, phi=10):
    """Run simulation scenario A from the paper"""
    # Simulation Parameters (Scenario A from Table I)
    dt = 0.01  # 100 Hz sampling freq
    duration = 10.0  # 10 seconds simulation
    steps = int(duration / dt)

    # Ideal parameters to estimate
    true_W = np.array([[1.02, 0.02, -0.05], [0.02, 0.95, -0.03], [-0.05, -0.03, 1.03]])
    true_b = np.array([10.0, 5.0, -20.0])  # mG
    true_mag_magnitude = 50.0  # mG

    # EKF noise parameters.  Gyro measurements use rad/s internally.
    sigma_g = np.radians(sigma_g_deg)

    # Storage for plotting
    estimated_W33 = []
    estimated_bx = []
    calibrated_fields = []

    # Generate True Magnetic Field Sequence (Sine wave rotation)
    h_cal_true = np.array([true_mag_magnitude, 0.0, 0.0])

    # Initialize Filter
    # In practice, you use the first raw reading as the initial h_p0 guess
    initial_mag = true_W @ h_cal_true + true_b
    ekf = MagGyroCalibrationEKF(
        dt=dt, h_p0=initial_mag, sigma_m=sigma_m, sigma_g=sigma_g, phi=phi
    )

    print("Starting Simulation...")
    for step in range(steps):
        t = step * dt

        # Synthesize gyro rotation (rad/s)
        gyro = np.array([np.sin(t), np.cos(t), np.sin(2 * t)]) * 0.5

        # Step the true magnetic field (perfect integration for simulation truth)
        h_cal_true = (np.eye(3) + skew_symmetric(gyro) * dt) @ h_cal_true
        # Normalize to prevent Euler integration drift in the simulation data
        h_cal_true = h_cal_true / np.linalg.norm(h_cal_true) * true_mag_magnitude

        # Synthesize noisy sensor readings
        gyro_meas = gyro + np.random.normal(0, sigma_g, 3)
        mag_meas = true_W @ h_cal_true + true_b + np.random.normal(0, sigma_m, 3)

        # 1. EKF Predict
        ekf.predict(gyro_meas)
        # 2. EKF Update
        ekf.update(mag_meas)

        # Store metrics
        estimated_W33.append(ekf.x[5])  # W33 is index 5
        estimated_bx.append(ekf.x[9])  # b_x is index 9
        calibrated_fields.append(ekf.x[0:3].copy())

    print("Simulation Complete.")
    print(f"True W33: {true_W[2, 2]:.3f} | Estimated W33: {ekf.x[5]:.3f}")
    print(f"True b_x: {true_b[0]:.3f} | Estimated b_x: {ekf.x[9]:.3f}")

    print("All estimated params:")
    print(f"True W diagonal: {true_W[0, 0]} {true_W[1, 1]} {true_W[2, 2]} | Estimated W: {ekf.x[3:6]}")
    print(f"True W off-diag: {true_W[0, 1]} {true_W[0, 2]} {true_W[1, 2]} | Estimated W: {ekf.x[6:9]}")
    print(f"True b: {true_b} | Estimated W: {ekf.x[9:]}")

    # Plotting the convergence
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(steps) * dt, estimated_W33, label="Estimated W33")
    plt.axhline(true_W[2, 2], color="r", linestyle="--", label="True W33")
    plt.title("W33 Convergence")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(steps) * dt, estimated_bx, label="Estimated b_x")
    plt.axhline(true_b[0], color="r", linestyle="--", label="True b_x")
    plt.title("Bias X Convergence")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    plot_calibrated_field(
        np.arange(steps) * dt, calibrated_fields, true_magnitude=true_mag_magnitude
    )
    plt.show()


def load_gyro_bias(calibration_filename):
    """Load a gyro bias from an imu_calibration.json file, in degrees/s."""
    if calibration_filename is None:
        return np.zeros(3)

    try:
        with open(calibration_filename, encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        gyro_bias_dps = np.asarray(calibration["gyro_bias_dps"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Could not read gyro_bias_dps from {calibration_filename}: {error}"
        ) from error

    if gyro_bias_dps.shape != (3,):
        raise ValueError("gyro_bias_dps must contain exactly three values")
    return gyro_bias_dps


def load_samples(data_filename):
    """Read valid gyro samples and optional fresh mag samples from a recording."""
    required_columns = {
        "received_unix_s", "valid", "gyro_x", "gyro_y", "gyro_z",
    }
    rows = []
    samples = []
    try:
        with open(data_filename, newline="", encoding="utf-8") as data_file:
            reader = csv.DictReader(data_file)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError("CSV does not have the required IMU sample columns")
            fieldnames = reader.fieldnames
            for row_index, row in enumerate(reader):
                rows.append(row)
                if row["valid"].strip().lower() != "true":
                    continue
                mag = None
                mag_available = row.get("mag_available", "true").strip().lower() == "true"
                if mag_available:
                    try:
                        mag = np.array([float(row[name]) for name in ("mag_x", "mag_y", "mag_z")])
                    except (KeyError, TypeError, ValueError):
                        mag_available = False
                # Legacy CSV files lack mag_fresh; their available samples are
                # treated as fresh to preserve their previous behavior.
                mag_fresh = row.get("mag_fresh", str(mag_available)).strip().lower() == "true"
                samples.append((
                    row_index,
                    float(row["received_unix_s"]),
                    np.array([float(row[name]) for name in ("gyro_x", "gyro_y", "gyro_z")]),
                    mag,
                    mag_fresh and mag is not None,
                ))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read IMU samples from {data_filename}: {error}") from error

    if not samples or not any(sample[4] for sample in samples):
        raise ValueError("No valid samples with fresh magnetometer data were found")
    return fieldnames, rows, samples


def calibrated_output_filename(data_filename):
    input_path = Path(data_filename)
    return input_path.with_name(f"{input_path.stem}_magcal{input_path.suffix}")


def save_calibrated_samples(output_filename, fieldnames, rows, calibrated_fields):
    """Write input-format rows, substituting calibrated magnetic-field values."""
    for row_index, calibrated_field in calibrated_fields.items():
        rows[row_index]["mag_x"] = f"{calibrated_field[0]:.9g}"
        rows[row_index]["mag_y"] = f"{calibrated_field[1]:.9g}"
        rows[row_index]["mag_z"] = f"{calibrated_field[2]:.9g}"

    try:
        with open(output_filename, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise ValueError(f"Could not write calibrated samples to {output_filename}: {error}") from error


def run_recording(data_filename, calibration_filename, sigma_m=0.5, sigma_g_deg=0.1, phi=10):
    """Run the EKF over recorded gyro and magnetometer samples."""
    fieldnames, rows, samples = load_samples(data_filename)
    gyro_bias_dps = load_gyro_bias(calibration_filename)
    sigma_g = np.radians(sigma_g_deg)

    timestamps = np.array([sample[1] for sample in samples])
    positive_dts = np.diff(timestamps)
    positive_dts = positive_dts[positive_dts > 0]
    if len(positive_dts) == 0:
        raise ValueError("Sample timestamps must include at least one increasing pair")
    default_dt = float(np.median(positive_dts))

    first_mag_index = next(index for index, sample in enumerate(samples) if sample[4])
    ekf = MagGyroCalibrationEKF(
        dt=default_dt, h_p0=samples[first_mag_index][3], sigma_m=sigma_m,
        sigma_g=sigma_g, phi=phi,
    )
    estimated_W33 = []
    estimated_bx = []
    calibrated_fields = []
    elapsed_s = []
    calibrated_rows = {}
    previous_timestamp = samples[first_mag_index][1]

    print(f"Starting EKF on {len(samples) - first_mag_index} recorded samples...")
    for row_index, timestamp, gyro_dps, mag, mag_fresh in samples[first_mag_index:]:
        dt = timestamp - previous_timestamp
        if dt > 0:
            ekf.dt = dt
            gyro_rad_s = np.radians(gyro_dps - gyro_bias_dps)
            ekf.predict(gyro_rad_s)
        if mag_fresh:
            ekf.update(mag)
        previous_timestamp = timestamp
        elapsed_s.append(timestamp - samples[0][0])
        estimated_W33.append(ekf.x[5])
        estimated_bx.append(ekf.x[9])
        calibrated_fields.append(ekf.x[0:3].copy())
        if mag is not None:
            calibrated_rows[row_index] = ekf.x[0:3].copy()

    print("EKF Complete.")
    print(f"Estimated W diagonal: {ekf.x[3:6]}")
    print(f"Estimated W off-diagonal: {ekf.x[6:9]}")
    print(f"Estimated magnetometer bias: {ekf.x[9:]}")
    output_filename = calibrated_output_filename(data_filename)
    save_calibrated_samples(output_filename, fieldnames, rows, calibrated_rows)
    print(f"Calibrated samples saved to {output_filename}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(elapsed_s, estimated_W33, label="Estimated W33")
    plt.title("W33 Estimate")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(elapsed_s, estimated_bx, label="Estimated b_x")
    plt.title("Magnetometer Bias X Estimate")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plot_calibrated_field(elapsed_s, calibrated_fields)
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calibrate a magnetometer with gyro data using an EKF."
    )
    parser.add_argument(
        "data_filename", nargs="?", metavar="IMU_SAMPLES.csv",
        help="Recorded samples in imu_samples.csv format; omit to run the simulation.",
    )
    parser.add_argument(
        "calibration_filename", nargs="?", metavar="IMU_CALIBRATION.json",
        help="Optional gyro calibration in imu_calibration.json format (zero bias if omitted).",
    )
    parser.add_argument("--sigma-m", type=float, default=0.5, help="Magnetometer noise standard deviation in μT (default: 0.5).")
    parser.add_argument("--sigma-g", type=float, default=0.1, metavar="DEG_PER_S", help="Gyro noise standard deviation in º/s (default: 0.1).")
    parser.add_argument("--phi", type=float, default=10.0, help="EKF gyro-noise scale factor (default: 10).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.sigma_m < 0 or args.sigma_g < 0 or args.phi < 0:
        raise SystemExit("--sigma-m, --sigma-g, and --phi must be non-negative")
    try:
        if args.data_filename is None:
            run_simulation(args.sigma_m, args.sigma_g, args.phi)
        else:
            run_recording(
                args.data_filename, args.calibration_filename,
                args.sigma_m, args.sigma_g, args.phi,
            )
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
