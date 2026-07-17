import argparse
from collections import deque
import json
from pathlib import Path
import queue
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_DISTANCE_M = 5.0
DEFAULT_Q_POS = 1e-6
DEFAULT_Q_VEL = 1e-3
DEFAULT_Q_BIAS = 1e-4
DEFAULT_R_UWB = 0.05**2
ACCEL_HISTORY_MS = 10_000
GRAVITY_G = 1.0
STILLNESS_WINDOW_MS = 2_000
GYRO_STILL_EPSILON_DPS = 4.0
GYRO_STABILITY_EPSILON_DPS = 5.0
ACCEL_STABILITY_EPSILON_G = 0.03
ACCEL_GRAVITY_EPSILON_G = 0.20
MIN_ACCEL_CALIBRATION_ORIENTATIONS = 6
DEFAULT_IMU_CALIBRATION_FILE = "imu_calibration.json"
DEFAULT_UWB_CALIBRATION_FILE = "uwb_calibration.json"
UWB_CALIBRATION_ANCHOR_IDS = (0, 1)

# ==============================================================================
# 1. STREAMING KALMAN FILTER IMPLEMENTATION
# ==============================================================================


class ImplicitImuCalibrator:
    """Learns IMU offsets from stationary intervals without a calibration UI.

    Gyro bias is the least-squares mean of gyro samples observed while still.
    Accelerometer bias is fitted from the constraint that static corrected
    acceleration vectors lie on a 1 g sphere. It needs several still device
    orientations before an accelerometer correction is valid.
    """

    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        self.window = deque()
        self.gyro_sum = np.zeros(3)
        self.gyro_sample_count = 0
        self.accel_orientation_samples = []
        self.gyro_bias = np.zeros(3)
        self.gyro_bias_valid = False
        self.accel_bias = np.zeros(3)
        self.accel_bias_valid = False
        self.was_still = False
        self._load_calibration()

    def update(self, timestamp_ms, accel_g, gyro_dps):
        """Returns bias-corrected acceleration, gyro, and current stillness."""
        accel_g = np.asarray(accel_g, dtype=float)
        gyro_dps = np.asarray(gyro_dps, dtype=float)
        self.window.append((timestamp_ms, accel_g, gyro_dps))
        cutoff_ms = timestamp_ms - STILLNESS_WINDOW_MS
        while self.window and self.window[0][0] < cutoff_ms:
            self.window.popleft()

        is_still = self._is_still(timestamp_ms)
        if is_still:
            window_accel = np.array([sample[1] for sample in self.window])
            window_gyro = np.array([sample[2] for sample in self.window])
            if not self.gyro_bias_valid:
                # Before an offset is known, quiet gyro variation—not its raw
                # magnitude—identifies a likely stationary human-held device.
                self.gyro_bias = window_gyro.mean(axis=0)
                self.gyro_sum = self.gyro_bias * len(window_gyro)
                self.gyro_sample_count = len(window_gyro)
                self.gyro_bias_valid = True
                print(f"IMU calibration: gyro bias = {self.gyro_bias} dps")
                self._save_calibration()
            else:
                # The mean is the least-squares solution for a constant gyro bias.
                self.gyro_sum += gyro_dps
                self.gyro_sample_count += 1
                self.gyro_bias = self.gyro_sum / self.gyro_sample_count

            if not self.was_still:
                self._add_accel_orientation(window_accel.mean(axis=0))
                self._fit_accel_bias()
        self.was_still = is_still

        return accel_g - self.accel_bias, gyro_dps - self.gyro_bias, is_still

    def _is_still(self, timestamp_ms):
        if not self.window or timestamp_ms - self.window[0][0] < STILLNESS_WINDOW_MS:
            return False

        accel = np.array([sample[1] for sample in self.window])
        raw_gyro = np.array([sample[2] for sample in self.window])
        gyro_deviation = np.linalg.norm(raw_gyro - raw_gyro.mean(axis=0), axis=1)
        accel_deviation = np.linalg.norm(accel - accel.mean(axis=0), axis=1)
        is_quiet_and_gravity_consistent = (
            np.max(gyro_deviation) < GYRO_STABILITY_EPSILON_DPS
            and np.max(accel_deviation) < ACCEL_STABILITY_EPSILON_G
            and abs(np.linalg.norm(accel.mean(axis=0)) - GRAVITY_G) < ACCEL_GRAVITY_EPSILON_G
        )
        if not self.gyro_bias_valid:
            return is_quiet_and_gravity_consistent

        corrected_gyro_norm = np.linalg.norm(raw_gyro - self.gyro_bias, axis=1)
        return (
            is_quiet_and_gravity_consistent
            and np.max(corrected_gyro_norm) < GYRO_STILL_EPSILON_DPS
        )

    def _add_accel_orientation(self, mean_accel_g):
        """Keeps only stationary poses that add a meaningfully new direction."""
        if any(np.linalg.norm(mean_accel_g - sample) < 0.15 for sample in self.accel_orientation_samples):
            return
        self.accel_orientation_samples.append(mean_accel_g)
        print(
            "IMU calibration: collected "
            f"{len(self.accel_orientation_samples)}/{MIN_ACCEL_CALIBRATION_ORIENTATIONS} still orientations"
        )

    def _fit_accel_bias(self):
        if len(self.accel_orientation_samples) < MIN_ACCEL_CALIBRATION_ORIENTATIONS:
            return

        samples = np.array(self.accel_orientation_samples)
        # ||a - b||^2 = g^2 becomes 2 a.b + d = ||a||^2,
        # where b is the accelerometer bias and d = g^2 - ||b||^2.
        design = np.column_stack((2.0 * samples, np.ones(len(samples))))
        solution, _, rank, _ = np.linalg.lstsq(design, np.sum(samples**2, axis=1), rcond=None)
        if rank == 4:
            self.accel_bias = solution[:3]
            self.accel_bias_valid = True
            print(f"IMU calibration: accelerometer bias = {self.accel_bias} g")
            self._save_calibration()

    def _load_calibration(self):
        """Loads a previously validated calibration, if present and well formed."""
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            accel_bias = np.asarray(data["accel_bias_g"], dtype=float)
            gyro_bias = np.asarray(data["gyro_bias_dps"], dtype=float)
            accel_bias_valid = bool(data.get("accel_bias_valid", True))
            if accel_bias.shape != (3,) or gyro_bias.shape != (3,) or not (
                np.all(np.isfinite(accel_bias)) and np.all(np.isfinite(gyro_bias))
            ):
                raise ValueError("bias vectors must contain three finite values")
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as error:
            print(f"Ignoring invalid IMU calibration file {self.calibration_path}: {error}")
            return

        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.gyro_bias_valid = True
        self.accel_bias_valid = accel_bias_valid
        print(f"Loaded IMU calibration from {self.calibration_path}")

    def _save_calibration(self):
        """Atomically persists the currently valid IMU bias estimates."""
        data = {
            "version": 1,
            "accel_bias_g": self.accel_bias.tolist(),
            "accel_bias_valid": self.accel_bias_valid,
            "gyro_bias_dps": self.gyro_bias.tolist(),
        }
        temporary_path = self.calibration_path.with_suffix(self.calibration_path.suffix + ".tmp")
        try:
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            temporary_path.replace(self.calibration_path)
            print(f"Saved IMU calibration to {self.calibration_path}")
        except OSError as error:
            print(f"Unable to save IMU calibration to {self.calibration_path}: {error}")


class KalmanFilter2DStreaming:
    def __init__(self, q_pos, q_vel, q_bias, r_uwb):
        """
        State vector x = [px, py, vx, vy, bx, by]^T
        Optimized for real-time data streams with varying dt.
        """
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0

        # Save tuning configurations for dynamic Q matrix generation
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_bias = q_bias
        self.R = np.eye(2) * r_uwb

        # Track timing history
        self.last_timestamp_ms = None

    def predict_stream(self, timestamp_ms, ax_g, ay_g):
        """
        Callback triggered immediately upon receiving a raw IMU packet.
        Handles dynamic dt calculations and g to m/s^2 conversion.
        """
        if self.last_timestamp_ms is None:
            self.last_timestamp_ms = timestamp_ms
            return

        # 1. Calculate dynamic dt from timestamp stream
        dt = (timestamp_ms - self.last_timestamp_ms) / 1000.0
        self.last_timestamp_ms = timestamp_ms

        if dt <= 0:
            return

        # 2. Convert acceleration from g's to m/s^2
        G_TO_METRIC = 9.80665
        ax_raw = ax_g * G_TO_METRIC
        ay_raw = ay_g * G_TO_METRIC

        # 3. Reconstruct dynamic F matrix
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = -0.5 * dt * dt
        F[1, 5] = -0.5 * dt * dt
        F[2, 4] = -dt
        F[3, 5] = -dt

        # 4. Reconstruct dynamic B matrix
        B = np.zeros((6, 2))
        B[0, 0] = 0.5 * dt * dt
        B[1, 1] = 0.5 * dt * dt
        B[2, 0] = dt
        B[3, 1] = dt

        # 5. Reconstruct dynamic Q matrix
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3

        Q = np.zeros((6, 6))
        Q[0, 0] = self.q_pos + 0.25 * dt4 * self.q_vel
        Q[1, 1] = self.q_pos + 0.25 * dt4 * self.q_vel
        Q[0, 2] = 0.5 * dt3 * self.q_vel
        Q[2, 0] = 0.5 * dt3 * self.q_vel
        Q[1, 3] = 0.5 * dt3 * self.q_vel
        Q[3, 1] = 0.5 * dt3 * self.q_vel
        Q[2, 2] = dt2 * self.q_vel
        Q[3, 3] = dt2 * self.q_vel
        Q[4, 4] = dt * self.q_bias
        Q[5, 5] = dt * self.q_bias

        # 6. Run standard Kalman Prediction
        u = np.array([[ax_raw], [ay_raw]])
        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + Q

    def update(self, x_uwb, y_uwb):
        """
        Triggered only when the UWB buffer pipeline emits an XY pairing.
        """
        z = np.array([[x_uwb], [y_uwb]])
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 1] = 1

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


# ==============================================================================
# 2. UWB STREAM HANDLING & GEOMETRIC BUFFER PIPELINE
# ==============================================================================


class UWBBufferPipeline:
    def __init__(self, d_anchors, kf_instance):
        self.d_anchors = d_anchors
        self.kf = kf_instance

        # State storage for individual incoming ranges
        self.r0 = 0.0
        self.r1 = 0.0
        self.r0_fresh = False
        self.r1_fresh = False

    def handle_uwb_stream(self, anchor_id, distance_m):
        """
        Callback representing your asynchronous UWB client engine packet parser.
        Accepts single scalar range packets.
        """
        if anchor_id == 0:
            self.r0 = distance_m
            self.r0_fresh = True
        elif anchor_id == 1:
            self.r1 = distance_m
            self.r1_fresh = True

        # Process only when a complete spatial calculation pair is ready
        if self.r0_fresh and self.r1_fresh:
            # Stage 1 Pre-processor: Closed-form trilateration
            x_uwb = (self.r0**2 - self.r1**2 + self.d_anchors**2) / (2 * self.d_anchors)
            y_uwb = np.sqrt(max(0.0, self.r0**2 - x_uwb**2))

            # Update the filter immediately
            self.kf.update(x_uwb, y_uwb)

            # Consume the samples
            self.r0_fresh = False
            self.r1_fresh = False
            return x_uwb, y_uwb

        return None, None


def parse_uwb_measurement(sample):
    """Returns an anchor ID and range from either structured or legacy UWB SSE data."""
    anchor_id = sample.get("anchor_id")
    distance_m = sample.get("distance_m")
    if anchor_id is None or distance_m is None:
        match = re.fullmatch(
            r"an(\d+):([0-9]+(?:\.[0-9]+)?)m", sample.get("report", "").strip()
        )
        if match is None:
            return None
        anchor_id, distance_m = match.groups()
    return int(anchor_id), float(distance_m)


def load_uwb_offsets(calibration_path):
    """Loads per-anchor additive range offsets, returning an empty map if unavailable."""
    path = Path(calibration_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        offsets = {
            int(anchor_id): float(values["offset_m"])
            for anchor_id, values in data["anchors"].items()
        }
        if not all(np.isfinite(offset) for offset in offsets.values()):
            raise ValueError("offsets must be finite")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as error:
        print(f"Ignoring invalid UWB calibration file {path}: {error}")
        return {}
    print(f"Loaded UWB range offsets from {path}")
    return offsets


# ==============================================================================
# 3. STREAM SIMULATOR HARNESS
# ==============================================================================


def run_simulation(d_anchors, q_pos, q_vel, q_bias, r_uwb):
    np.random.seed(42)

    total_time = 10.0
    fps_imu = 100
    fps_uwb = 10
    steps = int(total_time * fps_imu)

    accel_noise_std = 0.2  # m/s^2
    # True hardware bias configured in units of m/s^2
    accel_bias_metric = np.array([[0.15], [-0.25]])
    uwb_range_noise_std = 0.05

    anchor0_pos = np.array([0.0, 0.0])
    anchor1_pos = np.array([d_anchors, 0.0])

    # Instantiate Kalman Filter and the Buffer Pipeline Wrapper
    kf = KalmanFilter2DStreaming(q_pos, q_vel, q_bias, r_uwb)
    uwb_pipe = UWBBufferPipeline(d_anchors=d_anchors, kf_instance=kf)

    # Generate Ground Truth Circle
    t = np.linspace(0, total_time, steps)
    center_x, center_y = 2.5, 3.0
    radius = 1.5
    omega = 2 * np.pi / total_time

    true_px = center_x + radius * np.cos(omega * t)
    true_py = center_y + radius * np.sin(omega * t)
    true_ax = -radius * omega**2 * np.cos(omega * t)
    true_ay = -radius * omega**2 * np.sin(omega * t)

    est_positions = []
    uwb_positions = []
    est_biases = []

    # Stream Event Execution Loop
    for i in range(steps):
        # 1. Simulate a hardware clock timestamp in milliseconds
        current_time_ms = i * 10

        # 2. Simulate raw IMU hardware payload stream (in g's)
        # Add the bias and noise, then scale down to g's to match sensor output
        G_TO_METRIC = 9.80665
        ax_g = (
            true_ax[i] + accel_bias_metric[0, 0] + np.random.normal(0, accel_noise_std)
        ) / G_TO_METRIC
        ay_g = (
            true_ay[i] + accel_bias_metric[1, 0] + np.random.normal(0, accel_noise_std)
        ) / G_TO_METRIC

        # Fire high-rate IMU Stream Callback
        kf.predict_stream(current_time_ms, ax_g, ay_g)

        # 3. Simulate sequential asynchronous UWB packets every 10 IMU ticks
        calculated_xy = [None, None]
        if i % (fps_imu // fps_uwb) == 0:
            # Generate range data
            r0_true = np.linalg.norm(np.array([true_px[i], true_py[i]]) - anchor0_pos)
            r1_true = np.linalg.norm(np.array([true_px[i], true_py[i]]) - anchor1_pos)
            r0_noisy = r0_true + np.random.normal(0, uwb_range_noise_std)
            r1_noisy = r1_true + np.random.normal(0, uwb_range_noise_std)

            # Call streaming callbacks independently to simulate raw individual packets
            uwb_pipe.handle_uwb_stream(anchor_id=0, distance_m=r0_noisy)
            x_u, y_u = uwb_pipe.handle_uwb_stream(anchor_id=1, distance_m=r1_noisy)

            if x_u is not None:
                calculated_xy = [x_u, y_u]

        uwb_positions.append(calculated_xy)
        est_positions.append(kf.x[:2].copy())
        est_biases.append(kf.x[4:].copy())

    # Map arrays for evaluation plot
    est_positions = np.array(est_positions).squeeze()
    est_biases = np.array(est_biases).squeeze()
    uwb_positions_clean = np.array([pos for pos in uwb_positions if pos[0] is not None])

    # Draw verification plots
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(true_px, true_py, "g-", label="Ground Truth", linewidth=2)
    plt.scatter(
        uwb_positions_clean[:, 0],
        uwb_positions_clean[:, 1],
        color="red",
        alpha=0.5,
        s=15,
        label="Buffered UWB Out",
    )
    plt.plot(
        est_positions[:, 0],
        est_positions[:, 1],
        "b--",
        label="Streaming Kalman Estimate",
        linewidth=2,
    )
    plt.scatter(
        [anchor0_pos[0], anchor1_pos[0]],
        [anchor0_pos[1], anchor1_pos[1]],
        color="black",
        marker="^",
        s=100,
        label="Anchors",
    )
    plt.title("Streaming 2D Tracker Trajectory")
    plt.xlabel("X (metres)")
    plt.ylabel("Y (metres)")
    plt.legend()
    plt.grid(True)
    ax1.set_aspect("equal")

    plt.subplot(1, 2, 2)
    plt.axhline(accel_bias_metric[0, 0], color="r", linestyle=":", label="True X Bias")
    plt.plot(est_biases[:, 0], "r-", label="Estimated X Bias")
    plt.axhline(accel_bias_metric[1, 0], color="b", linestyle=":", label="True Y Bias")
    plt.plot(est_biases[:, 1], "b-", label="Estimated Y Bias")
    plt.title("Live Bias Convergence (m/s^2)")
    plt.xlabel("Streaming Updates")
    plt.ylabel("Bias")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class LiveTracker:
    """Consumes AtomS3R IMU and UWB SSE streams and displays live estimates."""

    def __init__(self, ip_address, d_anchors, q_pos, q_vel, q_bias, r_uwb, calibration_path,
                 uwb_calibration_path):
        self.base_url = f"http://{ip_address}"
        self.kf = KalmanFilter2DStreaming(q_pos, q_vel, q_bias, r_uwb)
        self.imu_calibrator = ImplicitImuCalibrator(calibration_path)
        self.uwb_offsets = load_uwb_offsets(uwb_calibration_path)
        self.uwb_pipe = UWBBufferPipeline(d_anchors, self.kf)
        self.d_anchors = d_anchors
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.est_positions = deque()
        self.uwb_positions = deque()
        self.latest_ranges = {}
        self.accel_samples = deque()
        self.gyro_samples = deque()

    def run(self):
        """Starts both SSE readers and maintains an interactive position plot."""
        threads = [
            threading.Thread(target=self._read_sse, args=("/imu/stream", self._handle_imu), daemon=True),
            threading.Thread(target=self._read_sse, args=("/uwb/stream", self._handle_uwb), daemon=True),
        ]
        for thread in threads:
            thread.start()

        plt.ion()
        figure, (position_axis, accel_axis, gyro_axis) = plt.subplots(3, 1, figsize=(8, 12))
        figure.subplots_adjust(right=0.78)
        try:
            while plt.fignum_exists(figure.number):
                self._draw(position_axis, accel_axis, gyro_axis)
                plt.pause(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            plt.ioff()

    def _read_sse(self, path, callback):
        """Reconnects to one SSE endpoint and forwards each JSON data event."""
        url = self.base_url + path
        while not self.stop_event.is_set():
            try:
                print(f"Connecting to {url}")
                request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    for raw_line in response:
                        if self.stop_event.is_set():
                            return
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            callback(json.loads(line[5:].strip()))
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                            print(f"Ignoring malformed {path} event: {error}")
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                if not self.stop_event.is_set():
                    print(f"{path} disconnected: {error}; retrying in 2 seconds")
                    self.stop_event.wait(2)

    def _handle_imu(self, sample):
        """Applies one valid g-unit IMU sample and records its predicted state."""
        if not sample.get("valid", False):
            return
        accel = sample["accel"]
        gyro = sample["gyro"]
        with self.lock:
            timestamp_ms = sample["timestamp_ms"]
            corrected_accel, corrected_gyro, _ = self.imu_calibrator.update(
                timestamp_ms,
                (accel["x"], accel["y"], accel["z"]),
                (gyro["x"], gyro["y"], gyro["z"]),
            )
            self.kf.predict_stream(timestamp_ms, corrected_accel[0], corrected_accel[1])
            self.est_positions.append((timestamp_ms, *self.kf.x[:2, 0]))
            self.accel_samples.append((timestamp_ms, *corrected_accel))
            self.gyro_samples.append((timestamp_ms, *corrected_gyro))
            self._trim_history(timestamp_ms)

    def _handle_uwb(self, sample):
        """Displays raw ranges, then pairs them for UWB-only and fused estimates."""
        measurement = parse_uwb_measurement(sample)
        if measurement is None:
            return
        anchor_id, raw_distance_m = measurement

        with self.lock:
            distance_m = raw_distance_m - self.uwb_offsets.get(anchor_id, 0.0)
            self.latest_ranges[anchor_id] = distance_m
            x_uwb, y_uwb = self.uwb_pipe.handle_uwb_stream(
                anchor_id, distance_m
            )
            if x_uwb is None:
                return
            timestamp_ms = sample.get("timestamp_ms", self.kf.last_timestamp_ms or 0)
            self.uwb_positions.append((timestamp_ms, x_uwb, y_uwb))
            self._trim_history(timestamp_ms)
            fused_x, fused_y = self.kf.x[:2, 0]
            print(
                f"ranges: r0={self.latest_ranges.get(0, float('nan')):.2f} m, "
                f"r1={self.latest_ranges.get(1, float('nan')):.2f} m | "
                f"UWB-only ({x_uwb:.2f}, {y_uwb:.2f}) m | "
                f"fused ({fused_x:.2f}, {fused_y:.2f}) m"
            )

    def _trim_history(self, latest_timestamp_ms):
        """Discards trajectory and acceleration samples older than the display window."""
        cutoff_ms = latest_timestamp_ms - ACCEL_HISTORY_MS
        for history in (self.est_positions, self.uwb_positions, self.accel_samples, self.gyro_samples):
            while history and history[0][0] < cutoff_ms:
                history.popleft()

    def _draw(self, position_axis, accel_axis, gyro_axis):
        """Renders the latest position estimates plus ten seconds of IMU data."""
        with self.lock:
            uwb_positions = np.array(self.uwb_positions)
            est_positions = np.array(self.est_positions)
            accel_samples = np.array(self.accel_samples)
            gyro_samples = np.array(self.gyro_samples)

        position_axis.clear()
        position_axis.scatter([0, self.d_anchors], [0, 0], color="black", marker="^", s=100, label="Anchors")
        if len(est_positions):
            position_axis.plot(est_positions[:, 1], est_positions[:, 2], "b-", label="Kalman estimate")
        if len(uwb_positions):
            position_axis.scatter(
                uwb_positions[:, 1], uwb_positions[:, 2], color="red", s=36,
                label="UWB fixes", zorder=3,
            )
        position_axis.set_title("Live UWB/IMU Tracker (last 10 seconds)")
        position_axis.set_xlabel("X (metres)")
        position_axis.set_ylabel("Y (metres)")
        position_axis.set_aspect("equal")
        position_axis.grid(True)
        position_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        accel_axis.clear()
        if len(accel_samples):
            latest_time_ms = accel_samples[-1, 0]
            time_s = (accel_samples[:, 0] - latest_time_ms) / 1000.0
            accel_axis.plot(time_s, accel_samples[:, 1], label="X")
            accel_axis.plot(time_s, accel_samples[:, 2], label="Y")
            accel_axis.plot(time_s, accel_samples[:, 3], label="Z")
        accel_axis.set_title("Bias-corrected accelerometer (last 10 seconds)")
        accel_axis.set_xlabel("Time relative to latest sample (s)")
        accel_axis.set_ylabel("Acceleration (g)")
        accel_axis.set_xlim(-ACCEL_HISTORY_MS / 1000.0, 0)
        accel_axis.grid(True)
        accel_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        gyro_axis.clear()
        if len(gyro_samples):
            latest_time_ms = gyro_samples[-1, 0]
            time_s = (gyro_samples[:, 0] - latest_time_ms) / 1000.0
            gyro_axis.plot(time_s, gyro_samples[:, 1], label="X")
            gyro_axis.plot(time_s, gyro_samples[:, 2], label="Y")
            gyro_axis.plot(time_s, gyro_samples[:, 3], label="Z")
        gyro_axis.set_title("Bias-corrected gyroscope (last 10 seconds)")
        gyro_axis.set_xlabel("Time relative to latest sample (s)")
        gyro_axis.set_ylabel("Angular velocity (°/s)")
        gyro_axis.set_xlim(-ACCEL_HISTORY_MS / 1000.0, 0)
        gyro_axis.grid(True)
        gyro_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


def run_uwb_calibration(ip_address, distances_m, samples_per_anchor, calibration_path):
    """Guides surveyed-distance UWB range calibration and saves per-anchor offsets."""
    events = queue.Queue()
    stop_event = threading.Event()
    url = f"http://{ip_address}/uwb/stream"

    def read_uwb_events():
        while not stop_event.is_set():
            try:
                request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    for raw_line in response:
                        if stop_event.is_set():
                            return
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            events.put(json.loads(line[5:].strip()))
                        except json.JSONDecodeError:
                            continue
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                if not stop_event.is_set():
                    print(f"UWB stream disconnected: {error}; retrying in 2 seconds")
                    stop_event.wait(2)

    reader = threading.Thread(target=read_uwb_events, daemon=True)
    reader.start()
    print("Connecting to the UWB stream and discarding any retained reports...")
    time.sleep(1)
    while not events.empty():
        events.get_nowait()

    residuals = {anchor_id: [] for anchor_id in UWB_CALIBRATION_ANCHOR_IDS}
    try:
        for anchor_id in UWB_CALIBRATION_ANCHOR_IDS:
            for distance_m in distances_m:
                input(
                    f"Place the tag {distance_m:g} m from anchor {anchor_id} with clear line of sight, "
                    "then press Enter to collect samples: "
                )
                while not events.empty():
                    events.get_nowait()
                samples = []
                while len(samples) < samples_per_anchor:
                    try:
                        sample = events.get(timeout=10)
                    except queue.Empty:
                        print("Waiting for UWB reports...")
                        continue
                    measurement = parse_uwb_measurement(sample)
                    if measurement is None or measurement[0] != anchor_id:
                        continue
                    measured_distance_m = measurement[1]
                    samples.append(measured_distance_m)
                    print(
                        f"anchor {anchor_id}, {distance_m:g} m: "
                        f"{len(samples)}/{samples_per_anchor} = {measured_distance_m:.3f} m"
                    )
                residuals[anchor_id].extend(value - distance_m for value in samples)
    except KeyboardInterrupt:
        print("UWB calibration cancelled; no file written")
        return
    finally:
        stop_event.set()

    offsets = {anchor_id: float(np.median(values)) for anchor_id, values in residuals.items()}
    data = {
        "version": 1,
        "distances_m": list(distances_m),
        "samples_per_anchor": samples_per_anchor,
        "anchors": {
            str(anchor_id): {
                "offset_m": offsets[anchor_id],
                "residual_std_m": float(np.std(values)),
            }
            for anchor_id, values in residuals.items()
        },
    }
    path = Path(calibration_path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    for anchor_id, values in residuals.items():
        print(
            f"Anchor {anchor_id}: offset={offsets[anchor_id]:+.3f} m, "
            f"residual std={np.std(values):.3f} m"
        )
    print(f"Saved UWB range calibration to {path}")


def parse_args():
    """Parses tracker configuration, selecting simulation when no IP is supplied."""
    parser = argparse.ArgumentParser(description="Track an AtomS3R tag using IMU and UWB streams.")
    parser.add_argument("--ip", help="AtomS3R IP address; omit to run the simulation")
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE_M,
                        help=f"distance between anchors in metres (default: {DEFAULT_DISTANCE_M})")
    parser.add_argument("--q-pos", type=float, default=DEFAULT_Q_POS,
                        help=f"position process noise (default: {DEFAULT_Q_POS:g})")
    parser.add_argument("--q-vel", type=float, default=DEFAULT_Q_VEL,
                        help=f"velocity process noise (default: {DEFAULT_Q_VEL:g})")
    parser.add_argument("--q-bias", type=float, default=DEFAULT_Q_BIAS,
                        help=f"accelerometer-bias process noise (default: {DEFAULT_Q_BIAS:g})")
    parser.add_argument("--r-uwb", type=float, default=DEFAULT_R_UWB,
                        help=f"UWB measurement variance in m^2 (default: {DEFAULT_R_UWB:g})")
    parser.add_argument("--imu-calibration", default=DEFAULT_IMU_CALIBRATION_FILE,
                        help="path for persistent IMU biases (default: imu_calibration.json)")
    parser.add_argument("--uwb-calibration", default=DEFAULT_UWB_CALIBRATION_FILE,
                        help="path for persistent UWB range offsets (default: uwb_calibration.json)")
    parser.add_argument("--uwb-calibrate", type=float, nargs="+", metavar="METRES",
                        help="run guided UWB calibration at these surveyed distances")
    parser.add_argument("--uwb-calibration-samples", type=int, default=10,
                        help="reports per anchor at each UWB calibration distance (default: 10)")
    args = parser.parse_args()
    if args.distance <= 0 or min(args.q_pos, args.q_vel, args.q_bias, args.r_uwb) < 0:
        parser.error("distance must be positive and noise values must be non-negative")
    if args.uwb_calibrate is not None and (not args.ip or min(args.uwb_calibrate) <= 0):
        parser.error("--uwb-calibrate requires --ip and positive surveyed distances")
    if args.uwb_calibration_samples <= 0:
        parser.error("--uwb-calibration-samples must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.uwb_calibrate is not None:
        run_uwb_calibration(args.ip, args.uwb_calibrate, args.uwb_calibration_samples,
                            args.uwb_calibration)
    elif args.ip:
        LiveTracker(args.ip, args.distance, args.q_pos, args.q_vel, args.q_bias, args.r_uwb,
                    args.imu_calibration, args.uwb_calibration).run()
    else:
        run_simulation(args.distance, args.q_pos, args.q_vel, args.q_bias, args.r_uwb)
