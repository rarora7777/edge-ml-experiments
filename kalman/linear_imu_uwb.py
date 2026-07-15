import argparse
import json
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

# ==============================================================================
# 1. STREAMING KALMAN FILTER IMPLEMENTATION
# ==============================================================================


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

    def __init__(self, ip_address, d_anchors, q_pos, q_vel, q_bias, r_uwb):
        self.base_url = f"http://{ip_address}"
        self.kf = KalmanFilter2DStreaming(q_pos, q_vel, q_bias, r_uwb)
        self.uwb_pipe = UWBBufferPipeline(d_anchors, self.kf)
        self.d_anchors = d_anchors
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.est_positions = []
        self.uwb_positions = []
        self.latest_ranges = {}

    def run(self):
        """Starts both SSE readers and maintains an interactive position plot."""
        threads = [
            threading.Thread(target=self._read_sse, args=("/imu/stream", self._handle_imu), daemon=True),
            threading.Thread(target=self._read_sse, args=("/uwb/stream", self._handle_uwb), daemon=True),
        ]
        for thread in threads:
            thread.start()

        plt.ion()
        figure, axis = plt.subplots(figsize=(7, 6))
        figure.subplots_adjust(right=0.72)
        try:
            while plt.fignum_exists(figure.number):
                self._draw(axis)
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
        with self.lock:
            self.kf.predict_stream(sample["timestamp_ms"], accel["x"], accel["y"])
            self.est_positions.append(tuple(self.kf.x[:2, 0]))

    def _handle_uwb(self, sample):
        """Displays raw ranges, then pairs them for UWB-only and fused estimates."""
        anchor_id = sample.get("anchor_id")
        distance_m = sample.get("distance_m")
        sequence = sample.get("sequence")
        if anchor_id is None or distance_m is None:
            # Compatibility with firmware that sends only the raw Unit UWB report.
            match = re.fullmatch(
                r"an(\d+):([0-9]+(?:\.[0-9]+)?)m", sample.get("report", "").strip()
            )
            if match is None:
                return
            anchor_id, distance_m = match.groups()

        with self.lock:
            anchor_id = int(anchor_id)
            distance_m = float(distance_m)
            self.latest_ranges[anchor_id] = distance_m
            x_uwb, y_uwb = self.uwb_pipe.handle_uwb_stream(
                anchor_id, distance_m
            )
            if x_uwb is None:
                return
            self.uwb_positions.append((x_uwb, y_uwb))
            fused_x, fused_y = self.kf.x[:2, 0]
            print(
                f"ranges: r0={self.latest_ranges.get(0, float('nan')):.2f} m, "
                f"r1={self.latest_ranges.get(1, float('nan')):.2f} m | "
                f"UWB-only ({x_uwb:.2f}, {y_uwb:.2f}) m | "
                f"fused ({fused_x:.2f}, {fused_y:.2f}) m"
            )

    def _draw(self, axis):
        """Renders the latest UWB fixes and Kalman estimates without retaining artists."""
        with self.lock:
            uwb_positions = np.array(self.uwb_positions)
            est_positions = np.array(self.est_positions)

        axis.clear()
        axis.scatter([0, self.d_anchors], [0, 0], color="black", marker="^", s=100, label="Anchors")
        if len(uwb_positions):
            axis.plot(uwb_positions[:, 0], uwb_positions[:, 1], "r.", label="UWB fixes")
        if len(est_positions):
            axis.plot(est_positions[:, 0], est_positions[:, 1], "b-", label="Kalman estimate")
        axis.set_title("Live UWB/IMU Tracker")
        axis.set_xlabel("X (metres)")
        axis.set_ylabel("Y (metres)")
        axis.set_aspect("equal")
        axis.grid(True)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


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
    args = parser.parse_args()
    if args.distance <= 0 or min(args.q_pos, args.q_vel, args.q_bias, args.r_uwb) < 0:
        parser.error("distance must be positive and noise values must be non-negative")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.ip:
        LiveTracker(args.ip, args.distance, args.q_pos, args.q_vel, args.q_bias, args.r_uwb).run()
    else:
        run_simulation(args.distance, args.q_pos, args.q_vel, args.q_bias, args.r_uwb)
