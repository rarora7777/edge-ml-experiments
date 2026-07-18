#!/usr/bin/env python3
"""Calibrate streamed magnetometer data and display live IMU orientation.

Example:
    python3 imu_realtime_orientation.py http://192.168.1.42/imu/stream \
        --calibration imu_calibration.json --algorithm madgwick --record imu.csv

    python3 imu_realtime_orientation.py imu.csv --algorithm ekf

Install dependencies first:
    python3 -m pip install ahrs numpy matplotlib
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Literal, TypeAlias, cast
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
from ahrs.filters import EKF, Madgwick
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from ekf_imu_mag import MagGyroCalibrationEKF
from imu_stream_to_csv import CSV_COLUMNS, sample_row


FloatArray: TypeAlias = NDArray[np.float64]
SseSample: TypeAlias = dict[str, Any]
OrientationAlgorithm: TypeAlias = Literal["ekf", "madgwick"]
OrientationFilter: TypeAlias = EKF | Madgwick
SensorSample: TypeAlias = tuple[float, FloatArray]


@dataclass(frozen=True)
class RuntimeArguments:
    """Command-line settings for the live orientation runner."""

    source: str
    calibration: Path | None
    algorithm: OrientationAlgorithm
    record: Path | None
    frequency: float
    reconnect_delay: float
    sigma_m: float
    sigma_g: float
    phi: float


BOX_VERTICES: Final[FloatArray] = np.array([
    [-1.0, -2.5, -0.5], [1.0, -2.5, -0.5], [1.0, 2.5, -0.5], [-1.0, 2.5, -0.5],
    [-1.0, -2.5, 0.5], [1.0, -2.5, 0.5], [1.0, 2.5, 0.5], [-1.0, 2.5, 0.5],
], dtype=float)
BOX_FACES: Final[tuple[tuple[int, int, int, int], ...]] = (
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)
# Opposing faces use related hues, making roll, pitch, and yaw easy to see.
BOX_FACE_COLORS: Final[tuple[str, ...]] = (
    "#1f77b4", "#6baed6", "#d62728", "#ff9896", "#2ca02c", "#98df8a",
)


def as_bool(value: object) -> bool:
    """Interpret common textual and numeric truth values."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def vector_from_sample(sample: SseSample, name: str) -> FloatArray:
    """Extract one three-component sensor value from an SSE sample."""
    values = sample[name]
    if not isinstance(values, dict):
        raise ValueError(f"invalid {name} vector")
    vector = np.asarray([values[axis] for axis in ("x", "y", "z")], dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"invalid {name} vector")
    return vector


def sample_dt(timestamp_ms: float, previous_timestamp_ms: float | None,
              default_dt: float) -> float:
    """Use device time when it is plausible, otherwise use a nominal period."""
    if previous_timestamp_ms is None:
        return default_dt
    try:
        dt = (float(timestamp_ms) - previous_timestamp_ms) / 1000.0
    except (TypeError, ValueError):
        return default_dt
    return dt if 0.0 < dt < 1.0 else default_dt


def is_stream_url(source: str) -> bool:
    """Return whether *source* is an HTTP(S) IMU event-stream URL."""
    return urlparse(source).scheme.lower() in {"http", "https"}


def sample_from_csv_row(row: dict[str, str | None]) -> SseSample:
    """Convert one recorded CSV row back into the SSE sample shape."""
    sample: SseSample = {
        "timestamp_ms": row.get("timestamp_ms", ""),
        "valid": row.get("valid", False),
        "mag_available": row.get("mag_available", False),
        "mag_fresh": row.get("mag_fresh", row.get("mag_available", False)),
        "accel": {"x": row.get("accel_x", ""), "y": row.get("accel_y", ""), "z": row.get("accel_z", "")},
        "gyro": {"x": row.get("gyro_x", ""), "y": row.get("gyro_y", ""), "z": row.get("gyro_z", "")},
    }
    if as_bool(sample["mag_available"]):
        sample["mag"] = {
            "x": row.get("mag_x", ""), "y": row.get("mag_y", ""), "z": row.get("mag_z", ""),
        }
    return sample


def recorded_samples(path: Path, default_dt: float) -> Iterator[SseSample]:
    """Yield CSV samples at their recorded device cadence for real-time playback."""
    previous_timestamp_ms: float | None = None
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"timestamp_ms", "valid", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing_columns))}")
        for row in reader:
            try:
                timestamp_ms = float(row["timestamp_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_timestamp_ms is not None:
                time.sleep(sample_dt(timestamp_ms, previous_timestamp_ms, default_dt))
            previous_timestamp_ms = timestamp_ms
            yield sample_from_csv_row(row)


def streamed_samples(url: str, reconnect_delay: float) -> Iterator[SseSample]:
    """Yield SSE IMU samples, reconnecting after recoverable stream failures."""
    while True:
        try:
            print(f"Connecting to {url}", file=sys.stderr)
            request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=30) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        sample = json.loads(line.removeprefix("data:").strip())
                    except json.JSONDecodeError as error:
                        print(f"Ignoring malformed event: {error}", file=sys.stderr)
                        continue
                    if isinstance(sample, dict):
                        yield sample
                    else:
                        print("Ignoring non-object IMU event", file=sys.stderr)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            print(f"Stream disconnected ({error}); retrying in {reconnect_delay:g}s", file=sys.stderr)
            time.sleep(reconnect_delay)


def source_samples(source: str, default_dt: float,
                   reconnect_delay: float) -> Iterator[SseSample]:
    """Choose URL streaming or real-time CSV playback for the requested source."""
    if is_stream_url(source):
        yield from streamed_samples(source, reconnect_delay)
    else:
        path = Path(source)
        print(f"Playing back {path} in real time", file=sys.stderr)
        yield from recorded_samples(path, default_dt)


def rotation_matrix(quaternion: FloatArray) -> FloatArray:
    """Return the body-to-world rotation matrix for a scalar-first quaternion."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class OrientationBox:
    """A non-blocking matplotlib 3D view of the IMU as a 2 x 5 x 1 box."""

    def __init__(self) -> None:
        """Create and show the orientation-box figure."""
        plt.ion()
        self.figure = plt.figure(figsize=(7, 7))
        self.axes: Axes3D = cast(Axes3D, self.figure.add_subplot(projection="3d"))
        self.axes.set_title("Live IMU orientation")
        self.axes.set_xlabel("World X")
        self.axes.set_ylabel("World Y")
        self.axes.set_zlabel("World Z (Down)")
        self.axes.set_xlim(-3, 3)
        self.axes.set_ylim(-3, 3)
        self.axes.set_zlim(-3, 3)
        # AHRS EKF uses NED: positive world Z points down, unlike the usual
        # matplotlib 3D convention where positive Z is drawn upward.
        self.axes.invert_zaxis()
        self.axes.set_box_aspect((1, 1, 1))
        self.axes.view_init(elev=25, azim=35)
        self.box = Poly3DCollection(
            [], facecolors=BOX_FACE_COLORS, edgecolors="black", alpha=0.75
        )
        self.axes.add_collection3d(self.box)
        self.update(np.array([1.0, 0.0, 0.0, 0.0]))
        plt.show(block=False)

    def update(self, quaternion: FloatArray) -> None:
        """Rotate the box to match the supplied scalar-first quaternion."""
        vertices = BOX_VERTICES @ rotation_matrix(quaternion).T
        self.box.set_verts([vertices[list(face)] for face in BOX_FACES])
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    @property
    def is_open(self) -> bool:
        """Return whether the orientation figure is still open."""
        return plt.fignum_exists(self.figure.number)


class SensorHistoryPlot:
    """Live 3D scatter plot of the latest calibrated sensor vectors."""

    WINDOW_S = 5.0
    REFRESH_S = 0.1
    SERIES: Final[tuple[tuple[str, str, float, str], ...]] = (
        ("Acceleration", "tab:blue", 1.0, "g"),
        ("Gyroscope", "tab:orange", 1.0 / 360.0, "rev/s"),
        ("Magnetometer", "tab:green", 1.0 / 50.0, "50 uT"),
    )

    def __init__(self) -> None:
        """Create and show the rolling sensor-vector scatter figure."""
        self.history: dict[str, deque[SensorSample]] = {
            name: deque() for name, _, _, _ in self.SERIES
        }
        self.last_refresh_s = -np.inf
        self.figure = plt.figure(figsize=(7, 7))
        self.axes: Axes3D = cast(Axes3D, self.figure.add_subplot(projection="3d"))
        self.axes.set_title("Calibrated Sensor Vectors (last 5 s; display-scaled)")
        self.axes.set_xlabel("Scaled X")
        self.axes.set_ylabel("Scaled Y")
        self.axes.set_zlabel("Scaled Z")
        self.axes.set_xlim(-2, 2)
        self.axes.set_ylim(-2, 2)
        self.axes.set_zlim(-2, 2)
        self.axes.set_box_aspect((1, 1, 1))
        self.axes.view_init(elev=25, azim=35)
        plt.show(block=False)

    @staticmethod
    def sampling_rate(samples: deque[SensorSample]) -> float | None:
        """Estimate the rate of a timestamped rolling sample buffer."""
        if len(samples) < 2:
            return None
        elapsed_s = samples[-1][0] - samples[0][0]
        return (len(samples) - 1) / elapsed_s if elapsed_s > 0 else None

    def append(self, timestamp_s: float, accel_g: FloatArray, gyro_dps: FloatArray,
               calibrated_mag_ut: FloatArray | None) -> None:
        """Append new sensor vectors and refresh the plot at a bounded rate."""
        values: dict[str, FloatArray] = {
            "Acceleration": accel_g,
            "Gyroscope": gyro_dps,
        }
        if calibrated_mag_ut is not None:
            values["Magnetometer"] = calibrated_mag_ut
        oldest_timestamp_s = timestamp_s - self.WINDOW_S
        for name, vector in values.items():
            self.history[name].append((timestamp_s, vector.copy()))
        for samples in self.history.values():
            while samples and samples[0][0] < oldest_timestamp_s:
                samples.popleft()
        if timestamp_s - self.last_refresh_s >= self.REFRESH_S:
            self.refresh()
            self.last_refresh_s = timestamp_s

    def refresh(self) -> None:
        """Redraw the scatter collections and their sampling-rate legend."""
        for collection in list(self.axes.collections):
            collection.remove()
        for name, color, scale, unit in self.SERIES:
            samples = self.history[name]
            if not samples:
                continue
            vectors = np.asarray([vector for _, vector in samples]) * scale
            rate_hz = self.sampling_rate(samples)
            rate_label = f"{rate_hz:.1f} Hz" if rate_hz is not None else "warming up"
            self.axes.scatter(
                xs=vectors[:, 0], ys=vectors[:, 1], zs=vectors[:, 2], s=10, color=color, # type: ignore
                alpha=0.7, label=f"{name} ({unit}, {rate_label})",
            )
        self.axes.legend(loc="upper left")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    @property
    def is_open(self) -> bool:
        """Return whether the sensor-history figure is still open."""
        return plt.fignum_exists(self.figure.number)


def load_calibration(path: Path | None) -> tuple[FloatArray, FloatArray]:
    """Load optional accelerometer and gyro bias vectors from calibration JSON."""
    accel_bias = np.zeros(3)
    gyro_bias = np.zeros(3)
    if path is None:
        return accel_bias, gyro_bias
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
        if calibration.get("accel_bias_valid", True):
            accel_bias = np.asarray(calibration["accel_bias_g"], dtype=float)
        if calibration.get("gyro_bias_valid", True):
            gyro_bias = np.asarray(calibration["gyro_bias_dps"], dtype=float)
        if accel_bias.shape != (3,) or gyro_bias.shape != (3,):
            raise ValueError("bias vectors must each contain exactly three values")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load calibration file {path}: {error}") from error
    print(f"Loaded calibration from {path}", file=sys.stderr)
    return accel_bias, gyro_bias


def create_orientation_filter(algorithm: OrientationAlgorithm, frequency: float) -> OrientationFilter:
    """Create the requested AHRS filter with its nominal IMU frequency."""
    if algorithm == "ekf":
        return EKF(
            frequency=frequency, mag=np.ones((1, 3)),
            noises=[4e-3 * 4e-3 * frequency, 1e-8 * frequency, 0.6 * 0.6],
        )
    return Madgwick(frequency=frequency)


def update_orientation(orientation_filter: OrientationFilter, quaternion: FloatArray,
                       gyro_rad_s: FloatArray, accel_g: FloatArray,
                       calibrated_mag_ut: FloatArray | None, mag_fresh: bool,
                       dt: float) -> FloatArray:
    """Update AHRS at IMU rate, using MARG only when a fresh mag sample arrives."""
    if isinstance(orientation_filter, EKF):
        # AHRS's EKF selects its 3- or 6-element measurement model from this
        # attribute, while retaining the same covariance matrix in either mode.
        orientation_filter.mag = calibrated_mag_ut if mag_fresh else None # type: ignore
        if mag_fresh and calibrated_mag_ut is not None:
            return orientation_filter.update(
                quaternion, gyro_rad_s, accel_g, mag=calibrated_mag_ut, dt=dt
            )
        return orientation_filter.update(quaternion, gyro_rad_s, accel_g, dt=dt)
    if mag_fresh and calibrated_mag_ut is not None:
        return orientation_filter.updateMARG(
            quaternion, gyro_rad_s, accel_g, calibrated_mag_ut, dt=dt
        )
    return orientation_filter.updateIMU(quaternion, gyro_rad_s, accel_g, dt=dt)


def run(args: RuntimeArguments) -> None:
    """Consume SSE samples, update filters, record data, and refresh both views."""
    accel_bias_g, gyro_bias_dps = load_calibration(args.calibration)
    nominal_dt = 1.0 / args.frequency
    sigma_g_rad_s = np.deg2rad(args.sigma_g)
    mag_ekf: MagGyroCalibrationEKF | None = None
    orientation_filter: OrientationFilter | None = None
    quaternion: FloatArray = np.array([1.0, 0.0, 0.0, 0.0])
    previous_timestamp_ms: float | None = None
    box = OrientationBox()
    sensor_history_plot = SensorHistoryPlot()
    processed = 0
    mag_updates = 0

    csv_file = None
    try:
        if args.record is not None:
            csv_file = args.record.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            if csv_file.tell() == 0:
                writer.writeheader()
            print(f"Recording raw samples to {args.record}", file=sys.stderr)
        else:
            writer = None

        for sample in source_samples(args.source, nominal_dt, args.reconnect_delay):
            if not box.is_open or not sensor_history_plot.is_open:
                return
            if writer is not None:
                writer.writerow(sample_row(sample))
                assert csv_file is not None
                csv_file.flush()
            if not as_bool(sample.get("valid", False)):
                continue

            try:
                accel_g = vector_from_sample(sample, "accel") - accel_bias_g
                gyro_dps = vector_from_sample(sample, "gyro") - gyro_bias_dps
                timestamp_ms = float(sample["timestamp_ms"])
            except (KeyError, TypeError, ValueError) as error:
                print(f"Ignoring invalid IMU event: {error}", file=sys.stderr)
                continue

            dt = sample_dt(timestamp_ms, previous_timestamp_ms, nominal_dt)
            previous_timestamp_ms = timestamp_ms
            gyro_rad_s = np.deg2rad(gyro_dps)
            mag_fresh = as_bool(sample.get("mag_fresh", False))
            raw_mag_ut: FloatArray | None = None
            if mag_fresh:
                try:
                    raw_mag_ut = vector_from_sample(sample, "mag")
                except (KeyError, TypeError, ValueError) as error:
                    print(f"Ignoring invalid magnetometer event: {error}", file=sys.stderr)
                    mag_fresh = False
            if orientation_filter is None:
                orientation_filter = create_orientation_filter(args.algorithm, args.frequency)
                print("Initialized orientation filter", file=sys.stderr)

            if mag_ekf is not None:
                mag_ekf.dt = dt
                mag_ekf.predict(gyro_rad_s)

            calibrated_mag_ut: FloatArray | None = None
            if mag_fresh and raw_mag_ut is not None and mag_ekf is None:
                mag_ekf = MagGyroCalibrationEKF(
                    dt=dt, h_p0=raw_mag_ut, sigma_m=args.sigma_m,
                    sigma_g=sigma_g_rad_s, phi=args.phi,
                )
                print("Initialized magnetometer calibration filter", file=sys.stderr)
            if mag_fresh and raw_mag_ut is not None and mag_ekf is not None:
                mag_ekf.update(raw_mag_ut)
                calibrated_mag_ut = mag_ekf.x[0:3].copy()
                mag_updates += 1
            if processed:
                quaternion = update_orientation(
                    orientation_filter, quaternion, gyro_rad_s, accel_g,
                    calibrated_mag_ut, mag_fresh, dt,
                )
            box.update(quaternion)
            sensor_history_plot.append(
                timestamp_ms / 1000.0, accel_g, gyro_dps, calibrated_mag_ut
            )
            processed += 1
    finally:
        if csv_file is not None:
            csv_file.close()
        print(f"Processed {processed} valid IMU samples ({mag_updates} fresh mag updates).", file=sys.stderr)


def parse_arguments() -> RuntimeArguments:
    """Parse and validate the command-line options for the live runner."""
    parser = argparse.ArgumentParser(description="Live mag calibration and AHRS orientation display.")
    parser.add_argument(
        "source",
        help="IMU SSE URL or CSV recorded by imu_stream_to_csv.py",
    )
    parser.add_argument("--calibration", type=Path,
                        help="optional JSON calibration containing accel_bias_g and gyro_bias_dps")
    parser.add_argument("--algorithm", choices=("ekf", "madgwick"), default="ekf",
                        help="AHRS algorithm to use (default: ekf)")
    parser.add_argument("--record", type=Path, help="optionally append raw stream samples to this CSV")
    parser.add_argument("--frequency", type=float, default=100.0,
                        help="nominal stream frequency in Hz (default: 100)")
    parser.add_argument("--reconnect-delay", type=float, default=1.0,
                        help="seconds to wait before reconnecting (default: 1)")
    parser.add_argument("--sigma-m", type=float, default=0.5,
                        help="magnetometer noise standard deviation in uT (default: 0.5)")
    parser.add_argument("--sigma-g", type=float, default=0.1,
                        help="gyro noise standard deviation in deg/s (default: 0.1)")
    parser.add_argument("--phi", type=float, default=10.0,
                        help="mag calibration EKF gyro-noise scale factor (default: 10)")
    args = parser.parse_args()
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.reconnect_delay < 0 or args.sigma_m < 0 or args.sigma_g < 0 or args.phi < 0:
        parser.error("noise values, --phi, and --reconnect-delay must be non-negative")
    return RuntimeArguments(
        source=args.source, calibration=args.calibration, algorithm=args.algorithm,
        record=args.record, frequency=args.frequency, reconnect_delay=args.reconnect_delay,
        sigma_m=args.sigma_m, sigma_g=args.sigma_g, phi=args.phi
    )


def main() -> None:
    """Run the real-time orientation application from command-line settings."""
    args = parse_arguments()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
