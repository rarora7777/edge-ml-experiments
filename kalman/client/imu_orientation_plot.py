#!/usr/bin/env python3
"""Estimate and plot IMU orientation from a CSV recorded by imu_stream_to_csv.py.

Examples:
    python3 imu_orientation_plot.py imu_samples.csv
    python3 imu_orientation_plot.py imu_samples.csv --algorithm madgwick \
        --calibration imu_calibration.json --save orientation.png

Install dependencies first:
    python3 -m pip install ahrs numpy matplotlib
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ahrs.filters import EKF, Madgwick


SENSOR_COLUMNS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
MAG_COLUMNS = ("mag_x", "mag_y", "mag_z")


def as_bool(value: str) -> bool:
    """Converts truthy strings to boolean True and other strings to False"""
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamps, accel, gyro, mag, and the fresh-mag flag from a recording."""
    rows = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(SENSOR_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            if not as_bool(row.get("valid", "true")):
                continue
            try:
                accel = [float(row[column]) for column in SENSOR_COLUMNS[:3]]
                gyro = [float(row[column]) for column in SENSOR_COLUMNS[3:6]]
                timestamp_ms = float(row["timestamp_ms"])
                mag = [float(row[column]) for column in MAG_COLUMNS]
            except (KeyError, TypeError, ValueError):
                # Samples without a magnetometer value are still usable by the
                # IMU-only AHRS update.
                try:
                    timestamp_ms = float(row["timestamp_ms"])
                    accel = [float(row[column]) for column in SENSOR_COLUMNS[:3]]
                    gyro = [float(row[column]) for column in SENSOR_COLUMNS[3:]]
                    mag = [np.nan, np.nan, np.nan]
                except (KeyError, TypeError, ValueError):
                    continue
            mag_fresh = as_bool(row.get("mag_fresh", row.get("mag_available", "true")))
            mag_fresh = mag_fresh and np.all(np.isfinite(mag))
            rows.append((timestamp_ms, accel, gyro, mag, mag_fresh))

    if len(rows) < 2:
        raise ValueError("recording needs at least two valid IMU samples")
    timestamps, accel, gyro, mag, mag_fresh = zip(*rows)
    return (np.asarray(timestamps), np.asarray(accel), np.asarray(gyro), np.asarray(mag),
            np.asarray(mag_fresh, dtype=bool))


def load_calibration(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Load this repository's accel_bias_g and gyro_bias_dps JSON fields."""
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


def sample_intervals(timestamps_ms: np.ndarray, default_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a monotonic plotting time vector and robust per-sample intervals."""
    dt = np.diff(timestamps_ms, prepend=timestamps_ms[0]) / 1000.0
    valid_dt = dt[(dt > 0.0) & (dt < 1.0)]
    nominal_dt = float(np.median(valid_dt)) if len(valid_dt) else default_dt
    dt[0] = nominal_dt
    dt[(dt <= 0.0) | (dt >= 1.0)] = nominal_dt
    return np.cumsum(dt) - dt[0], dt


def quaternion_to_world_axes(quaternions: np.ndarray) -> np.ndarray:
    """Return body X/Y/Z unit vectors expressed in the filter's world frame."""
    axes = np.empty((len(quaternions), 3, 3))
    for index, quaternion in enumerate(quaternions):
        w, x, y, z = quaternion / np.linalg.norm(quaternion)
        # This rotation maps IMU/body vectors into the AHRS world frame. Its
        # columns are the IMU X, Y, and Z axes, respectively, in world coords.
        axes[index] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return axes


def estimate_orientation(algorithm: str, accel_g: np.ndarray, gyro_dps: np.ndarray,
                         mag_ut: np.ndarray, mag_fresh: np.ndarray,
                         dt: np.ndarray) -> np.ndarray:
    """Run the selected AHRS filter and return scalar-first quaternions."""
    frequency = 1.0 / float(np.median(dt))
    # EKF fixes its measurement dimension at construction. The placeholder
    # selects its six-element accel+mag MARG observation model.
    orientation_filter = (
        EKF(frequency=frequency, mag=np.ones((1, 3)), noises=[4e-3*4e-3*frequency, 1e-8*frequency, 0.6*0.6])
        if algorithm == "ekf" else Madgwick(frequency=frequency)
    )
    quaternions = np.empty((len(accel_g), 4))
    quaternions[0] = np.array([1.0, 0.0, 0.0, 0.0])
    gyro_rad_s = np.deg2rad(gyro_dps)

    for index in range(1, len(accel_g)):
        if isinstance(orientation_filter, EKF):
            orientation_filter.mag = mag_ut[index] if mag_fresh[index] else None
            if mag_fresh[index]:
                quaternions[index] = orientation_filter.update(
                    quaternions[index - 1], gyro_rad_s[index], accel_g[index],
                    mag=mag_ut[index], dt=float(dt[index]))
            else:
                quaternions[index] = orientation_filter.update(
                    quaternions[index - 1], gyro_rad_s[index], accel_g[index], dt=float(dt[index]))
        elif mag_fresh[index]:
            quaternions[index] = orientation_filter.updateMARG(
                quaternions[index - 1], gyro_rad_s[index], accel_g[index], mag_ut[index],
                dt=float(dt[index]))
        else:
            quaternions[index] = orientation_filter.updateIMU(
                quaternions[index - 1], gyro_rad_s[index], accel_g[index], dt=float(dt[index]))
    return quaternions


def plot_results(time_s: np.ndarray, accel_g: np.ndarray, gyro_dps: np.ndarray,
                 mag_ut: np.ndarray, world_axes: np.ndarray, algorithm: str,
                 mag_was_flipped: bool, output: Path | None) -> None:
    """Plot recorded inputs and the IMU body axes represented in world coordinates."""
    figure, plots = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    input_series = (("Raw accelerometer", accel_g, "g"),
                    ("Raw gyroscope", gyro_dps, "deg/s"),
                    ("Raw magnetometer", mag_ut, "uT"))
    axis_names = ("IMU X axis in world frame", "IMU Y axis in world frame", "IMU Z axis in world frame")
    colors = ("tab:red", "tab:green", "tab:blue")

    for plot, (title, values, units) in zip(plots[0], input_series):
        for component, color, label in zip(range(3), colors, ("X", "Y", "Z")):
            plot.plot(time_s, values[:, component], color=color, label=label, linewidth=0.8)
        plot.set_title(title)
        plot.set_ylabel(units)
        plot.grid(True)
        plot.legend()

    for plot, axis_index, title in zip(plots[1], range(3), axis_names):
        for component, color, label in zip(range(3), colors, ("world X", "world Y", "world Z")):
            plot.plot(time_s, world_axes[:, component, axis_index], color=color, label=label, linewidth=0.9)
        plot.set_title(title)
        plot.set_xlabel("elapsed time (s)")
        plot.set_ylabel("unit-vector component")
        plot.set_ylim(-1.05, 1.05)
        plot.grid(True)
        plot.legend()

    flip_description = "mag X/Z flipped" if mag_was_flipped else "mag axes unmodified"
    figure.suptitle(f"{algorithm.upper()} orientation estimate ({flip_description})")
    figure.tight_layout()
    if output is not None:
        figure.savefig(output, dpi=160)
        print(f"Wrote plot to {output}", file=sys.stderr)
    else:
        plt.show()


def main() -> None:
    """Main entrypoint for the script"""
    parser = argparse.ArgumentParser(description="Run AHRS EKF or Madgwick on recorded IMU CSV data.")
    parser.add_argument("input", type=Path, help="CSV created by imu_stream_to_csv.py")
    parser.add_argument("--algorithm", choices=("ekf", "madgwick"), default="ekf",
                        help="AHRS algorithm to use (default: ekf)")
    parser.add_argument("--calibration", type=Path,
                        help="optional JSON calibration containing accel_bias_g and gyro_bias_dps")
    parser.add_argument("--flip-mag-xz", action="store_true",
                        help="flip magnetometer X/Z for legacy recordings made before the firmware fix")
    parser.add_argument("--save", type=Path, help="save the plot instead of opening a window")
    args = parser.parse_args()

    try:
        timestamps_ms, raw_accel_g, raw_gyro_dps, raw_mag_ut, mag_fresh = load_recording(args.input)
        accel_bias, gyro_bias = load_calibration(args.calibration)
        time_s, dt = sample_intervals(timestamps_ms, default_dt=0.01)
        corrected_accel_g = raw_accel_g - accel_bias
        corrected_gyro_dps = raw_gyro_dps - gyro_bias
        filter_mag_ut = raw_mag_ut.copy()
        if args.flip_mag_xz:
            filter_mag_ut[:, (0, 2)] *= -1.0
        quaternions = estimate_orientation(args.algorithm, corrected_accel_g,
                                           corrected_gyro_dps, filter_mag_ut, mag_fresh, dt)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Processed {len(time_s)} valid samples ({mag_fresh.sum()} fresh mag); median period is {np.median(dt) * 1000:.2f} ms",
          file=sys.stderr)
    plot_results(time_s, raw_accel_g, raw_gyro_dps, raw_mag_ut,
                 quaternion_to_world_axes(quaternions), args.algorithm,
                 args.flip_mag_xz, args.save)


if __name__ == "__main__":
    main()
