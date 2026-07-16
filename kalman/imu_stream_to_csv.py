#!/usr/bin/env python3
"""Record IMU server-sent events from an M5 tag to a CSV file.

Example:
    python3 imu_stream_to_csv.py http://192.168.1.42/imu/stream imu.csv
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request


CSV_COLUMNS = (
    "received_unix_s",
    "timestamp_ms",
    "valid",
    "mag_available",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
)


def sample_row(sample: dict) -> dict:
    """Convert an IMU SSE JSON event into one CSV row."""
    accel = sample.get("accel", {})
    gyro = sample.get("gyro", {})
    mag = sample.get("mag", {})
    return {
        "received_unix_s": f"{time.time():.6f}",
        "timestamp_ms": sample.get("timestamp_ms", ""),
        "valid": sample.get("valid", False),
        "mag_available": sample.get("mag_available", False),
        "accel_x": accel.get("x", ""),
        "accel_y": accel.get("y", ""),
        "accel_z": accel.get("z", ""),
        "gyro_x": gyro.get("x", ""),
        "gyro_y": gyro.get("y", ""),
        "gyro_z": gyro.get("z", ""),
        "mag_x": mag.get("x", ""),
        "mag_y": mag.get("y", ""),
        "mag_z": mag.get("z", ""),
    }


def record(url: str, output_path: str, reconnect_delay: float) -> None:
    """Append IMU events from *url* to *output_path* until interrupted."""
    with open(output_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if csv_file.tell() == 0:
            writer.writeheader()
            csv_file.flush()

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
                        writer.writerow(sample_row(sample))
                        csv_file.flush()
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                print(f"Stream disconnected ({error}); retrying in {reconnect_delay:g}s", file=sys.stderr)
                time.sleep(reconnect_delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record /imu/stream SSE events to CSV.")
    parser.add_argument("url", help="IMU stream URL, e.g. http://192.168.1.42/imu/stream")
    parser.add_argument("output", help="CSV file to create or append")
    parser.add_argument("--reconnect-delay", type=float, default=1.0,
                        help="seconds to wait before reconnecting (default: 1)")
    args = parser.parse_args()
    if args.reconnect_delay < 0:
        parser.error("--reconnect-delay must be non-negative")

    try:
        record(args.url, args.output, args.reconnect_delay)
    except KeyboardInterrupt:
        print("\nStopped recording.", file=sys.stderr)


if __name__ == "__main__":
    main()
