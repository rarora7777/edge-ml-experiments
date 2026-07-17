"""Tests for mixed-rate recorded-data handling in ekf_imu_mag.py."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from client import ekf_imu_mag


CSV_COLUMNS = (
    "received_unix_s", "timestamp_ms", "valid", "mag_available", "mag_fresh",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
)


class MixedRateMagEkfTest(unittest.TestCase):
    def test_predicts_at_imu_rate_and_updates_at_mag_rate(self):
        """A 100 Hz gyro stream must not re-use a 10 Hz mag measurement."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for index in range(11):
                    writer.writerow({
                        "received_unix_s": f"{index * 0.01:.3f}",
                        "timestamp_ms": index * 10,
                        "valid": "True",
                        "mag_available": "True",
                        "mag_fresh": "True" if index in (0, 10) else "False",
                        "accel_x": 0, "accel_y": 0, "accel_z": 1,
                        "gyro_x": 0, "gyro_y": 0, "gyro_z": 0,
                        "mag_x": 20, "mag_y": 5, "mag_z": -45,
                    })

            predict_calls = 0
            update_calls = 0
            original_predict = ekf_imu_mag.MagGyroCalibrationEKF.predict
            original_update = ekf_imu_mag.MagGyroCalibrationEKF.update

            def count_predict(filter_instance, gyro):
                nonlocal predict_calls
                predict_calls += 1
                return original_predict(filter_instance, gyro)

            def count_update(filter_instance, mag):
                nonlocal update_calls
                update_calls += 1
                return original_update(filter_instance, mag)

            with patch.object(ekf_imu_mag.MagGyroCalibrationEKF, "predict", count_predict), \
                 patch.object(ekf_imu_mag.MagGyroCalibrationEKF, "update", count_update), \
                 patch.object(ekf_imu_mag.plt, "show"):
                ekf_imu_mag.run_recording(str(input_path), calibration_filename=None)

            # The first fresh sample initializes the filter. Every subsequent
            # 100 Hz gyro sample predicts; only the two fresh samples update.
            self.assertEqual(predict_calls, 10)
            self.assertEqual(update_calls, 2)

            output_path = Path(temporary_directory) / "input_magcal.csv"
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                output_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(output_rows), 11)
            self.assertEqual(output_rows[5]["mag_fresh"], "False")
            self.assertNotEqual(output_rows[10]["mag_x"], "")


if __name__ == "__main__":
    unittest.main()
