from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 100
STEP_SIZE = 50
SAMPLE_RATE_HZ = 50.0
FEATURE_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
CLASS_LABELS = ["hand", "pocket"]


@dataclass
class DatasetBundle:
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray


def discover_csv_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.csv") if path.is_file())


def preprocess_and_segment(
    file_list: list[Path],
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for filepath in file_list:
        df = pd.read_csv(filepath)
        if len(df) < window_size:
            continue

        t_raw = df["timestamp_ms"].to_numpy(dtype=np.float32) / 1000.0
        t_raw = t_raw - t_raw[0]
        t_uniform = np.arange(0, t_raw[-1], 1.0 / SAMPLE_RATE_HZ, dtype=np.float32)

        resampled_data: dict[str, np.ndarray] = {}
        for col in FEATURE_COLUMNS:
            f_interp = interp1d(t_raw, df[col].to_numpy(dtype=np.float32), kind="linear", fill_value="extrapolate")
            resampled_data[col] = f_interp(t_uniform).astype(np.float32)

        f_label = interp1d(
            t_raw,
            df["label"].to_numpy(dtype=np.float32),
            kind="nearest",
            fill_value="extrapolate",
        )
        labels_uniform = f_label(t_uniform).astype(np.int64)

        for start in range(0, len(t_uniform) - window_size + 1, step_size):
            end = start + window_size
            segment = np.stack(
                [resampled_data[col][start:end] for col in FEATURE_COLUMNS],
                axis=0,
            ).astype(np.float32)
            majority_label = int(np.bincount(labels_uniform[start:end]).argmax())
            X_list.append(segment)
            y_list.append(majority_label)

    if not X_list:
        raise ValueError("No usable IMU windows were extracted from the provided CSV files.")

    return np.stack(X_list).astype(np.float32), np.asarray(y_list, dtype=np.int64)


def compute_normalization(X_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train_raw.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X_train_raw.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def normalize_segments(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def build_dataset_bundle(
    file_list: list[Path],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    X, y = preprocess_and_segment(file_list)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    mean, std = compute_normalization(X_train_raw)
    X_train = normalize_segments(X_train_raw, mean, std)
    X_val = normalize_segments(X_val_raw, mean, std)
    return DatasetBundle(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        y_train=y_train,
        y_val=y_val,
        mean=mean,
        std=std,
        X_train=X_train,
        X_val=X_val,
    )


def save_normalization(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32))


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)
