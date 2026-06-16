#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from imu_model_pipeline import CLASS_LABELS, load_normalization


def detect_layout(shape: list[int]) -> str:
    if shape == [1, 6, 100]:
        return "kImuInputLayoutNCW"
    if shape == [1, 100, 6]:
        return "kImuInputLayoutNWC"
    if shape == [1, 6, 100, 1]:
        return "kImuInputLayoutNCWH"
    if shape == [1, 100, 6, 1]:
        return "kImuInputLayoutNWCH"
    raise ValueError(f"Unsupported TFLite input shape for firmware export: {shape}")


def format_float_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(value):.9g}f" for value in values.tolist())


def write_model_data_header(path: Path, model_bytes: bytes) -> None:
    hex_bytes = ", ".join(f"0x{byte:02x}" for byte in model_bytes)
    path.write_text(
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        f"alignas(16) const unsigned char g_imu_model_tflite[] = {{{hex_bytes}}};\n"
        f"constexpr unsigned int g_imu_model_tflite_len = {len(model_bytes)};\n"
    )


def write_model_config_header(
    path: Path,
    input_shape: list[int],
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    layout_name = detect_layout(input_shape)
    mean_values = mean.reshape(-1)
    std_values = std.reshape(-1)
    path.write_text(
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        "enum ImuInputLayout : uint8_t {\n"
        "  kImuInputLayoutNCW = 0,\n"
        "  kImuInputLayoutNWC = 1,\n"
        "  kImuInputLayoutNCWH = 2,\n"
        "  kImuInputLayoutNWCH = 3,\n"
        "};\n\n"
        "constexpr bool kImuModelAvailable = true;\n"
        "constexpr int kImuModelWindowSize = 100;\n"
        "constexpr int kImuModelFeatureCount = 6;\n"
        "constexpr int kImuModelClassCount = 2;\n"
        "constexpr int kImuTensorArenaSize = 163840;\n"
        f"constexpr ImuInputLayout kImuModelInputLayout = {layout_name};\n"
        f"constexpr int kImuModelInputDims[{len(input_shape)}] = {{{', '.join(str(v) for v in input_shape)}}};\n"
        f"constexpr float kImuModelMean[kImuModelFeatureCount] = {{{format_float_array(mean_values)}}};\n"
        f"constexpr float kImuModelStd[kImuModelFeatureCount] = {{{format_float_array(std_values)}}};\n"
        f"constexpr float kImuModelInputScale = {input_scale:.9g}f;\n"
        f"constexpr int kImuModelInputZeroPoint = {input_zero_point};\n"
        f"constexpr float kImuModelOutputScale = {output_scale:.9g}f;\n"
        f"constexpr int kImuModelOutputZeroPoint = {output_zero_point};\n"
        f'constexpr const char* kImuModelLabels[kImuModelClassCount] = {{"{CLASS_LABELS[0]}", "{CLASS_LABELS[1]}"}};\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Arduino headers from the quantized TFLite model.")
    parser.add_argument("--tflite", default="training/artifacts/imu_classifier_1dcnn.int8.tflite", help="Path to the int8 TFLite model.")
    parser.add_argument("--normalization", default="training/artifacts/imu_classifier_1dcnn_normalization.npz", help="Saved mean/std stats.")
    parser.add_argument("--output-dir", default="firmware/m5stickc_imu_ble", help="Where to write generated Arduino headers.")
    args = parser.parse_args()

    tflite_path = Path(args.tflite).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = load_normalization(Path(args.normalization).resolve())

    model_bytes = tflite_path.read_bytes()
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"].tolist()
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    write_model_data_header(output_dir / "imu_model_data.h", model_bytes)
    write_model_config_header(
        output_dir / "imu_model_config.h",
        input_shape=input_shape,
        input_scale=float(input_scale),
        input_zero_point=int(input_zero_point),
        output_scale=float(output_scale),
        output_zero_point=int(output_zero_point),
        mean=mean,
        std=std,
    )
    print(f"Wrote firmware model assets to: {output_dir}")


if __name__ == "__main__":
    main()
