#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf

from imu_model_pipeline import build_dataset_bundle, discover_csv_files


def quantize_input(sample: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round(sample / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def dequantize_output(sample: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (sample.astype(np.float32) - zero_point) * scale


def reshape_for_tflite(sample: np.ndarray, input_shape: np.ndarray) -> np.ndarray:
    shape = input_shape.tolist()
    if shape == [1, 6, 100]:
        return np.expand_dims(sample, axis=0)
    if shape == [1, 100, 6]:
        return np.expand_dims(np.transpose(sample, (1, 0)), axis=0)
    if shape == [1, 6, 100, 1]:
        return np.expand_dims(np.expand_dims(sample, axis=-1), axis=0)
    if shape == [1, 100, 6, 1]:
        return np.expand_dims(np.expand_dims(np.transpose(sample, (1, 0)), axis=-1), axis=0)
    raise ValueError(f"Unsupported TFLite input shape: {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ONNX and int8 TFLite validation results.")
    parser.add_argument("--onnx", default="training/imu_classifier_1dcnn.onnx", help="Path to the source ONNX model.")
    parser.add_argument("--tflite", default="training/artifacts/imu_classifier_1dcnn.int8.tflite", help="Path to the int8 TFLite model.")
    parser.add_argument("--data-dir", default="data", help="Directory containing labeled CSV capture sessions.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of validation windows to evaluate.")
    args = parser.parse_args()

    csv_files = discover_csv_files(Path(args.data_dir).resolve())
    bundle = build_dataset_bundle(csv_files)
    X_val = bundle.X_val
    y_val = bundle.y_val

    if args.limit > 0:
        X_val = X_val[: args.limit]
        y_val = y_val[: args.limit]

    ort_session = ort.InferenceSession(str(Path(args.onnx).resolve()), providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name
    ort_logits = ort_session.run(None, {ort_input_name: X_val.astype(np.float32)})[0]
    ort_pred = ort_logits.argmax(axis=1)

    interpreter = tf.lite.Interpreter(model_path=str(Path(args.tflite).resolve()))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    tflite_logits = []
    for sample in X_val:
        sample_batch = reshape_for_tflite(sample, input_details["shape"])
        q_input = quantize_input(sample_batch, float(in_scale), int(in_zero_point))
        interpreter.set_tensor(input_details["index"], q_input)
        interpreter.invoke()
        q_output = interpreter.get_tensor(output_details["index"])
        tflite_logits.append(dequantize_output(q_output, float(out_scale), int(out_zero_point))[0])

    tflite_logits = np.asarray(tflite_logits, dtype=np.float32)
    tflite_pred = tflite_logits.argmax(axis=1)

    onnx_acc = float((ort_pred == y_val).mean())
    tflite_acc = float((tflite_pred == y_val).mean())
    agreement = float((ort_pred == tflite_pred).mean())
    logit_mae = float(np.mean(np.abs(ort_logits - tflite_logits)))

    print(f"Validation windows: {len(X_val)}")
    print(f"ONNX accuracy:    {onnx_acc:.4f}")
    print(f"TFLite accuracy:  {tflite_acc:.4f}")
    print(f"Prediction match: {agreement:.4f}")
    print(f"Mean abs logit diff: {logit_mae:.6f}")


if __name__ == "__main__":
    main()
