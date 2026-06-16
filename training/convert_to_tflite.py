#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf

from imu_model_pipeline import (
    CLASS_LABELS,
    build_dataset_bundle,
    discover_csv_files,
    save_normalization,
)
from onnx_imu_keras import build_keras_model_from_onnx


def representative_dataset(samples: np.ndarray):
    for sample in samples:
        yield [np.expand_dims(sample, axis=0).astype(np.float32)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the ONNX IMU model to int8 TFLite.")
    parser.add_argument("--onnx", default="training/imu_classifier_1dcnn.onnx", help="Path to the source ONNX model.")
    parser.add_argument("--data-dir", default="data", help="Directory containing labeled CSV capture sessions.")
    parser.add_argument("--output-dir", default="training/artifacts", help="Where to write conversion outputs.")
    parser.add_argument("--representative-samples", type=int, default=200, help="Calibration windows to use for int8 quantization.")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(data_dir)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    bundle = build_dataset_bundle(csv_files)
    stats_path = output_dir / "imu_classifier_1dcnn_normalization.npz"
    save_normalization(stats_path, bundle.mean, bundle.std)

    rep_count = min(args.representative_samples, len(bundle.X_train))
    rep_samples = np.transpose(bundle.X_train[:rep_count], (0, 2, 1))

    keras_model = build_keras_model_from_onnx(onnx_path)
    keras_sample = np.transpose(bundle.X_val[:8], (0, 2, 1))
    keras_logits = keras_model.predict(keras_sample, verbose=0)
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = ort_session.run(None, {ort_session.get_inputs()[0].name: bundle.X_val[:8]})[0]
    float_model_mae = float(np.mean(np.abs(ort_logits - keras_logits)))
    keras_path = output_dir / "imu_classifier_1dcnn.keras"
    keras_model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(rep_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = output_dir / "imu_classifier_1dcnn.int8.tflite"
    tflite_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    metadata = {
        "class_labels": CLASS_LABELS,
        "input_name": input_details["name"],
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "input_quantization": {
            "scale": float(input_details["quantization"][0]),
            "zero_point": int(input_details["quantization"][1]),
        },
        "output_name": output_details["name"],
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "output_quantization": {
            "scale": float(output_details["quantization"][0]),
            "zero_point": int(output_details["quantization"][1]),
        },
        "normalization_path": str(stats_path),
        "keras_path": str(keras_path),
        "representative_sample_count": rep_count,
        "validation_window_count": int(len(bundle.X_val)),
        "tflite_size_bytes": tflite_path.stat().st_size,
        "sanity_logits_shape": list(keras_logits.shape),
        "keras_vs_onnx_mae_first8": float_model_mae,
    }
    metadata_path = output_dir / "imu_classifier_1dcnn.int8.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Saved int8 TFLite model to: {tflite_path}")
    print(f"Saved Keras reconstruction to: {keras_path}")
    print(f"Saved normalization stats to: {stats_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Keras vs ONNX MAE (first 8 val windows): {float_model_mae:.8f}")
    print(f"Input shape: {metadata['input_shape']}  quant={metadata['input_quantization']}")
    print(f"Output shape: {metadata['output_shape']} quant={metadata['output_quantization']}")
    print(f"TFLite size: {metadata['tflite_size_bytes']} bytes")


if __name__ == "__main__":
    main()
