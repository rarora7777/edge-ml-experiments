from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import tensorflow as tf


def _load_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(onnx_path), load_external_data=True)
    return {
        initializer.name: numpy_helper.to_array(initializer).astype(np.float32)
        for initializer in model.graph.initializer
    }


def build_keras_model_from_onnx(onnx_path: Path) -> tf.keras.Model:
    weights = _load_initializers(onnx_path)

    inputs = tf.keras.Input(shape=(100, 6), name="imu_signal")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool2")(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, name="pool3")(x)
    x = tf.keras.layers.Permute((2, 1), name="flatten_order_fix")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc1")(x)
    outputs = tf.keras.layers.Dense(2, name="fc2")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="imu_classifier_1dcnn")

    model.get_layer("conv1").set_weights([
        np.transpose(weights["features.0.weight"], (2, 1, 0)),
        weights["features.0.bias"],
    ])
    model.get_layer("conv2").set_weights([
        np.transpose(weights["features.4.weight"], (2, 1, 0)),
        weights["features.4.bias"],
    ])
    model.get_layer("conv3").set_weights([
        np.transpose(weights["features.8.weight"], (2, 1, 0)),
        weights["features.8.bias"],
    ])
    model.get_layer("fc1").set_weights([
        weights["classifier.0.weight"].T,
        weights["classifier.0.bias"],
    ])
    model.get_layer("fc2").set_weights([
        weights["classifier.3.weight"].T,
        weights["classifier.3.bias"],
    ])
    return model
