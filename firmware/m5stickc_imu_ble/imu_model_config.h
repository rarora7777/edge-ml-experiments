#pragma once
#include <stdint.h>

enum ImuInputLayout : uint8_t {
  kImuInputLayoutNCW = 0,
  kImuInputLayoutNWC = 1,
  kImuInputLayoutNCWH = 2,
  kImuInputLayoutNWCH = 3,
};

constexpr bool kImuModelAvailable = true;
constexpr int kImuModelWindowSize = 100;
constexpr int kImuModelFeatureCount = 6;
constexpr int kImuModelClassCount = 2;
constexpr int kImuTensorArenaSize = 98304;
constexpr ImuInputLayout kImuModelInputLayout = kImuInputLayoutNWC;
constexpr int kImuModelInputDims[3] = {1, 100, 6};
constexpr float kImuModelMean[kImuModelFeatureCount] = {0.0969831347f, 0.373709321f, 0.345678747f, 27.2570133f, -22.9703827f, -24.829401f};
constexpr float kImuModelStd[kImuModelFeatureCount] = {0.65162015f, 0.713029146f, 0.4902713f, 141.189941f, 142.652451f, 118.285362f};
constexpr float kImuModelInputScale = 0.0750905126f;
constexpr int kImuModelInputZeroPoint = 23;
constexpr float kImuModelOutputScale = 0.155952126f;
constexpr int kImuModelOutputZeroPoint = -2;
constexpr const char* kImuModelLabels[kImuModelClassCount] = {"hand", "pocket"};
