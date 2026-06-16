#include <M5Unified.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <math.h>
#include <stdarg.h>

#include "imu_model_config.h"
#include "imu_model_data.h"

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Custom BLE UUIDs (matching the client script)
#define SERVICE_UUID           "23a00000-0df2-432d-a23e-63f59e9a1122"
#define CHARACTERISTIC_UUID    "23a00001-0df2-432d-a23e-63f59e9a1122"

// Color Constants in RGB565 format
#define COLOR_BLACK  0x0000U
#define COLOR_WHITE  0xFFFFU
#define COLOR_RED    0xF800U
#define COLOR_GREEN  0x07E0U
#define COLOR_BLUE   0x001FU
#define COLOR_CYAN   0x07FFU
#define COLOR_ORANGE 0xFD20U
#define COLOR_YELLOW 0xFFE0U

enum DeviceState {
  STATE_DISABLED_BEFORE_HAND = 0,
  STATE_HAND = 1,
  STATE_DISABLED_BEFORE_POCKET = 2,
  STATE_POCKET = 3
};

enum VisualState {
  V_DISABLED_HAND,
  V_PREPARING_HAND,
  V_STREAMING_HAND,
  V_DISABLED_POCKET,
  V_PREPARING_POCKET,
  V_STREAMING_POCKET
};

enum AppMode {
  APP_STREAM = 0,
  APP_EVAL = 1
};

struct __attribute__((packed)) IMUPacket {
  uint32_t timestamp;
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
  uint8_t label;
};

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
uint32_t packetCount = 0;

DeviceState currentState = STATE_DISABLED_BEFORE_HAND;
VisualState currentVisualState = V_DISABLED_HAND;
VisualState lastVisualState = V_DISABLED_HAND;
AppMode currentAppMode = APP_STREAM;
AppMode lastAppMode = APP_STREAM;

unsigned long stateTransitionTime = 0;
const unsigned long startDelay = 1000;
unsigned long lastSampleTime = 0;
unsigned long lastEvalTime = 0;
const unsigned long sampleInterval = 20;
const unsigned long evalInterval = 200;
unsigned long lastDisplayTime = 0;
const unsigned long displayInterval = 100;

float evalWindow[kImuModelFeatureCount][kImuModelWindowSize] = {0};
int evalSampleCount = 0;
int evalPrediction = -1;
float evalConfidence = 0.0f;
float evalLogits[kImuModelClassCount] = {0};
unsigned long evalInferenceDurationMs = 0;
bool evalRuntimeReady = false;
bool evalModelPresent = false;
bool evalInferenceOkay = false;
const char* evalStatus = "NO MODEL";
uint16_t lastEvalBgColor = COLOR_BLACK;

namespace {
const tflite::Model* gModel = nullptr;
tflite::MicroMutableOpResolver<9> gResolver;
tflite::MicroInterpreter* gInterpreter = nullptr;
TfLiteTensor* gInputTensor = nullptr;
TfLiteTensor* gOutputTensor = nullptr;
uint8_t* gTensorArena = nullptr;
}

namespace {
uint32_t gBootCount = 0;
constexpr bool kDebugSkipInitialDisplay = false;
}

void logDebug(const char* message) {
  Serial.println(message);
  Serial.flush();
}

void logDebugf(const char* fmt, ...) {
  char buffer[160];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  Serial.println(buffer);
  Serial.flush();
}

void setLed(bool turnOn) {
  auto board = M5.getBoard();
  int targetPin = 10;
  bool activeLevel = LOW;

  if (board == m5::board_t::board_M5StickCPlus2) {
    targetPin = 19;
    activeLevel = HIGH;
  }

  pinMode(targetPin, OUTPUT);
  digitalWrite(targetPin, turnOn ? activeLevel : !activeLevel);
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
  }
};

void resetEvalState() {
  memset(evalWindow, 0, sizeof(evalWindow));
  evalSampleCount = 0;
  evalPrediction = -1;
  evalConfidence = 0.0f;
  evalLogits[0] = 0.0f;
  evalLogits[1] = 0.0f;
  evalInferenceDurationMs = 0;
  evalInferenceOkay = false;
}

void drawStreamLayout(VisualState state) {
  uint16_t bgColor = COLOR_BLACK;
  uint16_t textColor = COLOR_WHITE;

  if (state == V_PREPARING_HAND || state == V_PREPARING_POCKET) {
    bgColor = COLOR_YELLOW;
    textColor = COLOR_BLACK;
  } else if (state == V_STREAMING_HAND) {
    bgColor = COLOR_CYAN;
    textColor = COLOR_BLACK;
  } else if (state == V_STREAMING_POCKET) {
    bgColor = COLOR_ORANGE;
    textColor = COLOR_BLACK;
  }

  M5.Display.fillScreen(bgColor);
  M5.Display.setTextColor(textColor, bgColor);
  M5.Display.setTextSize(1.4);
  M5.Display.drawString("IMU BLE Stream", 8, 5);
  M5.Display.setTextSize(1.1);
  M5.Display.drawString("BLE :", 8, 25);
  M5.Display.drawString("Mode:", 8, 40);

  if (state == V_STREAMING_HAND || state == V_STREAMING_POCKET) {
    M5.Display.drawString("Pkts:", 8, 55);
    M5.Display.drawString("Hz  :", 8, 68);
  } else if (state == V_PREPARING_HAND || state == V_PREPARING_POCKET) {
    M5.Display.drawString("Time:", 8, 55);
  } else {
    M5.Display.drawString("Next:", 8, 55);
  }
}

void updateStreamDisplay(VisualState state) {
  uint16_t bgColor = COLOR_BLACK;
  uint16_t defaultTextColor = COLOR_WHITE;

  if (state == V_PREPARING_HAND || state == V_PREPARING_POCKET) {
    bgColor = COLOR_YELLOW;
    defaultTextColor = COLOR_BLACK;
  } else if (state == V_STREAMING_HAND) {
    bgColor = COLOR_CYAN;
    defaultTextColor = COLOR_BLACK;
  } else if (state == V_STREAMING_POCKET) {
    bgColor = COLOR_ORANGE;
    defaultTextColor = COLOR_BLACK;
  }

  M5.Display.setTextSize(1.1);
  M5.Display.setCursor(45, 25);
  if (deviceConnected) {
    M5.Display.setTextColor(COLOR_GREEN, bgColor);
    M5.Display.print("CONNECTED   ");
  } else {
    M5.Display.setTextColor(COLOR_RED, bgColor);
    M5.Display.print("ADVERTISING ");
  }

  M5.Display.setCursor(45, 40);
  M5.Display.setTextColor(defaultTextColor, bgColor);
  switch (state) {
    case V_DISABLED_HAND:
    case V_DISABLED_POCKET:
      M5.Display.print("COLLECT OFF ");
      break;
    case V_PREPARING_HAND:
      M5.Display.print("PREP HAND   ");
      break;
    case V_PREPARING_POCKET:
      M5.Display.print("PREP POCKET ");
      break;
    case V_STREAMING_HAND:
      M5.Display.print("WALK HAND   ");
      break;
    case V_STREAMING_POCKET:
      M5.Display.print("WALK POCKET ");
      break;
  }

  if (state == V_STREAMING_HAND || state == V_STREAMING_POCKET) {
    static uint32_t lastDisplayPacketCount = 0;
    static unsigned long lastRateCalculationTime = 0;
    unsigned long now = millis();

    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(defaultTextColor, bgColor);
    M5.Display.printf("%u      ", packetCount);

    if (now - lastRateCalculationTime >= 1000) {
      float hz = (packetCount - lastDisplayPacketCount) * 1000.0f / (now - lastRateCalculationTime);
      M5.Display.setCursor(45, 68);
      M5.Display.printf("%.1f    ", hz);
      lastDisplayPacketCount = packetCount;
      lastRateCalculationTime = now;
    }
  } else if (state == V_PREPARING_HAND || state == V_PREPARING_POCKET) {
    unsigned long elapsed = millis() - stateTransitionTime;
    float remaining = (startDelay - elapsed) / 1000.0f;
    if (remaining < 0.0f) remaining = 0.0f;
    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(defaultTextColor, bgColor);
    M5.Display.printf("START %.1fs ", remaining);
  } else {
    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(COLOR_YELLOW, bgColor);
    if (state == V_DISABLED_HAND) {
      M5.Display.print("A=HAND B=EVAL");
    } else {
      M5.Display.print("A=POCK B=EVAL");
    }
  }
}

void drawEvalLayout() {
  uint16_t bgColor = lastEvalBgColor;
  uint16_t textColor = (bgColor == COLOR_YELLOW) ? COLOR_BLACK : COLOR_WHITE;

  M5.Display.fillScreen(bgColor);
  M5.Display.setTextColor(textColor, bgColor);
  M5.Display.setTextSize(1.4);
  M5.Display.drawString("IMU Eval", 8, 5);
  M5.Display.setTextSize(1.1);
  M5.Display.drawString("Stat:", 8, 25);
  M5.Display.drawString("Buf :", 8, 40);
  M5.Display.drawString("Pred:", 8, 55);
  M5.Display.drawString("Conf:", 8, 68);
  M5.Display.drawString("Inf :", 8, 81);
}

uint16_t currentEvalBackgroundColor() {
  if (!evalRuntimeReady) {
    return COLOR_RED;
  }
  if (evalSampleCount < kImuModelWindowSize) {
    return COLOR_BLUE;
  }
  if (!evalInferenceOkay || evalPrediction < 0) {
    return COLOR_RED;
  }
  if (evalConfidence < 0.60f) {
    return COLOR_YELLOW;
  }
  return (evalPrediction == 0) ? COLOR_GREEN : COLOR_ORANGE;
}

void updateEvalDisplay() {
  uint16_t bgColor = currentEvalBackgroundColor();
  uint16_t textColor = (bgColor == COLOR_YELLOW) ? COLOR_BLACK : COLOR_WHITE;

  if (bgColor != lastEvalBgColor) {
    lastEvalBgColor = bgColor;
    drawEvalLayout();
  }

  M5.Display.setTextSize(1.1);
  M5.Display.setTextColor(textColor, bgColor);

  M5.Display.setCursor(45, 25);
  M5.Display.print("            ");
  M5.Display.setCursor(45, 25);
  M5.Display.print(evalStatus);

  M5.Display.setCursor(45, 40);
  M5.Display.printf("%3d/%3d    ", evalSampleCount, kImuModelWindowSize);

  M5.Display.setCursor(45, 55);
  if (evalPrediction >= 0 && evalPrediction < kImuModelClassCount) {
    M5.Display.print(kImuModelLabels[evalPrediction]);
    M5.Display.print("      ");
  } else {
    M5.Display.print("--        ");
  }

  M5.Display.setCursor(45, 68);
  M5.Display.printf("%3d%%      ", (int)(evalConfidence * 100.0f));

  M5.Display.setCursor(45, 81);
  M5.Display.printf("%2lums      ", evalInferenceDurationMs);
}

void switchAppMode(AppMode newMode) {
  currentAppMode = newMode;
  lastAppMode = (AppMode)(1 - newMode);

  if (currentAppMode == APP_STREAM) {
    currentState = STATE_DISABLED_BEFORE_HAND;
    currentVisualState = V_DISABLED_HAND;
    lastVisualState = (VisualState)(-1);
    packetCount = 0;
  } else {
    resetEvalState();
    lastEvalBgColor = currentEvalBackgroundColor();
  }
}

void setupEvalRuntime() {
  logDebug("setupEvalRuntime: begin");
  evalModelPresent = kImuModelAvailable && g_imu_model_tflite_len > 0;
  logDebugf("setupEvalRuntime: model_present=%d model_len=%u", evalModelPresent ? 1 : 0, (unsigned)g_imu_model_tflite_len);
  if (!evalModelPresent) {
    evalStatus = "NO MODEL";
    evalRuntimeReady = false;
    logDebug("setupEvalRuntime: no model");
    return;
  }

  logDebug("setupEvalRuntime: registering ops");
  gResolver.AddExpandDims();
  gResolver.AddConv2D();
  gResolver.AddReshape();
  gResolver.AddMaxPool2D();
  gResolver.AddTranspose();
  gResolver.AddShape();
  gResolver.AddStridedSlice();
  gResolver.AddPack();
  gResolver.AddFullyConnected();

  logDebug("setupEvalRuntime: parsing model");
  gModel = tflite::GetModel(g_imu_model_tflite);
  if (gModel->version() != TFLITE_SCHEMA_VERSION) {
    evalStatus = "SCHEMA ERR";
    evalRuntimeReady = false;
    logDebugf("setupEvalRuntime: schema mismatch model=%d expected=%d", gModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  size_t largestBlock = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  logDebugf("setupEvalRuntime: allocating arena bytes=%u free_heap=%u max_alloc=%u", (unsigned)kImuTensorArenaSize, (unsigned)ESP.getFreeHeap(), (unsigned)largestBlock);
  gTensorArena = static_cast<uint8_t*>(heap_caps_malloc(kImuTensorArenaSize, MALLOC_CAP_8BIT));
  if (gTensorArena == nullptr) {
    evalStatus = "ARENA FAIL";
    evalRuntimeReady = false;
    logDebugf("setupEvalRuntime: arena allocation failed max_alloc=%u", (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    return;
  }
  logDebugf("setupEvalRuntime: arena_ptr=%p free_heap=%u", gTensorArena, (unsigned)ESP.getFreeHeap());

  logDebug("setupEvalRuntime: creating interpreter");
  gInterpreter = new tflite::MicroInterpreter(gModel, gResolver, gTensorArena, kImuTensorArenaSize);
  if (gInterpreter == nullptr || gInterpreter->AllocateTensors() != kTfLiteOk) {
    evalStatus = "ALLOC FAIL";
    evalRuntimeReady = false;
    logDebugf("setupEvalRuntime: AllocateTensors failed interpreter=%p free_heap=%u", gInterpreter, (unsigned)ESP.getFreeHeap());
    return;
  }

  gInputTensor = gInterpreter->input(0);
  gOutputTensor = gInterpreter->output(0);
  if (gInputTensor == nullptr || gOutputTensor == nullptr || gInputTensor->type != kTfLiteInt8 || gOutputTensor->type != kTfLiteInt8) {
    evalStatus = "INT8 ONLY";
    evalRuntimeReady = false;
    logDebugf("setupEvalRuntime: tensor type mismatch in=%d out=%d", gInputTensor ? gInputTensor->type : -1, gOutputTensor ? gOutputTensor->type : -1);
    return;
  }

  evalStatus = "READY";
  evalRuntimeReady = true;
  logDebugf("setupEvalRuntime: ready input_bytes=%d output_bytes=%d free_heap=%u", gInputTensor->bytes, gOutputTensor->bytes, (unsigned)ESP.getFreeHeap());
}

void appendEvalSample(float ax, float ay, float az, float gx, float gy, float gz) {
  const float values[kImuModelFeatureCount] = {ax, ay, az, gx, gy, gz};

  if (evalSampleCount < kImuModelWindowSize) {
    for (int ch = 0; ch < kImuModelFeatureCount; ++ch) {
      evalWindow[ch][evalSampleCount] = values[ch];
    }
    evalSampleCount++;
    return;
  }

  for (int ch = 0; ch < kImuModelFeatureCount; ++ch) {
    for (int t = 1; t < kImuModelWindowSize; ++t) {
      evalWindow[ch][t - 1] = evalWindow[ch][t];
    }
    evalWindow[ch][kImuModelWindowSize - 1] = values[ch];
  }
}

int8_t quantizeInputValue(float rawValue, int featureIdx) {
  float normalized = (rawValue - kImuModelMean[featureIdx]) / kImuModelStd[featureIdx];
  int quantized = (int)lrintf(normalized / kImuModelInputScale) + kImuModelInputZeroPoint;
  if (quantized < -128) quantized = -128;
  if (quantized > 127) quantized = 127;
  return (int8_t)quantized;
}

void populateEvalInputTensor() {
  int index = 0;
  for (int ch = 0; ch < kImuModelFeatureCount; ++ch) {
    for (int t = 0; t < kImuModelWindowSize; ++t) {
      int8_t value = quantizeInputValue(evalWindow[ch][t], ch);
      switch (kImuModelInputLayout) {
        case kImuInputLayoutNCW:
          gInputTensor->data.int8[index++] = value;
          break;
        case kImuInputLayoutNWC:
          gInputTensor->data.int8[t * kImuModelFeatureCount + ch] = value;
          break;
        case kImuInputLayoutNCWH:
          gInputTensor->data.int8[(ch * kImuModelWindowSize + t)] = value;
          break;
        case kImuInputLayoutNWCH:
          gInputTensor->data.int8[(t * kImuModelFeatureCount + ch)] = value;
          break;
      }
    }
  }
}

bool runEvalInference() {
  if (!evalRuntimeReady || evalSampleCount < kImuModelWindowSize) {
    return false;
  }

  populateEvalInputTensor();
  unsigned long startMs = millis();
  if (gInterpreter->Invoke() != kTfLiteOk) {
    evalStatus = "INFER FAIL";
    evalInferenceOkay = false;
    logDebug("runEvalInference: invoke failed");
    return false;
  }

  evalInferenceDurationMs = millis() - startMs;
  evalStatus = "RUNNING";

  for (int i = 0; i < kImuModelClassCount; ++i) {
    evalLogits[i] = (gOutputTensor->data.int8[i] - kImuModelOutputZeroPoint) * kImuModelOutputScale;
  }

  int bestIdx = 0;
  if (evalLogits[1] > evalLogits[0]) {
    bestIdx = 1;
  }
  evalPrediction = bestIdx;

  float maxLogit = evalLogits[0] > evalLogits[1] ? evalLogits[0] : evalLogits[1];
  float exp0 = expf(evalLogits[0] - maxLogit);
  float exp1 = expf(evalLogits[1] - maxLogit);
  float denom = exp0 + exp1;
  evalConfidence = (bestIdx == 0) ? (exp0 / denom) : (exp1 / denom);
  evalInferenceOkay = true;
  return true;
}


void setup() {
  Serial.begin(115200);
  delay(250);
  ++gBootCount;
  logDebug("");
  logDebug("==== boot ====");
  logDebugf("boot_count=%u reset_reason=%d free_heap=%u", (unsigned)gBootCount, (int)esp_reset_reason(), (unsigned)ESP.getFreeHeap());

  logDebug("setup: M5.begin start");
  auto cfg = M5.config();
  M5.begin(cfg);
  logDebugf("setup: M5.begin done free_heap=%u", (unsigned)ESP.getFreeHeap());
  logDebug("setup: before display rotation");
  if (!kDebugSkipInitialDisplay) {
    M5.Display.setRotation(3);
    logDebug("setup: display rotation set");
    logDebug("setup: before initial layout");
    drawStreamLayout(currentVisualState);
    logDebug("setup: initial layout drawn");
  } else {
    logDebug("setup: initial display skipped");
  }

  if (!M5.Imu.isEnabled()) {
    logDebug("setup: IMU not found");
    M5.Display.setTextColor(COLOR_RED, COLOR_BLACK);
    M5.Display.drawString("IMU Not Found!", 8, 35);
    while (1) {
      delay(100);
    }
  }
  logDebug("setup: IMU enabled");

  setLed(false);
  logDebug("setup: LED initialized");
  setupEvalRuntime();
  logDebugf("setup: eval runtime ready=%d status=%s", evalRuntimeReady ? 1 : 0, evalStatus);

  logDebug("setup: BLE init");
  BLEDevice::init("M5StickC-IMU");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  logDebug("setup: BLE advertising started");

  lastSampleTime = millis();
  lastEvalTime = millis();
  lastDisplayTime = millis();
  logDebug("setup: complete");
}

void loop() {
  M5.update();
  M5.Imu.update();

  unsigned long now = millis();

  if (M5.BtnB.wasPressed()) {
    switchAppMode(currentAppMode == APP_STREAM ? APP_EVAL : APP_STREAM);
  }

  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  M5.Imu.getAccel(&ax, &ay, &az);
  M5.Imu.getGyro(&gx, &gy, &gz);

  if (currentAppMode != lastAppMode) {
    if (currentAppMode == APP_STREAM) {
      drawStreamLayout(currentVisualState);
      lastVisualState = currentVisualState;
    } else {
      lastEvalBgColor = currentEvalBackgroundColor();
      drawEvalLayout();
      updateEvalDisplay();
    }
    lastAppMode = currentAppMode;
  }

  if (currentAppMode == APP_STREAM) {
    if (M5.BtnA.wasPressed()) {
      packetCount = 0;
      stateTransitionTime = now;
      switch (currentState) {
        case STATE_DISABLED_BEFORE_HAND:
          currentState = STATE_HAND;
          break;
        case STATE_HAND:
          currentState = STATE_DISABLED_BEFORE_POCKET;
          break;
        case STATE_DISABLED_BEFORE_POCKET:
          currentState = STATE_POCKET;
          break;
        case STATE_POCKET:
          currentState = STATE_DISABLED_BEFORE_HAND;
          break;
      }
    }

    bool isStreaming = false;
    unsigned long elapsed = now - stateTransitionTime;
    if (currentState == STATE_DISABLED_BEFORE_HAND) {
      currentVisualState = V_DISABLED_HAND;
    } else if (currentState == STATE_DISABLED_BEFORE_POCKET) {
      currentVisualState = V_DISABLED_POCKET;
    } else if (currentState == STATE_HAND) {
      if (elapsed < startDelay) {
        currentVisualState = V_PREPARING_HAND;
      } else {
        currentVisualState = V_STREAMING_HAND;
        isStreaming = true;
      }
    } else {
      if (elapsed < startDelay) {
        currentVisualState = V_PREPARING_POCKET;
      } else {
        currentVisualState = V_STREAMING_POCKET;
        isStreaming = true;
      }
    }

    if (currentVisualState != lastVisualState) {
      drawStreamLayout(currentVisualState);
      lastVisualState = currentVisualState;
    }

    if (now - lastSampleTime >= sampleInterval) {
      lastSampleTime = now;
      if (deviceConnected && isStreaming) {
        IMUPacket packet;
        packet.timestamp = now;
        packet.ax = ax;
        packet.ay = ay;
        packet.az = az;
        packet.gx = gx;
        packet.gy = gy;
        packet.gz = gz;
        packet.label = (currentState == STATE_HAND) ? 0 : 1;
        pCharacteristic->setValue((uint8_t*)&packet, sizeof(IMUPacket));
        pCharacteristic->notify();
        packetCount++;
      }
    }

    if (currentState == STATE_HAND || currentState == STATE_POCKET) {
      if (elapsed < startDelay) {
        bool flashState = (elapsed / 100) % 2 == 0;
        setLed(flashState);
      } else {
        setLed(true);
      }
    } else {
      setLed(false);
    }

    if (now - lastDisplayTime >= displayInterval) {
      lastDisplayTime = now;
      updateStreamDisplay(currentVisualState);
    }
  } else {
    if (M5.BtnA.wasPressed()) {
      resetEvalState();
    }

    if (now - lastSampleTime >= sampleInterval) {
      lastSampleTime = now;
      appendEvalSample(ax, ay, az, gx, gy, gz);
    }

    if (now - lastEvalTime >= evalInterval) {
      lastEvalTime = now;
      runEvalInference();
    }

    if (!evalRuntimeReady) {
      setLed((now / 250) % 2 == 0);
    } else if (evalSampleCount < kImuModelWindowSize) {
      setLed((now / 150) % 2 == 0);
    } else {
      setLed(evalInferenceOkay);
    }

    if (now - lastDisplayTime >= displayInterval) {
      lastDisplayTime = now;
      updateEvalDisplay();
    }
  }

  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    currentState = STATE_DISABLED_BEFORE_HAND;
    currentVisualState = V_DISABLED_HAND;
    packetCount = 0;
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
}
