#include <M5Unified.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

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

// State Machine Definitions
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

// Binary packet structure (packed to ensure exact struct size/alignment)
struct __attribute__((packed)) IMUPacket {
  uint32_t timestamp;  // ms timestamp
  float ax;            // Accelerometer X (g)
  float ay;            // Accelerometer Y (g)
  float az;            // Accelerometer Z (g)
  float gx;            // Gyroscope X (deg/s)
  float gy;            // Gyroscope Y (deg/s)
  float gz;            // Gyroscope Z (deg/s)
  uint8_t label;       // 0 = walking in hand, 1 = walking in pocket
};

// Global BLE and State variables
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
uint32_t packetCount = 0;

// Sequence control variables
DeviceState currentState = STATE_DISABLED_BEFORE_HAND;
VisualState currentVisualState = V_DISABLED_HAND;
VisualState lastVisualState = V_DISABLED_HAND;

unsigned long stateTransitionTime = 0;
const unsigned long startDelay = 1000; // 1 second prep delay

// Sampling parameters (50 Hz = 20ms intervals)
unsigned long lastSampleTime = 0;
const unsigned long sampleInterval = 20;

// Visual update rate (10 Hz = 100ms intervals to prevent LCD lag)
unsigned long lastDisplayTime = 0;
const unsigned long displayInterval = 100;

// Helper to set LED state dynamically based on the detected board
void setLed(bool turnOn) {
  auto board = M5.getBoard();
  int targetPin = 10;      // Default for StickC and StickC Plus
  bool activeLevel = LOW;  // Active low for CP2104 boards
  
  if (board == m5::board_t::board_M5StickCPlus2) {
    targetPin = 19;        // StickC Plus2 LED is on GPIO 19
    activeLevel = HIGH;    // Active high
  }
  
  pinMode(targetPin, OUTPUT);
  digitalWrite(targetPin, turnOn ? activeLevel : !activeLevel);
}

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
    }
};

// Render layout background and header structure based on state (no flicker)
void drawScreenLayout(VisualState state) {
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
  
  M5.Display.drawString("M5 IMU Data Stream", 8, 5);
  
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

// Draw dynamic values at 10Hz (overwriting previous characters using background color)
void updateDisplay(VisualState state, float ax, float ay, float az, float gx, float gy, float gz) {
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

  // 1. Connection status
  M5.Display.setCursor(45, 25);
  if (deviceConnected) {
    M5.Display.setTextColor(COLOR_GREEN, bgColor);
    M5.Display.print("CONNECTED   ");
  } else {
    M5.Display.setTextColor(COLOR_RED, bgColor);
    M5.Display.print("ADVERTISING ");
  }

  // 2. Mode name
  M5.Display.setCursor(45, 40);
  M5.Display.setTextColor(defaultTextColor, bgColor);
  switch (state) {
    case V_DISABLED_HAND:
    case V_DISABLED_POCKET:
      M5.Display.print("DISABLED    ");
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

  // 3. Conditional Row (Packets, countdown, or prompt)
  if (state == V_STREAMING_HAND || state == V_STREAMING_POCKET) {
    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(defaultTextColor, bgColor);
    M5.Display.printf("%u      ", packetCount);

    // Calculate actual frequency
    static uint32_t lastDisplayPacketCount = 0;
    static unsigned long lastRateCalculationTime = 0;
    unsigned long now = millis();
    
    if (now - lastRateCalculationTime >= 1000) {
      float hz = (packetCount - lastDisplayPacketCount) * 1000.0 / (now - lastRateCalculationTime);
      M5.Display.setCursor(45, 68);
      M5.Display.printf("%.1f    ", hz);
      
      lastDisplayPacketCount = packetCount;
      lastRateCalculationTime = now;
    }
  } else if (state == V_PREPARING_HAND || state == V_PREPARING_POCKET) {
    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(defaultTextColor, bgColor);
    
    unsigned long elapsed = millis() - stateTransitionTime;
    float remaining = (startDelay - elapsed) / 1000.0;
    if (remaining < 0) remaining = 0;
    
    M5.Display.printf("START IN %.1fs ", remaining);
  } else {
    M5.Display.setCursor(45, 55);
    M5.Display.setTextColor(COLOR_YELLOW, bgColor);
    if (state == V_DISABLED_HAND) {
      M5.Display.print("PRESS FOR HAND  ");
    } else {
      M5.Display.print("PRESS FOR POCKET");
    }
  }
}

void setup() {
  // Initialize M5Unified hardware
  auto cfg = M5.config();
  M5.begin(cfg);
  
  // Set up screen rotation (landscape)
  M5.Display.setRotation(3);
  drawScreenLayout(currentVisualState);

  // Verify IMU is active
  if (!M5.Imu.isEnabled()) {
    M5.Display.setTextColor(COLOR_RED, COLOR_BLACK);
    M5.Display.drawString("IMU Not Found!", 8, 35);
    while (1) {
      delay(100);
    }
  }

  setLed(false); // Off initially

  // Initialize BLE
  BLEDevice::init("M5StickC-IMU");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  // Client characteristic configuration descriptor
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  // Configure advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  lastSampleTime = millis();
  lastDisplayTime = millis();
}

void loop() {
  // Update button states and sensor readings
  M5.update();
  M5.Imu.update();

  unsigned long now = millis();

  // Handle Button A presses to advance the state machine
  if (M5.BtnA.wasPressed()) {
    packetCount = 0; // Reset packet counter for the new session
    stateTransitionTime = now;
    
    // Sequence cycle: DISABLED_BEFORE_HAND -> HAND -> DISABLED_BEFORE_POCKET -> POCKET -> DISABLED_BEFORE_HAND
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

  // Determine active visual state and check if streaming is allowed
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
  } else if (currentState == STATE_POCKET) {
    if (elapsed < startDelay) {
      currentVisualState = V_PREPARING_POCKET;
    } else {
      currentVisualState = V_STREAMING_POCKET;
      isStreaming = true;
    }
  }

  // Redraw full screen layout ONLY on transition changes (avoids flickering)
  if (currentVisualState != lastVisualState) {
    drawScreenLayout(currentVisualState);
    lastVisualState = currentVisualState;
  }

  // Retrieve IMU sensor readings
  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;
  M5.Imu.getAccel(&ax, &ay, &az);
  M5.Imu.getGyro(&gx, &gy, &gz);

  // Send data over BLE at 50Hz (only if connected and streaming is active)
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
      packet.label = (currentState == STATE_HAND) ? 0 : 1; // 0 for Hand, 1 for Pocket

      pCharacteristic->setValue((uint8_t*)&packet, sizeof(IMUPacket));
      pCharacteristic->notify();
      packetCount++;
    }
  }

  // Control Built-in LED dynamically for visual feedback
  if (currentState == STATE_HAND || currentState == STATE_POCKET) {
    if (elapsed < startDelay) {
      // Preparing: Flash LED rapidly (every 100ms)
      bool flashState = (elapsed / 100) % 2 == 0;
      setLed(flashState);
    } else {
      // Streaming: Solid LED ON
      setLed(true);
    }
  } else {
    // Disabled: LED OFF
    setLed(false);
  }

  // Update dynamic values on screen at 10Hz
  if (now - lastDisplayTime >= displayInterval) {
    lastDisplayTime = now;
    updateDisplay(currentVisualState, ax, ay, az, gx, gy, gz);
  }

  // Reset BLE advertising on disconnect transitions
  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // Allow BLE stack to clear
    pServer->startAdvertising();
    currentState = STATE_DISABLED_BEFORE_HAND; // Reset state machine on disconnect
    currentVisualState = V_DISABLED_HAND;
    packetCount = 0;
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
}
