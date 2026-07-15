#include <Arduino.h>
#include <M5Unified.h>
#include <esp_mac.h>

#ifndef UWB_ANCHOR_ID
#error "Set UWB_ANCHOR_ID to a unique anchor ID in platformio.ini."
#endif

namespace {

constexpr int kUwbRxPin = 33;  // Core2 Port A G33 receives UWB TX.
constexpr int kUwbTxPin = 32;  // Core2 Port A G32 sends data to UWB RX.
constexpr uint32_t kUwbBaudRate = 115200;
constexpr uint32_t kDisplayTimeoutMs = 10000;

HardwareSerial uwbSerial(2);
char statusText[96] = "Starting";
bool displayVisible = false;
uint32_t displayUntilMs = 0;

/** Draws the Core2's current battery percentage in the display header. */
void showBatteryLevel() {
    if (!displayVisible) {
        return;
    }
    M5.Display.fillRect(222, 8, 90, 24, TFT_DARKGREEN);
    M5.Display.setTextColor(TFT_WHITE, TFT_DARKGREEN);
    M5.Display.setTextSize(0.2);
    M5.Display.setCursor(226, 12);

    const int32_t batteryLevel = M5.Power.getBatteryLevel();
    if (batteryLevel >= 0) {
        M5.Display.printf("BAT %ld%%", static_cast<long>(batteryLevel));
    } else {
        M5.Display.print("BAT --");
    }
}

void drawAnchorDisplay() {
    M5.Display.fillScreen(TFT_DARKGREEN);
    M5.Display.setFont(&fonts::DejaVu72);
    M5.Display.setTextColor(TFT_WHITE, TFT_DARKGREEN);
    M5.Display.setTextSize(0.3);
    M5.Display.setCursor(68, 20);
    M5.Display.print("UWB ANCHOR");

    M5.Display.setTextSize(1.75);
    M5.Display.setCursor(130, 68);
    M5.Display.printf("%d", UWB_ANCHOR_ID);

    M5.Display.drawFastHLine(12, 188, 296, TFT_WHITE);
    showBatteryLevel();
}

void showStatus(const char* status) {
    strlcpy(statusText, status, sizeof(statusText));
    if (!displayVisible) {
        return;
    }
    M5.Display.fillRect(12, 198, 296, 30, TFT_DARKGREEN);
    M5.Display.setTextColor(TFT_WHITE, TFT_DARKGREEN);
    M5.Display.setTextSize(0.2);
    M5.Display.setCursor(12, 202);
    M5.Display.print("UWB: ");
    M5.Display.print(statusText);
}

/** Wakes the Core2 display and keeps its status visible for ten seconds. */
void wakeAnchorDisplay() {
    displayUntilMs = millis() + kDisplayTimeoutMs;
    if (displayVisible) {
        return;
    }
    M5.Display.wakeup();
    displayVisible = true;
    drawAnchorDisplay();
    showStatus(statusText);
}

/** Powers down the display once its ten-second visibility period expires. */
void updateAnchorDisplayPower() {
    if (displayVisible && static_cast<int32_t>(millis() - displayUntilMs) >= 0) {
        M5.Display.sleep();
        displayVisible = false;
    }
}

void printChipMac() {
    uint8_t mac[6] = {};
    if (esp_efuse_mac_get_default(mac) != ESP_OK) {
        Serial.println("ESP32 factory MAC: unavailable");
        return;
    }

    Serial.printf("ESP32 factory MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void sendUwbCommand(const char* command) {
    Serial.printf("UWB -> %s\n", command);
    showStatus(command);
    uwbSerial.print(command);
    uwbSerial.print("\r\n");
    delay(100);
}

void configureUwbAnchor() {
    while (uwbSerial.available() > 0) {
        uwbSerial.read();
    }

    char command[32] = {};
    snprintf(command, sizeof(command), "AT+anchor_tag=1,%d", UWB_ANCHOR_ID);
    sendUwbCommand(command);

    // The Unit UWB applies the selected role after a reset.
    sendUwbCommand("AT+RST");
    Serial.printf("UWB configured as fixed anchor ID %d\n", UWB_ANCHOR_ID);
}

void forwardUwbOutput() {
    static char line[128] = {};
    static size_t length = 0;

    while (uwbSerial.available() > 0) {
        const char character = static_cast<char>(uwbSerial.read());
        if (character == '\r') {
            continue;
        }
        if (character == '\n') {
            if (length > 0) {
                line[length] = '\0';
                Serial.printf("UWB <- %s\n", line);
                showStatus(line);
                length = 0;
            }
            continue;
        }
        if (length < sizeof(line) - 1) {
            line[length++] = character;
        } else {
            length = 0;
        }
    }
}

}  // namespace

void setup() {
    Serial.begin(115200);
    delay(500);

    auto config = M5.config();
    config.internal_imu = false;
    config.internal_mic = false;
    config.internal_spk = false;
    config.external_imu = false;
    config.external_rtc = false;
    M5.begin(config);
    M5.Display.setBrightness(80);
    M5.Display.sleep();
    showStatus("Booting");

    Serial.printf("UWB anchor station booting with ID %d\n", UWB_ANCHOR_ID);
    printChipMac();

    uwbSerial.begin(kUwbBaudRate, SERIAL_8N1, kUwbRxPin, kUwbTxPin);
    delay(500);
    configureUwbAnchor();
}

void loop() {
    M5.update();
    const auto touch = M5.Touch.getDetail();
    if (M5.BtnPWR.wasPressed() || M5.BtnPWR.wasClicked() || touch.wasPressed()) {
        Serial.println("Display wake request");
        wakeAnchorDisplay();
    }
    forwardUwbOutput();
    updateAnchorDisplayPower();

    static uint32_t lastBatteryUpdateMs = 0;
    if (displayVisible && millis() - lastBatteryUpdateMs >= 1000) {
        showBatteryLevel();
        lastBatteryUpdateMs = millis();
    }
    delay(5);
}
