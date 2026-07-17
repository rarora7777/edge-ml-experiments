/*
 * SPDX-FileCopyrightText: 2025 M5Stack Technology CO LTD
 * SPDX-License-Identifier: MIT
 *
 * AtomS3R-M12 camera or M5StickC Plus UWB/IMU tag server.
 */

#include <M5Unified.h>
#include <WiFi.h>
#if defined(TAG_M5STICKC_PLUS)
#include <M5_BMM150.h>
#endif
#if !defined(TAG_M5STICKC_PLUS)
#include <esp_camera.h>
#include <esp_timer.h>

#include <camera_pins.h>
#else
#include <nvs.h>
#endif
#include <wifi_credentials.h>

namespace {

WiFiServer server(80);
SemaphoreHandle_t imuMutex;
SemaphoreHandle_t uwbMutex;
HardwareSerial uwbSerial(1);

struct ImuSample {
    uint32_t timestampMs = 0;
    float accel[3] = {};
    float gyro[3] = {};
    float mag[3] = {};
    bool valid = false;
    bool magAvailable = false;
};

ImuSample latestImuSample;

struct UwbSample {
    uint32_t timestampMs = 0;
    uint32_t sequence = 0;
    int anchorId = -1;
    float distanceM = 0.0f;
    char report[128] = {};
    bool valid = false;
};

UwbSample latestUwbSample;
constexpr size_t kUwbHistoryLength = 32;
UwbSample uwbHistory[kUwbHistoryLength];
#if defined(TAG_M5STICKC_PLUS)
char latestUwbStatus[32] = "Starting";
uint32_t latestUwbStatusMs = 0;
constexpr uint32_t kDisplayTimeoutMs = 10000;
uint32_t stickDisplayUntilMs = 0;
uint32_t lastStickDisplayDrawMs = 0;
bool stickDisplayVisible = false;
#endif

#if defined(TAG_M5STICKC_PLUS)
constexpr int kImuSdaPin = 21;
constexpr int kImuSclPin = 22;
constexpr i2c_port_t kImuI2cPort = I2C_NUM_1;
constexpr m5::board_t kImuBoard = m5::board_t::board_M5StickCPlus;
constexpr int kMagSdaPin = 0;   // ENV-II HAT I2C SDA.
constexpr int kMagSclPin = 26;  // ENV-II HAT I2C SCL.
constexpr i2c_port_t kMagI2cPort = I2C_NUM_0;
constexpr uint8_t kBmm150Address = 0x10;
constexpr uint32_t kMagI2cFrequency = 400000;
constexpr int kUwbRxPin = 33;  // Port A G33 receives data from the UWB unit.
constexpr int kUwbTxPin = 32;  // Port A G32 sends commands to the UWB unit.
#else
constexpr int kImuSdaPin = IMU_SDA_GPIO_NUM;
constexpr int kImuSclPin = IMU_SCL_GPIO_NUM;
constexpr i2c_port_t kImuI2cPort = I2C_NUM_0;
constexpr m5::board_t kImuBoard = m5::board_t::board_M5AtomS3RCam;
constexpr int kUwbRxPin = 1;  // Port A G1 receives data from the UWB unit.
constexpr int kUwbTxPin = 2;  // Port A G2 sends commands to the UWB unit.

constexpr char kStreamContentType[] =
    "multipart/x-mixed-replace;boundary=123456789000000000000987654321";
constexpr char kStreamBoundary[] =
    "\r\n--123456789000000000000987654321\r\n";
constexpr char kStreamPart[] =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";
#endif
constexpr uint32_t kWiFiConnectTimeoutMs = 30000;
constexpr uint32_t kUwbBaudRate = 115200;

#if defined(TAG_M5STICKC_PLUS)
bmm150_dev stickMagnetometer = {};
bool stickMagnetometerAvailable = false;

/** BMM150 compensated-driver I2C read callback for the ENV-II HAT. */
int8_t readStickMagnetometer(uint8_t deviceAddress, uint8_t registerAddress,
                             uint8_t* data, uint16_t length) {
    return M5.Ex_I2C.readRegister(deviceAddress, registerAddress, data, length,
                                   kMagI2cFrequency)
        ? BMM150_OK : BMM150_E_DEV_NOT_FOUND;
}

/** BMM150 compensated-driver I2C write callback for the ENV-II HAT. */
int8_t writeStickMagnetometer(uint8_t deviceAddress, uint8_t registerAddress,
                              uint8_t* data, uint16_t length) {
    return M5.Ex_I2C.writeRegister(deviceAddress, registerAddress, data, length,
                                    kMagI2cFrequency)
        ? BMM150_OK : BMM150_E_DEV_NOT_FOUND;
}

void delayStickMagnetometer(uint32_t milliseconds) {
    delay(milliseconds);
}
#endif

/** Returns a readable label for an Arduino Wi-Fi connection state. */
const char* wifiStatusName(wl_status_t status) {
    switch (status) {
        case WL_IDLE_STATUS: return "idle";
        case WL_NO_SSID_AVAIL: return "SSID unavailable";
        case WL_SCAN_COMPLETED: return "scan completed";
        case WL_CONNECTED: return "connected";
        case WL_CONNECT_FAILED: return "connection failed";
        case WL_CONNECTION_LOST: return "connection lost";
        case WL_DISCONNECTED: return "disconnected";
        default: return "unknown";
    }
}

#if !defined(TAG_M5STICKC_PLUS)
camera_config_t cameraConfig = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sccb_sda = SIOD_GPIO_NUM,
    .pin_sccb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_LATEST,
    .sccb_i2c_port = 1,
};
#endif

/** Continuously reads the onboard IMU and publishes the most recent sample. */
void updateImuTask(void*) {
    for (;;) {
        if (M5.Imu.update()) {
            const auto sample = M5.Imu.getImuData();
            if (xSemaphoreTake(imuMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                latestImuSample.timestampMs = sample.usec / 1000;
                latestImuSample.accel[0] = sample.accel.x;
                latestImuSample.accel[1] = sample.accel.y;
                latestImuSample.accel[2] = sample.accel.z;
                latestImuSample.gyro[0] = sample.gyro.x;
                latestImuSample.gyro[1] = sample.gyro.y;
                latestImuSample.gyro[2] = sample.gyro.z;
#if !defined(TAG_M5STICKC_PLUS)
                latestImuSample.mag[0] = sample.mag.x;
                latestImuSample.mag[1] = sample.mag.y;
                latestImuSample.mag[2] = sample.mag.z;
                latestImuSample.magAvailable = true;
#else
                latestImuSample.magAvailable = stickMagnetometerAvailable;
#endif
                latestImuSample.valid = true;
                xSemaphoreGive(imuMutex);
            }
        }

#if defined(TAG_M5STICKC_PLUS)
        if (stickMagnetometerAvailable) {
            if (bmm150_read_mag_data(&stickMagnetometer) == BMM150_OK) {
                if (xSemaphoreTake(imuMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                    // M5_BMM150 applies the BMM150's factory trim compensation
                    // and reports microtesla. Do not apply or persist hard/soft-
                    // iron offsets here; the raw compensated field is streamed.
                    // ENV-II BMM150 axes differ from the onboard IMU: flip X/Z
                    // once here so every stream consumer receives IMU-frame data.
                    latestImuSample.mag[0] = -stickMagnetometer.data.x;
                    latestImuSample.mag[1] = stickMagnetometer.data.y;
                    latestImuSample.mag[2] = -stickMagnetometer.data.z;
                    latestImuSample.magAvailable = true;
                    xSemaphoreGive(imuMutex);
                }
            }
        }
#endif
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/** Initializes the IMU on its dedicated I2C bus, then starts its reader task. */
void initializeImuTask(void*) {
    log_i("IMU: configuring I2C port %d on SDA=%d SCL=%d",
          static_cast<int>(kImuI2cPort), kImuSdaPin, kImuSclPin);
    M5.In_I2C.setPort(kImuI2cPort, kImuSdaPin, kImuSclPin);

    // IMU_Class::begin() initializes the supplied I2C bus itself.
    log_i("IMU: initializing onboard sensor");
    if (!M5.Imu.begin(&M5.In_I2C, kImuBoard)) {
        log_e("IMU: initialization failed");
        vTaskDelete(nullptr);
        return;
    }

    M5.Imu.setCalibration(0, 0, 0);
    M5.Imu.clearOffsetData();

#if defined(TAG_M5STICKC_PLUS)
    log_i("BMM150: configuring ENV-II HAT I2C port %d on SDA=%d SCL=%d",
          static_cast<int>(kMagI2cPort), kMagSdaPin, kMagSclPin);
    if (!M5.Ex_I2C.begin(kMagI2cPort, kMagSdaPin, kMagSclPin)) {
        log_e("BMM150: I2C initialization failed");
    } else {
        // Use only the BMM150's factory trim compensation. This driver has no
        // persisted hard/soft-iron calibration state to load or apply.
        stickMagnetometer = {};
        stickMagnetometer.dev_id = kBmm150Address;
        stickMagnetometer.intf = BMM150_I2C_INTF;
        stickMagnetometer.read = readStickMagnetometer;
        stickMagnetometer.write = writeStickMagnetometer;
        stickMagnetometer.delay_ms = delayStickMagnetometer;
        if (bmm150_init(&stickMagnetometer) != BMM150_OK) {
            log_e("BMM150: initialization or factory-trim read failed");
        } else {
            stickMagnetometer.settings.pwr_mode = BMM150_NORMAL_MODE;
            if (bmm150_set_op_mode(&stickMagnetometer) != BMM150_OK) {
                log_e("BMM150: failed to enable normal measurement mode");
            } else {
                stickMagnetometer.settings.preset_mode = BMM150_PRESETMODE_ENHANCED;
                if (bmm150_set_presetmode(&stickMagnetometer) != BMM150_OK) {
                    log_e("BMM150: failed to configure enhanced measurement mode");
                } else {
                    stickMagnetometerAvailable = true;
                    log_i("BMM150: ENV-II HAT initialized with factory compensation");
                }
            }
        }
    }
#endif

    log_i("IMU: initialization succeeded (type=%d)",
          static_cast<int>(M5.Imu.getType()));
    xTaskCreatePinnedToCore(updateImuTask, "imu-read", 8192, nullptr, 1, nullptr, 0);
    vTaskDelete(nullptr);
}

/** Parses and stores one `an<id>:<distance>m` UWB range measurement. */
void storeUwbReport(const char* report) {
    int anchorId = -1;
    float distanceM = 0.0f;
    char extraCharacter = '\0';
    if (sscanf(report, "an%d:%fm%c", &anchorId, &distanceM, &extraCharacter) != 2 ||
        anchorId < 0 || distanceM < 0.0f) {
        return;
    }

    if (xSemaphoreTake(uwbMutex, pdMS_TO_TICKS(10)) != pdTRUE) {
        return;
    }

    const uint32_t sequence = latestUwbSample.sequence + 1;
    UwbSample& sample = uwbHistory[sequence % kUwbHistoryLength];
    sample.timestampMs = millis();
    sample.sequence = sequence;
    sample.anchorId = anchorId;
    sample.distanceM = distanceM;
    strlcpy(sample.report, report, sizeof(sample.report));
    sample.valid = true;
    latestUwbSample = sample;
    xSemaphoreGive(uwbMutex);
}

#if defined(TAG_M5STICKC_PLUS)
/** Stores the latest UWB UART line for the M5StickC Plus status display. */
void storeUwbStatus(const char* report) {
    if (xSemaphoreTake(uwbMutex, pdMS_TO_TICKS(10)) != pdTRUE) {
        return;
    }
    strlcpy(latestUwbStatus, report, sizeof(latestUwbStatus));
    latestUwbStatusMs = millis();
    xSemaphoreGive(uwbMutex);
}

/** Updates the M5StickC Plus display from the main task that owns M5Unified. */
void updateStickDisplay() {
    const uint32_t now = millis();
    const bool shouldShow = static_cast<int32_t>(stickDisplayUntilMs - now) > 0;
    if (!shouldShow) {
        if (stickDisplayVisible) {
            M5.Display.setBrightness(0);
            stickDisplayVisible = false;
        }
        return;
    }

    if (!stickDisplayVisible) {
        M5.Display.setBrightness(200);
        M5.Display.setRotation(0);
        M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
        M5.Display.setTextSize(2);
        stickDisplayVisible = true;
        lastStickDisplayDrawMs = 0;
    }
    if (now - lastStickDisplayDrawMs < 1000) {
        return;
    }
    lastStickDisplayDrawMs = now;

    char uwbStatus[sizeof(latestUwbStatus)];
    uint32_t statusMs = 0;
    if (xSemaphoreTake(uwbMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        strlcpy(uwbStatus, latestUwbStatus, sizeof(uwbStatus));
        statusMs = latestUwbStatusMs;
        xSemaphoreGive(uwbMutex);
    } else {
        strlcpy(uwbStatus, "Busy", sizeof(uwbStatus));
    }

    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(8, 12);
    const int32_t batteryLevel = M5.Power.getBatteryLevel();
    if (batteryLevel >= 0) {
        M5.Display.printf("BAT %ld%%", static_cast<long>(batteryLevel));
    } else {
        M5.Display.print("BAT --");
    }
    M5.Display.setCursor(8, 62);
    M5.Display.print("UWB");
    M5.Display.setTextSize(1);
    M5.Display.setCursor(8, 94);
    if (statusMs == 0 || millis() - statusMs > 5000) {
        M5.Display.print("Waiting for range");
    } else {
        M5.Display.print(uwbStatus);
    }
    M5.Display.setTextSize(2);
}

/** Keeps the M5StickC Plus display awake for ten seconds after Button A. */
void wakeStickDisplay() {
    stickDisplayUntilMs = millis() + kDisplayTimeoutMs;
}

/** Persists the correct panel type because PlatformIO exposes only m5stick-c. */
void selectStickCPlusPanel() {
    nvs_handle_t handle;
    if (nvs_open("M5GFX", NVS_READWRITE, &handle) != ESP_OK) {
        log_w("Display: unable to open M5GFX settings");
        return;
    }

    const uint32_t board = static_cast<uint32_t>(m5::board_t::board_M5StickCPlus);
    uint32_t storedBoard = 0;
    nvs_get_u32(handle, "AUTODETECT", &storedBoard);
    if (storedBoard != board) {
        nvs_set_u32(handle, "AUTODETECT", board);
        nvs_commit(handle);
    }
    nvs_close(handle);
}
#endif

/** Reads complete UWB UART lines without blocking the HTTP server. */
void updateUwbTask(void*) {
    char report[sizeof(latestUwbSample.report)] = {};
    size_t reportLength = 0;

    for (;;) {
        while (uwbSerial.available() > 0) {
            const char character = static_cast<char>(uwbSerial.read());
            if (character == '\r') {
                continue;
            }
            if (character == '\n') {
                if (reportLength > 0) {
                    report[reportLength] = '\0';
                    log_i("UWB: %s", report);
#if defined(TAG_M5STICKC_PLUS)
                    storeUwbStatus(report);
#endif
                    storeUwbReport(report);
                    reportLength = 0;
                }
                continue;
            }
            if (reportLength < sizeof(report) - 1) {
                report[reportLength++] = character;
            } else {
                // Discard an overlong report rather than publishing truncated data.
                reportLength = 0;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

/** Configures the Port A Unit UWB as a tag and starts its UART reader. */
void initializeUwb() {
    log_i("UWB: starting UART on Port A (RX=%d, TX=%d)", kUwbRxPin, kUwbTxPin);
    uwbSerial.begin(kUwbBaudRate, SERIAL_8N1, kUwbRxPin, kUwbTxPin);
    delay(100);
    while (uwbSerial.available() > 0) {
        uwbSerial.read();
    }

    // Unit UWB contains an STM32 controller. Configure it as the ranging tag.
    // Reset first: role changes are applied after the module has restarted.
    uwbSerial.print("AT+RST\r\n");
    delay(1000);
    uwbSerial.print("AT+anchor_tag=0,0\r\n");
    delay(100);
    uwbSerial.print("AT+interval=5\r\n");
    delay(100);
    uwbSerial.print("AT+switchdis=1\r\n");
    log_i("UWB: tag ranging enabled");

    xTaskCreatePinnedToCore(updateUwbTask, "uwb-read", 4096, nullptr, 1, nullptr, 0);
}

/** Connects to the configured Wi-Fi network, returning false after 30 seconds. */
bool connectToWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    log_i("WiFi: connecting");

    const uint32_t startedAt = millis();
    wl_status_t previousStatus = static_cast<wl_status_t>(-1);
    while (WiFi.status() != WL_CONNECTED && millis() - startedAt < kWiFiConnectTimeoutMs) {
        const wl_status_t status = WiFi.status();
        if (status != previousStatus) {
            log_i("WiFi: status=%d (%s)", static_cast<int>(status), wifiStatusName(status));
            previousStatus = status;
        }
        delay(500);
    }

    if (WiFi.status() != WL_CONNECTED) {
        log_e("WiFi: connection timed out; last status=%d (%s)",
              static_cast<int>(WiFi.status()), wifiStatusName(WiFi.status()));
        return false;
    }

    log_i("WiFi: connected, IP address: %s", WiFi.localIP().toString().c_str());
    return true;
}

/** Returns a mutex-protected snapshot of the latest IMU sample. */
ImuSample getImuSample() {
    ImuSample sample;
    if (xSemaphoreTake(imuMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        sample = latestImuSample;
        xSemaphoreGive(imuMutex);
    }
    return sample;
}

/** Returns a mutex-protected snapshot of the latest UWB range report. */
UwbSample getUwbSample() {
    UwbSample sample;
    if (xSemaphoreTake(uwbMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        sample = latestUwbSample;
        xSemaphoreGive(uwbMutex);
    }
    return sample;
}

/** Retrieves the next retained UWB line after a client-specific sequence number. */
bool getNextUwbSample(uint32_t afterSequence, UwbSample* sample) {
    if (xSemaphoreTake(uwbMutex, pdMS_TO_TICKS(10)) != pdTRUE) {
        return false;
    }

    const uint32_t latestSequence = latestUwbSample.sequence;
    if (latestSequence <= afterSequence) {
        xSemaphoreGive(uwbMutex);
        return false;
    }

    const uint32_t oldestSequence =
        latestSequence > kUwbHistoryLength ? latestSequence - kUwbHistoryLength + 1 : 1;
    const uint32_t nextSequence = max(afterSequence + 1, oldestSequence);
    *sample = uwbHistory[nextSequence % kUwbHistoryLength];
    xSemaphoreGive(uwbMutex);
    return true;
}

/** Writes a JSON-escaped string directly to an HTTP client. */
bool writeJsonString(WiFiClient& client, const char* value) {
    if (client.write('"') != 1) {
        return false;
    }
    for (const char* character = value; *character != '\0'; ++character) {
        switch (*character) {
            case '"':
            case '\\':
                if (client.write('\\') != 1 || client.write(*character) != 1) {
                    return false;
                }
                break;
            case '\n':
                if (client.print("\\n") != 2) {
                    return false;
                }
                break;
            case '\r':
                if (client.print("\\r") != 2) {
                    return false;
                }
                break;
            default:
                if (client.write(*character) != 1) {
                    return false;
                }
        }
    }
    return client.write('"') == 1;
}

/** Serves the IMU snapshot as a 50 Hz server-sent event stream. */
void streamImu(WiFiClient& client) {
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/event-stream");
    client.println("Cache-Control: no-cache");
    client.println("Connection: keep-alive");
    client.println("Access-Control-Allow-Origin: *");
    client.println();

    while (client.connected()) {
        const ImuSample sample = getImuSample();
#if defined(TAG_M5STICKC_PLUS)
        const int written = sample.magAvailable
            ? client.printf(
                  "event: imu\ndata: {\"timestamp_ms\":%lu,\"valid\":%s,\"mag_available\":true,"
                  "\"accel\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f},"
                  "\"gyro\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f},"
                  "\"mag\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f}}\n\n",
                  static_cast<unsigned long>(sample.timestampMs),
                  sample.valid ? "true" : "false",
                  sample.accel[0], sample.accel[1], sample.accel[2],
                  sample.gyro[0], sample.gyro[1], sample.gyro[2],
                  sample.mag[0], sample.mag[1], sample.mag[2])
            : client.printf(
                  "event: imu\ndata: {\"timestamp_ms\":%lu,\"valid\":%s,\"mag_available\":false,"
                  "\"accel\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f},"
                  "\"gyro\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f}}\n\n",
                  static_cast<unsigned long>(sample.timestampMs),
                  sample.valid ? "true" : "false",
                  sample.accel[0], sample.accel[1], sample.accel[2],
                  sample.gyro[0], sample.gyro[1], sample.gyro[2]);
        if (written <= 0) {
            break;
        }
#else
        if (client.printf(
                "event: imu\ndata: {\"timestamp_ms\":%lu,\"valid\":%s,\"mag_available\":true,"
                "\"accel\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f},"
                "\"gyro\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f},"
                "\"mag\":{\"x\":%.6f,\"y\":%.6f,\"z\":%.6f}}\n\n",
                static_cast<unsigned long>(sample.timestampMs),
                sample.valid ? "true" : "false",
                sample.accel[0], sample.accel[1], sample.accel[2],
                sample.gyro[0], sample.gyro[1], sample.gyro[2],
                sample.mag[0], sample.mag[1], sample.mag[2]) <= 0) {
            break;
        }
#endif
        delay(20);
    }

    client.stop();
    Serial.println("IMU stream ended");
}

/** Serves retained UWB anchor range measurements as a diagnostic event stream. */
void streamUwb(WiFiClient& client) {
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/event-stream");
    client.println("Cache-Control: no-cache");
    client.println("Connection: keep-alive");
    client.println("Access-Control-Allow-Origin: *");
    client.println();
    log_i("UWB stream started");

    uint32_t lastSequence = 0;
    uint32_t lastKeepaliveMs = millis();
    while (client.connected()) {
        UwbSample sample;
        if (!getNextUwbSample(lastSequence, &sample)) {
            if (millis() - lastKeepaliveMs >= 5000) {
                client.println(": keepalive");
                client.println();
                lastKeepaliveMs = millis();
            }
            delay(5);
            continue;
        }
        lastSequence = sample.sequence;

        if (client.printf("event: uwb\ndata: {\"timestamp_ms\":%lu,\"sequence\":%lu,\"anchor_id\":%d,\"distance_m\":%.3f,\"report\":",
                          static_cast<unsigned long>(sample.timestampMs),
                          static_cast<unsigned long>(sample.sequence),
                          sample.anchorId, sample.distanceM) <= 0 ||
            !writeJsonString(client, sample.report) || client.print("}\n\n") != 3) {
            break;
        }
    }

    client.stop();
    log_i("UWB stream ended");
}

#if !defined(TAG_M5STICKC_PLUS)
/** Serves camera frames as an MJPEG multipart stream until the client disconnects. */
void streamJpeg(WiFiClient& client) {
    client.println("HTTP/1.1 200 OK");
    client.printf("Content-Type: %s\r\n", kStreamContentType);
    client.println("Content-Disposition: inline; filename=capture.jpg");
    client.println("Access-Control-Allow-Origin: *");
    client.println();

    int64_t lastFrame = esp_timer_get_time();
    while (client.connected()) {
        camera_fb_t* frame = esp_camera_fb_get();
        if (frame == nullptr) {
            Serial.println("Camera capture failed");
            continue;
        }

        uint8_t* jpeg = frame->buf;
        size_t jpegLength = frame->len;

        client.print(kStreamBoundary);
        client.printf(kStreamPart, jpegLength);

        size_t remaining = jpegLength;
        bool sendOk = true;
        constexpr size_t kPacketLength = 8 * 1024;
        while (remaining > 0) {
            const size_t chunk = min(remaining, kPacketLength);
            if (client.write(jpeg, chunk) != chunk) {
                sendOk = false;
                break;
            }
            jpeg += chunk;
            remaining -= chunk;
        }

        const int64_t now = esp_timer_get_time();
        const int64_t frameTimeMs = max<int64_t>(1, (now - lastFrame) / 1000);
        lastFrame = now;
        Serial.printf("MJPG: %luKB %llums (%.1ffps)\r\n",
                      static_cast<unsigned long>(jpegLength / 1024),
                      static_cast<long long>(frameTimeMs),
                      1000.0 / static_cast<double>(frameTimeMs));

        esp_camera_fb_return(frame);

        if (!sendOk) {
            break;
        }
    }

    client.stop();
    Serial.println("Image stream ended");
}
#endif

/** Routes an HTTP connection to its requested camera, IMU, or UWB stream. */
void handleClient(WiFiClient client) {
    client.setTimeout(2000);
    const String requestLine = client.readStringUntil('\n');
    const bool imuRequest = requestLine.startsWith("GET /imu/stream ");
    const bool uwbRequest = requestLine.startsWith("GET /uwb/stream ");
#if !defined(TAG_M5STICKC_PLUS)
    const bool cameraRequest = requestLine.startsWith("GET /stream ");
#endif

    if (imuRequest) {
        streamImu(client);
    } else if (uwbRequest) {
        streamUwb(client);
#if !defined(TAG_M5STICKC_PLUS)
    } else if (cameraRequest) {
        streamJpeg(client);
#endif
    } else {
        client.println("HTTP/1.1 404 Not Found");
        client.println("Content-Type: text/plain");
        client.println("Connection: close");
        client.println();
        client.println(
#if defined(TAG_M5STICKC_PLUS)
            "Use /imu/stream or /uwb/stream"
#else
            "Use /stream, /imu/stream, or /uwb/stream"
#endif
        );
        client.stop();
    }
}

/** Owns one HTTP client connection in a dedicated FreeRTOS task. */
void clientTask(void* parameter) {
    WiFiClient client = *static_cast<WiFiClient*>(parameter);
    delete static_cast<WiFiClient*>(parameter);
    handleClient(client);
    vTaskDelete(nullptr);
}

}  // namespace

/** Initializes board hardware, sensors, Wi-Fi connection, and the HTTP server. */
void setup() {
    Serial.begin(115200);

#if defined(TAG_M5STICKC_PLUS)
    selectStickCPlusPanel();
    auto config = M5.config();
    config.internal_imu = false;  // Initialize MPU6886 on the explicit I2C bus below.
    M5.begin(config);
    log_i("M5StickC Plus tag mode: board=%d display=%d",
          static_cast<int>(M5.getBoard()), static_cast<int>(M5.Display.getBoard()));
    M5.Display.setBrightness(0);
#else
    pinMode(POWER_GPIO_NUM, OUTPUT);
    digitalWrite(POWER_GPIO_NUM, LOW);
    delay(500);

    const esp_err_t cameraError = esp_camera_init(&cameraConfig);
    if (cameraError != ESP_OK) {
        Serial.printf("Camera init failed: 0x%lx\n", cameraError);
        delay(1000);
        esp_restart();
    }
    Serial.println("Camera init succeeded");
#endif

    imuMutex = xSemaphoreCreateMutex();
    if (imuMutex == nullptr) {
        Serial.println("IMU: mutex allocation failed");
    } else {
        xTaskCreatePinnedToCore(initializeImuTask, "imu-init", 8192, nullptr, 1, nullptr, 0);
    }

    uwbMutex = xSemaphoreCreateMutex();
    if (uwbMutex == nullptr) {
        log_e("UWB: mutex allocation failed");
    } else {
        initializeUwb();
    }

    while (!connectToWifi()) {
        log_w("WiFi: retrying in 5 seconds");
        delay(5000);
    }
    server.begin();
    log_i("HTTP server listening on port 80");
}

/** Accepts HTTP connections and moves each one to an independent task. */
void loop() {
#if defined(TAG_M5STICKC_PLUS)
    M5.update();
    if (M5.BtnA.wasPressed() || M5.BtnA.wasClicked()) {
        log_i("Display: Button A wake request");
        wakeStickDisplay();
    }
    updateStickDisplay();
#endif

    WiFiClient client = server.available();
    if (!client) {
        delay(1);
        return;
    }

    auto* clientCopy = new WiFiClient(client);
    xTaskCreatePinnedToCore(clientTask, "http", 8192, clientCopy, 1, nullptr, 1);
}
