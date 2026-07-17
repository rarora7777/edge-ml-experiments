# AtomS3R-M12 / M5StickC Plus IMU and UWB Tag

PlatformIO firmware for an M5Stack AtomS3R-M12 or M5StickC Plus. The Atom
serves an MJPEG camera stream, BMI270/BMM150 IMU samples, and UWB ranges. The
StickC Plus serves its MPU6886 IMU samples and UWB ranges without a camera.

The device operates as the moving **UWB tag**. The fixed Core2 anchor firmware
is a separate PlatformIO project in [`anchors/`](anchors/README.md).

## Hardware

- M5Stack AtomS3R-M12 or M5StickC Plus
- M5Stack Unit UWB connected to Port A
- One or more separately configured UWB anchors

Port A uses UART at 115200 baud. AtomS3R uses `G1` as RX and `G2` as TX;
M5StickC Plus uses `G33` as RX and `G32` as TX. The firmware configures the
Unit UWB as a tag, enables ranging, and preserves each complete UART range
report as received from the unit.

## Wi-Fi setup

Credentials are deliberately not tracked. Create `include/wifi_credentials.h`:

```cpp
#pragma once

#define WIFI_SSID "your-2.4-ghz-network"
#define WIFI_PASSWORD "your-password"
```

Use [`include/wifi_credentials.h.example`](include/wifi_credentials.h.example)
as the template. The ESP32-S3 supports 2.4 GHz Wi-Fi only.

## Build, upload, and monitor

```sh
pio run -e atoms3r-m12
pio run -e atoms3r-m12 -t upload --upload-port /dev/cu.usbmodem101
pio device monitor --port /dev/cu.usbmodem101 --baud 115200
```

Replace the port with the device reported by `pio device list`.

For an M5StickC Plus, use the camera-less tag environment. Its IMU stream has
`"mag_available": false`; a BMM150 HAT is not initialized in this mode.

```sh
pio run -e m5stickc-plus
pio run -e m5stickc-plus -t upload --upload-port /dev/cu.usbserial-<port>
```

## HTTP streams

All endpoints use port `80` and allow cross-origin requests.

| Endpoint | Format | Contents |
| --- | --- | --- |
| `/stream` | MJPEG multipart | VGA JPEG camera frames |
| `/imu/stream` | Server-sent events | 100 Hz JSON acceleration and gyro samples; magnetometer updates at its sensor ODR |
| `/uwb/stream` | Server-sent events | Parsed UWB anchor ranges with SSE keepalives |

`/stream` is available only in the `atoms3r-m12` environment.

Example UWB event:

```text
event: uwb
data: {"timestamp_ms":1234,"sequence":42,"anchor_id":1,"distance_m":2.140,"report":"an1:2.14m"}
```

The firmware accepts Unit UWB reports in the form `an<id>:<distance>m`, emits
only those range reports, and retains the 32 most recent measurements for newly
connected clients. `report` preserves the original UART line for traceability.

## UWB topology

Use fixed anchors with unique UWB IDs and this AtomS3R device as the tag. Two
anchors provide two ranges but leave a mirrored-position ambiguity; use three
or more anchors for an unambiguous 2D position fix. The Unit UWB supports up
to four anchors and one active tag.

Build the first two anchor stations from the separate project:

```sh
cd anchors
pio run -e anchor-66 -t upload --upload-port /dev/cu.usbserial-<port>
pio run -e anchor-77 -t upload --upload-port /dev/cu.usbserial-<port>
```
