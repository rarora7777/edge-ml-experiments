# UWB Anchor Firmware

PlatformIO firmware for an M5Stack Core2 connected to an M5Stack Unit UWB on
Port A. It configures the Unit UWB as a fixed anchor over UART at 115200 baud.
The Core2 display shows the assigned anchor ID in large type and the latest UWB
command or module response below it.

## Anchor IDs

Use a small, explicit ID for each physical anchor. IDs are part of the UWB
ranging topology, so they must remain stable when an anchor is replaced or
reflashed.

| Physical anchor | PlatformIO environment | UWB ID |
| --- | --- | --- |
| Anchor 66 | `anchor-66` | `0` |
| Anchor 77 | `anchor-77` | `1` |

Add another environment with a distinct `-DUWB_ANCHOR_ID=<n>` for each added
anchor. Use three anchors for unambiguous 2D position fixes.

## Build and upload

```sh
pio run -e anchor-66
pio run -e anchor-66 -t upload --upload-port /dev/cu.usbserial-<port>
pio device monitor --port /dev/cu.usbserial-<port> --baud 115200
```

At boot, the firmware prints the ESP32 factory MAC address. Record it in your
deployment inventory alongside the physical anchor location and its UWB ID.
The factory MAC is a stable identity for the Core2 controller; the UWB ID is
the separate, intentionally assigned identity used by the ranging system.
