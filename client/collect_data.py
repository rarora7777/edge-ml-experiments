#!/usr/bin/env python3
import asyncio
import csv
import os
import struct
import sys
import time
import argparse
import random
import math
from datetime import datetime

# Custom BLE UUIDs (must match the firmware exactly)
SERVICE_UUID = "23a00000-0df2-432d-a23e-63f59e9a1122"
CHARACTERISTIC_UUID = "23a00001-0df2-432d-a23e-63f59e9a1122"
PACKET_FORMAT = "<IffffffB"  # uint32, float*6, uint8 (29 bytes)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# Statistics
packet_count = 0
start_time = 0
last_report_time = 0
fps_counter = 0
current_fps = 0.0
last_packet_data = None
csv_writer = None
csv_file = None

def parse_packet(data: bytearray):
    """Unpacks the binary packet received from the M5StickC."""
    if len(data) != PACKET_SIZE:
        raise ValueError(f"Expected {PACKET_SIZE} bytes, but got {len(data)} bytes.")
    
    # Unpack little-endian: uint32 (timestamp), 6x float (accel + gyro), uint8 (label)
    unpacked = struct.unpack(PACKET_FORMAT, data)
    timestamp_ms, ax, ay, az, gx, gy, gz, label = unpacked
    
    label_name = "hand" if label == 0 else "pocket"
    return {
        "timestamp_ms": timestamp_ms,
        "accel_x": ax,
        "accel_y": ay,
        "accel_z": az,
        "gyro_x": gx,
        "gyro_y": gy,
        "gyro_z": gz,
        "label": label,
        "label_name": label_name
    }

def print_stats(parsed_data, source="BLE"):
    """Prints a rolling summary of received IMU data in place."""
    global packet_count, start_time, last_report_time, fps_counter, current_fps
    
    packet_count += 1
    fps_counter += 1
    now = time.time()
    
    # Calculate sampling rate (FPS) every second
    if now - last_report_time >= 1.0:
        current_fps = fps_counter / (now - last_report_time)
        fps_counter = 0
        last_report_time = now
        
    elapsed = now - start_time
    
    # Format current label with color for readability
    label_str = parsed_data['label_name'].upper()
    if parsed_data['label'] == 0:
        label_color = "\033[96m"  # Cyan for HAND
    else:
        label_color = "\033[93m"  # Yellow/Orange for POCKET
    reset_color = "\033[0m"
    
    # Print status in place using carriage return
    sys.stdout.write(
        f"\r[{source}] Elapsed: {elapsed:6.1f}s | "
        f"Packets: {packet_count:6d} | "
        f"Rate: {current_fps:4.1f} Hz | "
        f"Label: {label_color}{label_str}{reset_color} (val: {parsed_data['label']}) | "
        f"Acc: ({parsed_data['accel_x']:+5.2f}, {parsed_data['accel_y']:+5.2f}, {parsed_data['accel_z']:+5.2f}) g | "
        f"Gyro: ({parsed_data['gyro_x']:+7.1f}, {parsed_data['gyro_y']:+7.1f}, {parsed_data['gyro_z']:+7.1f}) d/s"
    )
    sys.stdout.flush()

def handle_data(parsed_data, source="BLE"):
    """Writes parsed data to the CSV and updates the terminal display."""
    global csv_writer, csv_file

    assert csv_writer is not None, "CSV writer is not initialized."
    assert csv_file is not None, "CSV file is not initialized."
    
    # Write to CSV
    if csv_writer:
        csv_writer.writerow([
            parsed_data["timestamp_ms"],
            parsed_data["accel_x"],
            parsed_data["accel_y"],
            parsed_data["accel_z"],
            parsed_data["gyro_x"],
            parsed_data["gyro_y"],
            parsed_data["gyro_z"],
            parsed_data["label"],
            parsed_data["label_name"]
        ])
        # Flush frequently so data is persisted in case of interruption
        if packet_count % 10 == 0:
            csv_file.flush()
            
    print_stats(parsed_data, source)

def notification_handler(sender, data):
    """Callback for Bleak notifications."""
    try:
        parsed = parse_packet(data)
        handle_data(parsed, source="BLE")
    except Exception as e:
        print(f"\nError processing BLE notification: {e}")

async def run_mock_stream(sample_rate_hz=50):
    """Generates mock IMU walking data when hardware is not connected."""
    global start_time
    print("\n--- Running in MOCK Mode ---")
    print("Generates synthetic walking data with periodic hand/pocket label switches.")
    print("Press Ctrl+C to stop.\n")
    
    start_time = time.time()
    t = 0.0
    dt = 1.0 / sample_rate_hz
    mock_label = 0  # 0 = hand, 1 = pocket
    last_label_switch = time.time()
    
    # Base walking frequency in Hz (typical human walking step rate)
    walking_freq = 1.8 
    
    while True:
        # Toggle label every 12 seconds automatically
        current_time = time.time()
        if current_time - last_label_switch >= 12.0:
            mock_label = 1 if mock_label == 0 else 0
            last_label_switch = current_time
            sys.stdout.write(f"\n[MOCK] Auto-switching label to: {'POCKET' if mock_label == 1 else 'HAND'}\n")
            
        # Synthesize walking accelerations and rotations
        # Human walking creates periodic vertical (Z or Y depending on orientation) and forward/backward oscillations
        angle = 2 * math.pi * walking_freq * t
        
        # Add basic gait dynamics + sensor noise
        def noise(scale):
            return random.normalvariate(0, scale)
        
        if mock_label == 0:
            # WALKING IN HAND: Multi-axis movements, arm swing rotation (larger gyro signals)
            # Arm swing matches step frequency, with some phase offset
            ax = 0.1 * math.sin(angle) + noise(0.05)
            ay = 0.2 * math.cos(angle) + noise(0.05)
            az = 1.0 + 0.3 * math.sin(2 * angle) + noise(0.08)  # Vertical bounce (2x step frequency)
            
            # Arm swing angular velocity (gyro)
            gx = 40.0 * math.sin(angle) + noise(5)
            gy = 20.0 * math.cos(angle) + noise(5)
            gz = 15.0 * math.sin(angle) + noise(5)
        else:
            # WALKING IN POCKET: Tighter constraints, less rotation, higher vertical impacts (hip bounce)
            ax = 0.05 * math.sin(angle) + noise(0.03)
            ay = 0.15 * math.sin(angle) + noise(0.03)
            az = 1.0 + 0.5 * math.sin(2 * angle) + noise(0.05)  # More pronounced vertical hip movement
            
            # Limited angular velocity in pocket
            gx = 10.0 * math.sin(angle) + noise(2)
            gy = 8.0 * math.cos(angle) + noise(2)
            gz = 5.0 * math.sin(angle) + noise(2)
            
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        parsed = {
            "timestamp_ms": timestamp_ms,
            "accel_x": ax,
            "accel_y": ay,
            "accel_z": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
            "label": mock_label,
            "label_name": "hand" if mock_label == 0 else "pocket"
        }
        
        handle_data(parsed, source="MCK")
        t += dt
        await asyncio.sleep(dt)

async def run_ble_stream(device_name, timeout):
    """Discovers and connects to the M5StickC BLE device and handles streaming."""
    global start_time
    from bleak import BleakScanner, BleakClient
    # from bleak.exc import BleakError
    
    print(f"Scanning for BLE device with name containing '{device_name}'...")
    
    # Filter by name
    device = await BleakScanner.find_device_by_filter(
        lambda d, _: bool(d.name and device_name.lower() in d.name.lower()),
        timeout=timeout
    )
    
    if not device:
        print(f"\nError: Could not find device '{device_name}' within {timeout} seconds.")
        print("Please ensure your M5StickC is turned on, advertising, and within range.")
        return
        
    print(f"Found device: {device.name} [{device.address}]")
    print("Connecting...")
    
    # Connection callbacks
    def on_disconnect(client):
        print("\nDisconnected from M5StickC device.")
        
    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        print("Connected! Initializing services...")
        
        # Verify custom service and characteristic
        services = client.services
        char = services.get_characteristic(CHARACTERISTIC_UUID)
        if not char:
            print(f"Error: Characteristic {CHARACTERISTIC_UUID} not found on this device.")
            return
            
        print("M5StickC IMU Service found. Starting stream...")
        start_time = time.time()
        
        # Subscribe to notifications
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        
        print("Logging data. Press Ctrl+C to stop recording.\n")
        
        # Keep client connection alive
        while client.is_connected:
            await asyncio.sleep(0.5)

def main():
    global csv_writer, csv_file
    
    parser = argparse.ArgumentParser(description="M5StickC IMU BLE Data Collector")
    parser.add_argument("-o", "--output", help="Path to output CSV file (defaults to timestamped name)")
    parser.add_argument("-n", "--name", default="M5StickC-IMU", help="BLE device name to search for (default: M5StickC-IMU)")
    parser.add_argument("-t", "--timeout", type=float, default=15.0, help="Scan timeout in seconds (default: 15.0)")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode to test script functionality")
    args = parser.parse_args()
    
    # Ensure the data output directory exists
    output_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped CSV filename if not specified
    if args.output:
        csv_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"imu_data_{timestamp}.csv")
        
    print(f"Data will be saved to: {csv_path}")
    
    # Open CSV file and write header
    try:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_ms", 
            "accel_x", "accel_y", "accel_z", 
            "gyro_x", "gyro_y", "gyro_z", 
            "label", "label_name"
        ])
    except IOError as e:
        print(f"Error creating CSV file {csv_path}: {e}")
        sys.exit(1)
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        if args.mock:
            loop.run_until_complete(run_mock_stream())
        else:
            loop.run_until_complete(run_ble_stream(args.name, args.timeout))
    except KeyboardInterrupt:
        print("\n\nSession stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
    finally:
        # Clean shutdown: close file, close event loop
        if csv_file:
            csv_file.close()
            print(f"Data saved successfully to {csv_path}")
        loop.close()

if __name__ == "__main__":
    main()
