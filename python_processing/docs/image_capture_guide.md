# Image Capture Implementation Guide

## Overview

This document describes the implementation of image capture on the Raspberry Pi 4 for drone missions. The system supports two capture modes:

1. **Interval-based capture**: Images taken at fixed time intervals
2. **GPS-triggered capture**: Images taken when the drone reaches specific GPS coordinates

## Architecture

```
Raspberry Pi 4
├── Camera Module (Raspberry Pi Camera or Multispectral)
├── GPS Module (via Pixhawk/MAVLink or standalone GPS)
├── Flight Controller Connection (USB/Serial)
└── Capture Script (Python)
    ├── MAVLink Interface (DroneKit/pymavlink)
    ├── Camera Control (picamera2 or OpenCV)
    └── Image Storage (/home/pi/drone_images/)
```

## Prerequisites

### Hardware Setup

1. **Camera**
   - Raspberry Pi Camera Module v2/v3, or
   - USB camera (tested with Logitech C920), or
   - Multispectral camera (e.g., Parrot Sequoia)

2. **GPS/Telemetry**
   - Pixhawk flight controller connected via USB (`/dev/ttyUSB0`) or Serial (`/dev/serial0`)
   - Or standalone GPS module

3. **Storage**
   - Ensure sufficient SD card space (recommended: 32GB+)
   - External USB drive for longer missions (optional)

### Software Dependencies

Install required Python packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-picamera2 libatlas-base-dev

# For MAVLink communication
pip3 install dronekit pymavlink

# For image processing (optional, for onboard preprocessing)
pip3 install opencv-python numpy pillow

# For GPS coordinates (if using standalone GPS)
pip3 install gpsd-py3 pynmea2
```

### Serial Port Permissions

Grant permissions for serial port access:

```bash
sudo usermod -a -G dialout pi
sudo chmod 666 /dev/ttyUSB0  # Or /dev/serial0
```

**Note**: You may need to logout/login for group changes to take effect.

## Implementation Options

### Option 1: Interval-Based Capture

Captures images at fixed time intervals regardless of location.

#### Implementation

```python
#!/usr/bin/env python3
"""
Interval-based image capture script
Usage: python3 capture_interval.py --interval 5 --duration 3600
"""

import time
import argparse
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2

def capture_interval(camera, interval_seconds, duration_seconds, output_dir):
    """
    Capture images at fixed intervals
    
    Args:
        camera: Picamera2 instance
        interval_seconds: Time between captures (seconds)
        duration_seconds: Total mission duration (seconds)
        output_dir: Directory to save images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    capture_count = 0
    
    print(f"Starting interval capture: {interval_seconds}s interval, {duration_seconds}s duration")
    print(f"Output directory: {output_dir}")
    
    try:
        while time.time() - start_time < duration_seconds:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = output_dir / f"IMG_{timestamp}.jpg"
            
            # Capture image
            camera.capture_file(str(filename))
            capture_count += 1
            
            print(f"Captured: {filename.name} (#{capture_count})")
            
            # Wait for next interval
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\nCapture stopped by user")
    finally:
        print(f"\nTotal images captured: {capture_count}")
        print(f"Average interval: {(time.time() - start_time) / capture_count:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interval-based image capture")
    parser.add_argument("--interval", type=float, default=5.0, 
                       help="Capture interval in seconds (default: 5.0)")
    parser.add_argument("--duration", type=float, default=3600.0,
                       help="Mission duration in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--output", type=str, default="/home/pi/drone_images",
                       help="Output directory (default: /home/pi/drone_images)")
    
    args = parser.parse_args()
    
    # Initialize camera
    camera = Picamera2()
    camera.configure(camera.create_still_configuration())
    camera.start()
    time.sleep(2)  # Allow camera to warm up
    
    try:
        capture_interval(camera, args.interval, args.duration, args.output)
    finally:
        camera.stop()
```

#### Usage

```bash
# Capture every 5 seconds for 1 hour
python3 capture_interval.py --interval 5 --duration 3600

# Capture every 2 seconds for 30 minutes
python3 capture_interval.py --interval 2 --duration 1800 --output /mnt/usb/drone_images
```

#### Advantages
- Simple implementation
- Predictable storage requirements
- No GPS dependency

#### Disadvantages
- May capture redundant images in same area
- May miss important waypoints
- Less efficient storage usage

---

### Option 2: GPS-Triggered Capture

Captures images only when the drone reaches specified GPS waypoints or enters geofenced areas.

#### Implementation

```python
#!/usr/bin/env python3
"""
GPS-triggered image capture script
Connects to Pixhawk via MAVLink and captures images at specified GPS coordinates
Usage: python3 capture_gps_triggered.py --waypoints waypoints.json --tolerance 10
"""

import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2
from dronekit import connect, VehicleMode
import math

class GPSCaptureController:
    def __init__(self, vehicle, camera, waypoints, tolerance_meters=10.0):
        """
        GPS-triggered capture controller
        
        Args:
            vehicle: DroneKit vehicle instance
            camera: Picamera2 instance
            waypoints: List of {lat, lng, name} waypoints
            tolerance_meters: Distance threshold to trigger capture (meters)
        """
        self.vehicle = vehicle
        self.camera = camera
        self.waypoints = waypoints
        self.tolerance_meters = tolerance_meters
        self.captured_waypoints = set()
        self.output_dir = Path("/home/pi/drone_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two GPS coordinates using Haversine formula
        Returns distance in meters
        """
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (math.sin(dphi/2)**2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def check_proximity(self, current_lat, current_lng):
        """Check if current position is near any waypoint"""
        for waypoint in self.waypoints:
            waypoint_id = f"{waypoint['lat']}_{waypoint['lng']}"
            
            # Skip if already captured
            if waypoint_id in self.captured_waypoints:
                continue
            
            distance = self.haversine_distance(
                current_lat, current_lng,
                waypoint['lat'], waypoint['lng']
            )
            
            if distance <= self.tolerance_meters:
                return waypoint, distance
        
        return None, None
    
    def capture_at_waypoint(self, waypoint, gps_data):
        """Capture image at waypoint with GPS metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        waypoint_name = waypoint.get('name', 'WP')
        filename = self.output_dir / f"{waypoint_name}_{timestamp}.jpg"
        
        # Capture image
        self.camera.capture_file(str(filename))
        
        # Save GPS metadata
        metadata = {
            'filename': filename.name,
            'timestamp': datetime.now().isoformat(),
            'waypoint': waypoint,
            'gps': {
                'lat': gps_data['lat'],
                'lng': gps_data['lng'],
                'altitude': gps_data.get('altitude', 0),
                'heading': gps_data.get('heading', 0),
                'ground_speed': gps_data.get('ground_speed', 0)
            }
        }
        
        # Save metadata JSON
        metadata_file = filename.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        waypoint_id = f"{waypoint['lat']}_{waypoint['lng']}"
        self.captured_waypoints.add(waypoint_id)
        
        print(f"✓ Captured at waypoint '{waypoint_name}': {filename.name}")
        return filename, metadata
    
    def run(self):
        """Main capture loop"""
        print(f"GPS-triggered capture active")
        print(f"Waypoints: {len(self.waypoints)}")
        print(f"Tolerance: {self.tolerance_meters}m")
        print(f"Waiting for GPS lock...")
        
        # Wait for GPS
        while not self.vehicle.gps_0.fix_type >= 2:
            print("Waiting for GPS...")
            time.sleep(1)
        
        print("GPS lock acquired!")
        
        last_check_time = 0
        check_interval = 1.0  # Check GPS every second
        
        try:
            while True:
                current_time = time.time()
                
                # Throttle GPS checks
                if current_time - last_check_time < check_interval:
                    time.sleep(0.1)
                    continue
                
                last_check_time = current_time
                
                # Get current position
                if not self.vehicle.location.global_frame:
                    continue
                
                current_lat = self.vehicle.location.global_frame.lat
                current_lng = self.vehicle.location.global_frame.lng
                current_alt = self.vehicle.location.global_frame.alt
                
                # Check proximity to waypoints
                waypoint, distance = self.check_proximity(current_lat, current_lng)
                
                if waypoint:
                    gps_data = {
                        'lat': current_lat,
                        'lng': current_lng,
                        'altitude': current_alt,
                        'heading': self.vehicle.heading,
                        'ground_speed': self.vehicle.groundspeed
                    }
                    
                    self.capture_at_waypoint(waypoint, gps_data)
                
                # Print status
                if len(self.captured_waypoints) < len(self.waypoints):
                    remaining = len(self.waypoints) - len(self.captured_waypoints)
                    print(f"Position: {current_lat:.6f}, {current_lng:.6f} | "
                          f"Remaining waypoints: {remaining}")
                
                # Exit if all waypoints captured
                if len(self.captured_waypoints) >= len(self.waypoints):
                    print("\n✓ All waypoints captured!")
                    break
                    
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        finally:
            print(f"\nTotal waypoints captured: {len(self.captured_waypoints)}/{len(self.waypoints)}")


def load_waypoints(filepath):
    """Load waypoints from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPS-triggered image capture")
    parser.add_argument("--connection", type=str, default="/dev/ttyUSB0",
                       help="Pixhawk connection (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=57600,
                       help="Serial baud rate (default: 57600)")
    parser.add_argument("--waypoints", type=str, required=True,
                       help="JSON file with waypoints")
    parser.add_argument("--tolerance", type=float, default=10.0,
                       help="Waypoint capture tolerance in meters (default: 10.0)")
    
    args = parser.parse_args()
    
    # Load waypoints
    waypoints = load_waypoints(args.waypoints)
    print(f"Loaded {len(waypoints)} waypoints from {args.waypoints}")
    
    # Connect to vehicle
    print(f"Connecting to vehicle at {args.connection}...")
    vehicle = connect(args.connection, baud=args.baud, wait_ready=True)
    print(f"Connected to vehicle: {vehicle.version}")
    
    # Initialize camera
    camera = Picamera2()
    camera.configure(camera.create_still_configuration())
    camera.start()
    time.sleep(2)
    
    try:
        controller = GPSCaptureController(vehicle, camera, waypoints, args.tolerance)
        controller.run()
    finally:
        camera.stop()
        vehicle.close()
```

#### Waypoint File Format

Create a JSON file with waypoints (e.g., `waypoints.json`):

```json
[
  {
    "name": "Field_Section_A",
    "lat": 37.7749,
    "lng": -122.4194
  },
  {
    "name": "Field_Section_B",
    "lat": 37.7755,
    "lng": -122.4200
  },
  {
    "name": "Field_Section_C",
    "lat": 37.7730,
    "lng": -122.4210
  }
]
```

#### Usage

```bash
# Capture at GPS waypoints
python3 capture_gps_triggered.py --waypoints waypoints.json --tolerance 10

# Use different serial port
python3 capture_gps_triggered.py --connection /dev/serial0 --waypoints waypoints.json
```

#### Advantages
- Captures images only at important locations
- More efficient storage usage
- Better coverage of field areas
- Synchronized with mission waypoints

#### Disadvantages
- Requires GPS lock
- More complex implementation
- Depends on flight controller connection

---

### Option 3: Hybrid Mode (Recommended)

Combines interval-based and GPS-triggered capture for reliability.

```python
#!/usr/bin/env python3
"""
Hybrid capture: Interval + GPS-triggered
Ensures images every N seconds, but prioritizes waypoint captures
"""

# Implementation combines both methods
# - Always captures at waypoints (GPS-triggered)
# - Fills gaps with interval captures
# - Prevents duplicate captures near waypoints
```

## Configuration

### Camera Settings

Adjust camera configuration in the capture script:

```python
# High resolution for crop analysis
camera = Picamera2()
config = camera.create_still_configuration(
    main={"size": (4056, 3040)},  # Full resolution for Pi Camera v2
    buffer_count=3
)
camera.configure(config)
camera.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
```

### Storage Management

Monitor disk space and clean up old images:

```bash
# Check available space
df -h /home/pi

# Cleanup script (run before missions)
#!/bin/bash
# Remove images older than 7 days
find /home/pi/drone_images -name "*.jpg" -mtime +7 -delete
```

## Integration with Upload Script

After capture, images are automatically uploaded to the Flask server:

```python
# upload_images.py (runs after mission)
import requests
from pathlib import Path

UPLOAD_URL = "http://your-server-ip:5001/api/upload"
IMAGE_DIR = Path("/home/pi/drone_images")

for img_file in IMAGE_DIR.glob("*.jpg"):
    metadata_file = img_file.with_suffix('.json')
    
    with open(img_file, 'rb') as f, open(metadata_file, 'r') as m:
        files = {'image': f}
        data = {'gps': m.read()} if metadata_file.exists() else {}
        response = requests.post(UPLOAD_URL, files=files, data=data)
        print(f"Uploaded: {img_file.name}")
```

## Troubleshooting

### GPS Not Locking
- Ensure Pixhawk has clear sky view
- Check serial connection: `dmesg | grep tty`
- Verify baud rate matches flight controller settings

### Camera Not Detected
- Check camera connection: `vcgencmd get_camera`
- Enable camera: `sudo raspi-config` → Interface Options → Camera → Enable
- Test with: `libcamera-hello --list-cameras`

### Serial Port Issues
- Check permissions: `ls -l /dev/ttyUSB0`
- Add user to dialout group: `sudo usermod -a -G dialout pi`
- Test connection: `mavproxy.py --master=/dev/ttyUSB0 --baudrate=57600`

## Performance Considerations

- **Interval capture**: 1-2 images/second max (depends on storage write speed)
- **GPS-triggered**: Minimal overhead, captures only when needed
- **Storage**: JPEG compression recommended (balance quality vs size)
- **Processing**: Onboard preprocessing (resize/NDVI) adds latency

## Testing Without Drone

Test GPS-triggered capture using simulated coordinates:

```python
# Simulate GPS movement
simulated_waypoints = [
    {'lat': 37.7749, 'lng': -122.4194, 'name': 'Test_WP1'},
    {'lat': 37.7750, 'lng': -122.4195, 'name': 'Test_WP2'}
]

# Mock vehicle with simulated GPS
class MockVehicle:
    def __init__(self):
        self.location = type('obj', (object,), {
            'global_frame': type('obj', (object,), {
                'lat': 37.7749, 'lng': -122.4194, 'alt': 50
            })()
        })()
        self.heading = 90
        self.groundspeed = 5.0
```

## Next Steps

1. **Geofence Integration**: Add software geofencing (see `geofence_integration.md`)
2. **Real-time Upload**: Stream images during flight (requires strong WiFi)
3. **Onboard Processing**: Pre-compute NDVI before upload
4. **Mission Planning**: Generate waypoints from web app geofence

## References

- [DroneKit Documentation](https://dronekit-python.readthedocs.io/)
- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [MAVLink Protocol](https://mavlink.io/en/)
- [GPS Coordinate Calculations](https://en.wikipedia.org/wiki/Haversine_formula)

