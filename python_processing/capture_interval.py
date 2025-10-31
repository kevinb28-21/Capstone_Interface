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
        if capture_count > 0:
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

