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

