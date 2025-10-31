#!/usr/bin/env python3
"""
Upload images to Flask server after mission completion
Usage: python3 upload_images.py --server http://192.168.1.100:5001 --directory /home/pi/drone_images
"""

import requests
import argparse
import json
from pathlib import Path
from time import sleep

def upload_images(server_url, image_dir, delete_after_upload=False):
    """
    Upload images and metadata to Flask server
    
    Args:
        server_url: Base URL of Flask API (e.g., http://192.168.1.100:5001)
        image_dir: Directory containing images
        delete_after_upload: Whether to delete images after successful upload
    """
    image_dir = Path(image_dir)
    upload_url = f"{server_url}/api/upload"
    
    image_files = list(image_dir.glob("*.jpg"))
    total = len(image_files)
    
    print(f"Found {total} images to upload")
    print(f"Server: {server_url}")
    
    uploaded = 0
    failed = 0
    
    for idx, img_file in enumerate(image_files, 1):
        metadata_file = img_file.with_suffix('.json')
        
        try:
            # Read image
            with open(img_file, 'rb') as f:
                files = {'image': (img_file.name, f, 'image/jpeg')}
                
                # Read GPS metadata if available
                data = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as m:
                        gps_data = json.load(m)
                        data['gps'] = json.dumps(gps_data.get('gps', {}))
                
                # Upload
                response = requests.post(upload_url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    uploaded += 1
                    print(f"[{idx}/{total}] ✓ {img_file.name} (NDVI: {result.get('analysis', {}).get('ndvi', 'N/A')})")
                    
                    # Delete if requested
                    if delete_after_upload:
                        img_file.unlink()
                        if metadata_file.exists():
                            metadata_file.unlink()
                else:
                    failed += 1
                    print(f"[{idx}/{total}] ✗ {img_file.name}: {response.status_code}")
        
        except Exception as e:
            failed += 1
            print(f"[{idx}/{total}] ✗ {img_file.name}: {e}")
        
        # Small delay to avoid overwhelming server
        sleep(0.5)
    
    print(f"\nUpload complete: {uploaded} succeeded, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload images to Flask server")
    parser.add_argument("--server", type=str, default="http://localhost:5001",
                       help="Flask server URL (default: http://localhost:5001)")
    parser.add_argument("--directory", type=str, default="/home/pi/drone_images",
                       help="Image directory (default: /home/pi/drone_images)")
    parser.add_argument("--delete", action="store_true",
                       help="Delete images after successful upload")
    
    args = parser.parse_args()
    
    upload_images(args.server, args.directory, args.delete)

