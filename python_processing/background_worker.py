"""
Background Worker for Automated Image Processing
Monitors upload folder and automatically processes new images.
"""
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from image_processor import analyze_crop_health
import json
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', './processed')
PROCESSING_LOG = os.getenv('PROCESSING_LOG', './processing_log.json')


class ImageHandler(FileSystemEventHandler):
    """Handle new image files in upload folder"""
    
    def __init__(self):
        self.processed_files = self.load_processed_log()
    
    def load_processed_log(self):
        """Load list of already processed files"""
        if os.path.exists(PROCESSING_LOG):
            try:
                with open(PROCESSING_LOG, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_files', []))
            except:
                pass
        return set()
    
    def save_processed_log(self, filename):
        """Save processed file to log"""
        self.processed_files.add(filename)
        data = {'processed_files': list(self.processed_files)}
        with open(PROCESSING_LOG, 'w') as f:
            json.dump(data, f)
    
    def on_created(self, event):
        """Called when a new file is created"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image file
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        if file_path.suffix.lower() not in valid_extensions:
            return
        
        # Check if already processed
        if str(file_path) in self.processed_files:
            return
        
        # Wait a moment for file to finish writing
        time.sleep(1)
        
        # Process the image
        print(f"Processing new image: {file_path.name}")
        try:
            analysis = analyze_crop_health(str(file_path), use_tensorflow=False)
            
            # Save analysis results
            result_file = os.path.join(PROCESSED_FOLDER, f"{file_path.stem}_analysis.json")
            with open(result_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"✓ Analysis complete: {file_path.name}")
            print(f"  NDVI: {analysis['ndvi_mean']:.3f}, Status: {analysis['health_status']}")
            
            # Mark as processed
            self.save_processed_log(str(file_path))
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")


def main():
    """Run the background worker"""
    print(f"Monitoring folder: {UPLOAD_FOLDER}")
    print("Press Ctrl+C to stop")
    
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, UPLOAD_FOLDER, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
    print("Worker stopped")


if __name__ == '__main__':
    main()

