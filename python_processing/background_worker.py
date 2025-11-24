#!/usr/bin/env python3
"""
Background Job Service - Image Processing Worker
Monitors database for new image uploads and automatically processes them.

Status Flow:
    uploaded → processing → completed
    (or failed if error occurs)
"""
import time
import logging
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv
from image_processor import analyze_crop_health, calculate_savi, calculate_gndvi
from db_utils import (
    get_pending_images,
    set_processing_started,
    set_processing_completed,
    set_processing_failed,
    save_analysis,
    get_image_path,
    get_db_connection,
    return_db_connection,
    test_connection
)
from s3_utils import upload_to_s3, generate_s3_key, download_from_s3
import os

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('background_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
POLL_INTERVAL = int(os.getenv('WORKER_POLL_INTERVAL', '10'))  # seconds
BATCH_SIZE = int(os.getenv('WORKER_BATCH_SIZE', '5'))  # images per batch
# Use server uploads directory if images are stored there
SERVER_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server', 'uploads')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', SERVER_UPLOAD_DIR if os.path.exists(SERVER_UPLOAD_DIR) else './uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', './processed')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info("Shutdown signal received, stopping worker...")
    running = False


def repair_image_paths():
    """
    Repair missing file_paths in database for images that exist locally
    This runs once at startup to fix any existing issues
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Find images with missing file_path but existing files
            cur.execute("""
                SELECT id, filename, s3_stored, file_path
                FROM images 
                WHERE (file_path IS NULL OR file_path = '')
                AND processing_status IN ('uploaded', 'failed')
                AND filename IS NOT NULL
            """)
            
            repaired = 0
            for row in cur.fetchall():
                image_id, filename, s3_stored, existing_path = row
                
                # Try to find the file
                test_record = {
                    'id': image_id,
                    'filename': filename,
                    's3_stored': s3_stored,
                    'file_path': existing_path
                }
                
                file_path = get_image_path(test_record)
                if file_path and os.path.exists(file_path):
                    # Update database
                    try:
                        cur.execute("""
                            UPDATE images 
                            SET file_path = %s, s3_stored = false 
                            WHERE id = %s
                        """, (file_path, image_id))
                        repaired += 1
                        logger.info(f"Repaired file_path for image {image_id}: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to update path for {image_id}: {e}")
            
            conn.commit()
            if repaired > 0:
                logger.info(f"✓ Repaired {repaired} image file path(s)")
            else:
                logger.info("✓ No image paths needed repair")
        
        return_db_connection(conn)
    except Exception as e:
        logger.error(f"Error repairing image paths: {e}", exc_info=True)
        if conn:
            return_db_connection(conn)


def download_image_if_needed(image_record: dict) -> str:
    """
    Get local file path for image, downloading from S3 if needed
    Automatically repairs missing file_paths in database
    
    Args:
        image_record: Image record from database
    
    Returns:
        Local file path to image
    
    Raises:
        FileNotFoundError: If image cannot be found or downloaded
    """
    image_id = image_record.get('id')
    filename = image_record.get('filename', 'unknown')
    
    # Step 1: If stored in S3, try to download it
    if image_record.get('s3_stored') and image_record.get('s3_key'):
        logger.info(f"Downloading image from S3: {image_record['s3_key']}")
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download from S3
        if download_from_s3(image_record['s3_key'], local_path):
            if os.path.exists(local_path):
                # Update database with local path for future reference
                _update_file_path_in_db(image_id, local_path)
                return local_path
        else:
            logger.warning(f"Failed to download from S3, trying local path")
    
    # Step 2: Try to get local file path using robust resolution
    file_path = get_image_path(image_record)
    if file_path and os.path.exists(file_path):
        # Update database if file_path was missing or incorrect
        if not image_record.get('file_path') or image_record.get('file_path') != file_path:
            _update_file_path_in_db(image_id, file_path, set_s3_stored_false=True)
            logger.info(f"Auto-repaired file_path for image {image_id}: {file_path}")
        return file_path
    
    # Step 3: Last resort - try constructing path from filename in UPLOAD_FOLDER
    if filename:
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(local_path):
            _update_file_path_in_db(image_id, local_path, set_s3_stored_false=True)
            logger.info(f"Found file in UPLOAD_FOLDER, updated database: {local_path}")
            return local_path
    
    # If we get here, file doesn't exist anywhere
    error_msg = (
        f"Image file not found for {image_id} (filename: {filename}). "
        f"Tried locations: file_path={image_record.get('file_path')}, "
        f"server/uploads, python_processing/uploads, UPLOAD_FOLDER={UPLOAD_FOLDER}, "
        f"s3_stored={image_record.get('s3_stored')}"
    )
    raise FileNotFoundError(error_msg)


def _update_file_path_in_db(image_id: str, file_path: str, set_s3_stored_false: bool = False):
    """
    Helper function to update file_path in database
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if set_s3_stored_false:
                cur.execute("""
                    UPDATE images 
                    SET file_path = %s, s3_stored = false 
                    WHERE id = %s
                """, (file_path, image_id))
            else:
                cur.execute("""
                    UPDATE images 
                    SET file_path = %s 
                    WHERE id = %s
                """, (file_path, image_id))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to update file_path in database for {image_id}: {e}")
    finally:
        if conn:
            return_db_connection(conn)


def process_image(image_record: dict) -> bool:
    """
    Process a single image through the analysis pipeline
    
    Args:
        image_record: Image record from database
    
    Returns:
        True if successful, False otherwise
    """
    image_id = image_record['id']
    logger.info(f"Processing image {image_id}: {image_record.get('filename', 'unknown')}")
    
    try:
        # Step 1: Mark as processing
        if not set_processing_started(image_id):
            logger.error(f"Failed to mark image {image_id} as processing")
            return False
        
        # Step 2: Get local file path
        try:
            image_path = download_image_if_needed(image_record)
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {e}")
            set_processing_failed(image_id, str(e))
            return False
        
        # Step 3: Perform analysis (includes NDVI, SAVI, GNDVI)
        # Try to use TensorFlow model if available
        model_path = os.getenv('ONION_MODEL_PATH', './models/onion_crop_health_model.h5')
        use_tensorflow = os.path.exists(model_path) if model_path else False
        
        if use_tensorflow:
            logger.info(f"Using TensorFlow model for analysis: {model_path}")
        else:
            logger.info(f"TensorFlow model not found, using vegetation index analysis only")
        
        logger.info(f"Analyzing onion crop image: {image_path}")
        analysis_result = analyze_crop_health(image_path, use_tensorflow=use_tensorflow, model_path=model_path if use_tensorflow else None)
        
        # Extract TensorFlow model results if available
        tf_results = analysis_result.get('tensorflow', {})
        if tf_results and tf_results.get('model_loaded'):
            # Use TensorFlow classification if available
            analysis_result['health_status'] = tf_results.get('classification', analysis_result.get('health_status'))
            analysis_result['confidence'] = tf_results.get('confidence', 0.0)
            analysis_result['model_version'] = os.path.basename(model_path) if model_path else None
            analysis_result['analysis_type'] = 'tensorflow_onion_classification'
            logger.info(f"TensorFlow classification: {tf_results.get('classification')} (confidence: {tf_results.get('confidence', 0):.2%})")
        
        # Ensure all vegetation indices are calculated
        # (analyze_crop_health should include all, but double-check)
        if 'savi_mean' not in analysis_result:
            try:
                savi_result = calculate_savi(image_path)
                analysis_result.update({
                    'savi_mean': savi_result.get('savi_mean'),
                    'savi_std': savi_result.get('savi_std'),
                    'savi_min': savi_result.get('savi_min'),
                    'savi_max': savi_result.get('savi_max'),
                })
            except Exception as e:
                logger.warning(f"SAVI calculation failed: {e}")
        
        if 'gndvi_mean' not in analysis_result:
            try:
                gndvi_result = calculate_gndvi(image_path)
                analysis_result.update({
                    'gndvi_mean': gndvi_result.get('gndvi_mean'),
                    'gndvi_std': gndvi_result.get('gndvi_std'),
                    'gndvi_min': gndvi_result.get('gndvi_min'),
                    'gndvi_max': gndvi_result.get('gndvi_max'),
                })
            except Exception as e:
                logger.warning(f"GNDVI calculation failed: {e}")
        
        # Step 5: Upload processed image to S3 if available
        processed_path = analysis_result.get('processed_image_path')
        if processed_path and os.path.exists(processed_path):
            processed_s3_key = generate_s3_key(
                f"processed_{image_record['filename']}",
                prefix='processed'
            )
            processed_s3_url = upload_to_s3(processed_path, processed_s3_key, 'image/jpeg')
            if processed_s3_url:
                analysis_result['processed_s3_url'] = processed_s3_url
                logger.info(f"Uploaded processed image to S3: {processed_s3_url}")
        
        # Step 6: Save analysis to database
        if not save_analysis(image_id, analysis_result):
            logger.error(f"Failed to save analysis for image {image_id}")
            set_processing_failed(image_id, "Failed to save analysis")
            return False
        
        # Step 7: Mark as completed
        if not set_processing_completed(image_id):
            logger.error(f"Failed to mark image {image_id} as completed")
            return False
        
        logger.info(f"✓ Successfully processed image {image_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing image {image_id}: {e}", exc_info=True)
        set_processing_failed(image_id, str(e))
        return False


def process_batch():
    """Process a batch of pending images"""
    try:
        # Get pending images
        pending_images = get_pending_images(limit=BATCH_SIZE)
        
        if not pending_images:
            return 0
        
        logger.info(f"Found {len(pending_images)} pending image(s) to process")
        
        processed_count = 0
        for image_record in pending_images:
            if not running:
                break
            
            if process_image(image_record):
                processed_count += 1
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in process_batch: {e}", exc_info=True)
        return 0


def main():
    """Main worker loop"""
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("Background Image Processing Worker Starting")
    logger.info("=" * 60)
    
    # Test database connection
    if not test_connection():
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)
    
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
    logger.info(f"Batch size: {BATCH_SIZE} images")
    
    # Repair any existing image paths at startup
    logger.info("-" * 60)
    logger.info("Repairing image file paths...")
    repair_image_paths()
    logger.info("-" * 60)
    
    logger.info("Worker is running. Press Ctrl+C to stop.")
    logger.info("-" * 60)
    
    consecutive_errors = 0
    max_errors = 10
    
    while running:
        try:
            processed = process_batch()
            
            if processed > 0:
                logger.info(f"Processed {processed} image(s) in this batch")
                consecutive_errors = 0
            else:
                # No images to process, sleep longer
                time.sleep(POLL_INTERVAL)
            
            # Small delay between batches if we processed something
            if processed > 0:
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            running = False
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error in main loop: {e}", exc_info=True)
            
            if consecutive_errors >= max_errors:
                logger.error(f"Too many consecutive errors ({max_errors}). Exiting.")
                running = False
                break
            
            # Exponential backoff on errors
            sleep_time = min(POLL_INTERVAL * (2 ** consecutive_errors), 60)
            logger.info(f"Waiting {sleep_time} seconds before retry...")
            time.sleep(sleep_time)
    
    logger.info("Background worker stopped.")


if __name__ == '__main__':
    main()
