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
from multispectral_loader import validate_canonical_band_order
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

# Global model cache (loaded once)
_model_cache = {
    'model': None,
    'model_path': None,
    'model_type': None,
    'health_classes': None,
    'crop_classes': None
}


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


def recover_stuck_images():
    """
    Recover images that are stuck in 'processing' status (likely from a previous crash).
    Resets them to 'uploaded' so they can be processed again.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Find images stuck in processing status (older than 5 minutes)
            cur.execute("""
                SELECT id, filename, updated_at
                FROM images 
                WHERE processing_status = 'processing'
                AND (updated_at IS NULL OR updated_at < NOW() - INTERVAL '5 minutes')
            """)
            
            stuck_images = cur.fetchall()
            if stuck_images:
                logger.warning(f"Found {len(stuck_images)} image(s) stuck in 'processing' status")
                for row in stuck_images:
                    image_id, filename, updated_at = row
                    logger.info(f"Recovering stuck image {image_id} ({filename}), last updated: {updated_at}")
                    cur.execute("""
                        UPDATE images 
                        SET processing_status = 'uploaded',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (image_id,))
                conn.commit()
                logger.info(f"✓ Recovered {len(stuck_images)} stuck image(s) - reset to 'uploaded' status")
                return len(stuck_images)
            else:
                logger.info("✓ No stuck images found")
                return 0
        
        return_db_connection(conn)
    except Exception as e:
        logger.error(f"Error recovering stuck images: {e}", exc_info=True)
        if conn:
            return_db_connection(conn)
        return 0


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


def load_model_once():
    """
    Load model once and cache it globally.
    Called at worker startup.
    """
    global _model_cache
    
    if _model_cache['model'] is not None:
        return _model_cache
    
    multi_crop_model_path = os.getenv('MULTI_CROP_MODEL_PATH')
    multi_crop_model_dir = os.getenv('MULTI_CROP_MODEL_DIR', './models/multi_crop')
    use_multi_crop = os.getenv('USE_MULTI_CROP_MODEL', 'true').lower() == 'true'
    
    # Try to find multi-crop model
    if use_multi_crop and not multi_crop_model_path:
        import glob
        if os.path.exists(multi_crop_model_dir):
            model_files = glob.glob(os.path.join(multi_crop_model_dir, '*_final.h5'))
            if model_files:
                multi_crop_model_path = max(model_files, key=os.path.getmtime)
    
    # Fallback to single-crop model
    single_crop_model_path = os.getenv('ONION_MODEL_PATH', './models/onion_crop_health_model.h5')
    
    model_path = None
    model_type = None
    
    if use_multi_crop and multi_crop_model_path and os.path.exists(multi_crop_model_path):
        model_path = multi_crop_model_path
        model_type = 'multi_crop'
        logger.info(f"Loading multi-crop TensorFlow model: {model_path}")
    elif single_crop_model_path and os.path.exists(single_crop_model_path):
        model_path = single_crop_model_path
        model_type = 'single_crop'
        logger.info(f"Loading single-crop TensorFlow model: {model_path}")
    else:
        logger.info("No TensorFlow model found, will use vegetation index analysis only")
        _model_cache['model'] = None
        _model_cache['model_path'] = None
        _model_cache['model_type'] = None
        return _model_cache
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        # Load class names
        import glob
        import json
        model_dir = os.path.dirname(model_path)
        metadata_files = glob.glob(os.path.join(model_dir, '*_metadata.json'))
        if metadata_files:
            with open(max(metadata_files, key=os.path.getmtime), 'r') as f:
                metadata = json.load(f)
                health_classes = metadata.get('health_classes', [])
                crop_classes = metadata.get('crop_types', [])
        else:
            health_classes = None
            crop_classes = None
        
        _model_cache['model'] = model
        _model_cache['model_path'] = model_path
        _model_cache['model_type'] = model_type
        _model_cache['health_classes'] = health_classes
        _model_cache['crop_classes'] = crop_classes
        
        logger.info(f"✓ Model loaded successfully: {model_type}")
        return _model_cache
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        _model_cache['model'] = None
        return _model_cache


def process_image(image_record: dict) -> bool:
    """
    Process a single image through the analysis pipeline
    
    Args:
        image_record: Image record from database
    
    Returns:
        True if successful, False otherwise
    """
    image_id = image_record['id']
    filename = image_record.get('filename', 'unknown')
    logger.info(f"[{image_id}] Starting processing for {filename}")
    
    # Track if we've marked as processing (to ensure we mark as failed on error)
    marked_as_processing = False
    
    try:
        # Step 1: Mark as processing
        logger.info(f"[{image_id}] Marking as 'processing'")
        if not set_processing_started(image_id):
            logger.error(f"[{image_id}] Failed to mark image as processing")
            return False
        marked_as_processing = True
        
        # Step 2: Get local file path
        try:
            image_path = download_image_if_needed(image_record)
            logger.info(f"[{image_id}] Local file path: {image_path}")
        except FileNotFoundError as e:
            logger.error(f"[{image_id}] Image file not found: {e}")
            set_processing_failed(image_id, str(e))
            return False
        
        # Step 3: Perform analysis (includes NDVI, SAVI, GNDVI)
        # Use cached model if available
        use_tensorflow = _model_cache['model'] is not None
        model_path = _model_cache['model_path']
        use_multi_crop = _model_cache['model_type'] == 'multi_crop'
        
        logger.info(f"[{image_id}] Analyzing crop image: {image_path}")
        analysis_result = analyze_crop_health(
            image_path,
            use_tensorflow=use_tensorflow,
            model_path=model_path if use_tensorflow else None,
            use_multi_crop=use_multi_crop
        )
        
        # Log analysis summary
        ndvi_mean = analysis_result.get('ndvi_mean', 'N/A')
        savi_mean = analysis_result.get('savi_mean', 'N/A')
        gndvi_mean = analysis_result.get('gndvi_mean', 'N/A')
        health_status = analysis_result.get('health_status', 'N/A')
        logger.info(
            f"[{image_id}] Analysis result: NDVI={ndvi_mean}, SAVI={savi_mean}, "
            f"GNDVI={gndvi_mean}, health_status={health_status}"
        )
        
        # Extract TensorFlow model results if available
        tf_results = analysis_result.get('tensorflow', {})
        if tf_results and tf_results.get('model_loaded'):
            # Use TensorFlow classification if available
            if use_multi_crop:
                # Multi-crop model results
                analysis_result['health_status'] = tf_results.get('health_classification', analysis_result.get('health_status'))
                analysis_result['confidence'] = tf_results.get('health_confidence', 0.0)
                analysis_result['crop_type'] = tf_results.get('crop_type', 'unknown')
                analysis_result['crop_confidence'] = tf_results.get('crop_confidence', 0.0)
                analysis_result['model_version'] = os.path.basename(model_path) if model_path else None
                analysis_result['analysis_type'] = 'multi_crop_tensorflow'
                
                # Log processing path and bands
                processing_path = tf_results.get('processing_path', 'unknown')
                band_schema = tf_results.get('band_schema', {})
                bands_used = band_schema.get('bands', [])
                logger.info(
                    f"Multi-crop TensorFlow classification (path: {processing_path}, bands: {bands_used}): "
                    f"crop={tf_results.get('crop_type', 'unknown')} "
                    f"(confidence: {tf_results.get('crop_confidence', 0):.2%}), "
                    f"health={tf_results.get('health_classification')} "
                    f"(confidence: {tf_results.get('health_confidence', 0):.2%})"
                )
                
                # Add additional fields for database
                analysis_result['band_schema'] = band_schema
                analysis_result['health_topk'] = tf_results.get('health_topk', [])
                analysis_result['crop_topk'] = tf_results.get('crop_topk', [])
                analysis_result['inference_time_ms'] = tf_results.get('inference_time_ms')
                analysis_result['heuristic_fusion_score'] = tf_results.get('heuristic_fusion_score')
                
                # Check for missing required bands warning
                if processing_path == 'rgb' and 'NIR' not in bands_used:
                    logger.warning(
                        f"Image {image_id}: Multispectral model requested but NIR band not available. "
                        f"Using RGB path only. Band schema: {band_schema}"
                    )
            else:
                # Single-crop model results (legacy)
                analysis_result['health_status'] = tf_results.get('classification', analysis_result.get('health_status'))
                analysis_result['confidence'] = tf_results.get('confidence', 0.0)
                analysis_result['crop_type'] = tf_results.get('crop_type', 'onion')  # Legacy default
                analysis_result['model_version'] = os.path.basename(model_path) if model_path else None
                analysis_result['analysis_type'] = 'tensorflow_single_crop'
                logger.info(
                    f"TensorFlow classification: {tf_results.get('classification')} "
                    f"(confidence: {tf_results.get('confidence', 0):.2%})"
                )
        
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
        logger.info(f"[{image_id}] Saving analysis to database")
        if not save_analysis(image_id, analysis_result):
            logger.error(f"[{image_id}] Failed to save analysis for image")
            set_processing_failed(image_id, "Failed to save analysis")
            return False
        logger.info(f"[{image_id}] Saved analysis row successfully")
        
        # Step 7: Mark as completed
        logger.info(f"[{image_id}] Marking as 'completed'")
        if not set_processing_completed(image_id):
            logger.error(f"[{image_id}] Failed to mark image as completed - marking as failed")
            set_processing_failed(image_id, "Failed to update status to completed")
            return False
        
        logger.info(f"[{image_id}] ✓ Successfully processed image - status set to 'completed'")
        return True
        
    except Exception as e:
        logger.exception(f"[{image_id}] Error during processing: {e}")
        if marked_as_processing:
            set_processing_failed(image_id, str(e))
            logger.info(f"[{image_id}] Status set to 'failed' due to error")
        return False


def process_batch():
    """Process a batch of pending images"""
    try:
        # Get pending images
        logger.info("[POLL] Checking for pending images...")
        pending_images = get_pending_images(limit=BATCH_SIZE)
        
        if not pending_images:
            return 0
        
        logger.info(f"[POLL] Found {len(pending_images)} pending image(s) to process")
        
        processed_count = 0
        for image_record in pending_images:
            if not running:
                break
            
            # Process each image with individual error handling
            try:
                if process_image(image_record):
                    processed_count += 1
            except Exception as e:
                # Individual image processing error - already logged in process_image
                logger.error(f"Unexpected error processing image {image_record.get('id')}: {e}", exc_info=True)
                # Ensure status is set to failed
                try:
                    set_processing_failed(image_record.get('id'), f"Unexpected error: {str(e)}")
                except Exception as db_error:
                    logger.error(f"Failed to mark image as failed: {db_error}")
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in process_batch: {e}", exc_info=True)
        return 0


"""
================================================================================
TESTING CHECKLIST - Background Worker & Image Processing Pipeline
================================================================================

1. START SERVICES:
   - Start PostgreSQL: Ensure database is running and accessible
   - Start Node backend: cd server && npm run dev (or npm start)
   - Start Python background worker: cd python_processing && python background_worker.py
   - Start frontend: cd client && npm run dev

2. UPLOAD A TEST IMAGE:
   - From the UI (Analytics page), upload an image
   - Verify in the database:
     SELECT id, filename, processing_status, uploaded_at
     FROM images
     ORDER BY uploaded_at DESC
     LIMIT 5;
   - Expect: new image with processing_status = 'uploaded'

3. VERIFY WORKER PROCESSING:
   - Watch the worker logs for:
     - "[POLL] Checking for pending images..."
     - "[POLL] Found X pending image(s) to process"
     - "[{image_id}] Starting processing for {filename}"
     - "[{image_id}] Status set to 'processing'"
     - "[{image_id}] Analysis result: NDVI=..., SAVI=..., GNDVI=..., health_status=..."
     - "[{image_id}] Status set to 'completed'"
   - After a short delay (10-30 seconds), run:
     SELECT id, filename, processing_status, processed_at
     FROM images
     WHERE processing_status IN ('uploaded', 'processing', 'completed', 'failed')
     ORDER BY uploaded_at DESC;
   - Expect: status transitions from 'uploaded' → 'processing' → 'completed'
     (or 'failed' on error)

4. VERIFY ANALYSES TABLE:
   - Check analyses:
     SELECT a.*, i.filename, i.processing_status
     FROM analyses a
     JOIN images i ON a.image_id = i.id
     ORDER BY a.processed_at DESC
     LIMIT 10;
   - Expect: at least one row linked to the new image with:
     - ndvi_mean, savi_mean, gndvi_mean populated
     - health_status, health_score, confidence populated
     - processed_at timestamp set

5. VERIFY /api/images ENDPOINT:
   - Call via curl or browser: GET http://localhost:5050/api/images
   - Confirm that the image appears with:
     - processingStatus: 'completed' (after processing)
     - A non-null analysis object containing:
       - ndvi: { mean, std, min, max }
       - savi: { mean, std, min, max }
       - gndvi: { mean, std, min, max }
       - healthStatus, healthScore, confidence
       - modelVersion (if ML model was used)

6. VERIFY MODEL STATUS:
   - Ensure there is at least one model file in:
     - python_processing/models/multi_crop/*_final.h5 OR
     - python_processing/models/onion_crop_health_model.h5
   - Hit GET http://localhost:5050/api/ml/status:
     - Expect: model_available: true
     - model_type: 'multi_crop' or 'single_crop'
     - model_path: absolute path to model file
     - model_version: version string
   - Confirm UI:
     - On ML page, ModelTraining component shows "Model Status: Available"
     - Displays model type and version

7. FAILURE HANDLING TEST:
   - Intentionally break one image (e.g., delete the file after upload)
   - Upload the image and watch worker logs
   - Confirm:
     - Worker logs error with stack trace
     - images.processing_status becomes 'failed'
     - No stuck 'processing' rows remain
   - Check database:
     SELECT id, filename, processing_status
     FROM images
     WHERE processing_status = 'processing';
     - Expect: 0 rows (or rows older than 5 minutes that will be recovered)

8. RECOVERY MECHANISM TEST:
   - Manually set an image to 'processing' status:
     UPDATE images SET processing_status = 'processing', updated_at = NOW() - INTERVAL '10 minutes'
     WHERE id = '<some-image-id>';
   - Restart the worker
   - Verify:
     - Worker logs: "Found X image(s) stuck in 'processing' status"
     - Status is reset to 'uploaded'
     - Image is processed in the next poll cycle

================================================================================
"""


def main():
    """Main worker loop"""
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("Background Image Processing Worker Starting")
    logger.info("=" * 60)
    
    # Validate canonical band order at startup (fail fast if mismatch)
    try:
        validate_canonical_band_order()
    except RuntimeError as e:
        logger.error(f"Startup validation failed: {e}")
        sys.exit(1)
    
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
    
    # Recover any images stuck in 'processing' status
    logger.info("-" * 60)
    logger.info("Recovering stuck images...")
    recover_stuck_images()
    logger.info("-" * 60)
    
    # Load model once at startup
    logger.info("-" * 60)
    logger.info("Loading ML model...")
    model_cache = load_model_once()
    if model_cache['model'] is not None:
        logger.info(f"Model loaded: {model_cache['model_type']} at {model_cache['model_path']}")
    else:
        logger.warning("No model loaded - will use vegetation index analysis only")
    logger.info("-" * 60)
    
    logger.info("Worker is running. Press Ctrl+C to stop.")
    logger.info("-" * 60)
    
    consecutive_errors = 0
    max_errors = 10
    last_recovery_time = time.time()
    recovery_interval = 300  # Run recovery every 5 minutes
    
    while running:
        try:
            # Periodically recover stuck images
            current_time = time.time()
            if current_time - last_recovery_time >= recovery_interval:
                logger.info("-" * 60)
                logger.info("Running periodic recovery check...")
                recover_stuck_images()
                last_recovery_time = current_time
                logger.info("-" * 60)
            
            processed = process_batch()
            
            if processed > 0:
                logger.info(f"[POLL] Processed {processed} image(s) in this batch")
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
