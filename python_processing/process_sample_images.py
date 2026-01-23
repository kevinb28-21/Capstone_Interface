#!/usr/bin/env python3
"""
Process sample images to populate ML insights page.
This script processes images from the training data and saves them to the database
so they appear on the ML insights page with analysis data.
"""
import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import uuid
from datetime import datetime
import cv2
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from image_processor import analyze_crop_health
from db_utils import get_db_connection, return_db_connection, save_analysis, set_processing_completed
from s3_utils import generate_s3_key

load_dotenv()

def process_sample_images(num_images=5):
    """Process sample images and save to database."""
    
    # Get sample images from training data
    training_dir = Path("training_data/train")
    sample_images = []
    
    # Get a few images from different health categories
    health_categories = ['healthy', 'diseased', 'stressed', 'very_healthy', 'moderate']
    
    for category in health_categories:
        category_dir = training_dir / category
        if category_dir.exists():
            images = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.jpeg')) + list(category_dir.glob('*.png'))
            if images:
                sample_images.append(images[0])
                if len(sample_images) >= num_images:
                    break
    
    if not sample_images:
        print("No sample images found in training_data/train/")
        return
    
    print(f"Processing {len(sample_images)} sample images...")
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for img_path in sample_images:
                try:
                    # Copy image to server uploads directory
                    server_uploads = Path(__file__).parent.parent / "server" / "uploads"
                    server_uploads.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename
                    img_id = str(uuid.uuid4())
                    file_ext = img_path.suffix or '.jpg'
                    filename = f"{img_id}{file_ext}"
                    dest_path = server_uploads / filename
                    
                    # Copy image
                    shutil.copy2(img_path, dest_path)
                    print(f"  Copied {img_path.name} to {dest_path}")
                    
                    # Insert image record
                    cur.execute("""
                        INSERT INTO images (
                            id, filename, original_name, file_path, 
                            uploaded_at, processing_status, s3_stored
                        ) VALUES (
                            %s, %s, %s, %s,
                            CURRENT_TIMESTAMP, 'uploaded', false
                        )
                    """, (img_id, filename, img_path.name, str(dest_path)))
                    
                    # Commit image insert before processing
                    conn.commit()
                    
                    # Process image (without TensorFlow - uses vegetation indices)
                    print(f"  Analyzing {img_path.name}...")
                    try:
                        # Use original image path directly to avoid preprocessing issues
                        analysis_result = analyze_crop_health(
                            str(dest_path),
                            use_tensorflow=False,  # Use vegetation indices only
                            use_multi_crop=False
                        )
                    except Exception as e:
                        print(f"    Warning: {e}, trying with original image...")
                        # Try with original image if processed version fails
                        analysis_result = analyze_crop_health(
                            str(img_path),
                            use_tensorflow=False,
                            use_multi_crop=False
                        )
                    
                    # Save analysis
                    if save_analysis(img_id, analysis_result):
                        # Mark as completed
                        cur.execute("""
                            UPDATE images 
                            SET processing_status = 'completed',
                                processed_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (img_id,))
                        conn.commit()
                        
                        print(f"  ✓ Processed {img_path.name}")
                        print(f"    Health: {analysis_result.get('health_status')}")
                        ndvi_val = analysis_result.get('ndvi_mean')
                        savi_val = analysis_result.get('savi_mean')
                        gndvi_val = analysis_result.get('gndvi_mean')
                        print(f"    NDVI: {ndvi_val if ndvi_val is not None else 'N/A (RGB only)'}")
                        print(f"    SAVI: {savi_val if savi_val is not None else 'N/A (RGB only)'}")
                        print(f"    GNDVI: {gndvi_val if gndvi_val is not None else 'N/A (RGB only)'}")
                        print(f"    Health Score: {analysis_result.get('health_score', 0):.2f}")
                    else:
                        print(f"  ✗ Failed to save analysis for {img_path.name}")
                        cur.execute("""
                            UPDATE images 
                            SET processing_status = 'failed'
                            WHERE id = %s
                        """, (img_id,))
                        conn.commit()
                
                except Exception as e:
                    print(f"  ✗ Error processing {img_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Final commit (though each image is already committed)
            print(f"\n✓ Processed {len(sample_images)} images successfully")
            print(f"  Images are now available on the ML insights page")
            print(f"  Note: RGB images will show 'moderate' health status")
            print(f"        (NDVI/SAVI/GNDVI require NIR band for accurate analysis)")
    
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        return_db_connection(conn)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process sample images for ML insights page')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to process')
    args = parser.parse_args()
    
    process_sample_images(args.num_images)
