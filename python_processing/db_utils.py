"""
Database Utilities for PostgreSQL
Handles database connections and queries for image processing
"""
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import logging

# Register numpy adapters for psycopg2
try:
    import numpy as np
    psycopg2.extensions.register_adapter(np.integer, lambda x: psycopg2.extensions.AsIs(int(x)))
    psycopg2.extensions.register_adapter(np.floating, lambda x: psycopg2.extensions.AsIs(float(x)))
except ImportError:
    pass

load_dotenv()

logger = logging.getLogger(__name__)

# Database connection pool
_pool = None


def get_db_pool():
    """Get or create database connection pool"""
    global _pool
    if _pool is None:
        try:
            _pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'drone_analytics'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    return _pool


def get_db_connection():
    """Get a database connection from the pool"""
    pool = get_db_pool()
    return pool.getconn()


def return_db_connection(conn):
    """Return a connection to the pool"""
    pool = get_db_pool()
    pool.putconn(conn)


def get_pending_images(limit: int = 10) -> List[Dict]:
    """
    Get images that are pending processing
    
    Returns:
        List of image records with status 'uploaded' (excludes 'processing' to avoid duplicates)
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, filename, original_name, file_path, s3_url, s3_key, s3_stored,
                       captured_at, uploaded_at, processing_status
                FROM images
                WHERE processing_status = 'uploaded'
                ORDER BY uploaded_at ASC
                LIMIT %s
            """, (limit,))
            results = cur.fetchall()
            logger.debug(f"get_pending_images: Found {len(results)} images with status 'uploaded'")
            return results
    except Exception as e:
        logger.error(f"Error fetching pending images: {e}", exc_info=True)
        return []
    finally:
        if conn:
            return_db_connection(conn)


def update_image_status(image_id: str, status: str) -> bool:
    """
    Update image processing status
    
    Args:
        image_id: UUID of the image
        status: New status (uploaded, processing, completed, failed)
    
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE images
                SET processing_status = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (status, image_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating image status: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_db_connection(conn)


def set_processing_started(image_id: str) -> bool:
    """Mark image as processing started"""
    return update_image_status(image_id, 'processing')


def set_processing_completed(image_id: str) -> bool:
    """Mark image as processing completed"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE images
                SET processing_status = 'completed',
                    processed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (image_id,))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Error marking image as completed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_db_connection(conn)


def set_processing_failed(image_id: str, error_message: str = None) -> bool:
    """Mark image processing as failed"""
    return update_image_status(image_id, 'failed')


# Schema cache for column existence (reduces Postgres queries)
_schema_cache = {
    'has_gndvi': None,
    'has_crop_type': None,
    'has_ml_fields': None,
    'has_heuristic_fusion': None,
    'has_fusion_health': None,
    'has_fallback_reason': None,
    'initialized': False
}


def _check_column_exists(column_name: str, cur) -> bool:
    """Check if a column exists in the analyses table (with caching)"""
    # Check cache first
    cache_key = f'has_{column_name}'
    if _schema_cache.get(cache_key) is not None:
        return _schema_cache[cache_key]
    
    # Query database
    try:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='analyses' AND column_name=%s
        """, (column_name,))
        exists = cur.fetchone() is not None
        _schema_cache[cache_key] = exists
        return exists
    except Exception as e:
        logger.warning(f"Error checking column {column_name}: {e}")
        return False


def _initialize_schema_cache(cur):
    """Initialize schema cache by checking all columns at once"""
    if _schema_cache['initialized']:
        return
    
    try:
        # Check all columns in one query
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='analyses' 
            AND column_name IN ('gndvi_mean', 'crop_type', 'band_schema', 
                               'heuristic_fusion_score', 'fusion_health_score', 'fallback_reason')
        """)
        existing_columns = {row[0] for row in cur.fetchall()}
        
        _schema_cache['has_gndvi'] = 'gndvi_mean' in existing_columns
        _schema_cache['has_crop_type'] = 'crop_type' in existing_columns
        _schema_cache['has_ml_fields'] = 'band_schema' in existing_columns
        _schema_cache['has_heuristic_fusion'] = 'heuristic_fusion_score' in existing_columns
        _schema_cache['has_fusion_health'] = 'fusion_health_score' in existing_columns
        _schema_cache['has_fallback_reason'] = 'fallback_reason' in existing_columns
        _schema_cache['initialized'] = True
        
        logger.info(f"Schema cache initialized: {_schema_cache}")
    except Exception as e:
        logger.warning(f"Error initializing schema cache: {e}")
        # Fall back to individual checks
        _schema_cache['initialized'] = True


def _convert_to_python_type(value):
    """Convert numpy types to Python native types for database insertion"""
    if value is None:
        return None
    try:
        import numpy as np
        # Handle numpy scalar types
        if isinstance(value, np.generic):
            return value.item()  # Convert numpy scalar to Python native type
        if isinstance(value, (np.integer, np.floating)):
            return float(value) if isinstance(value, np.floating) else int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except (ImportError, AttributeError):
        pass
    # Ensure we return a Python native type
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    # Last resort: try to convert
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def save_analysis(image_id: str, analysis_data: Dict) -> bool:
    """
    Save analysis results to database
    
    Args:
        image_id: UUID of the image
        analysis_data: Dictionary with analysis results
    
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Initialize schema cache (once per connection)
            _initialize_schema_cache(cur)
            
            # Use cached column existence flags (reduces Postgres queries)
            has_gndvi = _schema_cache.get('has_gndvi', False)
            has_crop_type = _schema_cache.get('has_crop_type', False)
            has_ml_fields = _schema_cache.get('has_ml_fields', False)
            has_heuristic_fusion = _schema_cache.get('has_heuristic_fusion', False)
            has_fusion_health = _schema_cache.get('has_fusion_health', False)
            has_fallback_reason = _schema_cache.get('has_fallback_reason', False)
            
            # Build column lists and values based on available columns
            if has_gndvi and has_crop_type and has_ml_fields:
                # Full schema with all fields
                import json as json_module
                band_schema_json = json_module.dumps(analysis_data.get('band_schema')) if analysis_data.get('band_schema') else None
                health_topk_json = json_module.dumps(analysis_data.get('health_topk', [])) if analysis_data.get('health_topk') else None
                crop_topk_json = json_module.dumps(analysis_data.get('crop_topk', [])) if analysis_data.get('crop_topk') else None
                
                # Get fusion score value (prefer heuristic_fusion_score)
                heuristic_fusion_value = _convert_to_python_type(analysis_data.get('heuristic_fusion_score'))
                fusion_health_value = _convert_to_python_type(analysis_data.get('fusion_health_score')) or heuristic_fusion_value
                
                # Build column list dynamically based on what exists
                columns = [
                    'image_id', 'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max',
                    'savi_mean', 'savi_std', 'savi_min', 'savi_max',
                    'gndvi_mean', 'gndvi_std', 'gndvi_min', 'gndvi_max',
                    'health_score', 'health_status', 'summary',
                    'analysis_type', 'model_version', 'confidence',
                    'crop_type', 'crop_confidence',
                    'band_schema', 'health_topk', 'crop_topk',
                    'inference_time_ms'
                ]
                values = [
                    image_id,
                    _convert_to_python_type(analysis_data.get('ndvi_mean')),
                    _convert_to_python_type(analysis_data.get('ndvi_std')),
                    _convert_to_python_type(analysis_data.get('ndvi_min')),
                    _convert_to_python_type(analysis_data.get('ndvi_max')),
                    _convert_to_python_type(analysis_data.get('savi_mean')),
                    _convert_to_python_type(analysis_data.get('savi_std')),
                    _convert_to_python_type(analysis_data.get('savi_min')),
                    _convert_to_python_type(analysis_data.get('savi_max')),
                    _convert_to_python_type(analysis_data.get('gndvi_mean')),
                    _convert_to_python_type(analysis_data.get('gndvi_std')),
                    _convert_to_python_type(analysis_data.get('gndvi_min')),
                    _convert_to_python_type(analysis_data.get('gndvi_max')),
                    _convert_to_python_type(analysis_data.get('health_score')),
                    analysis_data.get('health_status'),
                    analysis_data.get('summary'),
                    analysis_data.get('analysis_type', 'ndvi_savi_gndvi'),
                    analysis_data.get('model_version'),
                    _convert_to_python_type(analysis_data.get('confidence')),
                    analysis_data.get('crop_type'),
                    _convert_to_python_type(analysis_data.get('crop_confidence')),
                    band_schema_json,
                    health_topk_json,
                    crop_topk_json,
                    analysis_data.get('inference_time_ms')
                ]
                
                # Add fusion score columns if they exist (backward compatibility)
                update_clauses = [
                    'ndvi_mean = EXCLUDED.ndvi_mean', 'ndvi_std = EXCLUDED.ndvi_std',
                    'ndvi_min = EXCLUDED.ndvi_min', 'ndvi_max = EXCLUDED.ndvi_max',
                    'savi_mean = EXCLUDED.savi_mean', 'savi_std = EXCLUDED.savi_std',
                    'savi_min = EXCLUDED.savi_min', 'savi_max = EXCLUDED.savi_max',
                    'gndvi_mean = EXCLUDED.gndvi_mean', 'gndvi_std = EXCLUDED.gndvi_std',
                    'gndvi_min = EXCLUDED.gndvi_min', 'gndvi_max = EXCLUDED.gndvi_max',
                    'health_score = EXCLUDED.health_score',
                    'health_status = EXCLUDED.health_status',
                    'summary = EXCLUDED.summary',
                    'analysis_type = EXCLUDED.analysis_type',
                    'model_version = EXCLUDED.model_version',
                    'confidence = EXCLUDED.confidence',
                    'crop_type = EXCLUDED.crop_type',
                    'crop_confidence = EXCLUDED.crop_confidence',
                    'band_schema = EXCLUDED.band_schema',
                    'health_topk = EXCLUDED.health_topk',
                    'crop_topk = EXCLUDED.crop_topk',
                    'inference_time_ms = EXCLUDED.inference_time_ms'
                ]
                
                if has_heuristic_fusion:
                    columns.append('heuristic_fusion_score')
                    values.append(heuristic_fusion_value)
                    update_clauses.append('heuristic_fusion_score = EXCLUDED.heuristic_fusion_score')
                
                if has_fusion_health:
                    columns.append('fusion_health_score')
                    values.append(fusion_health_value)
                    update_clauses.append('fusion_health_score = EXCLUDED.fusion_health_score')
                
                # Add fallback_reason if column exists
                if has_fallback_reason:
                    fallback_reason = analysis_data.get('fallback_reason')
                    if fallback_reason:
                        columns.append('fallback_reason')
                        values.append(fallback_reason)
                        update_clauses.append('fallback_reason = EXCLUDED.fallback_reason')
                
                columns.extend(['processed_image_path', 'processed_s3_url', 'processed_at'])
                values.extend([
                    analysis_data.get('processed_image_path'),
                    analysis_data.get('processed_s3_url'),
                    'CURRENT_TIMESTAMP'
                ])
                
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['%s'] * len(values))
                update_str = ', '.join(update_clauses) + ', updated_at = CURRENT_TIMESTAMP'
                
                cur.execute(f"""
                    INSERT INTO analyses ({columns_str})
                    VALUES ({placeholders})
                    ON CONFLICT (image_id) DO UPDATE SET {update_str}
                """, values)
            elif has_gndvi and has_crop_type:
                # Include GNDVI and crop_type columns (without ML fields)
                cur.execute("""
                    INSERT INTO analyses (
                        image_id, ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
                        savi_mean, savi_std, savi_min, savi_max,
                        gndvi_mean, gndvi_std, gndvi_min, gndvi_max,
                        health_score, health_status, summary,
                        analysis_type, model_version, confidence,
                        crop_type, crop_confidence,
                        processed_image_path, processed_s3_url,
                        processed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (image_id) DO UPDATE SET
                        ndvi_mean = EXCLUDED.ndvi_mean,
                        ndvi_std = EXCLUDED.ndvi_std,
                        ndvi_min = EXCLUDED.ndvi_min,
                        ndvi_max = EXCLUDED.ndvi_max,
                        savi_mean = EXCLUDED.savi_mean,
                        savi_std = EXCLUDED.savi_std,
                        savi_min = EXCLUDED.savi_min,
                        savi_max = EXCLUDED.savi_max,
                        gndvi_mean = EXCLUDED.gndvi_mean,
                        gndvi_std = EXCLUDED.gndvi_std,
                        gndvi_min = EXCLUDED.gndvi_min,
                        gndvi_max = EXCLUDED.gndvi_max,
                        health_score = EXCLUDED.health_score,
                        health_status = EXCLUDED.health_status,
                        summary = EXCLUDED.summary,
                        analysis_type = EXCLUDED.analysis_type,
                        model_version = EXCLUDED.model_version,
                        confidence = EXCLUDED.confidence,
                        crop_type = EXCLUDED.crop_type,
                        crop_confidence = EXCLUDED.crop_confidence,
                        processed_image_path = EXCLUDED.processed_image_path,
                        processed_s3_url = EXCLUDED.processed_s3_url,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    image_id,
                    _convert_to_python_type(analysis_data.get('ndvi_mean')),
                    _convert_to_python_type(analysis_data.get('ndvi_std')),
                    _convert_to_python_type(analysis_data.get('ndvi_min')),
                    _convert_to_python_type(analysis_data.get('ndvi_max')),
                    _convert_to_python_type(analysis_data.get('savi_mean')),
                    _convert_to_python_type(analysis_data.get('savi_std')),
                    _convert_to_python_type(analysis_data.get('savi_min')),
                    _convert_to_python_type(analysis_data.get('savi_max')),
                    _convert_to_python_type(analysis_data.get('gndvi_mean')),
                    _convert_to_python_type(analysis_data.get('gndvi_std')),
                    _convert_to_python_type(analysis_data.get('gndvi_min')),
                    _convert_to_python_type(analysis_data.get('gndvi_max')),
                    _convert_to_python_type(analysis_data.get('health_score')),
                    analysis_data.get('health_status'),
                    analysis_data.get('summary'),
                    analysis_data.get('analysis_type', 'ndvi_savi_gndvi'),
                    analysis_data.get('model_version'),
                    _convert_to_python_type(analysis_data.get('confidence')),
                    analysis_data.get('crop_type'),
                    _convert_to_python_type(analysis_data.get('crop_confidence')),
                    analysis_data.get('processed_image_path'),
                    analysis_data.get('processed_s3_url'),
                ))
            elif has_gndvi:
                # Include GNDVI columns (without crop_type)
                cur.execute("""
                    INSERT INTO analyses (
                        image_id, ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
                        savi_mean, savi_std, savi_min, savi_max,
                        gndvi_mean, gndvi_std, gndvi_min, gndvi_max,
                        health_score, health_status, summary,
                        analysis_type, model_version, confidence,
                        processed_image_path, processed_s3_url,
                        processed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (image_id) DO UPDATE SET
                        ndvi_mean = EXCLUDED.ndvi_mean,
                        ndvi_std = EXCLUDED.ndvi_std,
                        ndvi_min = EXCLUDED.ndvi_min,
                        ndvi_max = EXCLUDED.ndvi_max,
                        savi_mean = EXCLUDED.savi_mean,
                        savi_std = EXCLUDED.savi_std,
                        savi_min = EXCLUDED.savi_min,
                        savi_max = EXCLUDED.savi_max,
                        gndvi_mean = EXCLUDED.gndvi_mean,
                        gndvi_std = EXCLUDED.gndvi_std,
                        gndvi_min = EXCLUDED.gndvi_min,
                        gndvi_max = EXCLUDED.gndvi_max,
                        health_score = EXCLUDED.health_score,
                        health_status = EXCLUDED.health_status,
                        summary = EXCLUDED.summary,
                        analysis_type = EXCLUDED.analysis_type,
                        model_version = EXCLUDED.model_version,
                        confidence = EXCLUDED.confidence,
                        processed_image_path = EXCLUDED.processed_image_path,
                        processed_s3_url = EXCLUDED.processed_s3_url,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    image_id,
                    _convert_to_python_type(analysis_data.get('ndvi_mean')),
                    _convert_to_python_type(analysis_data.get('ndvi_std')),
                    _convert_to_python_type(analysis_data.get('ndvi_min')),
                    _convert_to_python_type(analysis_data.get('ndvi_max')),
                    _convert_to_python_type(analysis_data.get('savi_mean')),
                    _convert_to_python_type(analysis_data.get('savi_std')),
                    _convert_to_python_type(analysis_data.get('savi_min')),
                    _convert_to_python_type(analysis_data.get('savi_max')),
                    _convert_to_python_type(analysis_data.get('gndvi_mean')),
                    _convert_to_python_type(analysis_data.get('gndvi_std')),
                    _convert_to_python_type(analysis_data.get('gndvi_min')),
                    _convert_to_python_type(analysis_data.get('gndvi_max')),
                    _convert_to_python_type(analysis_data.get('health_score')),
                    analysis_data.get('health_status'),
                    analysis_data.get('summary'),
                    analysis_data.get('analysis_type', 'ndvi_savi_gndvi'),
                    analysis_data.get('model_version'),
                    _convert_to_python_type(analysis_data.get('confidence')),
                    analysis_data.get('processed_image_path'),
                    analysis_data.get('processed_s3_url'),
                ))
            else:
                # Fallback without GNDVI columns
                if has_crop_type:
                    # Include crop_type but not GNDVI
                    cur.execute("""
                        INSERT INTO analyses (
                            image_id, ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
                            savi_mean, savi_std, savi_min, savi_max,
                            health_score, health_status, summary,
                            analysis_type, model_version, confidence,
                            crop_type, crop_confidence,
                            processed_image_path, processed_s3_url,
                            processed_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s,
                            CURRENT_TIMESTAMP
                        )
                        ON CONFLICT (image_id) DO UPDATE SET
                            ndvi_mean = EXCLUDED.ndvi_mean,
                            ndvi_std = EXCLUDED.ndvi_std,
                            ndvi_min = EXCLUDED.ndvi_min,
                            ndvi_max = EXCLUDED.ndvi_max,
                            savi_mean = EXCLUDED.savi_mean,
                            savi_std = EXCLUDED.savi_std,
                            savi_min = EXCLUDED.savi_min,
                            savi_max = EXCLUDED.savi_max,
                            health_score = EXCLUDED.health_score,
                            health_status = EXCLUDED.health_status,
                            summary = EXCLUDED.summary,
                            analysis_type = EXCLUDED.analysis_type,
                            model_version = EXCLUDED.model_version,
                            confidence = EXCLUDED.confidence,
                            crop_type = EXCLUDED.crop_type,
                            crop_confidence = EXCLUDED.crop_confidence,
                            processed_image_path = EXCLUDED.processed_image_path,
                            processed_s3_url = EXCLUDED.processed_s3_url,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        image_id,
                        _convert_to_python_type(analysis_data.get('ndvi_mean')),
                        _convert_to_python_type(analysis_data.get('ndvi_std')),
                        _convert_to_python_type(analysis_data.get('ndvi_min')),
                        _convert_to_python_type(analysis_data.get('ndvi_max')),
                        _convert_to_python_type(analysis_data.get('savi_mean')),
                        _convert_to_python_type(analysis_data.get('savi_std')),
                        _convert_to_python_type(analysis_data.get('savi_min')),
                        _convert_to_python_type(analysis_data.get('savi_max')),
                        _convert_to_python_type(analysis_data.get('health_score')),
                        analysis_data.get('health_status'),
                        analysis_data.get('summary'),
                        analysis_data.get('analysis_type', 'ndvi_savi_gndvi'),
                        analysis_data.get('model_version'),
                        _convert_to_python_type(analysis_data.get('confidence')),
                        analysis_data.get('crop_type'),
                        _convert_to_python_type(analysis_data.get('crop_confidence')),
                        analysis_data.get('processed_image_path'),
                        analysis_data.get('processed_s3_url'),
                    ))
                else:
                    # No GNDVI, no crop_type
                    cur.execute("""
                        INSERT INTO analyses (
                            image_id, ndvi_mean, ndvi_std, ndvi_min, ndvi_max,
                            savi_mean, savi_std, savi_min, savi_max,
                            health_score, health_status, summary,
                            analysis_type, model_version, confidence,
                            processed_image_path, processed_s3_url,
                            processed_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            CURRENT_TIMESTAMP
                        )
                        ON CONFLICT (image_id) DO UPDATE SET
                            ndvi_mean = EXCLUDED.ndvi_mean,
                            ndvi_std = EXCLUDED.ndvi_std,
                            ndvi_min = EXCLUDED.ndvi_min,
                            ndvi_max = EXCLUDED.ndvi_max,
                            savi_mean = EXCLUDED.savi_mean,
                            savi_std = EXCLUDED.savi_std,
                            savi_min = EXCLUDED.savi_min,
                            savi_max = EXCLUDED.savi_max,
                            health_score = EXCLUDED.health_score,
                            health_status = EXCLUDED.health_status,
                            summary = EXCLUDED.summary,
                            analysis_type = EXCLUDED.analysis_type,
                            model_version = EXCLUDED.model_version,
                            confidence = EXCLUDED.confidence,
                            processed_image_path = EXCLUDED.processed_image_path,
                            processed_s3_url = EXCLUDED.processed_s3_url,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        image_id,
                        _convert_to_python_type(analysis_data.get('ndvi_mean')),
                        _convert_to_python_type(analysis_data.get('ndvi_std')),
                        _convert_to_python_type(analysis_data.get('ndvi_min')),
                        _convert_to_python_type(analysis_data.get('ndvi_max')),
                        _convert_to_python_type(analysis_data.get('savi_mean')),
                        _convert_to_python_type(analysis_data.get('savi_std')),
                        _convert_to_python_type(analysis_data.get('savi_min')),
                        _convert_to_python_type(analysis_data.get('savi_max')),
                        _convert_to_python_type(analysis_data.get('health_score')),
                        analysis_data.get('health_status'),
                        analysis_data.get('summary'),
                        analysis_data.get('analysis_type', 'ndvi_savi_gndvi'),
                        analysis_data.get('model_version'),
                        _convert_to_python_type(analysis_data.get('confidence')),
                        analysis_data.get('processed_image_path'),
                        analysis_data.get('processed_s3_url'),
                    ))
            
            # Store GNDVI in a JSON field or separate table if schema doesn't support it yet
            # For now, we'll store it in the summary or create a migration script
            
            # Save stress zones if provided
            if 'stress_zones' in analysis_data and analysis_data['stress_zones']:
                # Get analysis ID
                cur.execute("SELECT id FROM analyses WHERE image_id = %s", (image_id,))
                analysis_result = cur.fetchone()
                if analysis_result:
                    analysis_id = analysis_result[0]
                    # Delete existing stress zones
                    cur.execute("DELETE FROM stress_zones WHERE analysis_id = %s", (analysis_id,))
                    # Insert new stress zones
                    for zone in analysis_data['stress_zones']:
                        cur.execute("""
                            INSERT INTO stress_zones (
                                analysis_id, grid_x, grid_y, severity, ndvi_value, savi_value
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (analysis_id, grid_x, grid_y) DO UPDATE SET
                                severity = EXCLUDED.severity,
                                ndvi_value = EXCLUDED.ndvi_value,
                                savi_value = EXCLUDED.savi_value
                        """, (
                            analysis_id,
                            zone.get('x', zone.get('grid_x', 0)),
                            zone.get('y', zone.get('grid_y', 0)),
                            zone.get('severity', 0.5),
                            zone.get('ndvi', zone.get('ndvi_value')),
                            zone.get('savi', zone.get('savi_value'))
                        ))
            
            conn.commit()
            
            # Get the analysis ID for logging
            cur.execute("SELECT id FROM analyses WHERE image_id = %s", (image_id,))
            analysis_result = cur.fetchone()
            analysis_id = analysis_result[0] if analysis_result else None
            logger.info(f"[{image_id}] Saved analysis row with id {analysis_id}")
            
            return True
    except Exception as e:
        logger.error(f"Error saving analysis for image {image_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_db_connection(conn)


def get_image_path(image_record: Dict) -> Optional[str]:
    """
    Get local file path for an image record with robust path resolution
    Tries multiple locations in order of priority
    
    Returns:
        Local file path or None if not found
    """
    filename = image_record.get('filename')
    if not filename:
        return None
    
    # Priority 1: Use stored file_path if it exists and file exists
    stored_path = image_record.get('file_path')
    if stored_path and os.path.exists(stored_path):
        return stored_path
    
    # Priority 2: Try server/uploads directory (most common location)
    # This is where Node.js backend saves files
    server_uploads = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'server', 'uploads', filename
    )
    if os.path.exists(server_uploads):
        return server_uploads
    
    # Priority 3: Try configured UPLOAD_FOLDER
    upload_folder = os.getenv('UPLOAD_FOLDER', './uploads')
    if upload_folder and upload_folder != './uploads':  # Only if explicitly set
        constructed_path = os.path.join(upload_folder, filename)
        if os.path.exists(constructed_path):
            return constructed_path
    
    # Priority 4: Try python_processing/uploads (fallback)
    python_uploads = os.path.join(
        os.path.dirname(__file__), 
        'uploads', filename
    )
    if os.path.exists(python_uploads):
        return python_uploads
    
    # Priority 5: Try relative uploads folder
    relative_uploads = os.path.join('uploads', filename)
    if os.path.exists(relative_uploads):
        return os.path.abspath(relative_uploads)
    
    return None


def test_connection() -> bool:
    """Test database connection"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return_db_connection(conn)
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

