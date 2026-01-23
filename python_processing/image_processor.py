"""
Image Processing Module for Multi-Crop Plant Health Analysis
Performs NDVI, SAVI, and GNDVI calculation from true spectral bands.
Supports RGB and multispectral inputs with band-aware processing.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Import multispectral loader
try:
    from multispectral_loader import load_multispectral_image, detect_band_schema, create_band_mask
    HAS_MULTISPECTRAL = True
except ImportError:
    HAS_MULTISPECTRAL = False
    logger.warning("multispectral_loader not available, using RGB-only mode")


def calculate_ndvi(image_path: str, band_schema: Optional[Dict] = None, 
                   image_array: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate NDVI (Normalized Difference Vegetation Index) from true spectral bands.
    
    For RGB images: Returns None or error (NDVI requires NIR).
    For multispectral images: Uses true NIR band.
    
    Args:
        image_path: Path to the input image file (if image_array not provided)
        band_schema: Optional band schema dictionary with band order
        image_array: Optional pre-loaded image array (H, W, C)
        
    Returns:
        Dictionary with NDVI statistics, or None if NIR not available
    """
    # Load image if not provided
    if image_array is None:
        if HAS_MULTISPECTRAL:
            image_array, detected_schema = load_multispectral_image(
                image_path, target_size=None, dataset_name=None
            )
            if band_schema is None:
                band_schema = detected_schema
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_array = img_rgb.astype(np.float32) / 255.0
            if band_schema is None:
                band_schema = {'bands': ['R', 'G', 'B'], 'band_order': ['R', 'G', 'B']}
    
    # Get band indices
    band_order = band_schema.get('band_order', ['R', 'G', 'B'])
    
    # Check if required bands (R and NIR) are available
    if 'NIR' not in band_order:
        logger.warning("NDVI calculation requires NIR band, which is not available")
        return {
            'ndvi_mean': None,
            'ndvi_std': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'error': 'NIR band not available',
            'band_schema': band_schema
        }
    
    if 'R' not in band_order:
        logger.warning("NDVI calculation requires R band, which is not available")
        return {
            'ndvi_mean': None,
            'ndvi_std': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'error': 'R band not available',
            'band_schema': band_schema
        }
    
    # Extract bands
    try:
        red_idx = band_order.index('R')
        nir_idx = band_order.index('NIR')
    except ValueError:
        logger.warning(f"Required bands (R, NIR) not found in band_order: {band_order}")
        return {
            'ndvi_mean': None,
            'ndvi_std': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'error': 'Required bands not found',
            'band_schema': band_schema
        }
    
    # Extract red and NIR bands
    red = image_array[:, :, red_idx].astype(np.float32)
    nir = image_array[:, :, nir_idx].astype(np.float32)
    
    # Normalize if needed (assuming [0, 1] range)
    if np.max(red) > 1.0 or np.max(nir) > 1.0:
        red = red / 255.0
        nir = nir / 255.0
    
    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    denominator = red + nir + 1e-7  # Avoid division by zero
    ndvi = (nir - red) / denominator
    
    # Clip NDVI to valid range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)
    
    # Calculate statistics
    mean_ndvi = float(np.mean(ndvi))
    std_ndvi = float(np.std(ndvi))
    min_ndvi = float(np.min(ndvi))
    max_ndvi = float(np.max(ndvi))
    
    # Generate stress zones grid (10x10 for visualization)
    h, w = ndvi.shape
    grid_size = 10
    stress_zones = []
    
    cell_h, cell_w = h // grid_size, w // grid_size
    for y in range(grid_size):
        for x in range(grid_size):
            y_start, y_end = y * cell_h, (y + 1) * cell_h
            x_start, x_end = x * cell_w, (x + 1) * cell_w
            
            cell_ndvi = ndvi[y_start:y_end, x_start:x_end]
            cell_mean = float(np.mean(cell_ndvi))
            
            # Convert to severity (0-1, where 1 is most stressed)
            severity = 1.0 - np.clip((cell_mean + 1) / 2.0, 0, 1)
            
            if severity > 0.3:  # Only flag significant stress
                stress_zones.append({
                    'x': x,
                    'y': y,
                    'severity': round(severity, 2),
                    'ndvi': round(cell_mean, 3)
                })
    
    return {
        'ndvi_mean': round(mean_ndvi, 3),
        'ndvi_std': round(std_ndvi, 3),
        'ndvi_min': round(min_ndvi, 3),
        'ndvi_max': round(max_ndvi, 3),
        'stress_zones': stress_zones,
        'ndvi_map_shape': list(ndvi.shape),
        'stress_count': len(stress_zones),
        'band_schema': band_schema
    }


def calculate_gndvi(image_path: str, band_schema: Optional[Dict] = None,
                    image_array: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate GNDVI (Green Normalized Difference Vegetation Index) from true spectral bands.
    
    GNDVI = (NIR - Green) / (NIR + Green)
    
    GNDVI is particularly useful for early growth stages and is less sensitive to
    atmospheric conditions. Good for onion crops during early development.
    
    Requires NIR band - returns None for RGB-only images.
    
    Args:
        image_path: Path to the input image file (if image_array not provided)
        band_schema: Optional band schema dictionary
        image_array: Optional pre-loaded image array
        
    Returns:
        Dictionary with GNDVI statistics, or None if NIR not available
    """
    # Load image if not provided
    if image_array is None:
        if HAS_MULTISPECTRAL:
            image_array, detected_schema = load_multispectral_image(
                image_path, target_size=None, dataset_name=None
            )
            if band_schema is None:
                band_schema = detected_schema
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_array = img_rgb.astype(np.float32) / 255.0
            if band_schema is None:
                band_schema = {'bands': ['R', 'G', 'B'], 'band_order': ['R', 'G', 'B']}
    
    # Get band indices
    band_order = band_schema.get('band_order', ['R', 'G', 'B'])
    
    # Check if required bands (G and NIR) are available
    if 'NIR' not in band_order:
        logger.warning("GNDVI calculation requires NIR band, which is not available")
        return {
            'gndvi_mean': None,
            'gndvi_std': None,
            'gndvi_min': None,
            'gndvi_max': None,
            'error': 'NIR band not available',
            'band_schema': band_schema
        }
    
    if 'G' not in band_order:
        logger.warning("GNDVI calculation requires G band, which is not available")
        return {
            'gndvi_mean': None,
            'gndvi_std': None,
            'gndvi_min': None,
            'gndvi_max': None,
            'error': 'G band not available',
            'band_schema': band_schema
        }
    
    # Extract bands
    try:
        green_idx = band_order.index('G')
        nir_idx = band_order.index('NIR')
    except ValueError:
        logger.warning(f"Required bands (G, NIR) not found in band_order: {band_order}")
        return {
            'gndvi_mean': None,
            'gndvi_std': None,
            'gndvi_min': None,
            'gndvi_max': None,
            'error': 'Required bands not found',
            'band_schema': band_schema
        }
    
    # Extract green and NIR bands
    green = image_array[:, :, green_idx].astype(np.float32)
    nir = image_array[:, :, nir_idx].astype(np.float32)
    
    # Normalize if needed
    if np.max(green) > 1.0 or np.max(nir) > 1.0:
        green = green / 255.0
        nir = nir / 255.0
    
    # Calculate GNDVI: (NIR - Green) / (NIR + Green)
    denominator = green + nir + 1e-7
    gndvi = (nir - green) / denominator
    
    # Clip GNDVI to valid range [-1, 1]
    gndvi = np.clip(gndvi, -1, 1)
    
    # Calculate statistics
    mean_gndvi = float(np.mean(gndvi))
    std_gndvi = float(np.std(gndvi))
    min_gndvi = float(np.min(gndvi))
    max_gndvi = float(np.max(gndvi))
    
    return {
        'gndvi_mean': round(mean_gndvi, 3),
        'gndvi_std': round(std_gndvi, 3),
        'gndvi_min': round(min_gndvi, 3),
        'gndvi_max': round(max_gndvi, 3),
        'band_schema': band_schema
    }


def calculate_savi(image_path: str, L: float = 0.5, 
                  band_schema: Optional[Dict] = None,
                  image_array: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate SAVI (Soil-Adjusted Vegetation Index) from true spectral bands.
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    Where L is a soil adjustment factor (typically 0.5)
    
    Requires NIR band - returns None for RGB-only images.
    
    Args:
        image_path: Path to the input image file (if image_array not provided)
        L: Soil adjustment factor (default 0.5)
        band_schema: Optional band schema dictionary
        image_array: Optional pre-loaded image array
        
    Returns:
        Dictionary with SAVI statistics, or None if NIR not available
    """
    # Load image if not provided
    if image_array is None:
        if HAS_MULTISPECTRAL:
            image_array, detected_schema = load_multispectral_image(
                image_path, target_size=None, dataset_name=None
            )
            if band_schema is None:
                band_schema = detected_schema
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_array = img_rgb.astype(np.float32) / 255.0
            if band_schema is None:
                band_schema = {'bands': ['R', 'G', 'B'], 'band_order': ['R', 'G', 'B']}
    
    # Get band indices
    band_order = band_schema.get('band_order', ['R', 'G', 'B'])
    
    # Check if NIR is available
    if 'NIR' not in band_order:
        logger.warning("SAVI calculation requires NIR band, which is not available in RGB image")
        return {
            'savi_mean': None,
            'savi_std': None,
            'savi_min': None,
            'savi_max': None,
            'error': 'NIR band not available',
            'band_schema': band_schema
        }
    
    # Extract bands
    try:
        red_idx = band_order.index('R')
        nir_idx = band_order.index('NIR')
    except ValueError:
        logger.warning(f"Required bands (R, NIR) not found in band_order: {band_order}")
        return {
            'savi_mean': None,
            'savi_std': None,
            'savi_min': None,
            'savi_max': None,
            'error': 'Required bands not found',
            'band_schema': band_schema
        }
    
    # Extract red and NIR bands
    red = image_array[:, :, red_idx].astype(np.float32)
    nir = image_array[:, :, nir_idx].astype(np.float32)
    
    # Normalize if needed
    if np.max(red) > 1.0 or np.max(nir) > 1.0:
        red = red / 255.0
        nir = nir / 255.0
    
    # Calculate SAVI: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    denominator = red + nir + L + 1e-7
    savi = ((nir - red) / denominator) * (1 + L)
    
    # Clip SAVI to valid range (typically [-1, 1] but can exceed)
    savi = np.clip(savi, -2, 2)  # SAVI can exceed [-1, 1] range
    
    # Calculate statistics
    mean_savi = float(np.mean(savi))
    std_savi = float(np.std(savi))
    min_savi = float(np.min(savi))
    max_savi = float(np.max(savi))
    
    return {
        'savi_mean': round(mean_savi, 3),
        'savi_std': round(std_savi, 3),
        'savi_min': round(min_savi, 3),
        'savi_max': round(max_savi, 3),
        'band_schema': band_schema
    }


def classify_crop_health_tensorflow(image_path: str, model_path: Optional[str] = None) -> Dict:
    """
    Classify onion crop health using a trained TensorFlow model.
    
    Loads the trained model and runs inference on the input image.
    Optimized for onion (Allium cepa) crop health classification.
    
    Args:
        image_path: Path to the input image
        model_path: Optional path to saved TensorFlow model
                    (defaults to ./models/onion_crop_health_model.h5)
        
    Returns:
        Dictionary with classification results including:
        - classification: Predicted health category
        - confidence: Confidence score (0-1)
        - classes: All available class names
        - all_predictions: Probability distribution across all classes
        - model_loaded: Whether model was successfully loaded
        - crop_type: 'onion'
    """
    import os
    
    # Default model path
    if model_path is None:
        # Try environment variable first
        model_path = os.getenv('ONION_MODEL_PATH', './models/onion_crop_health_model.h5')
    
    # Check if model exists
    if not os.path.exists(model_path):
        return {
            'classification': 'model_not_found',
            'confidence': 0.0,
            'classes': ['very_healthy', 'healthy', 'moderate', 'poor', 'very_poor', 'diseased', 'stressed', 'weeds'],
            'all_predictions': {},
            'model_loaded': False,
            'crop_type': 'onion',
            'error': f'Model file not found: {model_path}'
        }
    
    try:
        # Load model
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        # Load class names
        model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(model_dir, 'onion_class_names.json')
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            # Default class names if file not found
            class_names = ['very_healthy', 'healthy', 'moderate', 'poor', 'very_poor', 'diseased', 'stressed', 'weeds']
        
        # Preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize to model input size (typically 224x224)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        predictions = model.predict(img, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Create prediction dictionary
        all_predictions = {}
        for i, class_name in enumerate(class_names):
            if i < len(predictions[0]):
                all_predictions[class_name] = float(predictions[0][i])
        
        return {
            'classification': class_names[predicted_idx] if predicted_idx < len(class_names) else 'unknown',
            'confidence': round(confidence, 3),
            'classes': class_names,
            'all_predictions': all_predictions,
            'model_loaded': True,
            'crop_type': 'onion',
            'model_path': model_path
        }
        
    except Exception as e:
        return {
            'classification': 'error',
            'confidence': 0.0,
            'classes': ['very_healthy', 'healthy', 'moderate', 'poor', 'very_poor', 'diseased', 'stressed', 'weeds'],
            'all_predictions': {},
            'model_loaded': False,
            'crop_type': 'onion',
            'error': str(e)
        }


def classify_multi_crop_health(
    image_path: str,
    model_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    band_schema: Optional[Dict] = None
) -> Dict:
    """
    Classify multi-crop plant health using band-aware model (no NIR approximation).
    
    Supports cherry tomatoes, onions, and corn with unified health taxonomy.
    Uses separate RGB and multispectral paths based on available bands.
    
    Args:
        image_path: Path to the input image
        model_path: Optional path to saved TensorFlow model
        dataset_name: Optional dataset name for band schema lookup
        band_schema: Optional pre-detected band schema
        
    Returns:
        Dictionary with classification results including:
        - health_classification: Predicted health category
        - health_confidence: Health classification confidence (0-1)
        - crop_type: Predicted crop type
        - crop_confidence: Crop type confidence (0-1)
        - all_health_predictions: Probability distribution across health classes
        - all_crop_predictions: Probability distribution across crop types
        - band_schema: Detected band schema
        - processing_path: 'rgb' or 'multispectral'
        - model_loaded: Whether model was successfully loaded
    """
    import os
    import glob
    import time
    
    start_time = time.time()
    
    # Default model path - try to find multi-crop model
    if model_path is None:
        model_path = os.getenv('MULTI_CROP_MODEL_PATH')
        if not model_path or not os.path.exists(model_path):
            multi_crop_dir = os.getenv('MULTI_CROP_MODEL_DIR', './models/multi_crop')
            if os.path.exists(multi_crop_dir):
                model_files = glob.glob(os.path.join(multi_crop_dir, '*_final.h5'))
                if model_files:
                    model_path = max(model_files, key=os.path.getmtime)
    
    if not model_path or not os.path.exists(model_path):
        return {
            'health_classification': 'model_not_found',
            'health_confidence': 0.0,
            'crop_type': 'unknown',
            'crop_confidence': 0.0,
            'all_health_predictions': {},
            'all_crop_predictions': {},
            'model_loaded': False,
            'error': f'Multi-crop model file not found: {model_path}'
        }
    
    try:
        # Load model (should be loaded once in background_worker, but handle here too)
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        # Load class names
        model_dir = os.path.dirname(model_path)
        metadata_files = glob.glob(os.path.join(model_dir, '*_metadata.json'))
        if metadata_files:
            with open(max(metadata_files, key=os.path.getmtime), 'r') as f:
                metadata = json.load(f)
                health_class_names = metadata.get('health_classes', UNIFIED_HEALTH_LABELS)
                crop_class_names = metadata.get('crop_types', CROP_TYPES)
        else:
            health_class_names = UNIFIED_HEALTH_LABELS
            crop_class_names = CROP_TYPES
        
        # Load image with multispectral support
        if HAS_MULTISPECTRAL:
            img, detected_schema = load_multispectral_image(
                image_path,
                target_size=(224, 224),
                dataset_name=dataset_name,
                band_order=None  # Auto-detect
            )
            if band_schema is None:
                band_schema = detected_schema
        else:
            # Fallback to OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            if band_schema is None:
                band_schema = {'bands': ['R', 'G', 'B'], 'band_order': ['R', 'G', 'B'], 'band_count': 3}
        
        # Determine processing path (check for fallback first)
        processing_path = band_schema.get('processing_path', None)
        fallback_reason = band_schema.get('fallback_reason', None)
        
        if processing_path is None:
            band_order = band_schema.get('band_order', [])
            has_nir = 'NIR' in band_order
            processing_path = 'multispectral' if has_nir else 'rgb'
        
        # Ensure fallback_reason is set if processing_path is rgb_fallback
        if processing_path == 'rgb_fallback' and fallback_reason is None:
            fallback_reason = band_schema.get('fallback_reason', 'Band schema validation failed')
        
        # Prepare inputs for band-aware model
        # RGB input (always 3 channels)
        if img.shape[2] >= 3:
            rgb_img = img[:, :, :3]
        else:
            # Pad if needed
            rgb_img = np.pad(img, ((0, 0), (0, 0), (0, 3 - img.shape[2])), mode='constant')
        
        # Multispectral input (4 channels: RGB+NIR)
        if img.shape[2] >= 4:
            ms_img = img[:, :, :4]
        elif img.shape[2] == 3:
            # RGB only - pad with zeros for NIR (model will use band_mask to ignore)
            ms_img = np.concatenate([img, np.zeros((img.shape[0], img.shape[1], 1))], axis=-1)
        else:
            ms_img = np.pad(img, ((0, 0), (0, 0), (0, 4 - img.shape[2])), mode='constant')
        
        # Create band mask using band-name keyed mapping
        if HAS_MULTISPECTRAL:
            from multispectral_loader import create_band_mask_array, STANDARD_MULTISPECTRAL_BANDS
            band_mask = create_band_mask_array(band_schema, STANDARD_MULTISPECTRAL_BANDS)
        else:
            # Default: RGB only
            band_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # [R, G, B, NIR]
        
        # Compute index features for fusion
        # Calculate indices from true bands
        ndvi_stats = calculate_ndvi(image_path, band_schema=band_schema, image_array=img)
        savi_stats = calculate_savi(image_path, band_schema=band_schema, image_array=img)
        gndvi_stats = calculate_gndvi(image_path, band_schema=band_schema, image_array=img)
        
        # Compute index features (exactly 12 features)
        from train_multi_crop_model_v2 import compute_index_features, INDEX_FEATURE_DIM
        index_features = compute_index_features(ndvi_stats, savi_stats, gndvi_stats)
        assert index_features.shape[0] == INDEX_FEATURE_DIM, f"Expected {INDEX_FEATURE_DIM} features, got {index_features.shape[0]}"
        
        # Prepare model inputs
        rgb_input = np.expand_dims(rgb_img, axis=0)
        ms_input = np.expand_dims(ms_img, axis=0)
        band_mask_input = np.expand_dims(band_mask, axis=0)
        index_features_input = np.expand_dims(index_features, axis=0)
        
        # Predict (band-aware model expects 4 inputs)
        # Check if model is band-aware (4 inputs) or legacy (1 input)
        if len(model.inputs) == 4:
            # Band-aware model
            predictions = model.predict(
                [rgb_input, ms_input, band_mask_input, index_features_input],
                verbose=0
            )
        else:
            # Legacy model - use RGB only
            logger.warning("Using legacy model (not band-aware), falling back to RGB path")
            if len(model.inputs) == 1:
                predictions = model.predict(rgb_input, verbose=0)
            else:
                raise ValueError(f"Unexpected model input count: {len(model.inputs)}")
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # Handle both single-output (legacy) and multi-output models
        if isinstance(predictions, list) and len(predictions) == 2:
            health_preds = predictions[0][0]
            crop_preds = predictions[1][0]
        else:
            health_preds = predictions[0]
            crop_preds = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Health classification
        health_predicted_idx = np.argmax(health_preds)
        health_confidence = float(health_preds[health_predicted_idx])
        health_classification = health_class_names[health_predicted_idx] if health_predicted_idx < len(health_class_names) else 'unknown'
        
        # Crop type classification
        crop_predicted_idx = np.argmax(crop_preds)
        crop_confidence = float(crop_preds[crop_predicted_idx])
        crop_type = crop_class_names[crop_predicted_idx] if crop_predicted_idx < len(crop_class_names) else 'unknown'
        
        # Top-k predictions
        health_topk = []
        health_sorted = np.argsort(health_preds)[::-1][:3]  # Top 3
        for idx in health_sorted:
            health_topk.append({
                'class': health_class_names[idx] if idx < len(health_class_names) else 'unknown',
                'confidence': float(health_preds[idx])
            })
        
        crop_topk = []
        crop_sorted = np.argsort(crop_preds)[::-1][:3]  # Top 3
        for idx in crop_sorted:
            crop_topk.append({
                'class': crop_class_names[idx] if idx < len(crop_class_names) else 'unknown',
                'confidence': float(crop_preds[idx])
            })
        
        # All predictions
        all_health_predictions = {}
        for i, class_name in enumerate(health_class_names):
            if i < len(health_preds):
                all_health_predictions[class_name] = float(health_preds[i])
        
        all_crop_predictions = {}
        for i, class_name in enumerate(crop_class_names):
            if i < len(crop_preds):
                all_crop_predictions[class_name] = float(crop_preds[i])
        
        # Optional heuristic fusion score (post-hoc, not used in model)
        heuristic_fusion_score = None
        if ndvi_stats.get('ndvi_mean') is not None:
            # Use NDVI as additional signal for heuristic score
            ndvi_mean = ndvi_stats.get('ndvi_mean', 0.0)
            # Normalize NDVI to [0, 1] (NDVI is typically [-1, 1])
            ndvi_normalized = (ndvi_mean + 1.0) / 2.0
            # Weighted combination: 70% ML, 30% NDVI
            heuristic_fusion_score = 0.7 * health_confidence + 0.3 * ndvi_normalized
        else:
            heuristic_fusion_score = health_confidence
        
        return {
            'health_classification': health_classification,
            'health_confidence': round(health_confidence, 3),
            'crop_type': crop_type,
            'crop_confidence': round(crop_confidence, 3),
            'all_health_predictions': all_health_predictions,
            'all_crop_predictions': all_crop_predictions,
            'health_topk': health_topk,
            'crop_topk': crop_topk,
            'band_schema': band_schema,
            'processing_path': processing_path,
            'model_loaded': True,
            'model_path': model_path,
            'inference_time_ms': inference_time_ms,
            'heuristic_fusion_score': round(heuristic_fusion_score, 3) if heuristic_fusion_score else None,
            'index_stats': {
                'ndvi': ndvi_stats,
                'savi': savi_stats,
                'gndvi': gndvi_stats
            }
        }
        
    except Exception as e:
        import traceback
        return {
            'health_classification': 'error',
            'health_confidence': 0.0,
            'crop_type': 'unknown',
            'crop_confidence': 0.0,
            'all_health_predictions': {},
            'all_crop_predictions': {},
            'model_loaded': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# Index feature computation moved to train_multi_crop_model_v2.compute_index_features
# This ensures consistency: exactly 12 features (4 stats * 3 indices)


def preprocess_image(image_path: str, output_path: Optional[str] = None, 
                     target_size: Optional[Tuple[int, int]] = None) -> str:
    """
    Preprocess image: resize, noise filtering, normalization.
    
    Args:
        image_path: Input image path
        output_path: Optional output path (if None, overwrites input)
        target_size: Optional (width, height) for resizing
        
    Returns:
        Path to processed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize if specified
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Save processed image
    if output_path is None:
        output_path = str(Path(image_path).with_suffix('.processed.jpg'))
    
    cv2.imwrite(output_path, img)
    return output_path


def analyze_crop_health(image_path: str, use_tensorflow: bool = False, 
                        model_path: Optional[str] = None,
                        use_multi_crop: bool = True,
                        model: Optional[object] = None,
                        dataset_name: Optional[str] = None) -> Dict:
    """
    Complete multi-crop plant health analysis pipeline.
    
    Combines NDVI, SAVI, and GNDVI calculation with TensorFlow classification.
    Supports cherry tomatoes, onions, and corn with unified health taxonomy.
    
    Args:
        image_path: Path to input image
        use_tensorflow: Whether to use TensorFlow model (if available)
        model_path: Path to TensorFlow model (None = auto-detect)
        use_multi_crop: Whether to prefer multi-crop model over single-crop
        channels: Number of input channels (3 for RGB, 4 for RGB+NIR)
        
    Returns:
        Complete analysis dictionary with NDVI, SAVI, GNDVI, and classification
    """
    import os
    
    # Preprocess (skip if already processed or if preprocessing fails)
    try:
        if image_path.endswith('.processed.jpg') or image_path.endswith('.processed.png'):
            processed_path = image_path  # Already processed
        else:
            processed_path = preprocess_image(image_path)
    except Exception as e:
        logger.warning(f"Preprocessing failed, using original image: {e}")
        processed_path = image_path  # Fallback to original
    
    # Calculate NDVI
    ndvi_results = calculate_ndvi(processed_path)
    
    # Calculate SAVI
    savi_results = calculate_savi(processed_path)
    
    # Calculate GNDVI
    gndvi_results = calculate_gndvi(processed_path)
    
    # TensorFlow classification (if enabled)
    tf_results = {}
    health_status = None
    health_score = None
    summary = None
    confidence = None
    crop_type = 'unknown'
    crop_confidence = None
    model_version = None
    analysis_type = 'ndvi_savi_gndvi'
    
    if use_tensorflow:
        # Try multi-crop model first if enabled
        if use_multi_crop:
            # Use pre-loaded model if provided, otherwise load from path
            if model is not None:
                # Model is already loaded, need to call inference directly
                # For now, still use classify_multi_crop_health but pass model
                tf_results = classify_multi_crop_health(
                    processed_path,
                    model_path=model_path,
                    dataset_name=dataset_name
                )
            else:
                tf_results = classify_multi_crop_health(
                    processed_path,
                    model_path=model_path,
                    dataset_name=dataset_name
                )
            if tf_results.get('model_loaded'):
                # Multi-crop model results
                health_status = tf_results.get('health_classification')
                confidence = tf_results.get('health_confidence', 0.0)
                crop_type = tf_results.get('crop_type', 'unknown')
                crop_confidence = tf_results.get('crop_confidence', 0.0)
                model_version = os.path.basename(model_path) if model_path else 'multi_crop_model'
                analysis_type = 'multi_crop_tensorflow'
            else:
                # Fallback to single-crop model
                tf_results = classify_crop_health_tensorflow(processed_path, model_path)
                if tf_results.get('model_loaded'):
                    health_status = tf_results.get('classification')
                    confidence = tf_results.get('confidence', 0.0)
                    crop_type = tf_results.get('crop_type', 'onion')  # Legacy models default to onion
                    model_version = os.path.basename(model_path) if model_path else 'tensorflow_model'
                    analysis_type = 'tensorflow_single_crop'
        else:
            # Use single-crop model
            tf_results = classify_crop_health_tensorflow(processed_path, model_path)
            if tf_results.get('model_loaded'):
                health_status = tf_results.get('classification')
                confidence = tf_results.get('confidence', 0.0)
                crop_type = tf_results.get('crop_type', 'onion')
                model_version = os.path.basename(model_path) if model_path else 'tensorflow_model'
                analysis_type = 'tensorflow_single_crop'
        
        if health_status:
            # Generate summary based on classification
            status_descriptions = {
                'very_healthy': 'Excellent crop health with high vegetation vigor',
                'healthy': 'Good crop health with adequate vegetation',
                'moderate': 'Moderate crop health requiring monitoring',
                'poor': 'Poor crop health requiring attention',
                'very_poor': 'Critical crop health requiring immediate intervention',
                'diseased': 'Signs of disease detected - treatment recommended',
                'stressed': 'Crop stress detected - review environmental conditions',
                'weeds': 'Significant weed presence affecting crop health',
                'unknown': 'Unable to determine crop health status'
            }
            summary = status_descriptions.get(health_status, f'Crop health: {health_status}')
            
            # Calculate health score from confidence and status
            base_scores = {
                'very_healthy': 0.9,
                'healthy': 0.75,
                'moderate': 0.5,
                'poor': 0.3,
                'very_poor': 0.1,
                'diseased': 0.2,
                'stressed': 0.4,
                'weeds': 0.35,
                'unknown': 0.0
            }
            health_score = base_scores.get(health_status, 0.5) * confidence
    
    # Fallback to NDVI-based classification if TensorFlow not used
    if not health_status:
        mean_ndvi = ndvi_results.get('ndvi_mean')
        
        # Handle case where NDVI cannot be calculated (RGB images without NIR)
        if mean_ndvi is None:
            # For RGB images, use a default moderate health status
            # This allows the system to still process and display images
            health_status = "moderate"
            summary = "RGB image processed - NIR band required for precise vegetation index analysis"
            health_score = 0.5  # Default moderate score
            confidence = 0.5  # Lower confidence for RGB-only analysis
        else:
            # Use NDVI for classification
            if mean_ndvi < 0.2:
                health_status = "very_poor"
                summary = "Critical attention needed"
            elif mean_ndvi < 0.4:
                health_status = "poor"
                summary = "Attention needed"
            elif mean_ndvi < 0.6:
                health_status = "moderate"
                summary = "Moderate health"
            elif mean_ndvi < 0.8:
                health_status = "healthy"
                summary = "Healthy"
            else:
                health_status = "very_healthy"
                summary = "Very healthy"
            
            # Calculate health score from NDVI
            health_score = min(1.0, max(0.0, (mean_ndvi + 0.2) / 1.0))
            confidence = 0.7  # Higher confidence when NDVI is available
        
        analysis_type = 'ndvi_savi_gndvi'
    
    # Combine results
    analysis = {
        **ndvi_results,
        **savi_results,
        **gndvi_results,
        'health_status': health_status,
        'health_score': health_score,
        'summary': summary,
        'tensorflow': tf_results if use_tensorflow else None,
        'confidence': confidence,
        'crop_type': crop_type,
        'crop_confidence': crop_confidence,
        'model_version': model_version,
        'processed_image_path': processed_path,
        'original_image_path': image_path,
        'analysis_type': analysis_type,
        'channels_used': channels if use_tensorflow else None
    }
    
    return analysis


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Analyzing: {image_path}")
    
    try:
        results = analyze_crop_health(image_path)
        print("\n=== Analysis Results ===")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

