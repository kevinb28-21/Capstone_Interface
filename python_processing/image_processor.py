"""
Image Processing Module
Performs NDVI calculation and crop health analysis using OpenCV and TensorFlow.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def calculate_ndvi(image_path: str) -> Dict:
    """
    Calculate NDVI (Normalized Difference Vegetation Index) from an RGB image.
    
    For RGB images (non-multispectral), this uses a pseudo-NDVI approximation
    based on Red and Green channels. For true NDVI, use NIR band if available.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Dictionary with NDVI statistics and processed image data
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract channels
    red = img_rgb[:, :, 0].astype(np.float32)
    green = img_rgb[:, :, 1].astype(np.float32)
    blue = img_rgb[:, :, 2].astype(np.float32)
    
    # Pseudo-NDVI approximation for RGB images
    # Using formula: (NIR - Red) / (NIR + Red)
    # Approximate NIR as: (2 * Green) - Red (common approximation)
    # Alternative: use Green channel as proxy
    nir_approx = (2.0 * green) - red
    nir_approx = np.clip(nir_approx, 0, 255)
    
    # Calculate NDVI
    denominator = red + nir_approx + 1e-7  # Avoid division by zero
    ndvi = (nir_approx - red) / denominator
    
    # Normalize to 0-1 range
    ndvi = np.clip(ndvi, -1, 1)
    ndvi_normalized = (ndvi + 1) / 2.0  # Scale to 0-1 for visualization
    
    # Calculate statistics
    mean_ndvi = float(np.mean(ndvi))
    std_ndvi = float(np.std(ndvi))
    min_ndvi = float(np.min(ndvi))
    max_ndvi = float(np.max(ndvi))
    
    # Health classification based on NDVI
    # NDVI ranges: -1 to 1, typically vegetation is 0.2-0.8
    if mean_ndvi < 0.2:
        health_status = "Very Poor"
        summary = "Critical attention needed"
    elif mean_ndvi < 0.4:
        health_status = "Poor"
        summary = "Attention needed"
    elif mean_ndvi < 0.6:
        health_status = "Moderate"
        summary = "Moderate health"
    elif mean_ndvi < 0.8:
        health_status = "Healthy"
        summary = "Healthy"
    else:
        health_status = "Very Healthy"
        summary = "Very healthy"
    
    # Identify stress zones (low NDVI regions)
    stress_threshold = mean_ndvi - std_ndvi
    stress_mask = ndvi < stress_threshold
    stress_coords = np.where(stress_mask)
    
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
    
    # Create NDVI visualization (color-coded)
    ndvi_visual = np.zeros((h, w, 3), dtype=np.uint8)
    # Color mapping: red (low) -> yellow -> green (high)
    ndvi_scaled = (ndvi_normalized * 255).astype(np.uint8)
    ndvi_visual[:, :, 0] = np.clip(255 - ndvi_scaled, 0, 255)  # Red channel
    ndvi_visual[:, :, 1] = ndvi_scaled  # Green channel
    ndvi_visual[:, :, 2] = np.clip(255 - ndvi_scaled * 0.5, 0, 255)  # Blue channel
    
    return {
        'ndvi_mean': round(mean_ndvi, 3),
        'ndvi_std': round(std_ndvi, 3),
        'ndvi_min': round(min_ndvi, 3),
        'ndvi_max': round(max_ndvi, 3),
        'health_status': health_status,
        'summary': summary,
        'stress_zones': stress_zones,
        'ndvi_map_shape': list(ndvi.shape),
        'stress_count': len(stress_zones)
    }


def classify_crop_health_tensorflow(image_path: str, model_path: Optional[str] = None) -> Dict:
    """
    Classify crop health using a TensorFlow model.
    
    This is a placeholder for TensorFlow model integration.
    Once you have a trained model, load it and run inference here.
    
    Args:
        image_path: Path to the input image
        model_path: Optional path to saved TensorFlow model
        
    Returns:
        Dictionary with classification results
    """
    # TODO: Load and run TensorFlow model when available
    # Example structure:
    # import tensorflow as tf
    # model = tf.keras.models.load_model(model_path)
    # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # predictions = model.predict(img_array)
    
    # For now, return placeholder structure
    return {
        'classification': 'pending_model',
        'confidence': 0.0,
        'classes': ['healthy', 'stressed', 'diseased', 'weeds'],
        'model_loaded': False
    }


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
                        model_path: Optional[str] = None) -> Dict:
    """
    Complete crop health analysis pipeline.
    
    Combines NDVI calculation and TensorFlow classification.
    
    Args:
        image_path: Path to input image
        use_tensorflow: Whether to use TensorFlow model (if available)
        model_path: Path to TensorFlow model
        
    Returns:
        Complete analysis dictionary
    """
    # Preprocess
    processed_path = preprocess_image(image_path)
    
    # Calculate NDVI
    ndvi_results = calculate_ndvi(processed_path)
    
    # TensorFlow classification (if enabled)
    tf_results = {}
    if use_tensorflow:
        tf_results = classify_crop_health_tensorflow(processed_path, model_path)
    
    # Combine results
    analysis = {
        **ndvi_results,
        'tensorflow': tf_results if use_tensorflow else None,
        'processed_image_path': processed_path,
        'original_image_path': image_path
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

