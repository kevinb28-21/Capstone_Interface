#!/usr/bin/env python3
"""
Multi-Crop Plant Health Model Training (Band-Aware Version)
Trains a TensorFlow model with separate RGB and multispectral paths.
No NIR approximation - uses true bands only.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import yaml
import time

# Add datasets to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from datasets.label_mapping import UNIFIED_HEALTH_LABELS, map_label, get_crop_type_from_label
from multispectral_loader import (
    load_multispectral_image,
    detect_band_schema,
    create_band_mask_array,
    STANDARD_MULTISPECTRAL_BANDS,
    STANDARD_RGB_BANDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Crop types
CROP_TYPES = ['cherry_tomato', 'onion', 'corn', 'unknown']

# Standard multispectral bands (4 channels)
STANDARD_MULTISPECTRAL_BANDS = ['R', 'G', 'B', 'NIR']
STANDARD_RGB_BANDS = ['R', 'G', 'B']

# Index feature dimension (exactly 12: 4 stats per index * 3 indices)
INDEX_FEATURE_DIM = 12

# Model configuration
DEFAULT_CONFIG = {
    'input_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'test_split': 0.1,
    'learning_rate': 0.001,
    'num_health_classes': len(UNIFIED_HEALTH_LABELS),
    'num_crop_classes': len(CROP_TYPES),
    'use_transfer_learning': True,
    'base_model': 'EfficientNetB0',
    'dropout_rate': 0.5,
    'data_augmentation': True,
    'fusion_dim': 128,  # Dimension for index fusion layer
}


def load_images_from_processed_folder(
    folder_path: str,
    target_size: Tuple[int, int] = (224, 224),
    dataset_name: Optional[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[Dict], List[str], List[str]]:
    """
    Load images from processed dataset folder with band schema detection.
    Standardizes all multispectral images to 4 channels [R, G, B, NIR].
    
    Returns:
        images: List of image arrays (standardized to 3 or 4 channels)
        health_labels: Health class indices
        crop_labels: Crop type indices
        band_schemas: List of band schema dicts for each image
        health_class_names: Health class names
        crop_class_names: Crop type names
    """
    folder = Path(folder_path)
    
    images = []
    health_labels = []
    crop_labels = []
    band_schemas = []
    
    health_class_to_idx = {name: idx for idx, name in enumerate(UNIFIED_HEALTH_LABELS)}
    crop_class_to_idx = {name: idx for idx, name in enumerate(CROP_TYPES)}
    
    # Iterate through crop types
    for crop_dir in folder.iterdir():
        if not crop_dir.is_dir():
            continue
        
        crop_type = crop_dir.name
        if crop_type not in crop_class_to_idx:
            logger.warning(f"Unknown crop type: {crop_type}, skipping")
            continue
        
        crop_idx = crop_class_to_idx[crop_type]
        
        # Iterate through health labels
        for health_dir in crop_dir.iterdir():
            if not health_dir.is_dir():
                continue
            
            health_label = health_dir.name
            if health_label not in health_class_to_idx:
                logger.warning(f"Unknown health label: {health_label}, skipping")
                continue
            
            health_idx = health_class_to_idx[health_label]
            
            # Load images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                image_files.extend(health_dir.glob(ext))
            
            logger.info(f"Loading {len(image_files)} images from {crop_type}/{health_label}...")
            
            for img_path in image_files:
                try:
                    # Load with multispectral support (standardizes to 4 channels)
                    img, band_schema = load_multispectral_image(
                        str(img_path),
                        target_size=target_size,
                        dataset_name=dataset_name
                    )
                    
                    # Ensure standardization: RGB (3) or RGB+NIR (4)
                    if img.shape[2] > 4:
                        logger.warning(f"Image {img_path.name} has {img.shape[2]} channels, limiting to 4")
                        img = img[:, :, :4]
                        band_schema['band_count'] = 4
                        band_schema['band_order'] = STANDARD_MULTISPECTRAL_BANDS
                    elif img.shape[2] == 4:
                        # Ensure band_order is correct
                        if band_schema.get('band_order') != STANDARD_MULTISPECTRAL_BANDS:
                            band_schema['band_order'] = STANDARD_MULTISPECTRAL_BANDS
                    elif img.shape[2] == 3:
                        # RGB only
                        if band_schema.get('band_order') != STANDARD_RGB_BANDS:
                            band_schema['band_order'] = STANDARD_RGB_BANDS
                            band_schema['band_count'] = 3
                    
                    images.append(img)
                    health_labels.append(health_idx)
                    crop_labels.append(crop_idx)
                    band_schemas.append(band_schema)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path.name}: {e}")
    
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Health classes: {UNIFIED_HEALTH_LABELS}")
    logger.info(f"Crop types: {CROP_TYPES}")
    
    # Count band schemas
    schema_counts = {}
    for schema in band_schemas:
        band_count = schema['band_count']
        schema_counts[band_count] = schema_counts.get(band_count, 0) + 1
    
    logger.info(f"Band schema distribution: {schema_counts}")
    
    return images, np.array(health_labels), np.array(crop_labels), band_schemas, UNIFIED_HEALTH_LABELS, CROP_TYPES


def create_band_aware_model(
    input_size: Tuple[int, int],
    num_health_classes: int,
    num_crop_classes: int,
    config: Dict
) -> models.Model:
    """
    Create a band-aware model with separate RGB and multispectral paths.
    Standardized to 4-channel multispectral: [R, G, B, NIR]
    Uses MobileNetV3Small for cost-effective CPU inference.
    
    Args:
        input_size: (height, width)
        num_health_classes: Number of health classes
        num_crop_classes: Number of crop types
        config: Training configuration
    
    Returns:
        Compiled Keras model
    """
    # RGB path input (3 channels)
    rgb_input = layers.Input(shape=(*input_size, 3), name='rgb_input')
    
    # Multispectral path input (4 channels: RGB+NIR)
    multispectral_input = layers.Input(shape=(*input_size, 4), name='multispectral_input')
    
    # Band mask input (4 values: [R, G, B, NIR] presence)
    band_mask_input = layers.Input(shape=(len(STANDARD_MULTISPECTRAL_BANDS),), name='band_mask')
    
    # Index features input (exactly 12: 4 stats * 3 indices)
    index_features_input = layers.Input(shape=(INDEX_FEATURE_DIM,), name='index_features')
    
    # Use MobileNetV3Small for cost-effective CPU inference (or EfficientNetB0 if specified)
    base_model_name = config.get('base_model', 'MobileNetV3Small')
    feature_dim = 256  # Target feature dimension for merging
    
    # RGB path
    if config.get('use_transfer_learning', True):
        if base_model_name == 'MobileNetV3Small':
            base_model_rgb = keras.applications.MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=(*input_size, 3),
                pooling='avg'
            )
        else:  # EfficientNetB0
            base_model_rgb = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(*input_size, 3),
                pooling='avg'
            )
        base_model_rgb.trainable = False
        rgb_features_raw = base_model_rgb(rgb_input)
        # Project to target dimension
        rgb_features = layers.Dense(feature_dim, activation='relu', name='rgb_projection')(rgb_features_raw)
    else:
        # Custom lightweight CNN
        x = layers.Conv2D(32, (3, 3), activation='relu')(rgb_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        rgb_features = layers.Dense(feature_dim, activation='relu')(x)
    
    # Multispectral path
    # Process RGB channels (first 3) with base model
    rgb_from_multispectral = layers.Lambda(lambda x: x[:, :, :, :3], name='extract_rgb_from_ms')(multispectral_input)
    
    if config.get('use_transfer_learning', True):
        if base_model_name == 'MobileNetV3Small':
            base_model_ms_rgb = keras.applications.MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=(*input_size, 3),
                pooling='avg'
            )
        else:
            base_model_ms_rgb = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(*input_size, 3),
                pooling='avg'
            )
        base_model_ms_rgb.trainable = False
        ms_rgb_features_raw = base_model_ms_rgb(rgb_from_multispectral)
        ms_rgb_features = layers.Dense(feature_dim, activation='relu', name='ms_rgb_projection')(ms_rgb_features_raw)
    else:
        x = layers.Conv2D(32, (3, 3), activation='relu')(rgb_from_multispectral)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        ms_rgb_features = layers.Dense(feature_dim, activation='relu')(x)
    
    # Process NIR channel separately (lightweight)
    nir_band = layers.Lambda(lambda x: x[:, :, :, 3:4], name='extract_nir')(multispectral_input)
    nir_conv = layers.Conv2D(16, (3, 3), activation='relu', name='nir_conv')(nir_band)
    nir_pool = layers.GlobalAveragePooling2D(name='nir_pool')(nir_conv)
    nir_dense = layers.Dense(64, activation='relu', name='nir_dense')(nir_pool)
    
    # Combine multispectral features
    ms_features_raw = layers.Concatenate(name='ms_concat')([ms_rgb_features, nir_dense])
    ms_features = layers.Dense(feature_dim, activation='relu', name='ms_projection')(ms_features_raw)
    
    # Apply band mask using band-name keyed approach
    # Extract NIR mask (index 3 in canonical order [R, G, B, NIR])
    nir_mask = layers.Lambda(lambda x: x[:, 3:4], name='extract_nir_mask')(band_mask_input)
    nir_mask_expanded = layers.Dense(feature_dim, activation='sigmoid', name='nir_mask_expand')(nir_mask)
    
    # Select features: RGB if no NIR, multispectral if NIR present
    # Both rgb_features and ms_features are projected to same dimension (feature_dim=256)
    rgb_selected = layers.Multiply(name='rgb_selected')([rgb_features, layers.Lambda(lambda x: 1.0 - x, name='invert_nir_mask')(nir_mask_expanded)])
    ms_selected = layers.Multiply(name='ms_selected')([ms_features, nir_mask_expanded])
    
    # Merge features using Add (both are same dimension: feature_dim=256)
    # This is CPU-light: element-wise addition of same-dimension vectors
    combined_features = layers.Add(name='merge_features')([rgb_selected, ms_selected])
    
    # Shared dense layers (compact)
    x = layers.Dense(256, activation='relu', name='shared_dense1')(combined_features)
    x = layers.Dropout(config.get('dropout_rate', 0.5), name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='shared_dense2')(x)
    x = layers.Dropout(config.get('dropout_rate', 0.5), name='dropout2')(x)
    
    # Index feature embedding (12 -> 64)
    index_embedding = layers.Dense(64, activation='relu', name='index_embedding')(index_features_input)
    
    # Concatenate image features with index features
    fused_features = layers.Concatenate(name='fuse_features')([x, index_embedding])
    fused_dense = layers.Dense(128, activation='relu', name='fused_dense')(fused_features)
    fused_dense = layers.Dropout(config.get('dropout_rate', 0.5), name='fused_dropout')(fused_dense)
    
    # Multi-output heads
    # Health classification head
    health_output = layers.Dense(num_health_classes, activation='softmax', name='health_class')(fused_dense)
    
    # Crop type classification head
    crop_output = layers.Dense(num_crop_classes, activation='softmax', name='crop_type')(fused_dense)
    
    # Create model
    model = models.Model(
        inputs=[rgb_input, multispectral_input, band_mask_input, index_features_input],
        outputs=[health_output, crop_output]
    )
    
    # Model summary
    logger.info(f"Model created with {model.count_params()} parameters")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
        loss={
            'health_class': 'sparse_categorical_crossentropy',
            'crop_type': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'health_class': 1.0,
            'crop_type': 0.5
        },
        metrics={
            'health_class': ['accuracy'],
            'crop_type': ['accuracy']
        }
    )
    
    return model


def compute_index_features(ndvi_stats: Dict, savi_stats: Dict, gndvi_stats: Dict) -> np.ndarray:
    """
    Compute index features for fusion layer.
    Always returns exactly 12 features (float32): mean, std, min, max for each of NDVI, SAVI, GNDVI.
    
    Missing index behavior:
    - If an index cannot be computed (e.g., missing NIR for NDVI), all 4 stats are set to 0.0
    - This ensures consistent feature vector length regardless of available bands
    - The model learns to handle missing indices through the band_mask input
    
    Args:
        ndvi_stats: NDVI statistics dict (may have None values if NIR/R missing)
        savi_stats: SAVI statistics dict (may have None values if NIR/R missing)
        gndvi_stats: GNDVI statistics dict (may have None values if NIR/G missing)
    
    Returns:
        Feature vector of shape (12,) dtype=np.float32
        Order: [NDVI_mean, NDVI_std, NDVI_min, NDVI_max, SAVI_mean, SAVI_std, SAVI_min, SAVI_max, 
                GNDVI_mean, GNDVI_std, GNDVI_min, GNDVI_max]
        Missing indices are zero-filled (all 4 stats = 0.0)
    """
    features = []
    
    # NDVI features (4 values: mean, std, min, max)
    # If NDVI cannot be computed (missing R or NIR), all stats are 0.0
    if ndvi_stats.get('ndvi_mean') is not None:
        features.extend([
            float(ndvi_stats.get('ndvi_mean', 0.0)),
            float(ndvi_stats.get('ndvi_std', 0.0)),
            float(ndvi_stats.get('ndvi_min', 0.0)),
            float(ndvi_stats.get('ndvi_max', 0.0))
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])  # Missing NIR/R - zero-fill
    
    # SAVI features (4 values)
    # If SAVI cannot be computed (missing R or NIR), all stats are 0.0
    if savi_stats.get('savi_mean') is not None:
        features.extend([
            float(savi_stats.get('savi_mean', 0.0)),
            float(savi_stats.get('savi_std', 0.0)),
            float(savi_stats.get('savi_min', 0.0)),
            float(savi_stats.get('savi_max', 0.0))
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])  # Missing NIR/R - zero-fill
    
    # GNDVI features (4 values)
    # If GNDVI cannot be computed (missing G or NIR), all stats are 0.0
    if gndvi_stats.get('gndvi_mean') is not None:
        features.extend([
            float(gndvi_stats.get('gndvi_mean', 0.0)),
            float(gndvi_stats.get('gndvi_std', 0.0)),
            float(gndvi_stats.get('gndvi_min', 0.0)),
            float(gndvi_stats.get('gndvi_max', 0.0))
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])  # Missing NIR/G - zero-fill
    
    # Ensure exactly 12 features (float32)
    if len(features) != INDEX_FEATURE_DIM:
        logger.error(f"Index features length mismatch: expected {INDEX_FEATURE_DIM}, got {len(features)}")
        # Pad or truncate to ensure correct length (should never happen, but safety check)
        if len(features) < INDEX_FEATURE_DIM:
            features.extend([0.0] * (INDEX_FEATURE_DIM - len(features)))
        else:
            features = features[:INDEX_FEATURE_DIM]
    
    return np.array(features, dtype=np.float32)

