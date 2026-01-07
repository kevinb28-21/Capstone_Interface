#!/usr/bin/env python3
"""
Multi-Crop Plant Health Model Training
Trains a TensorFlow model for multi-crop (cherry tomato, onion, corn) plant health classification.
Supports RGB and multispectral (RGB+NIR) inputs with graceful degradation.

Features:
- Multi-crop classification (cherry_tomato, onion, corn, unknown)
- Unified health taxonomy (9 classes)
- Multispectral support (RGB + NIR when available)
- Reproducible training (deterministic seeds, pinned requirements)
- Artifact versioning
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
import cv2
import logging
from datetime import datetime
import yaml

# Add datasets to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from datasets.label_mapping import UNIFIED_HEALTH_LABELS, map_label, get_crop_type_from_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Crop types
CROP_TYPES = ['cherry_tomato', 'onion', 'corn', 'unknown']

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
    'multispectral': True,  # Support multispectral inputs
    'channels': 4,  # RGB + NIR (default, will adapt if NIR unavailable)
    'use_transfer_learning': True,
    'base_model': 'EfficientNetB0',  # or 'ResNet50', 'MobileNetV2'
    'dropout_rate': 0.5,
    'data_augmentation': True,
}


def detect_image_channels(image_path: str) -> int:
    """
    Detect number of channels in an image.
    
    Returns:
        Number of channels (3 for RGB, 4 for RGB+NIR, etc.)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 3  # Default to RGB
    
    if len(img.shape) == 2:
        return 1  # Grayscale
    elif len(img.shape) == 3:
        return img.shape[2]  # RGB or RGB+NIR
    else:
        return 3  # Default


def load_image_multispectral(image_path: str, target_size: Tuple[int, int], 
                            channels: int = 3) -> np.ndarray:
    """
    Load image with support for RGB and multispectral (RGB+NIR) inputs.
    
    Args:
        image_path: Path to image
        target_size: Target size (height, width)
        channels: Expected number of channels (3 for RGB, 4 for RGB+NIR)
    
    Returns:
        Preprocessed image array
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Handle different channel configurations
    if len(img.shape) == 2:
        # Grayscale - convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # RGBA or RGB+NIR - handle appropriately
            if channels == 4:
                # Keep all 4 channels
                pass
            else:
                # Extract RGB only
                img = img[:, :, :3]
        elif img.shape[2] == 3:
            # RGB
            if channels == 4:
                # Need to add NIR channel - approximate or pad
                # For now, duplicate green channel as NIR approximation
                nir_approx = img[:, :, 1:2]  # Use green as NIR proxy
                img = np.concatenate([img, nir_approx], axis=2)
        else:
            # Unexpected number of channels
            if channels == 3:
                img = img[:, :, :3]
            elif channels == 4 and img.shape[2] < 4:
                # Pad with zeros or duplicate last channel
                padding = np.zeros((img.shape[0], img.shape[1], 4 - img.shape[2]), dtype=img.dtype)
                img = np.concatenate([img, padding], axis=2)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Ensure correct number of channels
    if channels == 4 and img.shape[2] == 3:
        # Add NIR approximation (use green channel as proxy)
        nir_approx = img[:, :, 1:2]
        img = np.concatenate([img, nir_approx], axis=2)
    elif channels == 3 and img.shape[2] == 4:
        # Take RGB only
        img = img[:, :, :3]
    
    return img


def load_images_from_processed_folder(folder_path: str, 
                                      target_size: Tuple[int, int] = (224, 224),
                                      channels: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load images from processed dataset folder structure.
    
    Expected structure:
    folder_path/
        crop_type/
            health_label/
                image1.jpg
                image2.jpg
    
    Args:
        folder_path: Path to processed dataset
        target_size: Target image size (height, width)
        channels: Number of channels (3 for RGB, 4 for RGB+NIR)
    
    Returns:
        images: Image arrays
        health_labels: Health class indices
        crop_labels: Crop type indices
        health_class_names: Health class names
        crop_class_names: Crop type names
    """
    folder = Path(folder_path)
    
    images = []
    health_labels = []
    crop_labels = []
    
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
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(health_dir.glob(ext))
            
            logger.info(f"Loading {len(image_files)} images from {crop_type}/{health_label}...")
            
            for img_path in image_files:
                try:
                    img = load_image_multispectral(str(img_path), target_size, channels)
                    images.append(img)
                    health_labels.append(health_idx)
                    crop_labels.append(crop_idx)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path.name}: {e}")
    
    images = np.array(images)
    health_labels = np.array(health_labels)
    crop_labels = np.array(crop_labels)
    
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Health classes: {UNIFIED_HEALTH_LABELS}")
    logger.info(f"Crop types: {CROP_TYPES}")
    
    return images, health_labels, crop_labels, UNIFIED_HEALTH_LABELS, CROP_TYPES


def create_multi_crop_model(input_shape: Tuple[int, int, int], 
                            num_health_classes: int,
                            num_crop_classes: int,
                            config: Dict) -> models.Model:
    """
    Create a multi-output model for crop type and health classification.
    
    Args:
        input_shape: (height, width, channels)
        num_health_classes: Number of health classes
        num_crop_classes: Number of crop types
        config: Training configuration
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation (only during training)
    if config.get('data_augmentation', True):
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)
    else:
        x = inputs
    
    # Base feature extractor
    if config.get('use_transfer_learning', True):
        base_model_name = config.get('base_model', 'EfficientNetB0')
        
        if base_model_name == 'EfficientNetB0':
            # EfficientNetB0 expects 3 channels, so we need to adapt for 4 channels
            if input_shape[2] == 4:
                # Create custom input layer for 4 channels
                # Split into RGB and NIR, process separately, then combine
                rgb = layers.Lambda(lambda x: x[:, :, :, :3])(x)
                nir = layers.Lambda(lambda x: x[:, :, :, 3:4])(x)
                
                # Process RGB with EfficientNet
                base_model_rgb = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(*input_shape[:2], 3),
                    pooling='avg'
                )
                base_model_rgb.trainable = False
                rgb_features = base_model_rgb(rgb)
                
                # Process NIR with simple CNN
                nir_conv = layers.Conv2D(32, (3, 3), activation='relu')(nir)
                nir_pool = layers.GlobalAveragePooling2D()(nir_conv)
                
                # Combine features
                x = layers.Concatenate()([rgb_features, nir_pool])
            else:
                # Standard RGB EfficientNet
                base_model = keras.applications.EfficientNetB0(
                    include_top=False,
                    weights='imagenet',
                    input_shape=input_shape,
                    pooling='avg'
                )
                base_model.trainable = False
                x = base_model(x)
        else:
            # Fallback to custom CNN
            x = layers.Conv2D(32, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.GlobalAveragePooling2D()(x)
    else:
        # Custom CNN architecture
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
    
    # Shared dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(config.get('dropout_rate', 0.5))(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(config.get('dropout_rate', 0.5))(x)
    
    # Multi-output heads
    # Health classification head
    health_output = layers.Dense(num_health_classes, activation='softmax', name='health_class')(x)
    
    # Crop type classification head
    crop_output = layers.Dense(num_crop_classes, activation='softmax', name='crop_type')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=[health_output, crop_output])
    
    # Compile with separate losses and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
        loss={
            'health_class': 'sparse_categorical_crossentropy',
            'crop_type': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'health_class': 1.0,
            'crop_type': 0.5  # Health is primary task
        },
        metrics={
            'health_class': ['accuracy'],
            'crop_type': ['accuracy']
        }
    )
    
    return model


def train_model(
    data_folder: str,
    output_dir: str = "./models/multi_crop",
    config: Optional[Dict] = None,
    channels: int = 3
):
    """
    Train multi-crop plant health classification model.
    
    Args:
        data_folder: Path to processed dataset folder
        output_dir: Directory to save model
        config: Training configuration (uses DEFAULT_CONFIG if None)
        channels: Number of input channels (3 for RGB, 4 for RGB+NIR)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    logger.info("=" * 60)
    logger.info("Multi-Crop Plant Health ML Model Training")
    logger.info("=" * 60)
    logger.info(f"Channels: {channels} ({'RGB+NIR' if channels == 4 else 'RGB'})")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    # Load images
    logger.info("\n1. Loading images...")
    images, health_labels, crop_labels, health_class_names, crop_class_names = load_images_from_processed_folder(
        data_folder,
        target_size=config['input_size'],
        channels=channels
    )
    
    if len(images) == 0:
        raise ValueError("No images found in dataset folder")
    
    logger.info(f"   Loaded {len(images)} images")
    logger.info(f"   Image shape: {images[0].shape}")
    logger.info(f"   Health classes: {len(health_class_names)}")
    logger.info(f"   Crop types: {len(crop_class_names)}")
    
    # Class distribution
    unique_health, counts_health = np.unique(health_labels, return_counts=True)
    unique_crop, counts_crop = np.unique(crop_labels, return_counts=True)
    
    logger.info("\n   Health class distribution:")
    for idx, count in zip(unique_health, counts_health):
        logger.info(f"     {health_class_names[idx]}: {count}")
    
    logger.info("\n   Crop type distribution:")
    for idx, count in zip(unique_crop, counts_crop):
        logger.info(f"     {crop_class_names[idx]}: {count}")
    
    # Split data
    logger.info("\n2. Splitting data...")
    X_temp, X_test, y_health_temp, y_health_test, y_crop_temp, y_crop_test = train_test_split(
        images, health_labels, crop_labels,
        test_size=config['test_split'],
        random_state=RANDOM_SEED,
        stratify=health_labels
    )
    
    X_train, X_val, y_health_train, y_health_val, y_crop_train, y_crop_val = train_test_split(
        X_temp, y_health_temp, y_crop_temp,
        test_size=config['validation_split'] / (1 - config['test_split']),
        random_state=RANDOM_SEED,
        stratify=y_health_temp
    )
    
    logger.info(f"   Train: {len(X_train)}")
    logger.info(f"   Validation: {len(X_val)}")
    logger.info(f"   Test: {len(X_test)}")
    
    # Create model
    logger.info("\n3. Creating model...")
    input_shape = images[0].shape
    model = create_multi_crop_model(
        input_shape,
        config['num_health_classes'],
        config['num_crop_classes'],
        config
    )
    model.summary()
    
    # Callbacks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"multi_crop_model_{timestamp}"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_path / f"{model_name}_best.h5"),
            save_best_only=True,
            monitor='val_health_class_accuracy',
            mode='max',
            save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_health_class_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Data augmentation
    if config.get('data_augmentation', True):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]
        )
        train_gen = datagen.flow(
            X_train,
            {'health_class': y_health_train, 'crop_type': y_crop_train},
            batch_size=config['batch_size']
        )
    else:
        train_gen = (X_train, {'health_class': y_health_train, 'crop_type': y_crop_train})
    
    # Train
    logger.info("\n4. Training model...")
    try:
        if config.get('data_augmentation', True):
            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // config['batch_size'],
                epochs=config['epochs'],
                validation_data=(
                    X_val,
                    {'health_class': y_health_val, 'crop_type': y_crop_val}
                ),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                X_train,
                {'health_class': y_health_train, 'crop_type': y_crop_train},
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                validation_data=(
                    X_val,
                    {'health_class': y_health_val, 'crop_type': y_crop_val}
                ),
                callbacks=callbacks,
                verbose=1
            )
    except KeyboardInterrupt:
        logger.info("\n⚠ Training interrupted by user")
        history = None
    
    # Load best model
    best_model_path = output_path / f"{model_name}_best.h5"
    if best_model_path.exists():
        try:
            logger.info("\nLoading best model from checkpoint...")
            model = keras.models.load_model(str(best_model_path))
            logger.info("✓ Best model loaded")
        except Exception as e:
            logger.warning(f"Could not load best model: {e}")
    
    # Evaluate
    logger.info("\n5. Evaluating model...")
    test_results = model.evaluate(
        X_test,
        {'health_class': y_health_test, 'crop_type': y_crop_test},
        verbose=0
    )
    
    # Extract metrics
    metric_names = model.metrics_names
    test_metrics = dict(zip(metric_names, test_results))
    
    logger.info("\n   Test Results:")
    for name, value in test_metrics.items():
        logger.info(f"     {name}: {value:.4f}")
    
    # Save final model
    final_model_path = output_path / f"{model_name}_final.h5"
    try:
        if best_model_path.exists() and best_model_path != final_model_path:
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"\n✓ Best model copied to: {final_model_path}")
        else:
            model.save(final_model_path)
            logger.info(f"\n✓ Model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'crop_types': crop_class_names,
        'health_classes': health_class_names,
        'input_shape': list(input_shape),
        'channels': channels,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'config': config,
        'random_seed': RANDOM_SEED,
        'model_path': str(final_model_path)
    }
    
    metadata_path = output_path / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Model metadata saved to: {metadata_path}")
    
    # Save class names
    class_names_path = output_path / f"{model_name}_classes.json"
    with open(class_names_path, 'w') as f:
        json.dump({
            'health_classes': health_class_names,
            'crop_types': crop_class_names
        }, f, indent=2)
    logger.info(f"✓ Class names saved to: {class_names_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return model, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-crop plant health model')
    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Path to processed dataset folder'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/multi_crop',
        help='Output directory for model'
    )
    parser.add_argument(
        '--channels',
        type=int,
        choices=[3, 4],
        default=3,
        help='Number of input channels (3 for RGB, 4 for RGB+NIR)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file (optional)'
    )
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    
    train_model(
        args.data_folder,
        args.output_dir,
        config=config,
        channels=args.channels
    )


