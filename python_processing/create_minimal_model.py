#!/usr/bin/env python3
"""
Create a minimal multi-crop model for demonstration purposes.
This creates a valid TensorFlow model file that can be detected by the system.
"""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    
    HAS_TENSORFLOW = True
except ImportError:
    print("TensorFlow not available. Creating model structure without TensorFlow...")
    HAS_TENSORFLOW = False

# Health classes
UNIFIED_HEALTH_LABELS = [
    'very_healthy', 'healthy', 'moderate', 'poor', 'very_poor',
    'diseased', 'stressed', 'weeds', 'unknown'
]

# Crop types
CROP_TYPES = ['cherry_tomato', 'onion', 'corn', 'unknown']

def create_minimal_model():
    """Create a minimal but valid TensorFlow model."""
    if not HAS_TENSORFLOW:
        print("Cannot create TensorFlow model without TensorFlow installed.")
        print("Please install TensorFlow: pip install tensorflow")
        return False
    
    print("Creating minimal multi-crop model...")
    
    # Create model directory
    model_dir = Path("models/multi_crop")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple model architecture
    # RGB input (3 channels)
    rgb_input = layers.Input(shape=(224, 224, 3), name='rgb_input')
    
    # Multispectral input (4 channels: RGB+NIR)
    ms_input = layers.Input(shape=(224, 224, 4), name='ms_input')
    
    # Band mask input (4 channels)
    band_mask_input = layers.Input(shape=(4,), name='band_mask')
    
    # Index features input (12 features: 4 stats * 3 indices)
    index_features_input = layers.Input(shape=(12,), name='index_features')
    
    # Process RGB path
    rgb_base = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=rgb_input,
        input_shape=(224, 224, 3)
    )
    rgb_base.trainable = False
    rgb_features = layers.GlobalAveragePooling2D()(rgb_base.output)
    rgb_features = layers.Dense(128, activation='relu')(rgb_features)
    
    # Process multispectral path
    ms_conv = layers.Conv2D(32, (3, 3), activation='relu')(ms_input)
    ms_pool = layers.GlobalAveragePooling2D()(ms_conv)
    ms_features = layers.Dense(128, activation='relu')(ms_pool)
    
    # Apply band mask to multispectral features
    band_mask_expanded = layers.Reshape((1, 1, 4))(band_mask_input)
    ms_features_masked = layers.Multiply()([ms_features, band_mask_input])
    
    # Process index features
    index_dense = layers.Dense(64, activation='relu')(index_features_input)
    
    # Concatenate all features
    combined = layers.Concatenate()([rgb_features, ms_features_masked, index_dense])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    
    # Health classification output
    health_output = layers.Dense(len(UNIFIED_HEALTH_LABELS), activation='softmax', name='health_class')(combined)
    
    # Crop type classification output
    crop_output = layers.Dense(len(CROP_TYPES), activation='softmax', name='crop_type')(combined)
    
    # Create model
    model = models.Model(
        inputs=[rgb_input, ms_input, band_mask_input, index_features_input],
        outputs=[health_output, crop_output],
        name='multi_crop_health_model'
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'health_class': 'categorical_crossentropy',
            'crop_type': 'categorical_crossentropy'
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
    
    # Create dummy data for a quick "training" (just to initialize weights properly)
    print("Initializing model with dummy data...")
    dummy_rgb = np.random.random((1, 224, 224, 3))
    dummy_ms = np.random.random((1, 224, 224, 4))
    dummy_mask = np.array([[1.0, 1.0, 1.0, 0.0]])  # RGB only
    dummy_indices = np.random.random((1, 12))
    
    dummy_health = np.zeros((1, len(UNIFIED_HEALTH_LABELS)))
    dummy_health[0, 1] = 1.0  # "healthy"
    
    dummy_crop = np.zeros((1, len(CROP_TYPES)))
    dummy_crop[0, 1] = 1.0  # "onion"
    
    # Run a few training steps
    for _ in range(5):
        model.train_on_batch(
            [dummy_rgb, dummy_ms, dummy_mask, dummy_indices],
            [dummy_health, dummy_crop]
        )
    
    # Save model
    model_filename = f"multi_crop_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_final.h5"
    model_path = model_dir / model_filename
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Create metadata
    metadata = {
        'model_version': model_filename.replace('_final.h5', ''),
        'training_date': datetime.now().isoformat(),
        'health_classes': UNIFIED_HEALTH_LABELS,
        'crop_types': CROP_TYPES,
        'input_size': [224, 224],
        'channels': 3,
        'architecture': 'band-aware multi-crop',
        'description': 'Minimal multi-crop model for demonstration'
    }
    
    metadata_filename = f"multi_crop_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_metadata.json"
    metadata_path = model_dir / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"\n✓ Model created successfully!")
    print(f"  Model file: {model_path}")
    print(f"  Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    return True

if __name__ == "__main__":
    success = create_minimal_model()
    if not success:
        print("\nNote: To create a fully trained model, install TensorFlow and run:")
        print("  python train_multi_crop_model_v2.py --data-folder training_data_organized --output-dir models/multi_crop --epochs 50")
