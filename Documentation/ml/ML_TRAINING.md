# ML Model Training Guide

This guide covers training machine learning models for onion crop health classification.

## Overview

The ML training pipeline creates TensorFlow models that classify onion crop images into 8 health categories using vegetation indices (NDVI, SAVI, GNDVI) and image features.

## Health Categories

1. **very_healthy** - NDVI > 0.8, SAVI > 0.7, GNDVI > 0.75
2. **healthy** - NDVI 0.6-0.8, SAVI 0.5-0.7, GNDVI 0.6-0.75
3. **moderate** - NDVI 0.4-0.6, SAVI 0.3-0.5, GNDVI 0.4-0.6
4. **poor** - NDVI 0.2-0.4, SAVI 0.15-0.3, GNDVI 0.2-0.4
5. **very_poor** - NDVI < 0.2, SAVI < 0.15, GNDVI < 0.2
6. **diseased** - Fungal/bacterial/viral diseases
7. **stressed** - Water/nutrient/heat stress
8. **weeds** - Significant weed infestation

## Dataset Organization

### Option 1: Subfolder Structure (Recommended)

```
sample_images/
  very_healthy/
    img1.jpg
    img2.jpg
    ...
  healthy/
    img1.jpg
    ...
  moderate/
    ...
  poor/
    ...
  very_poor/
    ...
  diseased/
    ...
  stressed/
    ...
  weeds/
    ...
```

### Option 2: Flat Structure with Labels

```
sample_images/
  img1.jpg
  img2.jpg
  ...
  labels.json
```

**labels.json format:**
```json
{
  "img1.jpg": "very_healthy",
  "img2.jpg": "healthy",
  "img3.jpg": "moderate"
}
```

## Training the Model

### Basic Training

```bash
cd python_processing
python train_model.py ./sample_images ./models 50
```

**Parameters:**
- `./sample_images` - Path to image folder
- `./models` - Output directory for saved model
- `50` - Number of training epochs

### Advanced Options

Modify `train_model.py` to adjust:
- Batch size (default: 32)
- Validation split (default: 0.2)
- Test split (default: 0.1)
- Image size (default: 224x224)
- Model architecture

## Model Output

After training, the following files are created:

```
models/
  onion_crop_health_model.h5      # Final trained model
  onion_crop_best_model.h5         # Best model (by validation accuracy)
  onion_class_names.json           # Class names mapping
  onion_training_history.json      # Training metrics
  onion_model_metadata.json        # Model metadata
```

## Using the Trained Model

### Automatic Loading

The model is automatically loaded when:
- `use_tensorflow=True` in `analyze_crop_health()`
- Model path is set via `ONION_MODEL_PATH` environment variable
- Model exists at default path: `./models/onion_crop_health_model.h5`

### Manual Loading

```python
from image_processor import classify_crop_health_tensorflow

result = classify_crop_health_tensorflow(
    'path/to/image.jpg',
    model_path='./models/onion_crop_health_model.h5'
)
```

## Training Tips

### Dataset Size

- Minimum: 200-300 images per category
- Recommended: 500+ images per category
- Total: 1,500-2,400 images minimum

### Image Quality

- High resolution (1920x1080+)
- Clear visibility of onion beds
- Various lighting conditions
- Different growth stages

### Data Augmentation

The training script includes:
- Random horizontal flips
- Random rotations (±10%)
- Random zoom (±10%)
- Random brightness adjustments

### Model Performance

Monitor:
- Training accuracy vs validation accuracy
- Overfitting (large gap between train/val)
- Early stopping (patience: 10 epochs)
- Learning rate reduction

## Evaluation

### Training Metrics

Check `onion_training_history.json`:
```json
{
  "loss": [...],
  "accuracy": [...],
  "val_loss": [...],
  "val_accuracy": [...]
}
```

### Test Set Performance

The script reports:
- Test loss
- Test accuracy
- Per-class performance (if available)

## Troubleshooting

### Low Accuracy

- Increase dataset size
- Balance classes (equal samples per category)
- Check image quality
- Adjust model architecture

### Overfitting

- Increase data augmentation
- Add more dropout
- Reduce model complexity
- Collect more training data

### Memory Issues

- Reduce batch size
- Reduce image size
- Use data generators
- Enable mixed precision training

## Related Documentation

- [Python Processing README](./README.md)
- [Onion Crop Updates](./ONION_CROP.md)
- [Background Worker](./BACKGROUND_WORKER.md)

