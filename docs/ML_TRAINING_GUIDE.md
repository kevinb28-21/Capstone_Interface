# Multi-Crop ML Training and Integration Guide

This guide provides step-by-step instructions for training and integrating the multi-crop plant health model into the production system.

## Overview

The multi-crop model supports:
- **Crops**: Cherry tomatoes, onions, corn
- **Health Classes**: very_healthy, healthy, moderate, poor, very_poor, diseased, stressed, weeds, unknown
- **Spectral Support**: RGB (3 channels) and RGB+NIR (4 channels) with graceful degradation

## Prerequisites

1. Python 3.8+ with required packages (see `requirements.txt`)
2. PostgreSQL database with schema applied
3. Access to datasets (see `docs/datasets.md`)

## Step 1: Dataset Preparation

### 1.1 Download Datasets

```bash
# Download PlantVillage dataset
python python_processing/datasets/download_plantvillage.py --output-dir ./data/plantvillage --method manual

# Download TOM2024 dataset (manual download required)
python python_processing/datasets/download_tom2024.py --output-dir ./data/tom2024

# Download other datasets as needed (see docs/datasets.md)
```

### 1.2 Parse and Organize Datasets

```bash
# Parse PlantVillage dataset
python python_processing/datasets/parse_plantvillage.py \
    --input-dir ./data/plantvillage \
    --output-dir ./data/processed/plantvillage \
    --crops cherry_tomato onion corn

# Parse TOM2024 dataset (onion)
python python_processing/datasets/parse_tom2024.py \
    --input-dir ./data/tom2024 \
    --output-dir ./data/processed/tom2024 \
    --crop onion
```

### 1.3 Combine Processed Datasets

Combine all processed datasets into a single training directory:

```bash
# Structure: ./data/training/
#   cherry_tomato/
#     very_healthy/
#     healthy/
#     ...
#   onion/
#     very_healthy/
#     ...
#   corn/
#     very_healthy/
#     ...

mkdir -p ./data/training
cp -r ./data/processed/plantvillage/* ./data/training/
cp -r ./data/processed/tom2024/* ./data/training/
# Add other datasets as needed
```

## Step 2: Database Migration

Apply the database migration to add crop_type support:

```bash
# Connect to PostgreSQL and run migration
psql -U postgres -d drone_analytics -f server/database/migration_add_crop_type.sql
```

Or via Python:

```python
import psycopg2
with open('server/database/migration_add_crop_type.sql', 'r') as f:
    sql = f.read()
conn = psycopg2.connect(...)
cur = conn.cursor()
cur.execute(sql)
conn.commit()
```

## Step 3: Model Training

### 3.1 Configure Training

Edit `python_processing/training_config.yaml` to adjust hyperparameters:

```yaml
model:
  channels: 3  # Use 4 for RGB+NIR if you have multispectral data
training:
  epochs: 50
  batch_size: 32
```

### 3.2 Train Model

```bash
# Train RGB model (3 channels)
python python_processing/train_multi_crop_model.py \
    --data-folder ./data/training \
    --output-dir ./models/multi_crop \
    --channels 3 \
    --epochs 50 \
    --batch-size 32 \
    --config python_processing/training_config.yaml

# Train multispectral model (4 channels) if you have RGB+NIR data
python python_processing/train_multi_crop_model.py \
    --data-folder ./data/training \
    --output-dir ./models/multi_crop \
    --channels 4 \
    --epochs 50 \
    --batch-size 32
```

### 3.3 Training Output

The training script will create:
- `multi_crop_model_YYYYMMDD_HHMMSS_final.h5` - Final trained model
- `multi_crop_model_YYYYMMDD_HHMMSS_best.h5` - Best model (checkpoint)
- `multi_crop_model_YYYYMMDD_HHMMSS_metadata.json` - Model metadata
- `multi_crop_model_YYYYMMDD_HHMMSS_classes.json` - Class names

## Step 4: Integration

### 4.1 Update Environment Variables

Set environment variables for the background worker:

```bash
# .env file or environment
USE_MULTI_CROP_MODEL=true
MULTI_CROP_MODEL_DIR=./models/multi_crop
MODEL_CHANNELS=3  # or 4 for RGB+NIR
```

Or specify model path directly:

```bash
MULTI_CROP_MODEL_PATH=./models/multi_crop/multi_crop_model_YYYYMMDD_HHMMSS_final.h5
```

### 4.2 Test Model Inference

Test the model on a sample image:

```python
from image_processor import analyze_crop_health

result = analyze_crop_health(
    'path/to/test_image.jpg',
    use_tensorflow=True,
    use_multi_crop=True,
    channels=3
)

print(f"Crop Type: {result['crop_type']} (confidence: {result.get('crop_confidence', 0):.2%})")
print(f"Health: {result['health_status']} (confidence: {result.get('confidence', 0):.2%})")
```

### 4.3 Deploy Background Worker

The background worker will automatically use the multi-crop model if:
1. `USE_MULTI_CROP_MODEL=true` (default)
2. Model found in `MULTI_CROP_MODEL_DIR` or `MULTI_CROP_MODEL_PATH` is set

```bash
# Start background worker
python python_processing/background_worker.py
```

## Step 5: Validation

### 5.1 Verify Database Records

Check that crop_type is being saved:

```sql
SELECT 
    a.crop_type,
    a.crop_confidence,
    a.health_status,
    a.confidence,
    i.filename
FROM analyses a
JOIN images i ON a.image_id = i.id
ORDER BY a.processed_at DESC
LIMIT 10;
```

### 5.2 Monitor Worker Logs

Check background worker logs for model usage:

```bash
tail -f python_processing/background_worker.log
```

Look for messages like:
```
Using multi-crop TensorFlow model: ./models/multi_crop/multi_crop_model_..._final.h5
Multi-crop TensorFlow classification: crop=onion (confidence: 95.2%), health=healthy (confidence: 87.3%)
```

## Troubleshooting

### Model Not Found

If the worker can't find the model:
1. Check `MULTI_CROP_MODEL_PATH` or `MULTI_CROP_MODEL_DIR` environment variables
2. Verify model file exists and is readable
3. Check file permissions

### Wrong Crop Type Predictions

1. Verify training data includes all three crops
2. Check class distribution in training data (should be balanced)
3. Retrain with more data for underrepresented crops

### Multispectral Issues

1. Ensure images have 4 channels (RGB+NIR) if using `channels=4`
2. Model will approximate NIR from green channel if RGB-only images provided
3. For best results, train separate models for RGB and RGB+NIR

### Database Errors

1. Ensure migration has been applied: `migration_add_crop_type.sql`
2. Check that `crop_type` and `crop_confidence` columns exist in `analyses` table
3. Verify database connection and permissions

## Model Performance

### Expected Metrics

After training, you should see:
- Health classification accuracy: >80% (target)
- Crop type accuracy: >85% (target)
- Confidence scores: >0.7 for high-confidence predictions

### Improving Performance

1. **More Training Data**: Add more images, especially for underrepresented classes
2. **Data Augmentation**: Enable augmentation in config (already enabled by default)
3. **Transfer Learning**: Use EfficientNetB0 or ResNet50 (already configured)
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs
5. **Multispectral Data**: Use RGB+NIR data when available for better accuracy

## Production Deployment

### Model Versioning

Each trained model includes:
- Timestamp in filename
- Metadata JSON with training parameters
- Class names JSON

Keep multiple model versions for:
- A/B testing
- Rollback capability
- Performance comparison

### Monitoring

Monitor:
- Model inference time
- Prediction confidence distributions
- Crop type distribution in predictions
- Error rates

### Updates

To update the model:
1. Train new model with additional data
2. Test on validation set
3. Deploy by updating `MULTI_CROP_MODEL_PATH`
4. Monitor performance
5. Rollback if needed

## Support

For issues or questions:
1. Check logs: `python_processing/background_worker.log`
2. Review dataset documentation: `docs/datasets.md`
3. Check model metadata: `models/multi_crop/*_metadata.json`


