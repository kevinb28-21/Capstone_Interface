# Model Training Instructions

## Current Status

✅ **Model structure created**: `models/multi_crop/multi_crop_model_demo_final.h5`
✅ **Metadata file created**: `models/multi_crop/multi_crop_model_demo_metadata.json`
✅ **Training data organized**: `training_data_organized/onion/` (305 images)

⚠️ **Note**: Full model training requires TensorFlow, which is not compatible with Python 3.13.
The current Python version (3.13) cannot install TensorFlow.

## To Train the Full Model

### Option 1: Use Python 3.11 or 3.12 (Recommended)

1. **Create a new virtual environment with Python 3.11 or 3.12**:
   ```bash
   # Using pyenv or similar
   pyenv install 3.11.9
   pyenv local 3.11.9
   
   # Or use system Python 3.11/3.12 if available
   python3.11 -m venv venv_tf
   source venv_tf/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   cd python_processing
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python train_multi_crop_model_v2.py \
     --data-folder training_data_organized \
     --output-dir models/multi_crop \
     --epochs 50 \
     --channels 3 \
     --batch-size 16
   ```

4. **The trained model will be saved as**:
   - `models/multi_crop/multi_crop_model_YYYYMMDD_HHMMSS_final.h5`
   - `models/multi_crop/multi_crop_model_YYYYMMDD_HHMMSS_metadata.json`

### Option 2: Use Docker (Alternative)

Create a Docker container with Python 3.11 and TensorFlow:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "train_multi_crop_model_v2.py", "--data-folder", "training_data_organized", "--output-dir", "models/multi_crop", "--epochs", "50"]
```

## Current Demo Model

The current `multi_crop_model_demo_final.h5` is a placeholder that:
- ✅ Is detected by `/api/ml/status` endpoint
- ✅ Allows the system to process images with vegetation indices
- ⚠️ Does not provide ML-based classification (uses NDVI fallback)

## Training Data

Training data is organized in: `training_data_organized/onion/`
- Structure: `onion/health_label/*.jpg`
- Health labels: very_healthy, healthy, moderate, poor, very_poor, diseased, stressed, weeds
- Total: 305 images (limited for quick training)

For full training, use all 48,763 images from `training_data/train/`.

## After Training

Once the model is trained:

1. **Verify model detection**:
   ```bash
   curl http://localhost:5050/api/ml/status
   ```
   Should return `model_available: true`

2. **Restart background worker** to load the new model:
   ```bash
   cd python_processing
   python background_worker.py
   ```

3. **Process images** - they will now use ML classification instead of NDVI fallback

4. **Check ML insights page** - should show:
   - Model Status: Available
   - Model Performance metrics
   - Recent ML Predictions with confidence scores
   - Selected Image Analysis with ML predictions
