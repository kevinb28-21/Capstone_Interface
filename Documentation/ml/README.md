# Python Processing Service

The Python processing service handles image analysis, vegetation index calculations, and machine learning model integration for onion crop health monitoring.

## Overview

This service provides:
- **NDVI Calculation** - Normalized Difference Vegetation Index
- **SAVI Calculation** - Soil-Adjusted Vegetation Index  
- **GNDVI Calculation** - Green Normalized Difference Vegetation Index
- **Onion Crop Health Classification** - 8 health categories
- **ML Model Integration** - TensorFlow-based classification
- **Background Processing** - Automated image analysis pipeline

## Components

### Core Modules

- **`image_processor.py`** - Core image processing functions
  - `calculate_ndvi()` - NDVI calculation
  - `calculate_savi()` - SAVI calculation
  - `calculate_gndvi()` - GNDVI calculation
  - `analyze_crop_health()` - Complete analysis pipeline
  - `classify_crop_health_tensorflow()` - ML model inference

- **`flask_api_db.py`** - Flask REST API with database integration
  - Image upload endpoint
  - Analysis retrieval endpoint
  - Health check endpoint

- **`background_worker.py`** - Automated processing worker
  - Monitors database for new uploads
  - Processes images automatically
  - Updates processing status

- **`db_utils.py`** - Database utilities
  - Connection pooling
  - Query functions
  - Status management

- **`s3_utils.py`** - S3 storage utilities
  - Image upload/download
  - URL generation

## Setup

### Installation

```bash
cd python_processing
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
FLASK_PORT=5001
FLASK_DEBUG=True
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your_password
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5

# S3 Configuration (optional)
S3_ENABLED=True
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# Model Configuration (optional)
ONION_MODEL_PATH=./models/onion_crop_health_model.h5
```

## Usage

### Running the Flask API

```bash
python flask_api_db.py
```

Or with gunicorn for production:

```bash
gunicorn -w 4 -b 0.0.0.0:5001 flask_api_db:app
```

### Running the Background Worker

```bash
python background_worker.py
```

Or as a systemd service:

```bash
sudo cp background_worker.service /etc/systemd/system/
sudo systemctl enable background_worker
sudo systemctl start background_worker
```

### Testing Image Processing

```bash
# Single image
python image_processor.py path/to/image.jpg

# Batch processing
python batch_test_ndvi.py /path/to/image/folder
```

### Training ML Model

```bash
python train_model.py ./sample_images ./models 50
```

## API Endpoints

### POST `/api/upload`

Upload an image for processing.

**Request:**
- `image` (file) - Image file
- `gps` (optional, JSON string) - GPS metadata

**Response:**
```json
{
  "id": "uuid",
  "filename": "timestamp_filename.jpg",
  "processing_status": "uploaded",
  "message": "Image uploaded successfully. Processing will begin shortly."
}
```

### GET `/api/data`

Retrieve processed images.

**Query Parameters:**
- `image_id` (optional) - Get specific image

**Response:**
```json
{
  "images": [
    {
      "id": "uuid",
      "filename": "image.jpg",
      "ndvi_mean": 0.65,
      "savi_mean": 0.58,
      "health_status": "Healthy",
      "summary": "Healthy - Onion crop with green foliage..."
    }
  ]
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "flask-image-processor",
  "database": "connected"
}
```

## Vegetation Indices

### NDVI (Normalized Difference Vegetation Index)

Formula: `(NIR - Red) / (NIR + Red)`

- Range: -1 to 1
- Higher values indicate healthier vegetation
- Onion-specific thresholds:
  - Very Healthy: > 0.8
  - Healthy: 0.6-0.8
  - Moderate: 0.4-0.6
  - Poor: 0.2-0.4
  - Very Poor: < 0.2

### SAVI (Soil-Adjusted Vegetation Index)

Formula: `((NIR - Red) / (NIR + Red + L)) * (1 + L)`

- Range: -1 to 1
- Accounts for soil background
- Useful for sparse canopies
- L factor: 0.5 (default)

### GNDVI (Green Normalized Difference Vegetation Index)

Formula: `(NIR - Green) / (NIR + Green)`

- Range: -1 to 1
- Better for early growth stages
- Less sensitive to atmospheric conditions
- Useful for onion crops during development

## Health Categories

1. **Very Healthy** - Optimal growing conditions
2. **Healthy** - Good health, normal conditions
3. **Moderate** - Some stress indicators
4. **Poor** - Attention needed
5. **Very Poor** - Critical intervention required
6. **Diseased** - Fungal/bacterial/viral diseases
7. **Stressed** - Water/nutrient/heat stress
8. **Weeds** - Significant weed infestation

## Processing Pipeline

1. **Upload** - Image uploaded via API
2. **Storage** - Saved to S3 (or local) and database
3. **Detection** - Background worker detects new image
4. **Processing** - NDVI, SAVI, GNDVI calculated
5. **Classification** - Health status determined
6. **Storage** - Results saved to database
7. **Completion** - Status updated to 'completed'

## Related Documentation

- [Background Worker](./BACKGROUND_WORKER.md) - Worker service details
- [ML Training](./ML_TRAINING.md) - Model training guide
- [Image Capture](./IMAGE_CAPTURE.md) - Capture implementation
- [Onion Crop Updates](./ONION_CROP.md) - Onion-specific features

