Python Processing Service

The Python processing service handles image analysis, vegetation index calculations, and machine learning model integration for onion crop health monitoring.

Overview

This service provides:
  - NDVI Calculation - Normalized Difference Vegetation Index
  - SAVI Calculation - Soil-Adjusted Vegetation Index  
  - GNDVI Calculation - Green Normalized Difference Vegetation Index
  - Onion Crop Health Classification - 8 health categories
  - ML Model Integration - TensorFlow-based classification
  - Background Processing - Automated image analysis pipeline

Components

Core Modules
  - imageprocessor.py - Core image processing functions
  - calculatendvi() - NDVI calculation
  - calculatesavi() - SAVI calculation
  - calculategndvi() - GNDVI calculation
  - analyzecrophealth() - Complete analysis pipeline
  - classifycrophealthtensorflow() - ML model inference
  - flaskapidb.py - Flask REST API with database integration
  - Image upload endpoint
  - Analysis retrieval endpoint
  - Health check endpoint
  - backgroundworker.py - Automated processing worker
  - Monitors database for new uploads
  - Processes images automatically
  - Updates processing status
  - dbutils.py - Database utilities
  - Connection pooling
  - Query functions
  - Status management
  - s3utils.py - S3 storage utilities
  - Image upload/download
  - URL generation

Setup

Installation

cd pythonprocessing
pip install -r requirements.txt

Configuration

Create .env file:

FLASKPORT=5001
FLASKDEBUG=True
DBHOST=localhost
DBPORT=5432
DBNAME=droneanalytics
DBUSER=postgres
DBPASSWORD=yourpassword
UPLOADFOLDER=./uploads
PROCESSEDFOLDER=./processed
WORKERPOLLINTERVAL=10
WORKERBATCHSIZE=5

S3 Configuration (optional)
S3ENABLED=True
S3BUCKETNAME=your-bucket-name
AWSACCESSKEYID=your-key
AWSSECRETACCESSKEY=your-secret
AWSREGION=us-east-1

Model Configuration (optional)
ONIONMODELPATH=./models/onioncrophealthmodel.h5

Usage

Running the Flask API

python flaskapidb.py

Or with gunicorn for production:

gunicorn -w 4 -b 0.0.0.0:5001 flaskapidb:app

Running the Background Worker

python backgroundworker.py

Or as a systemd service:

sudo cp backgroundworker.service /etc/systemd/system/
sudo systemctl enable backgroundworker
sudo systemctl start backgroundworker

Testing Image Processing

Single image
python imageprocessor.py path/to/image.jpg

Batch processing
python batchtestndvi.py /path/to/image/folder

Training ML Model

python trainmodel.py ./sampleimages ./models 50

API Endpoints

POST /api/upload

Upload an image for processing.

Request:
  - image (file) - Image file
  - gps (optional, JSON string) - GPS metadata

Response:
{
  "id": "uuid",
  "filename": "timestampfilename.jpg",
  "processingstatus": "uploaded",
  "message": "Image uploaded successfully. Processing will begin shortly."
}

GET /api/data

Retrieve processed images.

Query Parameters:
  - imageid (optional) - Get specific image

Response:
{
  "images": [
    {
      "id": "uuid",
      "filename": "image.jpg",
      "ndvimean": 0.65,
      "savimean": 0.58,
      "healthstatus": "Healthy",
      "summary": "Healthy - Onion crop with green foliage..."
    }
  ]
}

GET /api/health

Health check endpoint.

Response:
{
  "status": "ok",
  "service": "flask-image-processor",
  "database": "connected"
}

Vegetation Indices

NDVI (Normalized Difference Vegetation Index)

Formula: (NIR - Red) / (NIR + Red)
  - Range: -1 to 1
  - Higher values indicate healthier vegetation
  - Onion-specific thresholds:
  - Very Healthy: > 0.8
  - Healthy: 0.6-0.8
  - Moderate: 0.4-0.6
  - Poor: 0.2-0.4
  - Very Poor: < 0.2

SAVI (Soil-Adjusted Vegetation Index)

Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
  - Range: -1 to 1
  - Accounts for soil background
  - Useful for sparse canopies
  - L factor: 0.5 (default)

GNDVI (Green Normalized Difference Vegetation Index)

Formula: (NIR - Green) / (NIR + Green)
  - Range: -1 to 1
  - Better for early growth stages
  - Less sensitive to atmospheric conditions
  - Useful for onion crops during development

Health Categories
  Very Healthy - Optimal growing conditions
  Healthy - Good health, normal conditions
  Moderate - Some stress indicators
  Poor - Attention needed
  Very Poor - Critical intervention required
  Diseased - Fungal/bacterial/viral diseases
  Stressed - Water/nutrient/heat stress
  Weeds - Significant weed infestation

Processing Pipeline
  Upload - Image uploaded via API
  Storage - Saved to S3 (or local) and database
  Detection - Background worker detects new image
  Processing - NDVI, SAVI, GNDVI calculated
  Classification - Health status determined
  Storage - Results saved to database
  Completion - Status updated to 'completed'

Related Documentation
  - Background Worker - Worker service details
  - ML Training - Model training guide
  - Image Capture - Capture implementation
  - Onion Crop Updates - Onion-specific features