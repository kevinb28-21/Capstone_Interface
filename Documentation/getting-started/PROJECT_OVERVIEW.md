# Project Overview

## Drone Crop Health Platform

A full-stack web application for monitoring crop health using drone-captured images. The platform performs automated analysis using vegetation indices (NDVI, SAVI, GNDVI) and machine learning models, specifically optimized for onion crop monitoring.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│              (Vite + React-Leaflet Dashboard)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/REST API
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Node.js Backend                            │
│              (Express.js API Server)                         │
└───────────────┬───────────────────────┬─────────────────────┘
                │                       │
                │                       │
┌───────────────▼──────────┐  ┌────────▼──────────────────────┐
│    Python Flask API      │  │    PostgreSQL Database        │
│  (Image Processing)      │  │  (Image Metadata & Analysis)  │
└───────────────┬──────────┘  └───────────────────────────────┘
                │
                │
┌───────────────▼──────────────────────────────────────────────┐
│              Background Worker                                │
│    (Automated Image Processing Pipeline)                    │
└──────────────────────────────────────────────────────────────┘
                │
                │
┌───────────────▼──────────────────────────────────────────────┐
│                    AWS S3 Storage                            │
│              (Image & Processed File Storage)               │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### Image Processing
- **NDVI Calculation** - Normalized Difference Vegetation Index
- **SAVI Calculation** - Soil-Adjusted Vegetation Index
- **GNDVI Calculation** - Green Normalized Difference Vegetation Index
- **Onion-Specific Health Classification** - 8 health categories
- **ML Model Integration** - TensorFlow-based crop health classification

### Drone Integration
- **GPS-Triggered Capture** - Capture images based on location
- **Interval-Based Capture** - Time-based image capture
- **Telemetry Tracking** - Real-time drone position and route
- **Geofence Management** - Define and monitor flight boundaries

### Data Management
- **PostgreSQL Database** - Persistent storage for images, analyses, and metadata
- **S3 Storage** - Scalable image storage
- **Background Processing** - Automated analysis pipeline
- **Status Tracking** - Processing status monitoring

### Dashboard
- **Interactive Map** - Leaflet-based map visualization
- **Image Analytics** - View analysis results per image
- **Health Status** - Visual indicators for crop health
- **Real-time Updates** - Live telemetry and route tracking

## Technology Stack

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **React-Leaflet** - Map visualization
- **CSS** - Custom styling

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **Multer** - File upload handling
- **CORS** - Cross-origin resource sharing

### Image Processing
- **Python** - Processing language
- **Flask** - REST API framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **TensorFlow** - Machine learning

### Database
- **PostgreSQL** - Relational database
- **psycopg2** - Python database adapter

### Storage
- **AWS S3** - Object storage
- **boto3** - AWS SDK for Python

## Data Flow

1. **Image Upload**: Drone captures image → Uploads to Flask API
2. **Storage**: Image saved to S3 and metadata to PostgreSQL
3. **Processing**: Background worker detects new image → Processes with NDVI/SAVI/GNDVI
4. **Analysis**: Results saved to database with health classification
5. **Visualization**: Frontend fetches data → Displays on dashboard

## Crop Focus

Currently optimized for **Onion (Allium cepa)** crops with:
- Onion-specific health thresholds
- 8 health categories (Very Healthy, Healthy, Moderate, Poor, Very Poor, Diseased, Stressed, Weeds)
- Vegetation index ranges tuned for onion growth stages

## Development Status

- ✅ Core image processing (NDVI, SAVI, GNDVI)
- ✅ Database integration
- ✅ S3 storage
- ✅ Background worker
- ✅ Frontend dashboard
- ✅ ML model training pipeline
- 🔄 ML model integration (in progress)
- 🔄 Production deployment optimization

## Related Documentation

- [Getting Started Guide](./README.md)
- [Development Setup](../development/README.md)
- [API Documentation](../api/README.md)
- [Deployment Guide](../deployment/README.md)

