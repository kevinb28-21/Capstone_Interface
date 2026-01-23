# System Architecture

This document describes the overall architecture and design of the Drone Crop Health Platform.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         React Frontend (Vite + React-Leaflet)        │   │
│  │  - Dashboard UI                                       │   │
│  │  - Map Visualization                                  │   │
│  │  - Image Analytics                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/REST API
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Application Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Node.js Backend (Express.js)                 │   │
│  │  - API Gateway                                       │   │
│  │  - Image Upload Handling                             │   │
│  │  - Telemetry Management                              │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────┬───────────────────────┬─────────────────────┘
                │                       │
                │                       │
┌───────────────▼──────────┐  ┌────────▼──────────────────────┐
│   Processing Layer        │  │    Data Layer                 │
│  ┌──────────────────────┐ │  │  ┌────────────────────────┐  │
│  │  Python Flask API    │ │  │  │  PostgreSQL Database   │  │
│  │  - Image Upload     │ │  │  │  - Image Metadata       │  │
│  │  - Analysis Results  │ │  │  │  - Analysis Results     │  │
│  └──────────┬───────────┘ │  │  │  - GPS Coordinates      │  │
│             │             │  │  │  - Telemetry            │  │
│  ┌──────────▼───────────┐ │  │  └────────────────────────┘  │
│  │ Background Worker    │ │  │                               │
│  │ - NDVI Calculation   │ │  │                               │
│  │ - SAVI Calculation   │ │  │                               │
│  │ - GNDVI Calculation  │ │  │                               │
│  │ - Health Classification│ │  │                               │
│  └──────────────────────┘ │  │                               │
└───────────────────────────┘  └───────────────────────────────┘
                │
                │
┌───────────────▼──────────────────────────────────────────────┐
│                    Storage Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AWS S3 Storage                          │   │
│  │  - Original Images                                   │   │
│  │  - Processed Images                                  │   │
│  │  - Analysis Results                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend (React)

- **Framework**: React with Vite
- **Map Library**: React-Leaflet
- **Purpose**: User interface for viewing drone data and analysis results
- **Communication**: REST API calls to Node.js backend

### Backend (Node.js)

- **Framework**: Express.js
- **Purpose**: API gateway and business logic
- **Responsibilities**:
  - Image upload handling
  - Telemetry management
  - Route and geofence management
  - API routing

### Processing Service (Python)

- **Framework**: Flask
- **Purpose**: Image analysis and processing
- **Responsibilities**:
  - Vegetation index calculation (NDVI, SAVI, GNDVI)
  - Health classification
  - ML model inference
  - Image preprocessing

### Background Worker

- **Language**: Python
- **Purpose**: Automated image processing
- **Responsibilities**:
  - Monitor database for new uploads
  - Process images automatically
  - Update processing status
  - Handle S3 uploads/downloads

### Database (PostgreSQL)

- **Purpose**: Persistent data storage
- **Tables**:
  - `images` - Image metadata
  - `analyses` - Analysis results
  - `image_gps` - GPS coordinates
  - `telemetry` - Current drone state
  - `stress_zones` - Stress zone data

### Storage (AWS S3)

- **Purpose**: Scalable image storage
- **Structure**: Organized by date (`images/YYYY/MM/DD/`)
- **Content**: Original and processed images

## Data Flow

### Image Upload Flow

1. Drone captures image → Uploads to Flask API
2. Flask API saves to S3 and creates database record (status: 'uploaded')
3. Background worker detects new image
4. Worker downloads image from S3 (if needed)
5. Worker processes image (NDVI, SAVI, GNDVI)
6. Worker saves analysis results to database
7. Worker updates status to 'completed'
8. Frontend fetches and displays results

### Telemetry Flow

1. Drone sends telemetry → Node.js API
2. API updates database
3. Frontend polls API for updates
4. Map displays current position and route

## Technology Choices

### Why React?
- Component-based architecture
- Large ecosystem
- Good performance
- Easy to maintain

### Why Node.js?
- JavaScript across stack
- Fast development
- Good for API services
- Large package ecosystem

### Why Python for Processing?
- Excellent image processing libraries (OpenCV, NumPy)
- Strong ML ecosystem (TensorFlow)
- Scientific computing support

### Why PostgreSQL?
- Relational data structure
- ACID compliance
- Strong query capabilities
- Good performance

### Why S3?
- Scalable storage
- Cost-effective
- Integration with AWS services
- CDN capabilities

## Scalability Considerations

### Current Architecture
- Single server deployment
- Synchronous processing
- Direct database connections

### Future Improvements
- Load balancing for API servers
- Message queue for processing (RabbitMQ, SQS)
- Database connection pooling
- Caching layer (Redis)
- CDN for static assets

## Security Considerations

### Current State
- No authentication
- Direct database access
- No encryption at rest

### Recommended Improvements
- API authentication (JWT)
- Database encryption
- HTTPS/TLS
- Input validation
- Rate limiting
- CORS configuration

## Related Documentation

- [S3 Integration](./S3_INTEGRATION.md)
- [S3 Implementation](./S3_IMPLEMENTATION.md)
- [Getting Started](../getting-started/PROJECT_OVERVIEW.md)
- [Deployment Guide](../deployment/README.md)

