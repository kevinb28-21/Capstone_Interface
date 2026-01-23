# Flask API Documentation

Complete API reference for the Python Flask image processing service.

## Base URL

```
http://localhost:5001
```

## Endpoints

### Image Upload

#### POST `/api/upload`

Upload an image for processing. The image is saved with status 'uploaded' and processed by the background worker.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `image` (file) - Image file (jpg, png, tiff)
  - `gps` (optional, form field) - GPS metadata as JSON string

**Example:**
```bash
curl -X POST http://localhost:5001/api/upload \
  -F "image=@image.jpg" \
  -F 'gps={"latitude":43.6532,"longitude":-79.3832,"altitude":100,"bearing":45,"speed":5.0}'
```

**Response (202 Accepted):**
```json
{
  "id": "uuid-here",
  "filename": "1234567890_image.jpg",
  "path": "https://s3.amazonaws.com/bucket/images/...",
  "s3_url": "https://s3.amazonaws.com/bucket/images/...",
  "s3_stored": true,
  "processing_status": "uploaded",
  "message": "Image uploaded successfully. Processing will begin shortly.",
  "gps": {
    "latitude": 43.6532,
    "longitude": -79.3832,
    "altitude": 100
  }
}
```

**Status Codes:**
- `202` - Accepted (uploaded, processing will start)
- `400` - Bad Request (missing file, invalid type)
- `500` - Internal Server Error

### Data Retrieval

#### GET `/api/data`

Get processed image data and analyses.

**Query Parameters:**
- `image_id` (optional) - Get specific image by ID

**Example:**
```bash
# Get all images
curl http://localhost:5001/api/data

# Get specific image
curl http://localhost:5001/api/data?image_id=uuid-here
```

**Response (all images):**
```json
{
  "images": [
    {
      "id": "uuid-here",
      "filename": "image.jpg",
      "original_name": "image.jpg",
      "s3_url": "https://s3.amazonaws.com/...",
      "uploaded_at": "2024-01-01T12:00:00Z",
      "processing_status": "completed",
      "latitude": 43.6532,
      "longitude": -79.3832,
      "ndvi_mean": 0.65,
      "savi_mean": 0.58,
      "health_status": "Healthy",
      "summary": "Healthy - Onion crop with green foliage...",
      "analysis_type": "ndvi_savi_gndvi_onion"
    }
  ]
}
```

**Response (single image):**
```json
{
  "id": "uuid-here",
  "filename": "image.jpg",
  "ndvi_mean": 0.65,
  "savi_mean": 0.58,
  "health_status": "Healthy",
  ...
}
```

**Status Codes:**
- `200` - Success
- `404` - Image not found (when using image_id)

### Health Check

#### GET `/api/health`

Check service and database status.

**Response:**
```json
{
  "status": "ok",
  "service": "flask-image-processor",
  "database": "connected"
}
```

**Status Codes:**
- `200` - Service is healthy
- `503` - Service unavailable (if database disconnected)

### Image Serving

#### GET `/uploads/<filename>`

Serve uploaded images directly.

**Example:**
```
http://localhost:5001/uploads/1234567890_image.jpg
```

**Response:**
- Image file (binary)
- Content-Type: `image/jpeg`, `image/png`, etc.

**Status Codes:**
- `200` - Success
- `404` - File not found

## Processing Status

Images go through these statuses:

1. **uploaded** - Image uploaded, waiting for processing
2. **processing** - Currently being analyzed
3. **completed** - Analysis complete, results available
4. **failed** - Processing failed (error details in database)

## GPS Metadata Format

```json
{
  "latitude": 43.6532,
  "longitude": -79.3832,
  "altitude": 100.0,
  "bearing": 45.0,
  "speed": 5.0,
  "accuracy": 5.0,
  "timestamp": 1704110400000
}
```

## Error Responses

```json
{
  "error": "Error message description"
}
```

## Related Documentation

- [Python Processing README](../python-processing/README.md)
- [Background Worker](../python-processing/BACKGROUND_WORKER.md)

