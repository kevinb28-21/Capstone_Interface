# Node.js API Documentation

Complete API reference for the Express.js backend.

## Base URL

```
http://localhost:5000
```

## Endpoints

### Image Management

#### POST `/api/images`

Upload an image for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `image` (file) - Image file (jpg, png, tiff)
  - `gps` (optional, string) - GPS metadata as JSON string

**Example:**
```bash
curl -X POST http://localhost:5000/api/images \
  -F "image=@image.jpg" \
  -F 'gps={"latitude":43.6532,"longitude":-79.3832,"altitude":100}'
```

**Response:**
```json
{
  "id": "abc123",
  "filename": "1234567890_image.jpg",
  "path": "/uploads/1234567890_image.jpg",
  "s3Url": "https://bucket.s3.amazonaws.com/images/...",
  "s3Stored": true,
  "gps": {
    "latitude": 43.6532,
    "longitude": -79.3832,
    "altitude": 100
  },
  "createdAt": "2024-01-01T12:00:00Z",
  "analysis": {
    "ndvi_mean": 0.65,
    "health_status": "Healthy",
    "stress_zones": [...]
  }
}
```

#### GET `/api/images`

List all uploaded images.

**Response:**
```json
[
  {
    "id": "abc123",
    "filename": "image.jpg",
    "path": "/uploads/image.jpg",
    "createdAt": "2024-01-01T12:00:00Z",
    "analysis": {...}
  }
]
```

#### GET `/api/images/:id`

Get a specific image by ID.

**Response:**
```json
{
  "id": "abc123",
  "filename": "image.jpg",
  "path": "/uploads/image.jpg",
  "createdAt": "2024-01-01T12:00:00Z",
  "analysis": {...}
}
```

### Telemetry

#### POST `/api/telemetry`

Update drone telemetry.

**Request:**
```json
{
  "position": {
    "lat": 43.6532,
    "lng": -79.3832
  },
  "route": [
    {"lat": 43.6532, "lng": -79.3832},
    {"lat": 43.6542, "lng": -79.3842}
  ],
  "geofence": [
    {"lat": 43.6555, "lng": -79.391},
    {"lat": 43.6505, "lng": -79.391},
    {"lat": 43.6505, "lng": -79.3755},
    {"lat": 43.6555, "lng": -79.3755}
  ]
}
```

**Response:**
```json
{
  "position": {"lat": 43.6532, "lng": -79.3832},
  "route": [...],
  "geofence": [...]
}
```

#### GET `/api/telemetry`

Get current telemetry.

**Response:**
```json
{
  "position": {"lat": 43.6532, "lng": -79.3832},
  "route": [...],
  "geofence": [...]
}
```

## Error Responses

```json
{
  "error": "Error message"
}
```

## Status Codes

- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

