# API Documentation

Complete API reference for the Drone Crop Health Platform.

## API Overview

The platform consists of two main APIs:
1. **Node.js API** - Main backend API (Express.js)
2. **Python Flask API** - Image processing service

## Base URLs

- **Node.js API**: `http://localhost:5000`
- **Flask API**: `http://localhost:5001`

## Node.js API

See [Node.js API Documentation](./NODE_API.md) for complete details.

### Main Endpoints

- `POST /api/images` - Upload image
- `GET /api/images` - List all images
- `GET /api/images/:id` - Get specific image
- `POST /api/telemetry` - Update telemetry
- `GET /api/telemetry` - Get telemetry

## Flask API

See [Flask API Documentation](./FLASK_API.md) for complete details.

### Main Endpoints

- `POST /api/upload` - Upload image for processing
- `GET /api/data` - Get processed images
- `GET /api/health` - Health check
- `GET /uploads/<filename>` - Serve uploaded images

## Authentication

Currently, the APIs do not require authentication. For production, implement:
- API keys
- JWT tokens
- OAuth2

## Error Responses

All APIs return errors in this format:

```json
{
  "error": "Error message description"
}
```

**Status Codes:**
- `200` - Success
- `201` - Created
- `202` - Accepted (processing started)
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

## Rate Limiting

Currently no rate limiting implemented. For production, consider:
- Per-IP rate limits
- Per-user rate limits
- Request throttling

## Related Documentation

- [Node.js API](./NODE_API.md)
- [Flask API](./FLASK_API.md)
- [Development Guide](../development/README.md)

