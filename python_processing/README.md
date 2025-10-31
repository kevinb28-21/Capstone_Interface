# Python Image Processing Service

This directory contains the Python-based image processing pipeline as specified in the architecture.

## Components

- `image_processor.py`: Core image processing functions (NDVI calculation, preprocessing, TensorFlow integration)
- `flask_api.py`: Flask REST API for receiving images from Raspberry Pi
- `requirements.txt`: Python dependencies

## Setup

1. Install dependencies:
```bash
cd python_processing
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run the Flask server:
```bash
python flask_api.py
```

Or with gunicorn for production:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 flask_api:app
```

## Testing Image Processing

Test the NDVI calculation directly:
```bash
python image_processor.py path/to/test_image.jpg
```

## API Endpoints

- `POST /api/upload`: Upload image (multipart/form-data)
  - Fields: `image` (file), `gps` (optional JSON string)
  - Returns: Image ID and analysis results

- `GET /api/data`: Get processed images
  - Query: `?image_id=<id>` for single, or omit for all
  - Returns: List of images with analysis data

- `GET /uploads/<filename>`: Serve uploaded images

## Integration with Node.js Backend

The Node.js backend can proxy to this Flask service:

```javascript
// In Node.js server
app.post('/api/images', upload.single('image'), async (req, res) => {
  // Forward to Flask
  const formData = new FormData();
  formData.append('image', req.file.buffer, req.file.originalname);
  const flaskRes = await fetch('http://localhost:5001/api/upload', {
    method: 'POST',
    body: formData
  });
  const data = await flaskRes.json();
  res.json(data);
});
```

## Next Steps

1. Add PostgreSQL integration for persistent storage
2. Add S3 integration for image storage
3. Load TensorFlow model when available
4. Add background job worker for automated processing
5. Integrate with Raspberry Pi upload script

