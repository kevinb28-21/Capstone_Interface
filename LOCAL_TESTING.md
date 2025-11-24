# Local Testing Guide

This guide shows you how to run and test the image processing algorithm locally.

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Start all services
./run-local.sh

# Stop all services
./stop-local.sh
```

### Option 2: Manual Commands

See "Manual Setup" section below.

## Prerequisites

1. **PostgreSQL** must be running:
   ```bash
   brew services start postgresql@14
   ```

2. **Database** must exist:
   ```bash
   # Check connection
   PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "SELECT 1;"
   ```

## Starting Services Manually

### Terminal 1: Python Background Worker

```bash
cd /Users/kevinbhatt/Desktop/Projects/Capstone_Interface/python_processing
source venv/bin/activate
python3 background_worker.py
```

**What it does:**
- Polls database for new images every 10 seconds
- Processes images: calculates NDVI, SAVI, GNDVI
- Saves analysis results to database
- Logs to console and `/tmp/worker.log`

**Watch logs:**
```bash
tail -f /tmp/worker.log
```

### Terminal 2: Node.js Backend

```bash
cd /Users/kevinbhatt/Desktop/Projects/Capstone_Interface/server
npm run dev
```

**What it does:**
- Runs Express API on port 5050
- Handles image uploads
- Serves API endpoints
- Logs to console

**Test API:**
```bash
curl http://localhost:5050/api/health
```

### Terminal 3: React Frontend

```bash
cd /Users/kevinbhatt/Desktop/Projects/Capstone_Interface/client
npm run dev
```

**What it does:**
- Runs Vite dev server on port 5182
- Serves React application
- Hot reload on code changes

**Open in browser:**
```
http://localhost:5182
```

## Testing Image Processing

### 1. Upload an Image

1. Open http://localhost:5182
2. Navigate to "Upload" or "Analytics" page
3. Click "Upload Image" or drag & drop
4. Select an image file (JPG, PNG)

### 2. Monitor Processing

**In the UI:**
- Status will change: `uploaded` → `processing` → `completed`
- Processing time: ~10-15 seconds (including poll wait)

**In Terminal (Worker logs):**
```bash
tail -f /tmp/worker.log
```

You should see:
```
INFO - Found 1 pending image(s) to process
INFO - Processing image <id>: <filename>
INFO - Analyzing onion crop image: <path>
INFO - ✓ Successfully processed image <id>
```

### 3. Check Results

**In the UI:**
- Go to "Analytics" page
- View NDVI, SAVI, GNDVI values
- See health status and summary

**In Database:**
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
SELECT 
    filename, 
    processing_status,
    uploaded_at,
    processed_at
FROM images 
ORDER BY uploaded_at DESC 
LIMIT 5;
"
```

**View Analysis Results:**
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
SELECT 
    i.filename,
    a.ndvi_mean,
    a.savi_mean,
    a.gndvi_mean,
    a.health_status,
    a.summary
FROM images i
JOIN analyses a ON i.id = a.image_id
ORDER BY i.uploaded_at DESC
LIMIT 5;
"
```

## Troubleshooting

### Worker Not Processing Images

**Check worker is running:**
```bash
ps aux | grep background_worker.py
```

**Check database connection:**
```bash
cd python_processing
source venv/bin/activate
python3 -c "from db_utils import test_connection; print('OK' if test_connection() else 'FAILED')"
```

**Check for pending images:**
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
SELECT COUNT(*) FROM images WHERE processing_status = 'uploaded';
"
```

### Images Not Found Error

**Check file paths:**
```bash
ls -la server/uploads/
```

**Fix database paths:**
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
UPDATE images 
SET file_path = '/Users/kevinbhatt/Desktop/Projects/Capstone_Interface/server/uploads/' || filename,
    s3_stored = false 
WHERE file_path IS NULL;
"
```

### Backend Not Responding

**Check if running:**
```bash
curl http://localhost:5050/api/health
```

**Check logs:**
```bash
tail -f /tmp/server.log
```

**Restart:**
```bash
cd server
npm run dev
```

### Frontend Not Loading

**Check if running:**
```bash
curl http://localhost:5182
```

**Check logs:**
```bash
tail -f /tmp/frontend.log
```

**Clear cache and restart:**
```bash
cd client
rm -rf node_modules/.vite
npm run dev
```

## Performance Testing

### Test Processing Speed

**Upload multiple images:**
```bash
# Upload 5 images quickly
# Monitor processing time in logs
tail -f /tmp/worker.log | grep "Successfully processed"
```

**Expected times:**
- Small images (< 1MB): ~0.1-0.2 seconds analysis
- Medium images (1-5MB): ~0.5-2 seconds analysis
- Large images (> 5MB): ~2-5 seconds analysis
- Total time: +10 seconds (poll wait)

### Test Concurrent Processing

The worker processes images in batches (default: 5 images per batch).

**Monitor batch processing:**
```bash
tail -f /tmp/worker.log | grep -E "Found|Processed.*batch"
```

## Useful Commands

### View All Logs
```bash
tail -f /tmp/worker.log /tmp/server.log /tmp/frontend.log
```

### Check Service Status
```bash
# Worker
ps aux | grep background_worker

# Backend
ps aux | grep "node.*server"

# Frontend
ps aux | grep vite
```

### Reset Failed Images
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
UPDATE images 
SET processing_status = 'uploaded' 
WHERE processing_status = 'failed';
"
```

### Clear All Images (Testing)
```bash
PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "
DELETE FROM analyses;
DELETE FROM images;
"
```

## Next Steps

1. ✅ Test with different image sizes
2. ✅ Test with multiple images
3. ✅ Verify NDVI/SAVI/GNDVI calculations
4. ✅ Check health status classifications
5. ✅ Test error handling (invalid images, etc.)

