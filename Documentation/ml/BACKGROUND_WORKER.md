# Background Worker Implementation Summary

## ✅ What Has Been Implemented

### 1. Database Utilities (`python_processing/db_utils.py`)
- **Database connection pooling** for efficient connections
- **`get_pending_images()`** - Fetches images with status 'uploaded'
- **`update_image_status()`** - Updates processing status
- **`set_processing_started()`** - Marks image as 'processing'
- **`set_processing_completed()`** - Marks image as 'completed'
- **`set_processing_failed()`** - Marks image as 'failed'
- **`save_analysis()`** - Saves analysis results to database
- **`get_image_path()`** - Gets local file path for image

### 2. Background Worker (`python_processing/background_worker.py`)
- **Monitors database** for new uploads (polls every 10 seconds)
- **Automatic processing** of images with status 'uploaded'
- **Status tracking** through the pipeline
- **S3 integration** - Downloads from S3 if needed, uploads processed images
- **Error handling** - Marks failed images appropriately
- **Logging** - Comprehensive logging to file and console
- **Graceful shutdown** - Handles SIGINT/SIGTERM signals

### 3. Status Flow Implementation

```
uploaded → processing → completed
              ↓
           failed (if error)
```

**Status Definitions:**
- **uploaded**: Image saved to database, waiting for processing
- **processing**: Analysis has started
- **completed**: Analysis done, results saved, processed image uploaded
- **failed**: Error occurred during processing

### 4. Database Schema Updates
- **`processing_status`** field default changed to 'uploaded'
- Status values: `uploaded`, `processing`, `completed`, `failed`
- **`processed_at`** timestamp tracks when processing completed

### 5. Flask API with Database (`python_processing/flask_api_db.py`)
- **Saves images** with status 'uploaded' (doesn't process immediately)
- **Returns immediately** to client
- **Background worker** picks up and processes automatically
- **GPS data** saved to `image_gps` table

### 6. S3 Integration
- **`download_from_s3()`** - Downloads images from S3 for processing
- **Automatic upload** of processed images to S3
- **Fallback** to local storage if S3 not available

### 7. Systemd Service File
- **`background_worker.service`** - Systemd service configuration
- Auto-restart on failure
- Runs as daemon on system boot

### 8. SAVI Calculation
- **`calculate_savi()`** function added to `image_processor.py`
- Calculates Soil-Adjusted Vegetation Index
- Better for sparse vegetation (like onion beds)

## 📋 Files Created/Modified

### New Files:
1. `python_processing/db_utils.py` - Database utilities
2. `python_processing/background_worker.py` - Background worker service
3. `python_processing/flask_api_db.py` - Flask API with database integration
4. `python_processing/background_worker.service` - Systemd service file
5. `python_processing/BACKGROUND_WORKER_SETUP.md` - Setup guide

### Modified Files:
1. `python_processing/image_processor.py` - Added `calculate_savi()` function
2. `python_processing/s3_utils.py` - Added `download_from_s3()` function
3. `server/database/schema.sql` - Updated default status to 'uploaded'

## 🔄 How It Works

### Image Upload Flow:

1. **Client uploads image** → Flask API (`flask_api_db.py`)
2. **Image saved to database** with status `uploaded`
3. **S3 upload** (if configured)
4. **API returns immediately** to client
5. **Background worker** polls database every 10 seconds
6. **Worker finds** images with status `uploaded`
7. **Status changed** to `processing`
8. **Analysis performed** (NDVI, SAVI, health metrics)
9. **Results saved** to `analyses` table
10. **Processed image** uploaded to S3
11. **Status changed** to `completed`

### Status Tracking:

```sql
-- Check status of images
SELECT id, filename, processing_status, uploaded_at, processed_at
FROM images
ORDER BY uploaded_at DESC;

-- See pending queue
SELECT COUNT(*) FROM images WHERE processing_status = 'uploaded';

-- See processing
SELECT COUNT(*) FROM images WHERE processing_status = 'processing';

-- See completed
SELECT COUNT(*) FROM images WHERE processing_status = 'completed';

-- See failed
SELECT COUNT(*) FROM images WHERE processing_status = 'failed';
```

## 🚀 Setup on EC2

### 1. Install Dependencies
```bash
cd python_processing
pip3 install -r requirements.txt
```

### 2. Configure Environment
```bash
# Create .env file
nano .env
```

Add:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your-password
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
```

### 3. Test Worker
```bash
python3 background_worker.py
```

### 4. Set Up as Service
```bash
sudo cp background_worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable drone-worker
sudo systemctl start drone-worker
sudo systemctl status drone-worker
```

## 📊 Monitoring

### View Logs
```bash
# Systemd logs
sudo journalctl -u drone-worker -f

# File logs
tail -f background_worker.log
```

### Check Database Status
```sql
-- Processing queue
SELECT processing_status, COUNT(*) 
FROM images 
GROUP BY processing_status;
```

## ✨ Features

- ✅ **Automatic processing** - No manual intervention needed
- ✅ **Status tracking** - Know exactly where each image is
- ✅ **Error handling** - Failed images marked appropriately
- ✅ **S3 integration** - Works with S3 storage
- ✅ **Scalable** - Can run multiple workers
- ✅ **Resilient** - Auto-restarts on failure
- ✅ **Logging** - Comprehensive logging for debugging

## 🔧 Configuration

### Poll Interval
```env
WORKER_POLL_INTERVAL=10  # seconds between database polls
```

### Batch Size
```env
WORKER_BATCH_SIZE=5  # images to process per batch
```

## 📝 Next Steps

1. **Deploy to EC2** - Follow setup guide
2. **Test with real images** - Upload and verify processing
3. **Monitor logs** - Ensure everything works correctly
4. **Scale if needed** - Run multiple workers for high volume

The background worker is ready to automatically process all uploaded images! 🎉

