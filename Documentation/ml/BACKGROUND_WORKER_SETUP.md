# Background Worker Setup Guide

## Overview

The background worker service automatically monitors the database for new image uploads and processes them through the analysis pipeline.

## Status Flow

```
uploaded → processing → completed
              ↓
           failed (if error)
```

### Status Definitions

- **uploaded**: Image has been uploaded and saved to database, waiting for processing
- **processing**: Analysis has started
- **completed**: Fully processed (analysis completed + results saved + processed image uploaded to S3)
- **failed**: Processing encountered an error

## Setup Instructions

### 1. Install Dependencies

```bash
cd python_processing
pip install -r requirements.txt
```

Make sure `psycopg2-binary` is in requirements.txt:
```
psycopg2-binary==2.9.9
```

### 2. Configure Environment Variables

Create/update `.env` file:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your-password

# Worker Configuration
WORKER_POLL_INTERVAL=10  # seconds between polls
WORKER_BATCH_SIZE=5      # images to process per batch

# S3 Configuration
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket

# Upload Folders
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed
```

### 3. Test Database Connection

```bash
python3 -c "from db_utils import test_connection; print('Connected!' if test_connection() else 'Failed')"
```

### 4. Run Worker Manually (Testing)

```bash
cd python_processing
python3 background_worker.py
```

You should see:
```
Background Image Processing Worker Starting
Poll interval: 10 seconds
Batch size: 5 images
Worker is running. Press Ctrl+C to stop.
```

### 5. Set Up as Systemd Service (Production)

#### On EC2/Linux:

1. **Copy service file:**
   ```bash
   sudo cp background_worker.service /etc/systemd/system/drone-worker.service
   ```

2. **Update paths in service file** if needed:
   ```bash
   sudo nano /etc/systemd/system/drone-worker.service
   ```

3. **Reload systemd:**
   ```bash
   sudo systemctl daemon-reload
   ```

4. **Enable service (start on boot):**
   ```bash
   sudo systemctl enable drone-worker
   ```

5. **Start service:**
   ```bash
   sudo systemctl start drone-worker
   ```

6. **Check status:**
   ```bash
   sudo systemctl status drone-worker
   ```

7. **View logs:**
   ```bash
   sudo journalctl -u drone-worker -f
   ```

### 6. Service Management Commands

```bash
# Start
sudo systemctl start drone-worker

# Stop
sudo systemctl stop drone-worker

# Restart
sudo systemctl restart drone-worker

# View status
sudo systemctl status drone-worker

# View logs
sudo journalctl -u drone-worker -f

# View last 100 lines
sudo journalctl -u drone-worker -n 100
```

## How It Works

1. **Image Upload**: Flask API receives image, saves to database with status `uploaded`
2. **Worker Polls**: Background worker checks database every 10 seconds (configurable)
3. **Status Update**: Worker changes status to `processing`
4. **Analysis**: Worker runs OpenCV/TensorFlow analysis
5. **Save Results**: Analysis results saved to `analyses` table
6. **Upload Processed**: Processed image uploaded to S3
7. **Status Complete**: Status changed to `completed`

## Monitoring

### Check Processing Queue

```sql
-- See pending images
SELECT id, filename, processing_status, uploaded_at
FROM images
WHERE processing_status IN ('uploaded', 'processing')
ORDER BY uploaded_at ASC;

-- See completed images
SELECT id, filename, processing_status, processed_at
FROM images
WHERE processing_status = 'completed'
ORDER BY processed_at DESC
LIMIT 10;

-- See failed images
SELECT id, filename, processing_status, uploaded_at
FROM images
WHERE processing_status = 'failed'
ORDER BY uploaded_at DESC;
```

### Worker Logs

Logs are written to:
- `background_worker.log` (in working directory)
- Systemd journal (if running as service)

## Troubleshooting

### Worker not processing images

1. **Check database connection:**
   ```bash
   python3 -c "from db_utils import test_connection; test_connection()"
   ```

2. **Check for pending images:**
   ```sql
   SELECT COUNT(*) FROM images WHERE processing_status = 'uploaded';
   ```

3. **Check worker logs:**
   ```bash
   tail -f background_worker.log
   ```

### Images stuck in 'processing'

If images are stuck, manually reset:
```sql
UPDATE images 
SET processing_status = 'uploaded' 
WHERE processing_status = 'processing' 
AND processed_at IS NULL;
```

### Database connection errors

- Verify database is running: `sudo systemctl status postgresql`
- Check connection string in `.env`
- Verify database credentials

### S3 upload failures

- Check AWS credentials in `.env`
- Verify S3 bucket exists and is accessible
- Check IAM permissions

## Performance Tuning

### Adjust Poll Interval

In `.env`:
```env
WORKER_POLL_INTERVAL=5  # More frequent (faster processing, more DB queries)
WORKER_POLL_INTERVAL=30 # Less frequent (slower processing, fewer DB queries)
```

### Adjust Batch Size

```env
WORKER_BATCH_SIZE=10  # Process more images at once
WORKER_BATCH_SIZE=1   # Process one at a time (more controlled)
```

### Multiple Workers

For high volume, run multiple workers:
```bash
# Worker 1
python3 background_worker.py

# Worker 2 (different terminal)
WORKER_BATCH_SIZE=5 python3 background_worker.py
```

Note: Database handles concurrent updates safely.

## Integration with Flask API

The Flask API should use `flask_api_db.py` which:
- Saves images with status `uploaded`
- Does NOT process immediately
- Returns immediately to client
- Background worker picks up and processes

## Status Tracking in Frontend

Frontend can poll for status:
```javascript
// Check image processing status
const checkStatus = async (imageId) => {
  const response = await fetch(`/api/images/${imageId}`);
  const data = await response.json();
  return data.processing_status; // uploaded, processing, completed, failed
};
```

