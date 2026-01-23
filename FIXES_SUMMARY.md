# Critical Fixes Summary - Image Processing & ML Model Status

## Overview
Fixed two critical issues in the drone crop health monitoring platform:
1. **Images stuck in "processing" status** - Images never transitioned to "completed" or "failed"
2. **ML model status showing "Not Available"** - Model detection endpoint failed to find existing models

## Changes Made

### 1. Background Worker (`python_processing/background_worker.py`)

#### Enhanced Error Handling
- **Added tracking flag** (`marked_as_processing`) to ensure images are marked as failed if an error occurs after status is set to 'processing'
- **Improved exception handling** in `process_image()` to always call `set_processing_failed()` on any error
- **Fixed status transition bug**: If `set_processing_completed()` fails, the image is now marked as 'failed' instead of remaining in 'processing'

#### Improved Logging
- Added structured logging with image IDs: `[{image_id}] Starting processing for {filename}`
- Added detailed step-by-step logging:
  - `[{image_id}] Marking as 'processing'`
  - `[{image_id}] Local file path: {path}`
  - `[{image_id}] Analysis result: NDVI=..., SAVI=..., GNDVI=..., health_status=...`
  - `[{image_id}] Saved analysis row successfully`
  - `[{image_id}] Status set to 'completed'`
- Added poll-level logging: `[POLL] Checking for pending images...` and `[POLL] Found X pending image(s) to process`

#### Recovery Mechanism
- **New function `recover_stuck_images()`**: Automatically resets images stuck in 'processing' status (older than 5 minutes) back to 'uploaded'
- Runs at worker startup and periodically (every 5 minutes) during operation
- Prevents accumulation of stuck images from previous crashes

#### Enhanced Batch Processing
- Individual error handling for each image in a batch
- Ensures one failed image doesn't prevent others from processing
- Double-checks status updates even on unexpected errors

### 2. Database Utilities (`python_processing/db_utils.py`)

#### Improved `get_pending_images()`
- Added debug logging to track how many images are found
- Confirmed it only queries `processing_status = 'uploaded'` (excludes 'processing' to avoid duplicates)
- Added exception logging with stack traces

#### Enhanced `save_analysis()`
- Added logging after successful save: `[{image_id}] Saved analysis row with id {analysis_id}`
- Improved error logging with image ID context

### 3. Node.js Backend (`server/src/server.js`)

#### Fixed ML Status Endpoint (`/api/ml/status`)
- **Robust path resolution**: Uses `path.resolve()` instead of `path.join()` for absolute paths
- **Better error handling**: Checks if directories exist before attempting to read them
- **Enhanced logging**: Logs all path checks and model detection steps
- **Improved model detection**:
  - Checks multi-crop model directory first
  - Falls back to single-crop model if multi-crop not found
  - Handles both environment variable paths and default paths
  - Extracts model version from metadata files or filename
- **Consistent response format**: Always returns `model_available`, `model_type`, `model_path`, `model_version`

### 4. Testing Checklist
Added comprehensive testing checklist as comments in `background_worker.py` covering:
- Service startup verification
- Image upload and processing flow
- Database status transitions
- API endpoint verification
- Model status detection
- Failure handling
- Recovery mechanism

## Key Improvements

### Issue 1: Images Stuck in "Processing"
**Root Causes Identified:**
1. If `set_processing_completed()` failed, image remained in 'processing'
2. No recovery mechanism for images stuck from previous crashes
3. Insufficient error handling in batch processing

**Solutions Implemented:**
- ✅ Always mark as 'failed' if completion fails
- ✅ Recovery function resets stuck images to 'uploaded'
- ✅ Individual error handling per image in batch
- ✅ Comprehensive logging for debugging

### Issue 2: ML Model "Not Available"
**Root Causes Identified:**
1. Path resolution using relative paths instead of absolute
2. Missing existence checks before file operations
3. Insufficient logging to diagnose path issues

**Solutions Implemented:**
- ✅ Use `path.resolve()` for absolute path resolution
- ✅ Check directory existence before reading
- ✅ Enhanced logging for path detection
- ✅ Better error messages in response

## Verification Steps

### 1. Test Image Processing
```bash
# Start services
cd server && npm run dev
cd python_processing && python background_worker.py

# Upload image via UI, then check database:
psql -d drone_analytics -c "
  SELECT id, filename, processing_status, uploaded_at, processed_at
  FROM images
  ORDER BY uploaded_at DESC
  LIMIT 5;
"

# Verify status transitions: uploaded → processing → completed
```

### 2. Test ML Model Detection
```bash
# Ensure model file exists:
ls -la python_processing/models/multi_crop/*_final.h5
# OR
ls -la python_processing/models/onion_crop_health_model.h5

# Check API response:
curl http://localhost:5050/api/ml/status

# Should return:
# {
#   "model_available": true,
#   "model_type": "multi_crop" or "single_crop",
#   "model_path": "/absolute/path/to/model.h5",
#   "model_version": "..."
# }
```

### 3. Test Recovery Mechanism
```sql
-- Manually create a stuck image:
UPDATE images 
SET processing_status = 'processing', 
    updated_at = NOW() - INTERVAL '10 minutes'
WHERE id = '<some-image-id>';

-- Restart worker, should see:
-- "Found 1 image(s) stuck in 'processing' status"
-- "Recovered 1 stuck image(s) - reset to 'uploaded' status"
```

## Files Modified

1. `python_processing/background_worker.py`
   - Enhanced `process_image()` with better error handling
   - Added `recover_stuck_images()` function
   - Improved `process_batch()` with individual error handling
   - Enhanced logging throughout
   - Added testing checklist comments

2. `python_processing/db_utils.py`
   - Improved `get_pending_images()` logging
   - Enhanced `save_analysis()` logging

3. `server/src/server.js`
   - Fixed `/api/ml/status` endpoint path resolution
   - Added comprehensive logging
   - Improved error handling

## Expected Behavior After Fixes

### Image Processing Flow
1. Image uploaded → `processing_status = 'uploaded'`
2. Worker picks up image → `processing_status = 'processing'`
3. Analysis completes → Analysis saved to `analyses` table
4. Status updated → `processing_status = 'completed'`
5. **On any error** → `processing_status = 'failed'` (never stuck in 'processing')

### ML Model Status
1. Worker checks for models at startup and logs which model is loaded
2. `/api/ml/status` correctly detects models using absolute paths
3. Frontend displays "Model Status: Available" when `model_available: true`
4. Model type, version, and path are correctly returned

## Notes

- All changes maintain backward compatibility
- No database schema changes required
- Frontend components already handle the API responses correctly
- Recovery mechanism runs automatically (no manual intervention needed)
- Enhanced logging helps diagnose issues in production
