# PostgreSQL Database Schema Overview

## Database Structure

This schema stores all data for the Drone Crop Health Platform including image metadata, GPS coordinates, timestamps, processing status, telemetry, and geofences.

---

## Tables

### 1. **missions**
Stores flight session/mission information.

**Key Fields:**
- `id` (UUID) - Primary key
- `name` - Mission name
- `start_time`, `end_time` - Mission timestamps
- `status` - active, completed, cancelled

---

### 2. **images** ⭐
Stores image metadata and file information.

**Key Fields:**
- `id` (UUID) - Primary key
- `mission_id` - Links to mission
- `filename`, `original_name` - File names
- `file_path` - Local path (if not S3)
- `s3_url`, `s3_key` - S3 storage info
- `s3_stored` (BOOLEAN) - Whether stored in S3
- `file_size`, `mime_type` - File metadata
- `captured_at` - When drone captured image
- `uploaded_at` - When uploaded to server
- `processing_status` - pending, processing, completed, failed
- `processed_at` - When processing completed

**Indexes:**
- `mission_id`, `captured_at`, `uploaded_at`, `processing_status`

---

### 3. **image_gps** 📍
Stores GPS coordinates for each image.

**Key Fields:**
- `id` (UUID) - Primary key
- `image_id` (UNIQUE) - Links to image
- `latitude`, `longitude` - GPS coordinates
- `altitude` - Height in meters
- `accuracy` - GPS accuracy in meters
- `heading` - Bearing/heading (0-360°)
- `ground_speed`, `speed` - Speed in m/s
- `captured_at` - GPS timestamp

**Indexes:**
- `image_id`, `(latitude, longitude)` for location queries

---

### 4. **analyses** 🔬
Stores crop health analysis results.

**Key Fields:**
- `id` (UUID) - Primary key
- `image_id` (UNIQUE) - Links to image
- **NDVI Values:**
  - `ndvi`, `ndvi_mean` - Mean NDVI (0-1)
  - `ndvi_std`, `ndvi_min`, `ndvi_max` - Statistics
- **SAVI Values:**
  - `savi`, `savi_mean` - Mean SAVI (0-1)
  - `savi_std`, `savi_min`, `savi_max` - Statistics
- **Health Metrics:**
  - `health_score` (0-100) - Overall health
  - `health_status` - healthy, attention_needed, stressed, critical
- `summary` - Text summary
- `analysis_type` - placeholder, opencv, tensorflow
- `model_version` - ML model version
- `confidence` - Confidence score (0-1)
- `processed_image_path`, `processed_s3_url` - Processed image locations
- `processed_at` - Processing timestamp

**Indexes:**
- `image_id`, `health_status`, `processed_at`

---

### 5. **stress_zones** 🗺️
Stores detailed stress zone grid data.

**Key Fields:**
- `id` (UUID) - Primary key
- `analysis_id` - Links to analysis
- `grid_x`, `grid_y` - Grid coordinates (0-100)
- `severity` (0-1) - Stress severity
- `ndvi_value`, `savi_value` - Values at grid point

**Unique Constraint:**
- `(analysis_id, grid_x, grid_y)` - One point per grid location

**Indexes:**
- `analysis_id`, `severity`

---

### 6. **telemetry** 🚁
Stores current drone position and state.

**Key Fields:**
- `id` (UUID) - Primary key
- `latitude`, `longitude` - Current position
- `altitude` - Height in meters
- `heading` - Direction (0-360°)
- `ground_speed` - Speed in m/s
- `battery_level` (0-100) - Battery percentage
- `status` - idle, flying, landing, error
- `timestamp` - Current timestamp

**Indexes:**
- `timestamp`, `status`

---

### 7. **route_points** 🛤️
Stores drone flight path history.

**Key Fields:**
- `id` (UUID) - Primary key
- `latitude`, `longitude` - Route point position
- `altitude` - Height at point
- `sequence` - Order in route (0, 1, 2, ...)
- `timestamp` - When point was recorded

**Indexes:**
- `sequence`, `timestamp`

---

### 8. **geofences** 🗺️
Stores geofence definitions.

**Key Fields:**
- `id` (UUID) - Primary key
- `name` - Geofence name
- `description` - Description
- `is_active` (BOOLEAN) - Whether active

**Indexes:**
- `is_active`

---

### 9. **geofence_points** 📍
Stores points that form geofence boundaries.

**Key Fields:**
- `id` (UUID) - Primary key
- `geofence_id` - Links to geofence
- `latitude`, `longitude` - Point coordinates
- `sequence` - Order to form polygon

**Unique Constraint:**
- `(geofence_id, sequence)` - One point per sequence

**Indexes:**
- `geofence_id`, `(geofence_id, sequence)`

---

## Relationships

```
missions
  └── images (mission_id)
        ├── image_gps (image_id) [1:1]
        └── analyses (image_id) [1:1]
              └── stress_zones (analysis_id) [1:many]

geofences
  └── geofence_points (geofence_id) [1:many]

telemetry [standalone - current state]
route_points [standalone - historical path]
```

---

## Key Features

### Timestamps
- **captured_at**: When drone captured image
- **uploaded_at**: When uploaded to server
- **processed_at**: When analysis completed
- **created_at**: Record creation time
- **updated_at**: Last update time (auto-updated via triggers)

### Processing Status
Images have `processing_status` field:
- `pending` - Uploaded, waiting for processing
- `processing` - Currently being analyzed
- `completed` - Analysis finished
- `failed` - Processing error

### S3 Integration
- `s3_stored` (BOOLEAN) - Whether in S3
- `s3_url` - Full S3 URL
- `s3_key` - S3 object key
- `file_path` - Local path (fallback)

### GPS Data
- Stored in separate `image_gps` table
- Includes altitude, accuracy, heading, speed
- Indexed for location-based queries

### Health Metrics
- **NDVI**: Normalized Difference Vegetation Index
- **SAVI**: Soil-Adjusted Vegetation Index (better for sparse vegetation)
- **health_score**: Overall score (0-100)
- **health_status**: Categorical status

---

## Usage Examples

### Insert Image with GPS
```sql
-- Insert image
INSERT INTO images (filename, original_name, s3_url, s3_stored, captured_at)
VALUES ('img123.jpg', 'field_photo.jpg', 'https://bucket.s3...', TRUE, NOW())
RETURNING id;

-- Insert GPS data
INSERT INTO image_gps (image_id, latitude, longitude, altitude, heading)
VALUES ('image-uuid', 43.6532, -79.3832, 50.5, 90.0);
```

### Insert Analysis Results
```sql
-- Insert analysis
INSERT INTO analyses (image_id, ndvi_mean, savi_mean, health_score, health_status, summary)
VALUES ('image-uuid', 0.65, 0.58, 85.0, 'healthy', 'Crop health is excellent');

-- Insert stress zones
INSERT INTO stress_zones (analysis_id, grid_x, grid_y, severity, ndvi_value)
VALUES 
  ('analysis-uuid', 10, 20, 0.3, 0.55),
  ('analysis-uuid', 11, 20, 0.2, 0.60);
```

### Query Images with Analysis
```sql
SELECT 
  i.id,
  i.filename,
  i.s3_url,
  i.captured_at,
  g.latitude,
  g.longitude,
  a.ndvi_mean,
  a.savi_mean,
  a.health_status
FROM images i
LEFT JOIN image_gps g ON i.id = g.image_id
LEFT JOIN analyses a ON i.id = a.image_id
WHERE i.processing_status = 'completed'
ORDER BY i.captured_at DESC;
```

---

## Setup Instructions

1. **Create database:**
   ```bash
   createdb drone_analytics
   ```

2. **Run schema:**
   ```bash
   psql -U postgres -d drone_analytics -f server/database/schema.sql
   ```

3. **Verify:**
   ```sql
   \dt  -- List tables
   \d images  -- Describe images table
   ```

---

## Notes

- Uses **UUID** for primary keys (better for distributed systems)
- **Automatic timestamps** via triggers
- **Indexes** on frequently queried fields
- **Foreign keys** with CASCADE deletes
- **Unique constraints** prevent duplicates
- **Comments** on tables for documentation

