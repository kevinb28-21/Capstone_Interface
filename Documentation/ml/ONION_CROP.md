# Onion Crop Health Analysis Updates

This document summarizes all updates made to support onion (Allium cepa) crop health monitoring with NDVI, SAVI, and GNDVI vegetation indices.

## Overview

The system has been updated to focus specifically on onion crop health assessment with:
- **NDVI** (Normalized Difference Vegetation Index)
- **SAVI** (Soil-Adjusted Vegetation Index) 
- **GNDVI** (Green Normalized Difference Vegetation Index)

## Health Categories

The system now uses onion-specific health categories:

1. **Very Healthy** - NDVI > 0.8, SAVI > 0.7, GNDVI > 0.75
   - Dark green, vigorous foliage
   - Optimal growing conditions

2. **Healthy** - NDVI 0.6-0.8, SAVI 0.5-0.7, GNDVI 0.6-0.75
   - Green, healthy-looking foliage
   - Good canopy coverage

3. **Moderate** - NDVI 0.4-0.6, SAVI 0.3-0.5, GNDVI 0.4-0.6
   - Light green to yellow-green foliage
   - Some visible stress

4. **Poor** - NDVI 0.2-0.4, SAVI 0.15-0.3, GNDVI 0.2-0.4
   - Yellowing or pale green foliage
   - Reduced canopy coverage

5. **Very Poor** - NDVI < 0.2, SAVI < 0.15, GNDVI < 0.2
   - Brown, yellow, or severely stressed foliage
   - Sparse canopy coverage

6. **Diseased** - Fungal/bacterial/viral diseases
7. **Stressed** - Water/nutrient/heat stress
8. **Weeds** - Significant weed infestation

## Updated Files

### Core Processing

1. **`image_processor.py`**
   - Added `calculate_gndvi()` function
   - Updated health classification to be onion-specific
   - Updated `analyze_crop_health()` to include all three indices
   - Updated `classify_crop_health_tensorflow()` to load and use trained models

2. **`background_worker.py`**
   - Updated to calculate and save GNDVI values
   - Enhanced logging for onion crop analysis

3. **`db_utils.py`**
   - Updated `save_analysis()` to save GNDVI values (with fallback if columns don't exist)

4. **`flask_api_db.py`**
   - Updated queries to include analysis type information

### New Scripts

1. **`batch_test_ndvi.py`**
   - Batch processing script for testing NDVI/SAVI/GNDVI on sample images
   - Generates comprehensive reports with statistics
   - Usage: `python batch_test_ndvi.py <image_folder> [output_dir]`

2. **`train_model.py`**
   - ML model training script for onion crop health classification
   - Supports 8 health categories
   - Usage: `python train_model.py <image_folder> [output_dir] [epochs]`

3. **`database_migration_add_gndvi.sql`**
   - Database migration to add GNDVI columns
   - Run with: `psql -U postgres -d drone_analytics -f database_migration_add_gndvi.sql`

## Database Migration

To add GNDVI support to your database, run:

```bash
psql -U postgres -d drone_analytics -f python_processing/database_migration_add_gndvi.sql
```

This adds:
- `gndvi`, `gndvi_mean`, `gndvi_std`, `gndvi_min`, `gndvi_max` columns to `analyses` table
- `gndvi_value` column to `stress_zones` table
- Indexes for efficient querying

## Usage

### Testing Image Processing Algorithm

```bash
cd python_processing
python batch_test_ndvi.py /path/to/onion/images
```

This will:
- Process all images in the folder
- Calculate NDVI, SAVI, and GNDVI for each
- Generate individual JSON files
- Create a CSV summary
- Generate a text report with statistics

### Training ML Model

Organize your images in subfolders:

```
sample_images/
  very_healthy/
    img1.jpg
    img2.jpg
  healthy/
    img1.jpg
  moderate/
    img1.jpg
  poor/
    img1.jpg
  very_poor/
    img1.jpg
  diseased/
    img1.jpg
  stressed/
    img1.jpg
  weeds/
    img1.jpg
```

Then train:

```bash
python train_model.py ./sample_images ./models 50
```

### Using Trained Model

The trained model will be saved to `./models/onion_crop_health_model.h5`. The system will automatically use it when:
- `use_tensorflow=True` is set in `analyze_crop_health()`
- Model path is set via `ONION_MODEL_PATH` environment variable or passed directly

## Expected Vegetation Index Ranges

For onion crops, typical ranges are:

| Health Status | NDVI Range | SAVI Range | GNDVI Range |
|--------------|------------|------------|-------------|
| Very Healthy  | > 0.8      | > 0.7      | > 0.75      |
| Healthy       | 0.6-0.8    | 0.5-0.7    | 0.6-0.75    |
| Moderate      | 0.4-0.6    | 0.3-0.5    | 0.4-0.6     |
| Poor          | 0.2-0.4    | 0.15-0.3   | 0.2-0.4     |
| Very Poor     | < 0.2      | < 0.15     | < 0.2       |

## Notes

- GNDVI is particularly useful for early growth stages of onion crops
- The system automatically detects if GNDVI columns exist in the database
- All analysis results include `crop_type: 'onion'` for identification
- The background worker automatically processes uploaded images with all three indices

