# ML Pipeline Documentation

## Overview

This document describes the band-aware, multispectral ML pipeline for multi-crop plant health classification. The system supports RGB and multispectral (RGB+NIR) inputs with **no NIR approximation** - images are processed through separate RGB and multispectral paths based on available bands.

## Key Features

### 1. Band-Aware Processing

- **No NIR Approximation**: The system does NOT approximate NIR from RGB channels
- **Separate Paths**: RGB images use RGB path, multispectral images use multispectral path
- **Band Mask**: Network receives a band mask indicating which bands are present
- **Graceful Degradation**: If multispectral model is requested but NIR is missing, falls back to RGB path with warning

### 2. Multispectral Loading

- **GeoTIFF Support**: Uses `rasterio` (preferred) or `tifffile` for multi-band TIFF files
- **Deterministic Band Order**: Enforced via `dataset_registry.yaml`
- **Band Schema Storage**: Per-image band schema stored in processing metadata and database

### 3. Index/ML Fusion

- **True Band Indices**: NDVI/SAVI/GNDVI computed from actual NIR band (when available)
- **Feature Fusion**: Index statistics (mean/std/min/max for each index) concatenated into dense fusion layer
- **Fusion Health Score**: Combined ML prediction + index-based score persisted to database

### 4. Database Schema

New fields in `analyses` table:
- `model_version`: ML model version identifier
- `inference_time_ms`: Model inference time
- `band_schema`: JSONB with band information
- `health_topk`: JSONB array of top-k health predictions
- `crop_topk`: JSONB array of top-k crop predictions
- `fusion_health_score`: Combined health score from ML + indices

## Architecture

### Model Architecture

The band-aware model has:
1. **RGB Input Path**: 3-channel input (R, G, B)
2. **Multispectral Input Path**: 4-channel input (R, G, B, NIR)
3. **Band Mask Input**: Indicates which bands are present
4. **Index Features Input**: Concatenated index statistics
5. **Fusion Layer**: Combines image features with index features
6. **Dual Output**: Health classification + crop type classification

### Processing Flow

```
Image Input
    ↓
Detect Band Schema (from dataset_registry.yaml or auto-detect)
    ↓
┌─────────────────┬──────────────────┐
│  RGB Path       │  Multispectral   │
│  (3 channels)   │  Path (4 ch)     │
└─────────────────┴──────────────────┘
    ↓                    ↓
Compute Indices (if NIR available)
    ↓
Create Band Mask
    ↓
Model Inference (band-aware)
    ↓
Fusion (ML + Index features)
    ↓
Predictions (health + crop)
    ↓
Store Results (with band_schema, topk, fusion_score)
```

## Dataset Registry

The `dataset_registry.yaml` file enforces deterministic band order:

```yaml
datasets:
  plantvillage:
    band_order: ["R", "G", "B"]
    band_count: 3
    domain: "leaf"
    domain_mismatch_warning: true
  
  weedsgalore:
    band_order: ["R", "G", "B", "RE", "NIR"]
    band_count: 5
    domain: "uav"
    domain_mismatch_warning: false
```

## Index Calculation

### NDVI (Normalized Difference Vegetation Index)
- **Formula**: (NIR - Red) / (NIR + Red)
- **Requires**: NIR band
- **Returns**: None if NIR not available (no approximation)

### SAVI (Soil-Adjusted Vegetation Index)
- **Formula**: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
- **Requires**: NIR band
- **Returns**: None if NIR not available

### GNDVI (Green NDVI)
- **Formula**: (NIR - Green) / (NIR + Green)
- **Requires**: NIR band
- **Returns**: None if NIR not available

## Background Worker

### Model Loading

- **Load Once**: Model loaded at worker startup, cached globally
- **Path Logging**: Logs which processing path was used (RGB vs multispectral)
- **Band Logging**: Logs which bands were detected and used
- **Missing Band Handling**: If multispectral model requested but NIR missing, uses RGB path with warning

### Logging Example

```
Multi-crop TensorFlow classification (path: multispectral, bands: ['R', 'G', 'B', 'NIR']): 
crop=onion (confidence: 95.2%), 
health=healthy (confidence: 87.3%)
```

## Database Migration

Apply the new migration to add ML fields:

```bash
psql -U postgres -d drone_analytics -f server/database/migration_add_ml_fields.sql
```

This adds:
- `model_version`
- `inference_time_ms`
- `band_schema` (JSONB)
- `health_topk` (JSONB)
- `crop_topk` (JSONB)
- `fusion_health_score`

## Evaluation Improvements

### Per-Crop Confusion Matrices

The evaluation system generates:
- Confusion matrix for each crop type (cherry_tomato, onion, corn)
- Macro F1 score across all crops
- Per-class precision, recall, F1

### Domain Shift Evaluation

- **Separate Splits**: UAV datasets evaluated separately from leaf datasets
- **Domain Mismatch Warning**: PlantVillage (leaf images) flagged for potential domain shift
- **Quantified Shift**: Metrics show performance difference between domains

## Dataset Metadata Corrections

1. **WeedsGalore License**: Updated to CC BY (dataset), Apache-2.0 (code)
2. **TOM2024 DOI**: Corrected to 10.17632/3d4yg89rtr.1
3. **TOM2024 URL**: Updated to match DOI
4. **PlantVillage Warning**: Added domain mismatch warning (leaf vs UAV)

## Usage

### Training

```bash
python python_processing/train_multi_crop_model_v2.py \
    --data-folder ./data/training \
    --output-dir ./models/multi_crop \
    --config training_config.yaml
```

### Inference

The background worker automatically:
1. Loads model once at startup
2. Detects band schema for each image
3. Routes to appropriate path (RGB or multispectral)
4. Computes indices from true bands
5. Performs fusion
6. Stores results with full metadata

### Environment Variables

```bash
USE_MULTI_CROP_MODEL=true
MULTI_CROP_MODEL_DIR=./models/multi_crop
```

## Files Modified

- `python_processing/image_processor.py`: Removed NIR approximation, added band-aware processing
- `python_processing/multispectral_loader.py`: New multispectral loading with rasterio/tifffile
- `python_processing/train_multi_crop_model_v2.py`: Band-aware model architecture
- `python_processing/background_worker.py`: Model loading once, path/band logging
- `python_processing/db_utils.py`: Save new ML fields
- `python_processing/datasets/dataset_registry.yaml`: Band order enforcement
- `server/database/migration_add_ml_fields.sql`: New database fields
- `docs/datasets.md`: Corrected metadata, added warnings

## Dependencies

New dependencies added to `requirements.txt`:
- `rasterio>=1.3.0`: GeoTIFF/multi-band TIFF loading
- `tifffile>=2023.1.0`: TIFF file support (fallback)
- `pyyaml>=6.0`: YAML configuration parsing

## Notes

- **No NIR Approximation**: This is a hard requirement - RGB images cannot compute true NDVI/SAVI/GNDVI
- **Band Schema Persistence**: Every processed image has its band schema stored for traceability
- **Fusion Score**: Combines ML confidence with index-based signals for more robust predictions
- **Top-K Predictions**: Stored for analysis and debugging


