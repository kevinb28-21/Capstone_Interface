# Dataset Download and Processing Scripts

This directory contains scripts for downloading and parsing agricultural crop health datasets for multi-crop model training.

## Overview

The scripts in this directory:
1. Download datasets from various sources
2. Parse and organize datasets into a unified structure
3. Map dataset-specific labels to the unified health taxonomy

## Unified Health Taxonomy

All datasets are mapped to:
```
{ very_healthy, healthy, moderate, poor, very_poor, diseased, stressed, weeds, unknown }
```

See `label_mapping.py` for mapping details.

## Dataset Structure

After processing, datasets are organized as:
```
processed_dataset/
  crop_type/
    health_label/
      image1.jpg
      image2.jpg
```

Where:
- `crop_type`: cherry_tomato, onion, corn, unknown
- `health_label`: One of the unified health taxonomy labels

## Available Scripts

### Download Scripts

- `download_plantvillage.py` - Downloads PlantVillage dataset
- `download_tom2024.py` - Instructions for TOM2024 dataset
- `download_weedsgalore.py` - (To be implemented) WeedsGalore dataset
- `download_cherry_tomato.py` - (To be implemented) Cherry tomato multispectral dataset

### Parse Scripts

- `parse_plantvillage.py` - Parses PlantVillage dataset
- `parse_tom2024.py` - Parses TOM2024 dataset
- `parse_weedsgalore.py` - (To be implemented) Parses WeedsGalore dataset

### Utilities

- `label_mapping.py` - Unified label mapping system
- `__init__.py` - Package initialization

## Usage

### 1. Download Dataset

```bash
# PlantVillage (manual download recommended)
python download_plantvillage.py --output-dir ./data/plantvillage --method manual

# TOM2024 (manual download required)
python download_tom2024.py --output-dir ./data/tom2024
```

### 2. Parse Dataset

```bash
# Parse PlantVillage
python parse_plantvillage.py \
    --input-dir ./data/plantvillage \
    --output-dir ./data/processed/plantvillage \
    --crops cherry_tomato onion corn

# Parse TOM2024 (onion)
python parse_tom2024.py \
    --input-dir ./data/tom2024 \
    --output-dir ./data/processed/tom2024 \
    --crop onion
```

### 3. Combine for Training

After parsing, combine all processed datasets:

```bash
mkdir -p ./data/training
cp -r ./data/processed/plantvillage/* ./data/training/
cp -r ./data/processed/tom2024/* ./data/training/
# Add other datasets as needed
```

## Adding New Datasets

To add support for a new dataset:

1. **Create download script**: `download_<dataset_name>.py`
   - Provide download instructions or automated download
   - Follow the pattern of existing download scripts

2. **Create parse script**: `parse_<dataset_name>.py`
   - Parse dataset structure
   - Map labels using `label_mapping.py`
   - Organize into `crop_type/health_label/` structure

3. **Update label mappings**: Add dataset-specific mappings to `label_mapping.py`

4. **Update documentation**: Add dataset to `docs/datasets.md`

## Label Mapping

The `label_mapping.py` module provides:
- `map_label(original_label, dataset_name)` - Map dataset label to unified taxonomy
- `get_crop_type_from_label(label, dataset_name)` - Infer crop type from label
- `validate_mapping(mapped_label)` - Validate mapped label
- `add_custom_mapping(dataset_name, original_label, mapped_label)` - Add custom mappings

Example:
```python
from datasets.label_mapping import map_label

# Map PlantVillage label
mapped = map_label('Tomato___Early_blight', 'plantvillage')
# Returns: 'diseased'

# Map TOM2024 label
mapped = map_label('Fusarium', 'tom2024')
# Returns: 'diseased'
```

## Dataset Requirements

For a dataset to be usable:
1. Must have images (JPG, PNG, JPEG)
2. Must have labels (in filenames, subdirectories, or separate file)
3. Must be mappable to unified health taxonomy
4. Should include crop type information (or be inferable)

## Output Statistics

Each parse script generates:
- `statistics.json` - Dataset statistics (counts by crop and health label)
- `mapping_log.json` - Label mapping log (original -> mapped)

## Notes

- Some datasets require manual download (e.g., TOM2024 from Mendeley)
- Large datasets may take significant time to download and process
- Ensure sufficient disk space (some datasets are 100+ GB)
- Processed datasets are typically much smaller than raw datasets

## See Also

- `docs/datasets.md` - Complete dataset documentation
- `docs/ML_TRAINING_GUIDE.md` - Training guide
- `../train_multi_crop_model.py` - Training script



