Dataset Download and Processing Scripts

This directory contains scripts for downloading and parsing agricultural crop health datasets for multi-crop model training.

Overview

The scripts in this directory:
  Download datasets from various sources
  Parse and organize datasets into a unified structure
  Map dataset-specific labels to the unified health taxonomy

Unified Health Taxonomy

All datasets are mapped to:
{ veryhealthy, healthy, moderate, poor, verypoor, diseased, stressed, weeds, unknown }

See labelmapping.py for mapping details.

Dataset Structure

After processing, datasets are organized as:
processeddataset/
  croptype/
    healthlabel/
      image1.jpg
      image2.jpg

Where:
  - croptype: cherrytomato, onion, corn, unknown
  - healthlabel: One of the unified health taxonomy labels

Available Scripts

Download Scripts
  - downloadplantvillage.py - Downloads PlantVillage dataset
  - downloadtom2024.py - Instructions for TOM2024 dataset
  - downloadweedsgalore.py - (To be implemented) WeedsGalore dataset
  - downloadcherrytomato.py - (To be implemented) Cherry tomato multispectral dataset

Parse Scripts
  - parseplantvillage.py - Parses PlantVillage dataset
  - parsetom2024.py - Parses TOM2024 dataset
  - parseweedsgalore.py - (To be implemented) Parses WeedsGalore dataset

Utilities
  - labelmapping.py - Unified label mapping system
  - init.py - Package initialization

Usage
  Download Dataset

PlantVillage (manual download recommended)
python downloadplantvillage.py --output-dir ./data/plantvillage --method manual

TOM2024 (manual download required)
python downloadtom2024.py --output-dir ./data/tom2024
  Parse Dataset

Parse PlantVillage
python parseplantvillage.py \
    --input-dir ./data/plantvillage \
    --output-dir ./data/processed/plantvillage \
    --crops cherrytomato onion corn

Parse TOM2024 (onion)
python parsetom2024.py \
    --input-dir ./data/tom2024 \
    --output-dir ./data/processed/tom2024 \
    --crop onion
  Combine for Training

After parsing, combine all processed datasets:

mkdir -p ./data/training
cp -r ./data/processed/plantvillage/ ./data/training/
cp -r ./data/processed/tom2024/ ./data/training/
Add other datasets as needed

Adding New Datasets

To add support for a new dataset:
  Create download script: download<datasetname>.py
  - Provide download instructions or automated download
  - Follow the pattern of existing download scripts
  Create parse script: parse<datasetname>.py
  - Parse dataset structure
  - Map labels using labelmapping.py
  - Organize into croptype/healthlabel/ structure
  Update label mappings: Add dataset-specific mappings to labelmapping.py
  Update documentation: Add dataset to documentation/datasets.md

Label Mapping

The labelmapping.py module provides:
  - maplabel(originallabel, datasetname) - Map dataset label to unified taxonomy
  - getcroptypefromlabel(label, datasetname) - Infer crop type from label
  - validatemapping(mappedlabel) - Validate mapped label
  - addcustommapping(datasetname, originallabel, mappedlabel) - Add custom mappings

Example:
from datasets.labelmapping import maplabel

Map PlantVillage label
mapped = maplabel('Tomato__Earlyblight', 'plantvillage')
Returns: 'diseased'

Map TOM2024 label
mapped = maplabel('Fusarium', 'tom2024')
Returns: 'diseased'

Dataset Requirements

For a dataset to be usable:
  Must have images (JPG, PNG, JPEG)
  Must have labels (in filenames, subdirectories, or separate file)
  Must be mappable to unified health taxonomy
  Should include crop type information (or be inferable)

Output Statistics

Each parse script generates:
  - statistics.json - Dataset statistics (counts by crop and health label)
  - mappinglog.json - Label mapping log (original -> mapped)

Notes
  - Some datasets require manual download (e.g., TOM2024 from Mendeley)
  - Large datasets may take significant time to download and process
  - Ensure sufficient disk space (some datasets are 100+ GB)
  - Processed datasets are typically much smaller than raw datasets

See Also
  - documentation/datasets.md - Complete dataset documentation
  - documentation/MLTRAININGGUIDE.md - Training guide
  - ../trainmulticrop_model.py - Training script