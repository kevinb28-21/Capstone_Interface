#!/usr/bin/env python3
"""
Parse TOM2024 Dataset
Parses TOM2024 dataset and maps labels to unified health taxonomy.

Usage:
    python parse_tom2024.py --input_dir ./data/tom2024 --output_dir ./data/processed/tom2024 --crop onion
"""
import os
import argparse
import shutil
from pathlib import Path
import logging
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.label_mapping import map_label, get_crop_type_from_label, UNIFIED_HEALTH_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_tom2024_dataset(input_dir: str, output_dir: str, crop: str = 'onion'):
    """
    Parse TOM2024 dataset for specified crop.
    
    Args:
        input_dir: Path to TOM2024 dataset root
        output_dir: Path to output processed dataset
        crop: Crop to extract ('onion', 'tomato', 'maize')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TOM2024 structure may vary, try common patterns
    crop_dirs = []
    
    # Pattern 1: Direct crop folders
    crop_folder = input_path / crop
    if crop_folder.exists():
        crop_dirs.append(crop_folder)
    
    # Pattern 2: Case variations
    for subdir in input_path.iterdir():
        if subdir.is_dir() and crop.lower() in subdir.name.lower():
            crop_dirs.append(subdir)
    
    if not crop_dirs:
        logger.warning(f"No {crop} directory found. Searching all subdirectories...")
        crop_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    stats = {
        'total_images': 0,
        'by_health': {label: 0 for label in UNIFIED_HEALTH_LABELS},
        'mapping_log': []
    }
    
    for crop_dir in crop_dirs:
        logger.info(f"Processing directory: {crop_dir.name}")
        
        # Look for label subdirectories or files with labels in names
        label_dirs = [d for d in crop_dir.iterdir() if d.is_dir()]
        
        if not label_dirs:
            # Flat structure - try to infer labels from filenames
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(crop_dir.glob(ext))
            
            for img_file in image_files:
                # Try to extract label from filename
                filename_lower = img_file.stem.lower()
                mapped_label = 'unknown'
                
                # Simple keyword matching
                if 'healthy' in filename_lower:
                    mapped_label = 'very_healthy'
                elif 'fusarium' in filename_lower:
                    mapped_label = 'diseased'
                elif 'alternaria' in filename_lower:
                    mapped_label = 'diseased'
                elif 'pest' in filename_lower or 'caterpillar' in filename_lower:
                    mapped_label = 'stressed'
                elif 'infest' in filename_lower or 'damage' in filename_lower:
                    mapped_label = 'poor'
                else:
                    mapped_label = map_label(img_file.stem, 'tom2024')
                
                # Create output directory
                health_output_dir = output_path / crop / mapped_label
                health_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy image
                new_filename = f"{crop}_{mapped_label}_{img_file.name}"
                dest_path = health_output_dir / new_filename
                shutil.copy2(img_file, dest_path)
                
                stats['total_images'] += 1
                stats['by_health'][mapped_label] += 1
        else:
            # Subdirectory structure
            for label_dir in label_dirs:
                label_name = label_dir.name
                mapped_label = map_label(label_name, 'tom2024')
                
                # Create output directory
                health_output_dir = output_path / crop / mapped_label
                health_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(label_dir.glob(ext))
                
                copied = 0
                for img_file in image_files:
                    new_filename = f"{crop}_{mapped_label}_{img_file.name}"
                    dest_path = health_output_dir / new_filename
                    shutil.copy2(img_file, dest_path)
                    copied += 1
                    stats['total_images'] += 1
                    stats['by_health'][mapped_label] += 1
                
                stats['mapping_log'].append({
                    'original': label_name,
                    'crop': crop,
                    'mapped_label': mapped_label,
                    'images': copied
                })
                
                logger.info(f"  {label_name} -> {mapped_label}: {copied} images")
    
    # Save statistics
    stats_path = output_path / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("PARSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"\nBy health label:")
    for label, count in stats['by_health'].items():
        if count > 0:
            logger.info(f"  {label}: {count}")
    logger.info(f"\nStatistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse TOM2024 dataset')
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to TOM2024 dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--crop',
        type=str,
        choices=['onion', 'tomato', 'maize'],
        default='onion',
        help='Crop to extract'
    )
    
    args = parser.parse_args()
    
    parse_tom2024_dataset(
        args.input_dir,
        args.output_dir,
        crop=args.crop
    )


if __name__ == '__main__':
    main()


