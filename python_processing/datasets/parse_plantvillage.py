#!/usr/bin/env python3
"""
Parse PlantVillage Dataset
Parses PlantVillage dataset and maps labels to unified health taxonomy.

Usage:
    python parse_plantvillage.py --input_dir ./data/plantvillage --output_dir ./data/processed/plantvillage
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


def parse_plantvillage_dataset(input_dir: str, output_dir: str, crops: list = None):
    """
    Parse PlantVillage dataset and organize by unified health labels.
    
    Args:
        input_dir: Path to PlantVillage dataset root
        output_dir: Path to output processed dataset
        crops: List of crops to include (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output structure: output_dir/crop_type/health_label/
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all class directories (format: Crop___Class)
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        # Try alternative structure (color or segmented subdirectories)
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                class_dirs.extend([d for d in subdir.iterdir() if d.is_dir()])
    
    logger.info(f"Found {len(class_dirs)} class directories")
    
    stats = {
        'total_images': 0,
        'by_crop': {},
        'by_health': {label: 0 for label in UNIFIED_HEALTH_LABELS},
        'mapping_log': []
    }
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Extract crop type and original label
        if '___' in class_name:
            parts = class_name.split('___')
            crop_part = parts[0]
            label_part = '___'.join(parts[1:])
        else:
            crop_part = None
            label_part = class_name
        
        # Map to unified health label
        mapped_label = map_label(class_name, 'plantvillage')
        
        # Infer crop type
        crop_type = get_crop_type_from_label(class_name, 'plantvillage')
        if crop_type is None:
            # Try to infer from crop_part
            if crop_part:
                crop_lower = crop_part.lower()
                if 'tomato' in crop_lower:
                    crop_type = 'cherry_tomato'
                elif 'onion' in crop_lower:
                    crop_type = 'onion'
                elif 'corn' in crop_lower or 'maize' in crop_lower:
                    crop_type = 'corn'
                else:
                    crop_type = 'unknown'
            else:
                crop_type = 'unknown'
        
        # Filter by requested crops
        if crops and crop_type not in crops and crop_type != 'unknown':
            continue
        
        # Create output directory structure
        crop_output_dir = output_path / crop_type / mapped_label
        crop_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(class_dir.glob(ext))
        
        copied = 0
        for img_file in image_files:
            try:
                # Create unique filename to avoid conflicts
                new_filename = f"{crop_type}_{mapped_label}_{img_file.name}"
                dest_path = crop_output_dir / new_filename
                
                shutil.copy2(img_file, dest_path)
                copied += 1
                stats['total_images'] += 1
            except Exception as e:
                logger.warning(f"Failed to copy {img_file}: {e}")
        
        # Update stats
        if crop_type not in stats['by_crop']:
            stats['by_crop'][crop_type] = {}
        if mapped_label not in stats['by_crop'][crop_type]:
            stats['by_crop'][crop_type][mapped_label] = 0
        stats['by_crop'][crop_type][mapped_label] += copied
        stats['by_health'][mapped_label] += copied
        
        # Log mapping
        stats['mapping_log'].append({
            'original': class_name,
            'crop_type': crop_type,
            'mapped_label': mapped_label,
            'images': copied
        })
        
        logger.info(
            f"Processed {class_name}: {copied} images -> {crop_type}/{mapped_label}"
        )
    
    # Save mapping log
    mapping_log_path = output_path / 'mapping_log.json'
    with open(mapping_log_path, 'w') as f:
        json.dump(stats['mapping_log'], f, indent=2)
    
    # Save statistics
    stats_path = output_path / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'total_images': stats['total_images'],
            'by_crop': stats['by_crop'],
            'by_health': stats['by_health']
        }, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("PARSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"\nBy crop type:")
    for crop, labels in stats['by_crop'].items():
        total = sum(labels.values())
        logger.info(f"  {crop}: {total} images")
        for label, count in labels.items():
            logger.info(f"    - {label}: {count}")
    logger.info(f"\nBy health label:")
    for label, count in stats['by_health'].items():
        if count > 0:
            logger.info(f"  {label}: {count}")
    logger.info(f"\nMapping log saved to: {mapping_log_path}")
    logger.info(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse PlantVillage dataset')
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to PlantVillage dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--crops',
        type=str,
        nargs='+',
        choices=['cherry_tomato', 'onion', 'corn'],
        help='Crops to include (default: all)'
    )
    
    args = parser.parse_args()
    
    parse_plantvillage_dataset(
        args.input_dir,
        args.output_dir,
        crops=args.crops
    )


if __name__ == '__main__':
    main()


