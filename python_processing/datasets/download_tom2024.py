#!/usr/bin/env python3
"""
Download TOM2024 Dataset
Downloads the TOM2024 (Tomato, Onion, Maize) dataset from Mendeley.

Usage:
    python download_tom2024.py --output_dir ./data/tom2024
"""
import os
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_via_manual_instructions(output_dir: str):
    """
    Print manual download instructions for TOM2024.
    """
    logger.info("=" * 60)
    logger.info("TOM2024 DATASET DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("Dataset: TOM2024: Tomato, Onion, and Maize Images")
    logger.info("License: CC BY 4.0")
    logger.info("")
    logger.info("Download Steps:")
    logger.info("1. Visit: https://data.mendeley.com/datasets/5e25daa0230741c695c9b8f86104078a")
    logger.info("2. Click 'Download' (requires free Mendeley account)")
    logger.info("3. Extract the archive to:")
    logger.info(f"   {output_dir}")
    logger.info("")
    logger.info("Alternative source:")
    logger.info("https://pmc.ncbi.nlm.nih.gov/articles/PMC11871477/")
    logger.info("")
    logger.info("Dataset Structure:")
    logger.info("  - Contains onion, tomato, and maize images")
    logger.info("  - Labels include: healthy, Fusarium, Alternaria, pest-infested")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download TOM2024 dataset')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/tom2024',
        help='Output directory for dataset'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    download_via_manual_instructions(str(output_dir))
    logger.info(f"\nDataset should be located at: {output_dir}")
    logger.info("Next step: Run parse_tom2024.py to process the dataset")


if __name__ == '__main__':
    main()



