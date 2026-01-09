#!/usr/bin/env python3
"""
Download PlantVillage Dataset
Downloads the PlantVillage dataset from Kaggle or GitHub.

Usage:
    python download_plantvillage.py --output_dir ./data/plantvillage
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_via_kaggle(output_dir: str, kaggle_dataset: str = "abdallahalidev/plantvillage-dataset"):
    """
    Download PlantVillage dataset via Kaggle API.
    
    Requires: pip install kaggle
    Setup: Place kaggle.json in ~/.kaggle/ (from Kaggle account settings)
    """
    try:
        import kaggle
        logger.info(f"Downloading PlantVillage dataset from Kaggle to {output_dir}...")
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=output_dir,
            unzip=True
        )
        logger.info("✓ Download complete")
        return True
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {e}")
        return False


def download_via_git(output_dir: str, repo_url: str = "https://github.com/spMohanty/PlantVillage-Dataset.git"):
    """
    Download PlantVillage dataset via Git clone.
    """
    try:
        logger.info(f"Cloning PlantVillage dataset from GitHub to {output_dir}...")
        subprocess.run(
            ['git', 'clone', repo_url, output_dir],
            check=True,
            capture_output=True
        )
        logger.info("✓ Clone complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {e}")
        logger.error(f"Git output: {e.stderr.decode() if e.stderr else 'N/A'}")
        return False
    except FileNotFoundError:
        logger.error("Git not found. Please install Git or use Kaggle download method.")
        return False


def download_via_manual_instructions(output_dir: str):
    """
    Print manual download instructions.
    """
    logger.info("=" * 60)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    logger.info("2. Click 'Download' (requires Kaggle account)")
    logger.info("3. Extract the archive to:")
    logger.info(f"   {output_dir}")
    logger.info("")
    logger.info("OR")
    logger.info("")
    logger.info("1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
    logger.info("2. Download as ZIP or clone the repository")
    logger.info("3. Extract to:")
    logger.info(f"   {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download PlantVillage dataset')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/plantvillage',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['kaggle', 'git', 'manual'],
        default='manual',
        help='Download method'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading PlantVillage dataset to: {output_dir}")
    
    if args.method == 'kaggle':
        success = download_via_kaggle(str(output_dir))
        if not success:
            logger.warning("Kaggle download failed. Trying manual instructions...")
            download_via_manual_instructions(str(output_dir))
    elif args.method == 'git':
        success = download_via_git(str(output_dir))
        if not success:
            logger.warning("Git clone failed. Trying manual instructions...")
            download_via_manual_instructions(str(output_dir))
    else:
        download_via_manual_instructions(str(output_dir))
    
    logger.info(f"\nDataset should be located at: {output_dir}")
    logger.info("Next step: Run parse_plantvillage.py to process the dataset")


if __name__ == '__main__':
    main()



