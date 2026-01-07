#!/usr/bin/env python3
"""
Model Evaluation Script with tf.data batching
Generates per-crop confusion matrices, macro F1, and separate UAV vs leaf evaluation.
Uses tf.data for scalable, memory-efficient evaluation.
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from datasets.label_mapping import UNIFIED_HEALTH_LABELS
from multispectral_loader import load_dataset_registry, create_band_mask_array, STANDARD_MULTISPECTRAL_BANDS
from train_multi_crop_model_v2 import compute_index_features, INDEX_FEATURE_DIM, CROP_TYPES
from image_processor import calculate_ndvi, calculate_savi, calculate_gndvi

CROP_TYPES = ['cherry_tomato', 'onion', 'corn', 'unknown']


def _load_and_preprocess_image(image_path: str, band_schema: Dict, target_size: Tuple[int, int] = (224, 224)):
    """
    Load and preprocess a single image from file path.
    Returns preprocessed inputs for model.
    """
    from multispectral_loader import load_multispectral_image
    
    # Load image
    img, detected_schema = load_multispectral_image(
        image_path,
        target_size=target_size,
        band_schema=band_schema
    )
    
    # RGB input (3 channels)
    if img.shape[2] >= 3:
        rgb_img = img[:, :, :3]
    else:
        rgb_img = np.pad(img, ((0, 0), (0, 0), (0, 3 - img.shape[2])), mode='constant')
    
    # Multispectral input (4 channels)
    if img.shape[2] >= 4:
        ms_img = img[:, :, :4]
    elif img.shape[2] == 3:
        ms_img = np.concatenate([img, np.zeros((img.shape[0], img.shape[1], 1))], axis=-1)
    else:
        ms_img = np.pad(img, ((0, 0), (0, 0), (0, 4 - img.shape[2])), mode='constant')
    
    # Band mask
    band_mask = create_band_mask_array(detected_schema, STANDARD_MULTISPECTRAL_BANDS)
    
    # Compute index features
    ndvi_stats = calculate_ndvi(image_path, band_schema=detected_schema, image_array=img)
    savi_stats = calculate_savi(image_path, band_schema=detected_schema, image_array=img)
    gndvi_stats = calculate_gndvi(image_path, band_schema=detected_schema, image_array=img)
    index_features = compute_index_features(ndvi_stats, savi_stats, gndvi_stats)
    
    return rgb_img, ms_img, band_mask, index_features


def create_evaluation_dataset(
    image_paths: List[str],
    health_labels: np.ndarray,
    crop_labels: np.ndarray,
    band_schemas: List[Dict],
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224)
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset from file paths for scalable evaluation.
    Loads images on-the-fly during batching.
    
    Args:
        image_paths: List of image file paths
        health_labels: Health class labels
        crop_labels: Crop type labels
        band_schemas: Band schema dicts (one per image)
        batch_size: Batch size for evaluation
        target_size: Target image size
    
    Returns:
        tf.data.Dataset with batched inputs
    """
    # Use from_tensor_slices + map for better throughput (more efficient than from_generator)
    # Create dataset from file paths and metadata
    dataset = tf.data.Dataset.from_tensor_slices({
        'image_path': [str(p) for p in image_paths],
        'band_schema_idx': list(range(len(band_schemas))),
        'health_label': health_labels.astype(np.int32),
        'crop_label': crop_labels.astype(np.int32)
    })
    
    # Load and preprocess images in parallel using py_function
    def load_image_wrapper(x):
        """Wrapper for loading image from path"""
        img_path = x['image_path'].numpy().decode('utf-8')
        schema_idx = x['band_schema_idx'].numpy()
        schema = band_schemas[schema_idx]
        
        try:
            rgb_img, ms_img, band_mask, index_features = _load_and_preprocess_image(
                img_path, schema, target_size
            )
            return {
                'rgb_input': rgb_img,
                'multispectral_input': ms_img,
                'band_mask': band_mask,
                'index_features': index_features,
                'health_label': x['health_label'],
                'crop_label': x['crop_label']
            }
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return zero-filled inputs on error
            return {
                'rgb_input': np.zeros((*target_size, 3), dtype=np.float32),
                'multispectral_input': np.zeros((*target_size, 4), dtype=np.float32),
                'band_mask': np.zeros(4, dtype=np.float32),
                'index_features': np.zeros(12, dtype=np.float32),
                'health_label': x['health_label'],
                'crop_label': x['crop_label']
            }
    
    # Map with py_function for parallel loading
    dataset = dataset.map(
        lambda x: tf.py_function(
            func=load_image_wrapper,
            inp=[x],
            Tout={
                'rgb_input': tf.float32,
                'multispectral_input': tf.float32,
                'band_mask': tf.float32,
                'index_features': tf.float32,
                'health_label': tf.int32,
                'crop_label': tf.int32
            }
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set output shapes explicitly
    dataset = dataset.map(
        lambda x: {
            'rgb_input': tf.ensure_shape(x['rgb_input'], (*target_size, 3)),
            'multispectral_input': tf.ensure_shape(x['multispectral_input'], (*target_size, 4)),
            'band_mask': tf.ensure_shape(x['band_mask'], (4,)),
            'index_features': tf.ensure_shape(x['index_features'], (12,)),
            'health_label': x['health_label'],
            'crop_label': x['crop_label']
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch for optimal throughput
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def evaluate_per_crop(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    health_class_names: List[str],
    crop_class_names: List[str]
) -> Dict:
    """
    Evaluate model per crop type with confusion matrices using tf.data.
    
    Returns:
        Dictionary with per-crop metrics and confusion matrices
    """
    results = {}
    
    # Collect all predictions and labels
    all_health_preds = []
    all_health_labels = []
    all_crop_preds = []
    all_crop_labels = []
    
    # Predict in batches
    logger.info("Running batched predictions...")
    for batch in dataset:
        inputs = {
            'rgb_input': batch['rgb_input'],
            'multispectral_input': batch['multispectral_input'],
            'band_mask': batch['band_mask'],
            'index_features': batch['index_features']
        }
        
        predictions = model.predict(inputs, verbose=0)
        
        if isinstance(predictions, list) and len(predictions) == 2:
            health_preds = predictions[0]
            crop_preds = predictions[1]
        else:
            health_preds = predictions
            crop_preds = np.zeros((predictions.shape[0], len(crop_class_names)))
        
        all_health_preds.extend(np.argmax(health_preds, axis=1))
        all_health_labels.extend(batch['health_label'].numpy())
        all_crop_preds.extend(np.argmax(crop_preds, axis=1))
        all_crop_labels.extend(batch['crop_label'].numpy())
    
    all_health_preds = np.array(all_health_preds)
    all_health_labels = np.array(all_health_labels)
    all_crop_preds = np.array(all_crop_preds)
    all_crop_labels = np.array(all_crop_labels)
    
    # Evaluate per crop
    for crop_idx, crop_name in enumerate(crop_class_names):
        crop_mask = all_crop_labels == crop_idx
        if not np.any(crop_mask):
            logger.warning(f"No samples for crop: {crop_name}")
            continue
        
        crop_health_labels = all_health_labels[crop_mask]
        crop_health_preds = all_health_preds[crop_mask]
        
        logger.info(f"Evaluating {crop_name}: {len(crop_health_labels)} samples")
        
        # Confusion matrix
        cm = confusion_matrix(
            crop_health_labels,
            crop_health_preds,
            labels=range(len(health_class_names))
        )
        
        # Metrics
        f1_macro = f1_score(
            crop_health_labels,
            crop_health_preds,
            average='macro',
            labels=range(len(health_class_names)),
            zero_division=0
        )
        f1_weighted = f1_score(
            crop_health_labels,
            crop_health_preds,
            average='weighted',
            labels=range(len(health_class_names)),
            zero_division=0
        )
        
        # Per-class metrics
        report = classification_report(
            crop_health_labels,
            crop_health_preds,
            labels=range(len(health_class_names)),
            target_names=health_class_names,
            output_dict=True,
            zero_division=0
        )
        
        results[crop_name] = {
            'confusion_matrix': cm.tolist(),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': report,
            'num_samples': len(crop_health_labels)
        }
    
    # Overall macro F1
    overall_f1_macro = f1_score(
        all_health_labels,
        all_health_preds,
        average='macro',
        labels=range(len(health_class_names)),
        zero_division=0
    )
    results['overall'] = {
        'f1_macro': float(overall_f1_macro),
        'num_samples': len(all_health_labels)
    }
    
    return results


def evaluate_by_domain(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    domains: List[str],
    health_class_names: List[str],
    batch_size: int = 32
) -> Dict:
    """
    Evaluate separately for UAV vs leaf datasets using tf.data.
    
    Returns:
        Dictionary with domain-specific metrics
    """
    results = {}
    
    # Collect predictions and labels with domain info
    all_health_preds = []
    all_health_labels = []
    all_domains = []
    
    # Predict in batches
    logger.info("Running batched predictions for domain evaluation...")
    batch_idx = 0
    for batch in dataset:
        inputs = {
            'rgb_input': batch['rgb_input'],
            'multispectral_input': batch['multispectral_input'],
            'band_mask': batch['band_mask'],
            'index_features': batch['index_features']
        }
        
        predictions = model.predict(inputs, verbose=0)
        
        if isinstance(predictions, list) and len(predictions) == 2:
            health_preds = predictions[0]
        else:
            health_preds = predictions
        
        batch_size = health_preds.shape[0]
        all_health_preds.extend(np.argmax(health_preds, axis=1))
        all_health_labels.extend(batch['health_label'].numpy())
        
        # Map batch indices to domain labels
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        all_domains.extend(domains[start_idx:end_idx])
        batch_idx += 1
    
    all_health_preds = np.array(all_health_preds)
    all_health_labels = np.array(all_health_labels)
    all_domains = np.array(all_domains)
    
    # Group by domain
    uav_indices = np.where(all_domains == 'uav')[0]
    leaf_indices = np.where(all_domains == 'leaf')[0]
    
    for domain_name, indices in [('uav', uav_indices), ('leaf', leaf_indices)]:
        if len(indices) == 0:
            logger.warning(f"No samples for domain: {domain_name}")
            continue
        
        domain_health_labels = all_health_labels[indices]
        domain_health_preds = all_health_preds[indices]
        
        logger.info(f"Evaluating {domain_name} domain: {len(domain_health_labels)} samples")
        
        # Metrics
        f1_macro = f1_score(
            domain_health_labels,
            domain_health_preds,
            average='macro',
            labels=range(len(health_class_names)),
            zero_division=0
        )
        f1_weighted = f1_score(
            domain_health_labels,
            domain_health_preds,
            average='weighted',
            labels=range(len(health_class_names)),
            zero_division=0
        )
        accuracy = np.mean(domain_health_labels == domain_health_preds)
        
        # Confusion matrix
        cm = confusion_matrix(
            domain_health_labels,
            domain_health_preds,
            labels=range(len(health_class_names))
        )
        
        results[domain_name] = {
            'confusion_matrix': cm.tolist(),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'accuracy': float(accuracy),
            'num_samples': len(domain_health_labels)
        }
    
    # Compute domain shift metric
    if 'uav' in results and 'leaf' in results:
        f1_diff = results['uav']['f1_macro'] - results['leaf']['f1_macro']
        results['domain_shift'] = {
            'f1_macro_difference': float(f1_diff),
            'relative_difference': float(f1_diff / max(results['uav']['f1_macro'], 0.001))
        }
    
    return results


def load_test_data(data_folder: str, dataset_name: Optional[str] = None):
    """
    Load test data file paths with domain information.
    Returns file paths instead of pre-loaded images for scalability.
    
    Returns:
        image_paths, health_labels, crop_labels, band_schemas, domains, health_class_names, crop_class_names
    """
    from pathlib import Path
    
    folder = Path(data_folder)
    image_paths = []
    health_labels = []
    crop_labels = []
    band_schemas = []
    
    from datasets.label_mapping import UNIFIED_HEALTH_LABELS
    from train_multi_crop_model_v2 import CROP_TYPES
    
    health_class_to_idx = {name: idx for idx, name in enumerate(UNIFIED_HEALTH_LABELS)}
    crop_class_to_idx = {name: idx for idx, name in enumerate(CROP_TYPES)}
    
    # Iterate through crop types
    for crop_dir in folder.iterdir():
        if not crop_dir.is_dir():
            continue
        
        crop_type = crop_dir.name
        if crop_type not in crop_class_to_idx:
            logger.warning(f"Unknown crop type: {crop_type}, skipping")
            continue
        
        crop_idx = crop_class_to_idx[crop_type]
        
        # Iterate through health labels
        for health_dir in crop_dir.iterdir():
            if not health_dir.is_dir():
                continue
            
            health_label = health_dir.name
            if health_label not in health_class_to_idx:
                logger.warning(f"Unknown health label: {health_label}, skipping")
                continue
            
            health_idx = health_class_to_idx[health_label]
            
            # Get image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                image_files.extend(health_dir.glob(ext))
            
            logger.info(f"Found {len(image_files)} images from {crop_type}/{health_label}...")
            
            for img_path in image_files:
                # Detect band schema
                schema = detect_band_schema(str(img_path), dataset_name)
                image_paths.append(str(img_path))
                health_labels.append(health_idx)
                crop_labels.append(crop_idx)
                band_schemas.append(schema)
    
    # Determine domain for each image
    registry = load_dataset_registry()
    domains = []
    for schema in band_schemas:
        domain = schema.get('domain', 'unknown')
        domains.append(domain)
    
    logger.info(f"Loaded {len(image_paths)} image paths")
    
    return image_paths, np.array(health_labels), np.array(crop_labels), band_schemas, domains, UNIFIED_HEALTH_LABELS, CROP_TYPES


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate multi-crop model with tf.data batching')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data folder')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--dataset-name', type=str, help='Dataset name for band schema lookup')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    # Load test data (file paths, not images)
    logger.info(f"Loading test data paths: {args.test_data}")
    image_paths, health_labels, crop_labels, band_schemas, domains, health_class_names, crop_class_names = load_test_data(
        args.test_data,
        dataset_name=args.dataset_name
    )
    
    # Create evaluation dataset from file paths
    logger.info("Creating evaluation dataset from file paths...")
    eval_dataset = create_evaluation_dataset(
        image_paths, health_labels, crop_labels, band_schemas, 
        batch_size=args.batch_size, target_size=(224, 224)
    )
    
    # Evaluate per crop
    logger.info("Evaluating per crop...")
    per_crop_results = evaluate_per_crop(
        model, eval_dataset, health_class_names, crop_class_names
    )
    
    # Evaluate by domain
    logger.info("Evaluating by domain...")
    domain_results = evaluate_by_domain(
        model, eval_dataset, domains, health_class_names, batch_size=args.batch_size
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'per_crop': per_crop_results,
        'by_domain': domain_results,
        'health_classes': health_class_names,
        'crop_types': crop_class_names
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\nPer-Crop F1 Macro Scores:")
    for crop_name, crop_results in per_crop_results.items():
        if crop_name != 'overall':
            print(f"  {crop_name}: {crop_results['f1_macro']:.4f} ({crop_results['num_samples']} samples)")
    
    if 'overall' in per_crop_results:
        print(f"\nOverall F1 Macro: {per_crop_results['overall']['f1_macro']:.4f}")
    
    print("\nDomain-Specific F1 Macro Scores:")
    for domain_name, domain_results in domain_results.items():
        if domain_name != 'domain_shift':
            print(f"  {domain_name}: {domain_results['f1_macro']:.4f} ({domain_results['num_samples']} samples)")
    
    if 'domain_shift' in domain_results:
        print(f"\nDomain Shift (UAV - Leaf): {domain_results['domain_shift']['f1_macro_difference']:.4f}")


if __name__ == '__main__':
    main()
