"""
Unified Label Mapping System
Maps dataset-specific labels to the unified health taxonomy.
"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Unified health taxonomy (MANDATORY)
UNIFIED_HEALTH_LABELS = [
    'very_healthy',
    'healthy',
    'moderate',
    'poor',
    'very_poor',
    'diseased',
    'stressed',
    'weeds',
    'unknown'
]

# Dataset-specific label mappings
LABEL_MAPPINGS = {
    # PlantVillage mappings
    'plantvillage': {
        # Tomato
        'Tomato___healthy': 'very_healthy',
        'Tomato___Early_blight': 'diseased',
        'Tomato___Late_blight': 'diseased',
        'Tomato___Septoria_leaf_spot': 'diseased',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'diseased',
        'Tomato___Bacterial_spot': 'diseased',
        'Tomato___Target_Spot': 'diseased',
        'Tomato___Tomato_mosaic_virus': 'diseased',
        'Tomato___Leaf_Mold': 'diseased',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'stressed',
        
        # Corn/Maize
        'Corn_(maize)___healthy': 'very_healthy',
        'Corn_(maize)___Common_rust': 'diseased',
        'Corn_(maize)___Northern_Leaf_Blight': 'diseased',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'diseased',
        'Corn_(maize)___Northern_Leaf_Spot': 'diseased',
        
        # Onion (if present)
        'Onion___healthy': 'very_healthy',
        'Onion___Downy_mildew': 'diseased',
    },
    
    # TOM2024 mappings
    'tom2024': {
        'healthy': 'very_healthy',
        'Healthy': 'very_healthy',
        'HEALTHY': 'very_healthy',
        'Fusarium': 'diseased',
        'fusarium': 'diseased',
        'Alternaria': 'diseased',
        'alternaria': 'diseased',
        'Caterpillar': 'stressed',
        'caterpillar': 'stressed',
        'Pest': 'stressed',
        'pest': 'stressed',
        'Infested': 'poor',
        'infested': 'poor',
        'Damage': 'poor',
        'damage': 'poor',
    },
    
    # WeedsGalore mappings
    'weedsgalore': {
        'maize': 'very_healthy',
        'Maize': 'very_healthy',
        'MAIZE': 'very_healthy',
        'corn': 'very_healthy',
        'Corn': 'very_healthy',
        'weed': 'weeds',
        'Weed': 'weeds',
        'WEED': 'weeds',
        'weeds': 'weeds',
        'Weeds': 'weeds',
        'stressed_maize': 'stressed',
        'Stressed_Maize': 'stressed',
        'diseased_maize': 'diseased',
        'Diseased_Maize': 'diseased',
    },
    
    # Cherry Tomato (Dryad) mappings
    'cherry_tomato_dryad': {
        'high_vigor': 'very_healthy',
        'High_Vigor': 'very_healthy',
        'normal_growth': 'healthy',
        'Normal_Growth': 'healthy',
        'moderate_stress': 'moderate',
        'Moderate_Stress': 'moderate',
        'high_stress': 'stressed',
        'High_Stress': 'stressed',
        'disease': 'diseased',
        'Disease': 'diseased',
        'diseased': 'diseased',
    },
    
    # Generic fallback mappings
    'generic': {
        'healthy': 'very_healthy',
        'Healthy': 'very_healthy',
        'HEALTHY': 'very_healthy',
        'normal': 'healthy',
        'Normal': 'healthy',
        'good': 'healthy',
        'Good': 'healthy',
        'moderate': 'moderate',
        'Moderate': 'moderate',
        'fair': 'moderate',
        'Fair': 'moderate',
        'poor': 'poor',
        'Poor': 'poor',
        'bad': 'poor',
        'Bad': 'poor',
        'very_poor': 'very_poor',
        'Very_Poor': 'very_poor',
        'critical': 'very_poor',
        'Critical': 'very_poor',
        'diseased': 'diseased',
        'Diseased': 'diseased',
        'disease': 'diseased',
        'Disease': 'diseased',
        'stressed': 'stressed',
        'Stressed': 'stressed',
        'stress': 'stressed',
        'Stress': 'stressed',
        'weeds': 'weeds',
        'Weeds': 'weeds',
        'weed': 'weeds',
        'Weed': 'weeds',
    }
}


def map_label(original_label: str, dataset_name: str, crop_type: Optional[str] = None) -> str:
    """
    Map a dataset-specific label to the unified health taxonomy.
    
    Args:
        original_label: Original label from the dataset
        dataset_name: Name of the dataset (e.g., 'plantvillage', 'tom2024')
        crop_type: Optional crop type for context (e.g., 'cherry_tomato', 'onion', 'corn')
    
    Returns:
        Mapped label from UNIFIED_HEALTH_LABELS
    """
    # Normalize input
    original_label = str(original_label).strip()
    dataset_name = dataset_name.lower()
    
    # Try dataset-specific mapping first
    if dataset_name in LABEL_MAPPINGS:
        mapping = LABEL_MAPPINGS[dataset_name]
        if original_label in mapping:
            return mapping[original_label]
        
        # Try case-insensitive match
        for key, value in mapping.items():
            if key.lower() == original_label.lower():
                return value
    
    # Try generic mapping
    if original_label in LABEL_MAPPINGS['generic']:
        return LABEL_MAPPINGS['generic'][original_label]
    
    # Case-insensitive generic match
    for key, value in LABEL_MAPPINGS['generic'].items():
        if key.lower() == original_label.lower():
            return value
    
    # If no mapping found, log warning and return 'unknown'
    logger.warning(
        f"No mapping found for label '{original_label}' from dataset '{dataset_name}'. "
        f"Using 'unknown'. Consider adding to label mappings."
    )
    return 'unknown'


def get_crop_type_from_label(label: str, dataset_name: str) -> Optional[str]:
    """
    Infer crop type from label or dataset name.
    
    Args:
        label: Label string (may contain crop information)
        dataset_name: Dataset name
    
    Returns:
        Crop type: 'cherry_tomato', 'onion', 'corn', or None
    """
    label_lower = label.lower()
    dataset_lower = dataset_name.lower()
    
    # Check label for crop indicators
    if 'tomato' in label_lower or 'tomato' in dataset_lower:
        return 'cherry_tomato'
    if 'onion' in label_lower or 'onion' in dataset_lower:
        return 'onion'
    if 'corn' in label_lower or 'maize' in label_lower or 'corn' in dataset_lower or 'maize' in dataset_lower:
        return 'corn'
    
    return None


def validate_mapping(mapped_label: str) -> bool:
    """
    Validate that a mapped label is in the unified taxonomy.
    
    Args:
        mapped_label: Label to validate
    
    Returns:
        True if valid, False otherwise
    """
    return mapped_label in UNIFIED_HEALTH_LABELS


def get_all_mappings() -> Dict[str, Dict[str, str]]:
    """
    Get all label mappings.
    
    Returns:
        Dictionary of all mappings
    """
    return LABEL_MAPPINGS.copy()


def add_custom_mapping(dataset_name: str, original_label: str, mapped_label: str):
    """
    Add a custom label mapping at runtime.
    
    Args:
        dataset_name: Dataset name
        original_label: Original label
        mapped_label: Mapped label (must be in UNIFIED_HEALTH_LABELS)
    """
    if mapped_label not in UNIFIED_HEALTH_LABELS:
        raise ValueError(f"Mapped label '{mapped_label}' not in unified taxonomy: {UNIFIED_HEALTH_LABELS}")
    
    dataset_name = dataset_name.lower()
    if dataset_name not in LABEL_MAPPINGS:
        LABEL_MAPPINGS[dataset_name] = {}
    
    LABEL_MAPPINGS[dataset_name][original_label] = mapped_label
    logger.info(f"Added custom mapping: {dataset_name}:{original_label} -> {mapped_label}")


if __name__ == '__main__':
    # Test mappings
    print("Testing label mappings...")
    
    test_cases = [
        ('plantvillage', 'Tomato___healthy', 'very_healthy'),
        ('plantvillage', 'Tomato___Early_blight', 'diseased'),
        ('tom2024', 'healthy', 'very_healthy'),
        ('tom2024', 'Fusarium', 'diseased'),
        ('weedsgalore', 'maize', 'very_healthy'),
        ('weedsgalore', 'weed', 'weeds'),
        ('cherry_tomato_dryad', 'high_vigor', 'very_healthy'),
        ('unknown_dataset', 'healthy', 'very_healthy'),
        ('unknown_dataset', 'unknown_label', 'unknown'),
    ]
    
    for dataset, original, expected in test_cases:
        result = map_label(original, dataset)
        status = "✓" if result == expected else "✗"
        print(f"{status} {dataset}: '{original}' -> '{result}' (expected: '{expected}')")
    
    print(f"\nUnified labels: {UNIFIED_HEALTH_LABELS}")


