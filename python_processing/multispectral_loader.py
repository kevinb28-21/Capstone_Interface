"""
Multispectral Image Loader
Supports multi-band TIFF loading with deterministic band order.
Standardized to 4-channel multispectral: [R, G, B, NIR]
Uses tifffile (preferred for cost) or rasterio (optional) as fallback.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import yaml

logger = logging.getLogger(__name__)

# Try to import tifffile first (preferred - lightweight, no GDAL dependency)
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    logger.warning("tifffile not available, will use rasterio or OpenCV fallback")

# Try to import rasterio (optional - heavier dependency)
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.debug("rasterio not available (optional)")

# Fallback to OpenCV
import cv2

# Standard multispectral schema: 4 channels [R, G, B, NIR]
STANDARD_MULTISPECTRAL_BANDS = ["R", "G", "B", "NIR"]
STANDARD_RGB_BANDS = ["R", "G", "B"]


def load_dataset_registry(registry_path: Optional[str] = None) -> Dict:
    """Load dataset registry YAML file."""
    if registry_path is None:
        registry_path = Path(__file__).parent / "datasets" / "dataset_registry.yaml"
    
    registry_path = Path(registry_path)
    if not registry_path.exists():
        logger.warning(f"Dataset registry not found at {registry_path}, using defaults")
        return {
            'datasets': {
                'default': {
                    'band_order': STANDARD_RGB_BANDS,
                    'band_count': 3,
                    'domain': 'unknown',
                    'domain_mismatch_warning': False
                }
            },
            'canonical_band_order': STANDARD_MULTISPECTRAL_BANDS
        }
    
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def validate_canonical_band_order() -> bool:
    """
    Validate that dataset_registry.yaml canonical_band_order matches Python constant.
    Called at worker startup - fails fast if mismatch detected.
    
    Returns:
        True if valid, raises RuntimeError if mismatch
    """
    registry = load_dataset_registry()
    yaml_canonical = registry.get('canonical_band_order', [])
    
    if yaml_canonical != STANDARD_MULTISPECTRAL_BANDS:
        error_msg = (
            f"CRITICAL: Canonical band order mismatch!\n"
            f"  Python constant: {STANDARD_MULTISPECTRAL_BANDS}\n"
            f"  YAML registry:   {yaml_canonical}\n"
            f"  Worker cannot start - fix dataset_registry.yaml or Python constant."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"✓ Canonical band order validated: {STANDARD_MULTISPECTRAL_BANDS}")
    return True


def detect_band_schema(image_path: str, dataset_name: Optional[str] = None) -> Dict:
    """
    Detect band schema for an image.
    
    Returns:
        Dictionary with band_schema information including source_band_indices
    """
    registry = load_dataset_registry()
    image_path = Path(image_path)
    
    # Try to get dataset-specific schema
    if dataset_name and dataset_name in registry.get('datasets', {}):
        dataset_config = registry['datasets'][dataset_name]
        schema = {
            'bands': dataset_config['band_order'],
            'band_count': dataset_config['band_count'],
            'band_order': dataset_config['band_order'],
            'domain': dataset_config.get('domain', 'unknown'),
            'domain_mismatch_warning': dataset_config.get('domain_mismatch_warning', False),
            'source_band_order': dataset_config.get('source_band_order', dataset_config['band_order'])
        }
        return schema
    
    # Detect from file
    band_count = _detect_band_count(image_path)
    
    # Standardize to RGB (3) or RGB+NIR (4)
    if band_count == 3:
        return {
            'bands': STANDARD_RGB_BANDS,
            'band_count': 3,
            'band_order': STANDARD_RGB_BANDS,
            'domain': 'unknown',
            'domain_mismatch_warning': False,
            'source_band_order': STANDARD_RGB_BANDS
        }
    elif band_count >= 4:
        # Standardize to 4 channels: [R, G, B, NIR]
        # If more bands, drop RE and keep first 4
        return {
            'bands': STANDARD_MULTISPECTRAL_BANDS,
            'band_count': 4,
            'band_order': STANDARD_MULTISPECTRAL_BANDS,
            'domain': 'uav',
            'domain_mismatch_warning': False,
            'source_band_order': [f'Band_{i}' for i in range(band_count)]  # Unknown source order
        }
    else:
        return {
            'bands': STANDARD_RGB_BANDS,
            'band_count': 3,
            'band_order': STANDARD_RGB_BANDS,
            'domain': 'unknown',
            'domain_mismatch_warning': False,
            'source_band_order': STANDARD_RGB_BANDS
        }


def _detect_band_count(image_path: Path) -> int:
    """Detect number of bands in an image file."""
    suffix = image_path.suffix.lower()
    
    # Try tifffile first (preferred - lightweight)
    if HAS_TIFFFILE and suffix in ['.tif', '.tiff']:
        try:
            img = tifffile.imread(str(image_path))
            if len(img.shape) == 2:
                return 1
            elif len(img.shape) == 3:
                # Check if channels-first or channels-last
                if img.shape[0] < img.shape[2]:
                    # Likely channels-first
                    return img.shape[0]
                else:
                    # Likely channels-last
                    return img.shape[2] if img.shape[2] <= 10 else 3
            else:
                return 3
        except Exception as e:
            logger.debug(f"tifffile failed: {e}")
    
    # Try rasterio (optional - for GeoTIFF)
    if HAS_RASTERIO and suffix in ['.tif', '.tiff', '.gtif']:
        try:
            with rasterio.open(str(image_path)) as src:
                return src.count
        except Exception as e:
            logger.debug(f"rasterio failed: {e}")
    
    # Fallback to OpenCV
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return 3
        if len(img.shape) == 2:
            return 1
        elif len(img.shape) == 3:
            return img.shape[2]
        else:
            return 3
    except Exception as e:
        logger.warning(f"Failed to detect band count: {e}")
        return 3


def _reorder_bands_to_standard(
    img: np.ndarray,
    source_band_order: List[str],
    target_band_order: List[str]
) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Reorder bands from source order to standard order [R, G, B, NIR].
    Zero-fills missing bands and tracks which bands are missing.
    
    Returns:
        Reordered image array, source_band_indices mapping, and missing_bands list
    """
    source_band_indices = {}
    target_indices = []
    missing_bands = []
    
    # Create mapping from source to target
    for target_band in target_band_order:
        if target_band in source_band_order:
            source_idx = source_band_order.index(target_band)
            target_indices.append(source_idx)
            source_band_indices[target_band] = source_idx
        else:
            # Band not in source - will zero-fill
            target_indices.append(None)
            source_band_indices[target_band] = None
            missing_bands.append(target_band)
    
    # Reorder bands (zero-fill missing bands, do NOT approximate)
    reordered_bands = []
    for i, target_band in enumerate(target_band_order):
        if target_indices[i] is not None:
            reordered_bands.append(img[:, :, target_indices[i]])
        else:
            # Missing band - ALWAYS zero-fill (no approximation)
            reordered_bands.append(np.zeros_like(img[:, :, 0]))
            logger.debug(f"Band {target_band} not in source, zero-filling")
    
    reordered_img = np.stack(reordered_bands, axis=-1)
    
    return reordered_img, source_band_indices, missing_bands


def validate_band_schema(band_schema: Dict, required_bands: List[str] = None) -> Tuple[bool, str]:
    """
    Validate band schema is mappable to standard schema.
    Checks if bands can be mapped to STANDARD_MULTISPECTRAL_BANDS.
    
    Returns:
        (is_valid, error_message)
    """
    if required_bands is None:
        required_bands = STANDARD_MULTISPECTRAL_BANDS
    
    available_bands = band_schema.get('band_order', [])
    
    # Check if schema is mappable (at least some bands overlap)
    if not available_bands:
        return False, "Empty band_order in schema"
    
    # Check for invalid band names
    valid_bands = set(STANDARD_MULTISPECTRAL_BANDS + STANDARD_RGB_BANDS)
    invalid_bands = [b for b in available_bands if b not in valid_bands]
    if invalid_bands:
        return False, f"Invalid band names: {invalid_bands}. Valid: {valid_bands}"
    
    # Schema is mappable if it has at least R or G (for RGB) or NIR (for multispectral)
    has_rgb = any(b in available_bands for b in STANDARD_RGB_BANDS)
    has_nir = 'NIR' in available_bands
    
    if not has_rgb and not has_nir:
        return False, f"Schema has no mappable bands. Available: {available_bands}"
    
    return True, ""


def load_multispectral_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    dataset_name: Optional[str] = None,
    band_order: Optional[List[str]] = None,
    require_nir: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Load multispectral image with deterministic band order.
    Standardized to 4-channel multispectral: [R, G, B, NIR]
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        dataset_name: Optional dataset name for registry lookup
        band_order: Optional explicit band order (overrides registry)
        require_nir: If True, switch to RGB path if NIR missing
    
    Returns:
        Tuple of (image_array, band_schema_dict)
        - image_array: Shape (H, W, C) normalized to [0, 1], C=3 or 4
        - band_schema_dict: Band schema with source_band_indices
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    
    # Get band schema from registry or detect
    if band_order:
        band_schema = {
            'bands': band_order,
            'band_count': len(band_order),
            'band_order': band_order,
            'domain': 'unknown',
            'domain_mismatch_warning': False,
            'source_band_order': band_order
        }
    else:
        band_schema = detect_band_schema(str(image_path), dataset_name)
    
    source_band_order = band_schema.get('source_band_order', band_schema['band_order'])
    target_band_order = band_schema['band_order']
    
    # Validate schema is mappable
    is_valid, error_msg = validate_band_schema(band_schema)
    if not is_valid:
        # Log structured warning for traceability
        image_id_str = getattr(image_path, 'name', str(image_path))
        dataset_str = f"dataset={dataset_name}" if dataset_name else "dataset=unknown"
        logger.warning(
            f"Band schema validation failed: {error_msg}. "
            f"Image: {image_id_str}, {dataset_str}. "
            f"Forcing RGB fallback path."
        )
        target_band_order = STANDARD_RGB_BANDS
        band_schema['band_order'] = STANDARD_RGB_BANDS
        band_schema['band_count'] = 3
        band_schema['missing_bands'] = ['NIR']
        band_schema['fallback_reason'] = error_msg
        band_schema['processing_path'] = 'rgb_fallback'
    
    # Determine if multispectral (has NIR) or RGB only
    has_nir = 'NIR' in band_schema.get('band_order', [])
    
    # Standardize to 4 channels if multispectral (has NIR)
    if has_nir:
        # Always output 4 channels [R, G, B, NIR] for multispectral
        # Missing bands will be zero-filled
        target_band_order = STANDARD_MULTISPECTRAL_BANDS
        band_schema['band_order'] = STANDARD_MULTISPECTRAL_BANDS
        band_schema['band_count'] = 4
        
        # Track missing bands
        available_bands = source_band_order
        missing_bands = [b for b in STANDARD_MULTISPECTRAL_BANDS if b not in available_bands]
        band_schema['missing_bands'] = missing_bands
        
        # Track dropped bands (bands in source but not in standard)
        dropped_bands = [b for b in available_bands if b not in STANDARD_MULTISPECTRAL_BANDS]
        if dropped_bands:
            band_schema['dropped_bands'] = dropped_bands
        
        # Set processing path
        band_schema['processing_path'] = 'multispectral'
        
        if missing_bands:
            logger.info(f"Multispectral image missing bands: {missing_bands}, will zero-fill")
    elif require_nir:
        logger.warning(f"NIR required but not available. Switching to RGB path.")
        target_band_order = STANDARD_RGB_BANDS
        band_schema['band_order'] = STANDARD_RGB_BANDS
        band_schema['band_count'] = 3
        band_schema['missing_bands'] = ['NIR']
        band_schema['processing_path'] = 'rgb_fallback'
        band_schema['fallback_reason'] = 'NIR required but not available'
    else:
        # RGB only
        target_band_order = STANDARD_RGB_BANDS
        band_schema['band_order'] = STANDARD_RGB_BANDS
        band_schema['band_count'] = 3
        band_schema['missing_bands'] = ['NIR']
        band_schema['processing_path'] = 'rgb'
    
    img = None
    source_band_indices = {}
    
    # Try tifffile first (preferred - lightweight)
    if HAS_TIFFFILE and suffix in ['.tif', '.tiff']:
        try:
            img = tifffile.imread(str(image_path))
            
            # Handle different shapes
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            elif len(img.shape) == 3:
                # Check if channels-first or channels-last
                if img.shape[0] < img.shape[2] and img.shape[0] <= 10:
                    # Likely channels-first, transpose
                    img = np.transpose(img, (1, 2, 0))
            
            # Limit to required bands
            if img.shape[2] > band_schema['band_count']:
                img = img[:, :, :band_schema['band_count']]
            
            # Reorder to standard if needed
            if source_band_order != target_band_order:
                img, source_band_indices, missing_bands = _reorder_bands_to_standard(
                    img, source_band_order, target_band_order
                )
                band_schema['missing_bands'] = missing_bands
            
            # Resize
            if img.shape[:2] != target_size:
                img_resized = []
                for i in range(img.shape[2]):
                    band = cv2.resize(img[:, :, i], target_size, interpolation=cv2.INTER_LINEAR)
                    img_resized.append(band)
                img = np.stack(img_resized, axis=-1)
            
            # Normalize
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
                img_max = np.max(img)
                if img_max > 0:
                    img = img / img_max
            
        except Exception as e:
            logger.warning(f"tifffile failed for {image_path}: {e}, trying fallback")
            img = None
    
    # Try rasterio (optional fallback)
    if img is None and HAS_RASTERIO and suffix in ['.tif', '.tiff', '.gtif']:
        try:
            with rasterio.open(str(image_path)) as src:
                img_bands = []
                num_bands = min(src.count, band_schema['band_count'])
                resize_shape = target_size if target_size is not None else (img.shape[0], img.shape[1])
                for i in range(1, num_bands + 1):
                    band = src.read(i, out_shape=resize_shape, 
                                   resampling=Resampling.bilinear)
                    img_bands.append(band)
                
                img = np.stack(img_bands, axis=-1)
                
            # Reorder if needed
            if source_band_order != target_band_order:
                img, source_band_indices, missing_bands = _reorder_bands_to_standard(
                    img, source_band_order, target_band_order
                )
                band_schema['missing_bands'] = missing_bands
                
                # Normalize
                if img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                elif img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.float32)
                    img_max = np.max(img)
                    if img_max > 0:
                        img = img / img_max
        except Exception as e:
            logger.warning(f"rasterio failed for {image_path}: {e}, trying OpenCV fallback")
            img = None
    
    # Fallback to OpenCV (RGB only)
    if img is None:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Handle grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
            # Resize (only if target_size is specified)
            if target_size is not None and img.shape[:2] != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Limit to RGB if more channels
            if img.shape[2] > 3:
                img = img[:, :, :3]
                target_band_order = STANDARD_RGB_BANDS
                band_schema['band_order'] = STANDARD_RGB_BANDS
                band_schema['band_count'] = 3
                logger.warning(f"Image has {img.shape[2]} channels, limiting to RGB")
            
            source_band_indices = {band: i for i, band in enumerate(STANDARD_RGB_BANDS)}
            band_schema['missing_bands'] = ['NIR']  # RGB only, NIR missing
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    # Ensure correct number of channels (standardize to 4 for multispectral, 3 for RGB)
    if has_nir and img.shape[2] < 4:
        # Multispectral but missing bands - zero-fill to 4 channels
        padding = np.zeros((img.shape[0], img.shape[1], 4 - img.shape[2]), dtype=img.dtype)
        img = np.concatenate([img, padding], axis=-1)
        logger.debug(f"Multispectral image padded to 4 channels [R,G,B,NIR]")
    elif not has_nir and img.shape[2] < 3:
        # RGB but missing bands - zero-fill to 3 channels
        padding = np.zeros((img.shape[0], img.shape[1], 3 - img.shape[2]), dtype=img.dtype)
        img = np.concatenate([img, padding], axis=-1)
        logger.debug(f"RGB image padded to 3 channels [R,G,B]")
    elif img.shape[2] > band_schema['band_count']:
        # More bands than expected - limit to expected count
        img = img[:, :, :band_schema['band_count']]
        logger.debug(f"Image limited to {band_schema['band_count']} channels")
    
    # Update band_schema with full provenance
    band_schema['source_band_indices'] = source_band_indices
    band_schema['band_order'] = target_band_order
    band_schema['band_count'] = img.shape[2]
    
    # Ensure all provenance fields are set
    if 'bands' not in band_schema:
        band_schema['bands'] = target_band_order
    if 'source_band_order' not in band_schema:
        band_schema['source_band_order'] = source_band_order
    if 'missing_bands' not in band_schema:
        band_schema['missing_bands'] = []
    if 'processing_path' not in band_schema:
        band_schema['processing_path'] = 'rgb' if not has_nir else 'multispectral'
    
    # Final validation
    is_valid, error_msg = validate_band_schema(band_schema, target_band_order)
    if not is_valid:
        logger.warning(f"Band schema validation failed: {error_msg}")
    
    return img, band_schema


def create_band_mask(band_schema: Dict, required_bands: List[str] = None) -> Dict[str, float]:
    """
    Create a band mask using band-name keyed mapping (not positional).
    Explicitly tracks missing bands.
    
    Args:
        band_schema: Band schema dictionary (may include 'missing_bands')
        required_bands: List of required band names (default: STANDARD_MULTISPECTRAL_BANDS)
    
    Returns:
        Dictionary mapping band names to presence (1.0) or absence (0.0)
    """
    if required_bands is None:
        required_bands = STANDARD_MULTISPECTRAL_BANDS
    
    available_bands = band_schema.get('band_order', [])
    missing_bands = band_schema.get('missing_bands', [])
    
    # Create mask: 1.0 if band is available, 0.0 if missing
    mask = {}
    for band in required_bands:
        if band in available_bands:
            mask[band] = 1.0
        elif band in missing_bands:
            mask[band] = 0.0
        else:
            # Not explicitly tracked - check if it's in available bands
            mask[band] = 1.0 if band in available_bands else 0.0
    
    return mask


def create_band_mask_array(band_schema: Dict, required_bands: List[str] = None) -> np.ndarray:
    """
    Create a band mask array in the specified order.
    Defaults to STANDARD_MULTISPECTRAL_BANDS order: [R, G, B, NIR].
    
    Args:
        band_schema: Band schema dictionary
        required_bands: List of required band names (default: STANDARD_MULTISPECTRAL_BANDS)
    
    Returns:
        Array of shape (len(required_bands),) with 1.0 for present, 0.0 for absent
        Returns mask in the order specified by required_bands
    """
    if required_bands is None:
        required_bands = STANDARD_MULTISPECTRAL_BANDS
    
    # Warn if different from standard (but don't crash)
    if required_bands != STANDARD_MULTISPECTRAL_BANDS:
        logger.warning(
            f"create_band_mask_array called with non-standard band order: {required_bands}. "
            f"Expected: {STANDARD_MULTISPECTRAL_BANDS}. Returning mask in requested order."
        )
    
    mask_dict = create_band_mask(band_schema, required_bands)
    # Return mask in the requested order
    mask_array = np.array([mask_dict[band] for band in required_bands], dtype=np.float32)
    
    return mask_array
