"""
Preprocess images for YOLO training by converting 4-channel TIFF to 3-channel RGB.

This script handles geospatial TIFF files that may have 4 channels (RGBA or extra bands)
and converts them to standard 3-channel RGB format that YOLOv8 expects.

Usage:
    python preprocess_images.py
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_dataset_images(dataset_dir="../../dataset/yolo_buildings"):
    """
    Convert all TIFF images in the dataset to 3-channel RGB format.
    
    Args:
        dataset_dir: Path to the dataset directory containing images/
    """
    dataset_dir = Path(dataset_dir).resolve()
    images_dir = dataset_dir / "images"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return False
    
    logger.info(f"Processing dataset: {dataset_dir}")
    
    try:
        import rasterio
        import numpy as np
        from PIL import Image as PILImage
    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        logger.error("Install with: pip install rasterio pillow numpy")
        return False
    
    total_converted = 0
    
    # Process each split (train, val, test)
    for split in ["train", "val", "test"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            logger.info(f"Split directory not found: {split_dir}")
            continue
        
        logger.info(f"\nProcessing {split} split...")
        tiff_files = sorted(list(split_dir.glob("*.tif")) + list(split_dir.glob("*.tiff")))
        
        if not tiff_files:
            logger.info(f"  No TIFF files found in {split}")
            continue
        
        logger.info(f"  Found {len(tiff_files)} TIFF files")
        
        for img_path in tiff_files:
            try:
                with rasterio.open(img_path) as src:
                    num_channels = src.count
                    
                    if num_channels <= 3:
                        logger.info(f"  ✓ {img_path.name}: {num_channels} channels (no conversion needed)")
                        continue
                    
                    logger.info(f"  Converting {img_path.name}: {num_channels} channels → 3 channels (RGB)")
                    
                    # Read first 3 bands
                    data = src.read([1, 2, 3])
                    
                    # Normalize to 0-255 range if needed
                    if data.dtype != np.uint8:
                        data_min = data.min()
                        data_max = data.max()
                        if data_max > data_min:
                            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                        else:
                            data = data.astype(np.uint8)
                    
                    # Convert from (3, H, W) to (H, W, 3)
                    rgb_array = np.transpose(data, (1, 2, 0))
                    rgb_img = PILImage.fromarray(rgb_array, mode='RGB')
                    
                    # Save as PNG (preserves quality and compatibility)
                    new_path = img_path.with_suffix('.png')
                    rgb_img.save(new_path)
                    logger.info(f"    Saved as: {new_path.name}")
                    
                    # Remove original TIFF with retry logic (handles file locking)
                    import time
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            img_path.unlink()
                            logger.info(f"    Removed: {img_path.name}")
                            total_converted += 1
                            break
                        except PermissionError:
                            if attempt < max_retries - 1:
                                logger.warning(f"    File locked, retrying... ({attempt + 1}/{max_retries})")
                                time.sleep(0.5)
                            else:
                                logger.warning(f"    Could not remove {img_path.name} (file is locked). Please remove manually.")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to convert {img_path.name}: {str(e)}")
                continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete! Converted {total_converted} images.")
    logger.info(f"{'='*60}")
    
    return True


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/yolo_buildings"
    success = preprocess_dataset_images(dataset_path)
    sys.exit(0 if success else 1)
