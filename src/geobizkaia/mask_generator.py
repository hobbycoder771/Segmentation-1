"""Mask Generator Module - Converts polygons to binary segmentation masks."""

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from typing import Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MaskGenerator:
    """Generates binary segmentation masks from polygon geometries."""
    
    def __init__(self, mask_value: int = 255):
        self.mask_value = mask_value
    
    def polygon_to_mask(
        self,
        geometries: List[Polygon],
        image_width: int,
        image_height: int,
        image_bounds: Tuple[float, float, float, float],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Convert polygon geometries to binary mask."""
        
        if not geometries:
            logger.warning("No geometries provided, creating empty mask")
            return np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Create transform from image bounds to pixel coordinates
        xmin, ymin, xmax, ymax = image_bounds
        transform = rasterio.transform.from_bounds(
            xmin, ymin, xmax, ymax, image_width, image_height
        )
        
        # Filter valid geometries
        valid_geoms = [geom for geom in geometries if geom.is_valid and not geom.is_empty]
        
        if not valid_geoms:
            logger.warning("No valid geometries found")
            return np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Create shapes for rasterization
        shapes = [(geom, self.mask_value) for geom in valid_geoms]
        
        try:
            mask = rasterize(
                shapes,
                out_shape=(image_height, image_width),
                transform=transform,
                default_value=0,
                dtype=np.uint8,
            )
            
            if output_path:
                self._save_mask(mask, output_path)
            
            return mask
        
        except Exception as e:
            logger.error(f"Failed to rasterize geometries: {e}")
            return np.zeros((image_height, image_width), dtype=np.uint8)
    
    def _save_mask(self, mask: np.ndarray, output_path: str) -> None:
        """Save mask to PNG file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            img = Image.fromarray(mask, mode='L')
            img.save(output_path)
            logger.info(f"Saved mask to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save mask: {e}")
    
    def get_mask_stats(self, mask: np.ndarray) -> dict:
        """Get statistics about a mask."""
        if mask.size == 0:
            return {
                "total_pixels": 0,
                "building_pixels": 0,
                "background_pixels": 0,
                "building_percentage": 0.0,
            }
        
        total = mask.size
        building = np.sum(mask == self.mask_value)
        background = np.sum(mask == 0)
        
        return {
            "total_pixels": int(total),
            "building_pixels": int(building),
            "background_pixels": int(background),
            "building_percentage": float(100.0 * building / total),
        }
