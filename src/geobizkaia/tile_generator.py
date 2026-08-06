"""Tile generator for large images with overlapping support."""

import logging
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TileGenerator:
    """Generate tiles from large images and masks with overlap support."""
    
    def __init__(self, tile_size=512, overlap=0.2):
        """Initialize tile generator."""
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        logger.info(f"Tile: size={tile_size}, overlap={overlap*100:.0f}%, stride={self.stride}")
    
    def generate_tile_coords(self, image_height, image_width):
        """Generate tile coordinates."""
        coords = []
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                y_start = y
                x_start = x
                y_end = min(y + self.tile_size, image_height)
                x_end = min(x + self.tile_size, image_width)
                
                if x_end - x_start < self.tile_size and x_end == image_width:
                    x_start = max(0, x_end - self.tile_size)
                if y_end - y_start < self.tile_size and y_end == image_height:
                    y_start = max(0, y_end - self.tile_size)
                
                coords.append((y_start, x_start, y_end, x_end))
                
                x += self.stride
                if x >= image_width:
                    break
            
            y += self.stride
            if y >= image_height:
                break
        
        return coords
    
    def tile_image(self, image_path, output_dir, prefix):
        """Tile a GeoTIFF image."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tiles_info = []
        
        with rasterio.open(image_path) as src:
            image = src.read()
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            
            logger.info(f"Tiling: {Path(image_path).name} ({height}x{width})")
            
            coords = self.generate_tile_coords(height, width)
            
            for tile_idx, (y_start, x_start, y_end, x_end) in enumerate(coords):
                tile_data = image[:, y_start:y_end, x_start:x_end]
                tile_height = y_end - y_start
                tile_width = x_end - x_start
                
                if tile_height < self.tile_size or tile_width < self.tile_size:
                    padded = np.zeros((image.shape[0], self.tile_size, self.tile_size), dtype=image.dtype)
                    padded[:, :tile_height, :tile_width] = tile_data
                    tile_data = padded
                
                tile_name = f"{prefix}_{Path(image_path).stem}_tile_{tile_idx:04d}.tif"
                tile_path = output_dir / tile_name
                
                tile_transform = rasterio.transform.Affine(
                    transform.a, transform.b, transform.c + x_start * transform.a,
                    transform.d, transform.e, transform.f + y_start * transform.e
                )
                
                with rasterio.open(
                    tile_path, 'w',
                    driver='GTiff',
                    height=self.tile_size,
                    width=self.tile_size,
                    count=image.shape[0],
                    dtype=image.dtype,
                    crs=crs,
                    transform=tile_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(tile_data)
                
                tiles_info.append(str(tile_path))
            
            logger.info(f"Generated {len(coords)} tiles")
        
        return tiles_info
    
    def tile_mask(self, mask_path, output_dir, prefix):
        """Tile a mask image (PNG)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        height, width = mask_array.shape[:2]
        
        logger.info(f"Tiling mask: {Path(mask_path).name} ({height}x{width})")
        
        tiles_paths = []
        coords = self.generate_tile_coords(height, width)
        
        for tile_idx, (y_start, x_start, y_end, x_end) in enumerate(coords):
            tile_data = mask_array[y_start:y_end, x_start:x_end]
            tile_height = y_end - y_start
            tile_width = x_end - x_start
            
            if tile_height < self.tile_size or tile_width < self.tile_size:
                padded = np.zeros((self.tile_size, self.tile_size), dtype=mask_array.dtype)
                padded[:tile_height, :tile_width] = tile_data
                tile_data = padded
            
            tile_name = f"{prefix}_{Path(mask_path).stem}_tile_{tile_idx:04d}.png"
            tile_path = output_dir / tile_name
            
            tile_img = Image.fromarray(tile_data.astype(np.uint8), mode='L')
            tile_img.save(tile_path)
            
            tiles_paths.append(str(tile_path))
        
        logger.info(f"Generated {len(coords)} mask tiles")
        
        return tiles_paths
    
    def tile_dataset(self, images_dir, masks_dir, output_images_dir, output_masks_dir, prefix="tile"):
        """Tile all images and masks in a dataset directory."""
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        output_images_dir = Path(output_images_dir)
        output_masks_dir = Path(output_masks_dir)
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_masks_dir.mkdir(parents=True, exist_ok=True)
        
        total_tiles = 0
        tile_count = 0
        
        image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
        
        for image_path in image_files:
            mask_name = image_path.stem + ".png"
            mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                logger.warning(f"Mask not found: {image_path.name}")
                continue
            
            self.tile_image(image_path, output_images_dir, prefix)
            self.tile_mask(mask_path, output_masks_dir, prefix)
            
            tile_count += 1
            total_tiles += len(self.generate_tile_coords(*Image.open(mask_path).size[::-1]))
        
        # Clean up original files (keep only tiles)
        for image_path in image_files:
            image_path.unlink()  # Delete original image
            mask_name = image_path.stem + ".png"
            mask_path = masks_dir / mask_name
            if mask_path.exists():
                mask_path.unlink()  # Delete original mask
        
        logger.info(f"Tiled {tile_count} pairs into {total_tiles} tiles (originals removed)")
        return tile_count, total_tiles
