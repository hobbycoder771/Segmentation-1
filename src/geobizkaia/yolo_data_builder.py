"""
YOLO Training Data Builder for Building Detection

This module provides functionality to:
1. Extract building geometries from a geopackage
2. Download imagery for each extent from ArcGIS MapServer
3. Convert vector data to YOLO annotations
4. Organize the dataset in YOLO format (train/val/test splits)
"""

import os
import geopandas as gpd
import pandas as pd
import requests
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLODataBuilder:
    """Build YOLO training dataset for building detection from geospatial data."""
    
    def __init__(self, 
                 imagery_url,
                 extents_gpkg_path,
                 extents_layer_name,
                 buildings_gpkg_path,
                 buildings_layer_name,
                 output_dir="../../dataset/yolo_buildings",
                 image_size=4096,
                 crs="EPSG:3857"):
        """
        Initialize YOLO Data Builder.
        
        Parameters
        ----------
        imagery_url : str
            ArcGIS MapServer export URL (without parameters)
        extents_gpkg_path : str
            Path to geopackage containing extent geometries
        extents_layer_name : str
            Layer name for extents in geopackage
        buildings_gpkg_path : str
            Path to geopackage containing building geometries
        buildings_layer_name : str
            Layer name for buildings in geopackage
        output_dir : str
            Output directory for YOLO dataset
        image_size : int
            Size of downloaded images (square)
        crs : str
            Coordinate reference system (EPSG code)
        """
        self.imagery_url = imagery_url
        self.extents_gpkg = extents_gpkg_path
        self.extents_layer = extents_layer_name
        self.buildings_gpkg = buildings_gpkg_path
        self.buildings_layer = buildings_layer_name
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.crs = crs
        
        # Setup output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Initialize data storage
        self.extents_gdf = None
        self.buildings_gdf = None
        
    def setup_directories(self):
        """Create necessary output directories."""
        logger.info("Setting up output directories...")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load extents and buildings from geopackages."""
        import fiona
        
        logger.info("Loading extents from geopackage...")
        self.extents_gdf = gpd.read_file(self.extents_gpkg, layer=self.extents_layer)
        logger.info(f"Loaded {len(self.extents_gdf)} extents")
        
        logger.info("Loading buildings from geopackage...")
        
        # Check if buildings_layer is a single layer or pattern
        try:
            # Try to load as single layer first
            self.buildings_gdf = gpd.read_file(self.buildings_gpkg, layer=self.buildings_layer)
            logger.info(f"Loaded {len(self.buildings_gdf)} buildings from layer '{self.buildings_layer}'")
        except Exception as e:
            # If single layer fails, try to load all layers matching pattern
            logger.info(f"Single layer '{self.buildings_layer}' not found, attempting to load multiple layers...")
            
            try:
                # Get all available layers
                available_layers = fiona.listlayers(self.buildings_gpkg)
                logger.info(f"Available layers in geopackage: {available_layers}")
                
                # Load all layers that contain the pattern (or all if pattern is generic)
                building_layers = [layer for layer in available_layers if 'building' in layer.lower()]
                
                if not building_layers:
                    raise ValueError(f"No layers containing 'building' found in {self.buildings_gpkg}")
                
                logger.info(f"Found {len(building_layers)} building layers: {building_layers}")
                
                # Load and concatenate all building layers
                gdfs = []
                for layer in building_layers:
                    logger.info(f"  Loading layer: {layer}")
                    gdf = gpd.read_file(self.buildings_gpkg, layer=layer)
                    gdfs.append(gdf)
                
                self.buildings_gdf = gpd.GeoDataFrame(
                    pd.concat(gdfs, ignore_index=True),
                    crs=gdfs[0].crs
                )
                logger.info(f"Loaded {len(self.buildings_gdf)} buildings from {len(building_layers)} layers")
                
            except Exception as e2:
                logger.error(f"Failed to load buildings: {e2}")
                raise
        
        # Ensure same CRS
        if self.buildings_gdf.crs != self.extents_gdf.crs:
            logger.info(f"Reprojecting buildings to {self.extents_gdf.crs}")
            self.buildings_gdf = self.buildings_gdf.to_crs(self.extents_gdf.crs)
    
    def download_imagery(self, extent_bounds, extent_id):
        """
        Download imagery for a given extent from ArcGIS MapServer.
        
        Parameters
        ----------
        extent_bounds : tuple
            (minx, miny, maxx, maxy) bounds in map coordinates
        extent_id : int
            Unique identifier for the extent
        
        Returns
        -------
        str or None
            Path to downloaded georeferenced GeoTIFF, or None if failed
        """
        try:
            minx, miny, maxx, maxy = extent_bounds
            
            # ArcGIS MapServer parameters
            params = {
                "bbox": f"{minx},{miny},{maxx},{maxy}",
                "bboxSR": "3857",
                "imageSR": "3857",
                "size": f"{self.image_size},{self.image_size}",
                "format": "png32",
                "transparent": "false",
                "f": "json",
            }
            
            logger.info(f"Downloading imagery for extent {extent_id}...")
            response = requests.get(self.imagery_url, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                logger.error(f"API Error: {result['error']}")
                return None
            
            image_url = result.get("href")
            if not image_url:
                logger.error(f"No image URL in response for extent {extent_id}")
                return None
            
            # Download the actual image
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            
            # Save as PNG first
            png_path = self.images_dir / f"extent_{extent_id:04d}.png"
            with open(png_path, "wb") as f:
                f.write(image_response.content)
            
            # Convert PNG to GeoTIFF
            geotiff_path = self.images_dir / f"extent_{extent_id:04d}.tif"
            self._png_to_geotiff(str(png_path), extent_bounds, str(geotiff_path))
            
            # Remove PNG
            png_path.unlink()
            
            logger.info(f"Saved georeferenced image: {geotiff_path}")
            return str(geotiff_path)
            
        except Exception as e:
            logger.error(f"Failed to download imagery for extent {extent_id}: {e}")
            return None
    
    def _png_to_geotiff(self, png_file, extent_bounds, output_tif):
        """Convert PNG to georeferenced GeoTIFF."""
        with rasterio.open(png_file) as src:
            image = src.read()
            width = src.width
            height = src.height
            profile = src.profile
        
        minx, miny, maxx, maxy = extent_bounds
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        profile.update(
            driver="GTiff",
            crs="EPSG:3857",
            transform=transform,
            compress="lzw",
        )
        
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(image)
    
    def get_buildings_in_extent(self, extent_geometry):
        """
        Get all buildings that intersect with the extent.
        
        Parameters
        ----------
        extent_geometry : shapely.Geometry
            The extent geometry
        
        Returns
        -------
        GeoDataFrame
            Buildings intersecting the extent
        """
        return self.buildings_gdf[self.buildings_gdf.geometry.intersects(extent_geometry)]
    
    def buildings_to_yolo_annotations(self, geotiff_path, extent_geometry, extent_id):
        """
        Convert building polygons to YOLO format annotations.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]
        
        Parameters
        ----------
        geotiff_path : str
            Path to the GeoTIFF image
        extent_geometry : shapely.Geometry
            The extent geometry
        extent_id : int
            Unique identifier for the extent
        
        Returns
        -------
        str or None
            Path to annotation file, or None if failed
        """
        try:
            # Get buildings in this extent
            buildings = self.get_buildings_in_extent(extent_geometry)
            
            if len(buildings) == 0:
                logger.warning(f"No buildings found in extent {extent_id}")
                return None
            
            # Open GeoTIFF to get transform
            with rasterio.open(geotiff_path) as src:
                transform = src.transform
                image_width = src.width
                image_height = src.height
            
            # Convert buildings to pixel coordinates
            annotations = []
            for idx, building in buildings.iterrows():
                geom = building.geometry
                
                # Skip if geometry is empty or invalid
                if geom.is_empty:
                    continue
                
                # Get bounding box of building
                minx, miny, maxx, maxy = geom.bounds
                
                # Transform to pixel coordinates
                from rasterio.windows import Window
                from rasterio.transform import xy
                
                # Get pixel coordinates
                col_min, row_max = ~transform * (minx, miny)
                col_max, row_min = ~transform * (maxx, maxy)
                
                # Normalize to image size
                x_center = ((col_min + col_max) / 2) / image_width
                y_center = ((row_min + row_max) / 2) / image_height
                width = (col_max - col_min) / image_width
                height = (row_max - row_min) / image_height
                
                # Clip to [0, 1]
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                # Class ID 0 for buildings
                annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            if not annotations:
                logger.warning(f"No valid annotations for extent {extent_id}")
                return None
            
            # Save annotations
            label_path = self.labels_dir / f"extent_{extent_id:04d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))
            
            logger.info(f"Created annotations for extent {extent_id}: {len(annotations)} buildings")
            return str(label_path)
            
        except Exception as e:
            logger.error(f"Failed to create annotations for extent {extent_id}: {e}")
            return None
    
    def process_extents(self, limit=None):
        """
        Process all extents: download imagery and create annotations.
        
        Parameters
        ----------
        limit : int, optional
            Limit number of extents to process (for testing)
        """
        if self.extents_gdf is None or self.buildings_gdf is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        extents = self.extents_gdf.head(limit) if limit else self.extents_gdf
        
        successful = 0
        for idx, (_, extent_row) in enumerate(extents.iterrows()):
            extent_id = idx
            extent_geom = extent_row.geometry
            extent_bounds = extent_geom.bounds
            
            # Download imagery
            geotiff_path = self.download_imagery(extent_bounds, extent_id)
            if not geotiff_path:
                continue
            
            # Create annotations
            label_path = self.buildings_to_yolo_annotations(
                geotiff_path, extent_geom, extent_id
            )
            
            if label_path:
                successful += 1
        
        logger.info(f"Successfully processed {successful}/{len(extents)} extents")
    
    def create_dataset_yaml(self, train_ratio=0.7, val_ratio=0.15):
        """
        Create YOLO dataset.yaml configuration file.
        
        Parameters
        ----------
        train_ratio : float
            Proportion for training set
        val_ratio : float
            Proportion for validation set
        """
        test_ratio = 1 - train_ratio - val_ratio
        
        yaml_content = f"""# YOLO Building Detection Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: building

train_ratio: {train_ratio}
val_ratio: {val_ratio}
test_ratio: {test_ratio}
"""
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        return str(yaml_path)
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        Split processed images and labels into train/val/test sets.
        
        Parameters
        ----------
        train_ratio : float
            Proportion for training set
        val_ratio : float
            Proportion for validation set
        """
        logger.info("Splitting dataset...")
        
        # Get all image files
        image_files = sorted(list(self.images_dir.glob("extent_*.tif")))
        
        if not image_files:
            logger.error("No images found to split")
            return
        
        # Shuffle and split
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(image_files)
        
        n_images = len(image_files)
        train_count = int(n_images * train_ratio)
        val_count = int(n_images * val_ratio)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Create split directories
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Move files to split directories
        for img_path in train_files:
            label_name = img_path.stem + ".txt"
            label_path = self.labels_dir / label_name
            
            shutil.move(str(img_path), str(self.images_dir / "train" / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / "train" / label_name))
        
        for img_path in val_files:
            label_name = img_path.stem + ".txt"
            label_path = self.labels_dir / label_name
            
            shutil.move(str(img_path), str(self.images_dir / "val" / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / "val" / label_name))
        
        for img_path in test_files:
            label_name = img_path.stem + ".txt"
            label_path = self.labels_dir / label_name
            
            shutil.move(str(img_path), str(self.images_dir / "test" / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(self.labels_dir / "test" / label_name))
        
        logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    def run_pipeline(self, limit=None, split=True):
        """
        Run the complete pipeline.
        
        Parameters
        ----------
        limit : int, optional
            Limit number of extents to process
        split : bool
            Whether to split dataset into train/val/test
        """
        logger.info("Starting YOLO data building pipeline...")
        start_time = datetime.now()
        
        self.setup_directories()
        self.load_data()
        self.process_extents(limit=limit)
        
        if split:
            self.split_dataset()
        
        self.create_dataset_yaml()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline completed in {duration}")


if __name__ == "__main__":
    # Example usage
    builder = YOLODataBuilder(
        imagery_url="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2024/MapServer/export",
        extents_gpkg_path="../../data/vector/extents/extent.gpkg",
        extents_layer_name="extent",  # Change to your actual layer name
        buildings_gpkg_path="../../data/vector/carto/karto.gpkg",
        buildings_layer_name="buildings",  # Change to your actual layer name
        output_dir="../../dataset/yolo_buildings"
    )
    
    # Run with limit for testing
    builder.run_pipeline(limit=5, split=False)
