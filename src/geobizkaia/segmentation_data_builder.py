import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import requests
import rasterio
import json
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationDataBuilder:
    def __init__(self, imagery_url, extents_gpkg_path, extents_layer_name,
                 objects_gpkg_path, objects_layer_name,
                 output_dir="../../dataset/yolo_buildings_seg", image_size=4096,
                 crs="EPSG:3857", feature_server_url=None,
                 carto_output_path=None, mask_value=255):
        self.imagery_url = imagery_url
        self.extents_gpkg = extents_gpkg_path
        self.extents_layer = extents_layer_name
        self.objects_gpkg = objects_gpkg_path
        self.objects_layer = objects_layer_name
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.crs = crs
        self.mask_value = mask_value
        self.feature_server_url = feature_server_url
        self.carto_output_path = carto_output_path
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.extents_gdf = None
        self.objects_gdf = None

    def setup_directories(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directories created")
        
        # Clear karto.gpkg at the beginning of pipeline if using FeatureServer
        if self.feature_server_url and self.carto_output_path:
            self._clear_karto_gpkg()
    
    def _clear_karto_gpkg(self):
        """Clear karto.gpkg at the beginning of pipeline.
        
        This ensures only data from the current extents is in karto.gpkg.
        Removes old layers and recreates the file if possible.
        """
        try:
            gpkg_path = Path(self.carto_output_path) / "karto.gpkg"
            if not gpkg_path.exists():
                logger.info(f"karto.gpkg does not exist yet, will be created during processing")
                return
            
            # Try to delete old layers from the gpkg using SQL
            import sqlite3
            try:
                conn = sqlite3.connect(str(gpkg_path))
                cursor = conn.cursor()
                
                # Find and remove old layer tables
                old_layers = ["buildings_tile_0", "buildings_tile_1", "buildings_tile_2", 
                             "buildings_tile_3", "buildings_tile_4", "buildings_tile_5",
                             "buildings_tile_6", "buildings_tile_7", "buildings_tile_8", 
                             "buildings_tile_9"]
                
                removed_layers = []
                for layer in old_layers:
                    try:
                        # Drop the layer table
                        cursor.execute(f"DROP TABLE IF EXISTS [{layer}];")
                        # Remove spatial index tables
                        for idx_table in [f"rtree_{layer}_geom", f"rtree_{layer}_geom_rowid", 
                                         f"rtree_{layer}_geom_node", f"rtree_{layer}_geom_parent"]:
                            cursor.execute(f"DROP TABLE IF EXISTS [{idx_table}];")
                        # Remove from gpkg_contents
                        cursor.execute("DELETE FROM gpkg_contents WHERE table_name = ?;", (layer,))
                        removed_layers.append(layer)
                    except:
                        pass
                
                conn.commit()
                conn.close()
                
                if removed_layers:
                    logger.info(f"Removed {len(removed_layers)} old layers from karto.gpkg")
                else:
                    logger.info(f"No old layers to remove from karto.gpkg")
                    
            except Exception as e:
                logger.warning(f"Failed to clean up old layers: {e}")
                # Fall back to deleting the entire file
                try:
                    gpkg_path.unlink()
                    logger.info(f"Deleted karto.gpkg for fresh start")
                except:
                    logger.warning(f"Could not delete karto.gpkg, old layers may persist")
                    
        except Exception as e:
            logger.warning(f"Failed to clear karto.gpkg: {e}")
    
    def fetch_vectors_from_featureserver(self, extent_bounds, extent_id):
        """Fetch vector data from FeatureServer for a specific extent.
        
        Returns a GeoDataFrame with the clipped vector data, or None if fetch fails.
        Also saves the data to karto.gpkg for persistence.
        """
        if not self.feature_server_url:
            return None
        
        try:
            minx, miny, maxx, maxy = extent_bounds
            
            # Transform from EPSG:3857 to EPSG:25830 for the query
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:25830", always_xy=True)
            xmin2, ymin2 = transformer.transform(minx, miny)
            xmax2, ymax2 = transformer.transform(maxx, maxy)
            
            # Build geometry for spatial query
            geometry = {
                "xmin": xmin2,
                "ymin": ymin2,
                "xmax": xmax2,
                "ymax": ymax2,
                "spatialReference": {"wkid": 25830},
            }
            
            # Query FeatureServer
            params = {
                "f": "geojson",
                "where": "1=1",
                "geometry": json.dumps(geometry),
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "inSR": 25830,
                "outSR": 3857,
                "returnGeometry": "true",
                "outFields": "*",
            }
            
            logger.info(f"Querying FeatureServer for extent {extent_id}...")
            response = requests.get(self.feature_server_url, params=params, timeout=30)
            response.raise_for_status()
            
            geojson = response.json()
            features = geojson.get("features", [])
            logger.info(f"Retrieved {len(features)} features from FeatureServer for extent {extent_id}")
            
            if not features:
                return None
            
            # Convert to GeoDataFrame
            rows = []
            for feat in features:
                attrs = feat["properties"]
                attrs["geometry"] = shape(feat["geometry"])
                rows.append(attrs)
            
            gdf = gpd.GeoDataFrame(rows, crs="EPSG:3857")
            
            # Save to karto.gpkg with layer name based on extent_id
            if self.carto_output_path:
                self._save_to_karto_gpkg(gdf, extent_id)
            
            return gdf
            
        except Exception as e:
            logger.warning(f"Failed to fetch vectors from FeatureServer: {e}")
            return None
    
    def _save_to_karto_gpkg(self, gdf, extent_id):
        """Save clipped vector data to karto.gpkg.
        
        Creates a layer for each extent with the clipped building data.
        """
        try:
            if not self.carto_output_path:
                return
            
            gpkg_path = Path(self.carto_output_path) / "karto.gpkg"
            gpkg_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create layer name based on extent_id
            layer_name = f"extent_{extent_id:04d}"
            
            # Write new data to the layer (creates or overwrites)
            gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode="a")
            logger.info(f"Saved layer '{layer_name}' to {gpkg_path}")
        
        except Exception as e:
            logger.warning(f"Failed to save clipped data to karto.gpkg: {e}")

    def load_data(self):
        logger.info("Loading extents...")
        self.extents_gdf = gpd.read_file(self.extents_gpkg, layer=self.extents_layer)
        logger.info(f"Loaded {len(self.extents_gdf)} extents")

        # Load static objects only if not using dynamic FeatureServer fetching
        if not self.feature_server_url:
            logger.info("Loading objects from static dataset...")
            try:
                self.objects_gdf = gpd.read_file(self.objects_gpkg, layer=self.objects_layer)
            except:
                import pyogrio
                all_layers = pyogrio.list_layers(self.objects_gpkg)[:,0].tolist()
                matching = [l for l in all_layers if self.objects_layer.lower() in l.lower()]
                gdfs = [gpd.read_file(self.objects_gpkg, layer=l) for l in matching]
                self.objects_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            
            logger.info(f"Loaded {len(self.objects_gdf)} objects")
            if self.objects_gdf.crs != self.extents_gdf.crs:
                self.objects_gdf = self.objects_gdf.to_crs(self.extents_gdf.crs)
        else:
            logger.info("Using dynamic FeatureServer for vector data (static dataset not loaded)")

    def download_imagery(self, extent_bounds, extent_id):
        try:
            minx, miny, maxx, maxy = extent_bounds
            params = {"bbox": f"{minx},{miny},{maxx},{maxy}","bboxSR": "3857",
                     "imageSR": "3857", "size": f"{self.image_size},{self.image_size}",
                     "format": "png32", "transparent": "false", "f": "json"}
            
            response = requests.get(self.imagery_url, params=params, timeout=30)
            result = response.json()
            image_url = result.get("href")
            if not image_url:
                return None
            
            image_response = requests.get(image_url, timeout=30)
            png_path = self.images_dir / f"extent_{extent_id:04d}.png"
            with open(png_path, "wb") as f:
                f.write(image_response.content)
            
            geotiff_path = self.images_dir / f"extent_{extent_id:04d}.tif"
            self._png_to_geotiff(str(png_path), extent_bounds, str(geotiff_path))
            png_path.unlink()
            logger.info(f"Downloaded extent {extent_id}")
            return str(geotiff_path)
        except Exception as e:
            logger.warning(f"Download failed {extent_id}: {e}")
            return None

    def _png_to_geotiff(self, png_file, extent_bounds, output_tif):
        with rasterio.open(png_file) as src:
            image = src.read()[:3] if src.count > 3 else src.read()
            width, height = src.width, src.height
            profile = src.profile

        minx, miny, maxx, maxy = extent_bounds
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        profile.update(driver="GTiff", crs="EPSG:3857", transform=transform, count=min(3, image.shape[0]))
        
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(image)

    def generate_mask(self, extent_bounds, extent_id, objects_for_extent):
        try:
            if objects_for_extent is None or len(objects_for_extent) == 0:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                minx, miny, maxx, maxy = extent_bounds
                transform = from_bounds(minx, miny, maxx, maxy, self.image_size, self.image_size)
                geometries = [(g, self.mask_value) for g in objects_for_extent.geometry if g.is_valid]
                mask = rasterize(geometries, out_shape=(self.image_size, self.image_size),
                                 transform=transform, default_value=0, dtype=np.uint8)

            mask_path = self.masks_dir / f"extent_{extent_id:04d}.png"
            Image.fromarray(mask, mode="L").save(mask_path)
            logger.info(f"Generated mask {extent_id}")
            return str(mask_path)
        except Exception as e:
            logger.warning(f"Mask generation failed {extent_id}: {e}")
            return None

    def process_extents(self, limit=None):
        extents = self.extents_gdf.head(limit) if limit else self.extents_gdf
        successful = 0

        for idx, (_, extent_row) in enumerate(extents.iterrows()):
            extent_id = idx
            extent_geom = extent_row.geometry
            extent_bounds = extent_geom.bounds

            geotiff_path = self.download_imagery(extent_bounds, extent_id)
            if not geotiff_path:
                continue

            # Try to fetch vectors from FeatureServer first (dynamic clipping)
            if self.feature_server_url:
                objects_for_extent = self.fetch_vectors_from_featureserver(extent_bounds, extent_id)
                if objects_for_extent is not None:
                    logger.info(f"Using {len(objects_for_extent)} features from FeatureServer for extent {extent_id}")
                else:
                    # Fall back to static dataset if FeatureServer fails
                    logger.warning(f"FeatureServer fetch failed for extent {extent_id}, falling back to static data")
                    objects_for_extent = self.objects_gdf[self.objects_gdf.geometry.intersects(extent_geom)] if self.objects_gdf is not None else None
            else:
                # Use static dataset if no FeatureServer URL provided
                objects_for_extent = self.objects_gdf[self.objects_gdf.geometry.intersects(extent_geom)] if self.objects_gdf is not None else None
            
            mask_path = self.generate_mask(extent_bounds, extent_id, objects_for_extent)
            if mask_path:
                successful += 1

        logger.info(f"Processed {successful}/{len(extents)} extents")

    def create_dataset_yaml(self):
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

train_masks: masks/train
val_masks: masks/val
test_masks: masks/test

nc: 1
names:
  0: building
"""
        (self.output_dir / "data.yaml").write_text(yaml_content)
        logger.info("Created data.yaml")

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        image_files = sorted(list(self.images_dir.glob("extent_*.tif")))
        if not image_files:
            for split in ["train", "val", "test"]:
                (self.images_dir / split).mkdir(parents=True, exist_ok=True)
                (self.masks_dir / split).mkdir(parents=True, exist_ok=True)
            return

        np.random.seed(42)
        np.random.shuffle(image_files)
        n = len(image_files)
        
        # Ensure at least 1 sample in train if n > 0
        train_count = max(1, int(n * train_ratio))
        val_count = max(0, int(n * val_ratio))
        
        # Adjust if counts exceed total
        if train_count + val_count > n:
            val_count = max(0, n - train_count)
        
        logger.info(f"Split: train={train_count}, val={val_count}, test={n-train_count-val_count}")
        
        for split in ["train", "val", "test"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.masks_dir / split).mkdir(parents=True, exist_ok=True)
        
        for files, split in [(image_files[:train_count], "train"),
                              (image_files[train_count:train_count+val_count], "val"),
                              (image_files[train_count+val_count:], "test")]:
            for img in files:
                mask_file = self.masks_dir / (img.stem + ".png")
                shutil.move(str(img), str(self.images_dir / split / img.name))
                if mask_file.exists():
                    shutil.move(str(mask_file), str(self.masks_dir / split / mask_file.name))

    def run_pipeline(self, limit=None, split=True):
        logger.info("Starting pipeline...")
        self.setup_directories()
        self.load_data()
        self.process_extents(limit=limit)
        if split:
            self.split_dataset()
        else:
            # Test mode: move all files to train folder
            for s in ["train", "val", "test"]:
                (self.images_dir / s).mkdir(parents=True, exist_ok=True)
                (self.masks_dir / s).mkdir(parents=True, exist_ok=True)
            
            # Move all images and masks to train folder
            for img in self.images_dir.glob("extent_*.tif"):
                shutil.move(str(img), str(self.images_dir / "train" / img.name))
            for mask in self.masks_dir.glob("extent_*.png"):
                shutil.move(str(mask), str(self.masks_dir / "train" / mask.name))
            
        self.create_dataset_yaml()
        logger.info("Pipeline complete")
