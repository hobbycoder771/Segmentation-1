import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image

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
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.extents_gdf = None
        self.objects_gdf = None

    def setup_directories(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directories created")

    def load_data(self):
        logger.info("Loading extents...")
        self.extents_gdf = gpd.read_file(self.extents_gpkg, layer=self.extents_layer)
        logger.info(f"Loaded {len(self.extents_gdf)} extents")

        logger.info("Loading objects...")
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
            self.generate_label_file(extent_id, str(mask_path))
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

            objects_for_extent = self.objects_gdf[self.objects_gdf.geometry.intersects(extent_geom)]
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
        train_count, val_count = int(n * train_ratio), int(n * val_ratio)

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
