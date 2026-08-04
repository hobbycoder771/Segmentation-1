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
from shapely.geometry import box, shape
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import shutil
import json
from pyproj import Transformer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YOLODataBuilder:
    """Build YOLO training dataset for building detection from geospatial data."""

    def __init__(
        self,
        imagery_url,
        extents_gpkg_path,
        extents_layer_name,
        buildings_gpkg_path,
        buildings_layer_name,
        output_dir="../../dataset/yolo_buildings",
        image_size=4096,
        crs="EPSG:3857",
        feature_server_url=None,
        carto_output_path=None,
        enable_tiling=False,
        tile_size=512,
        tile_overlap=0,
    ):
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
        feature_server_url : str, optional
            ArcGIS FeatureServer URL for vector data clipping
        carto_output_path : str, optional
            Output path for clipped vector data (geopackage)
        enable_tiling : bool
            Whether to tile large images into smaller tiles
        tile_size : int
            Size of tiles when tiling is enabled (e.g., 512, 640)
        tile_overlap : int
            Overlap between tiles in pixels (default: 0)
        """
        self.imagery_url = imagery_url
        self.extents_gpkg = extents_gpkg_path
        self.extents_layer = extents_layer_name
        self.buildings_gpkg = buildings_gpkg_path
        self.buildings_layer = buildings_layer_name
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.crs = crs
        self.feature_server_url = feature_server_url
        self.carto_output_path = carto_output_path
        self.enable_tiling = enable_tiling
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # Setup output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # Initialize data storage
        self.extents_gdf = None
        self.buildings_gdf = None
        self.tile_counter = 0  # Global counter for unique tile IDs

    def setup_directories(self):
        """Create necessary output directories."""
        logger.info("Setting up output directories...")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load extents and buildings from geopackages."""
        
        def get_layer_names(gpkg_path):
            """Get available layers in a geopackage using pyogrio."""
            try:
                import pyogrio
                return pyogrio.list_layers(gpkg_path)[:,0].tolist()
            except:
                # Fallback: try reading with geopandas and catching the warning
                try:
                    gpd.read_file(gpkg_path)
                    return ['buildings']
                except:
                    return []

        logger.info("Loading extents from geopackage...")
        self.extents_gdf = gpd.read_file(self.extents_gpkg, layer=self.extents_layer)
        logger.info(f"Loaded {len(self.extents_gdf)} extents")

        logger.info("Loading buildings from geopackage...")

        # Check if buildings_layer is a single layer or pattern
        try:
            # Try to load as single layer first
            self.buildings_gdf = gpd.read_file(
                self.buildings_gpkg, layer=self.buildings_layer
            )
            logger.info(
                f"Loaded {len(self.buildings_gdf)} buildings from layer '{self.buildings_layer}'"
            )
        except Exception as e:
            # If single layer fails, try to load all layers matching pattern
            logger.info(
                f"Single layer '{self.buildings_layer}' not found, attempting to load multiple layers..."
            )

            try:
                # Get all available layers
                available_layers = get_layer_names(self.buildings_gpkg)
                logger.info(f"Available layers in geopackage: {available_layers}")

                # Load all layers that contain the pattern (or all if pattern is generic)
                building_layers = [
                    layer for layer in available_layers if "building" in layer.lower()
                ]

                if not building_layers:
                    # If no building layers found and FeatureServer is configured, skip buildings loading
                    if self.feature_server_url:
                        logger.warning(
                            f"No building layers found in {self.buildings_gpkg}, but FeatureServer is configured. "
                            "Buildings will be loaded from FeatureServer during processing."
                        )
                        # Create empty GeoDataFrame with default CRS
                        self.buildings_gdf = gpd.GeoDataFrame(
                            geometry=[], crs=self.crs
                        )
                        return
                    else:
                        raise ValueError(
                            f"No layers containing 'building' found in {self.buildings_gpkg}"
                        )

                logger.info(
                    f"Found {len(building_layers)} building layers: {building_layers}"
                )

                # Load and concatenate all building layers
                gdfs = []
                for layer in building_layers:
                    logger.info(f"  Loading layer: {layer}")
                    gdf = gpd.read_file(self.buildings_gpkg, layer=layer)
                    gdfs.append(gdf)

                self.buildings_gdf = gpd.GeoDataFrame(
                    pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs
                )
                logger.info(
                    f"Loaded {len(self.buildings_gdf)} buildings from {len(building_layers)} layers"
                )

            except Exception as e2:
                logger.error(f"Failed to load buildings: {e2}")
                # If FeatureServer is configured, allow continuing without buildings
                if self.feature_server_url:
                    logger.warning(
                        "Creating empty buildings GeoDataFrame. Buildings will be loaded from FeatureServer."
                    )
                    self.buildings_gdf = gpd.GeoDataFrame(
                        geometry=[], crs=self.crs
                    )
                else:
                    raise

        # Ensure same CRS
        if self.buildings_gdf is not None and self.buildings_gdf.crs != self.extents_gdf.crs:
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
            # Drop alpha channel if present
            if image.shape[0] == 4:
                image = image[:3]
            
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
            count=image.shape[0],  # Update count to match actual channels
        )

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(image)

    def tile_image(self, geotiff_path, extent_id):
        """
        Split a large GeoTIFF into smaller tiles.

        Parameters
        ----------
        geotiff_path : str
            Path to the input GeoTIFF
        extent_id : int
            Unique identifier for the extent

        Returns
        -------
        list of str
            Paths to the created tile GeoTIFFs
        """
        try:
            with rasterio.open(geotiff_path) as src:
                image = src.read()
                profile = src.profile
                transform = src.transform
                image_width = src.width
                image_height = src.height

            tile_paths = []
            stride = self.tile_size - self.tile_overlap

            # Calculate number of tiles
            tiles_x = (image_width - self.tile_overlap + stride - 1) // stride
            tiles_y = (image_height - self.tile_overlap + stride - 1) // stride

            logger.info(
                f"Tiling extent {extent_id}: {tiles_x}×{tiles_y} tiles "
                f"({self.tile_size}×{self.tile_size})"
            )

            tile_id = 0
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Calculate tile boundaries
                    col_start = tx * stride
                    row_start = ty * stride
                    col_end = min(col_start + self.tile_size, image_width)
                    row_end = min(row_start + self.tile_size, image_height)

                    # Extract tile
                    tile_image = image[
                        :, row_start:row_end, col_start:col_end
                    ]

                    # Calculate geospatial bounds for this tile
                    tile_transform = rasterio.transform.Affine(
                        transform.a,
                        transform.b,
                        transform.c + col_start * transform.a,
                        transform.d,
                        transform.e,
                        transform.f + row_start * transform.e,
                    )

                    # Update profile for tile
                    tile_profile = profile.copy()
                    tile_profile.update(
                        width=tile_image.shape[2],
                        height=tile_image.shape[1],
                        transform=tile_transform,
                    )

                    # Save tile
                    tile_path = (
                        self.images_dir
                        / f"extent_{extent_id:04d}_tile_{tile_id:03d}.tif"
                    )
                    with rasterio.open(tile_path, "w", **tile_profile) as dst:
                        dst.write(tile_image)

                    tile_paths.append(str(tile_path))
                    tile_id += 1

            logger.info(
                f"Created {len(tile_paths)} tiles for extent {extent_id}"
            )
            return tile_paths

        except Exception as e:
            logger.error(f"Failed to tile image for extent {extent_id}: {e}")
            return []

    def tile_annotations(self, annotations, geotiff_path, extent_id):
        """
        Split YOLO annotations to match tiled images.

        Parameters
        ----------
        annotations : list of str
            YOLO format annotations from the original image
        geotiff_path : str
            Path to the original GeoTIFF
        extent_id : int
            Unique identifier for the extent

        Returns
        -------
        dict
            Mapping of tile paths to their annotations
        """
        try:
            with rasterio.open(geotiff_path) as src:
                image_width = src.width
                image_height = src.height

            tile_annotations = {}
            stride = self.tile_size - self.tile_overlap

            # Parse annotations
            parsed_annotations = []
            for ann in annotations:
                parts = ann.split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                parsed_annotations.append(
                    (class_id, x_center, y_center, width, height)
                )

            # Calculate number of tiles
            tiles_x = (image_width - self.tile_overlap + stride - 1) // stride
            tiles_y = (image_height - self.tile_overlap + stride - 1) // stride

            tile_id = 0
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Calculate tile boundaries in pixel coordinates
                    col_start = tx * stride
                    row_start = ty * stride
                    col_end = min(col_start + self.tile_size, image_width)
                    row_end = min(row_start + self.tile_size, image_height)

                    tile_width = col_end - col_start
                    tile_height = row_end - row_start

                    # Find annotations that intersect this tile
                    tile_anns = []
                    for class_id, x_center, y_center, width, height in parsed_annotations:
                        # Convert normalized coordinates to pixel coordinates
                        px_center = x_center * image_width
                        py_center = y_center * image_height
                        px_width = width * image_width
                        px_height = height * image_height

                        # Check intersection with tile
                        left = px_center - px_width / 2
                        right = px_center + px_width / 2
                        top = py_center - px_height / 2
                        bottom = py_center + px_height / 2

                        # Check if bounding box intersects tile
                        if (
                            right > col_start
                            and left < col_end
                            and bottom > row_start
                            and top < row_end
                        ):
                            # Clip bounding box to tile boundaries
                            clipped_left = max(left, col_start)
                            clipped_right = min(right, col_end)
                            clipped_top = max(top, row_start)
                            clipped_bottom = min(bottom, row_end)

                            # Convert back to normalized coordinates relative to tile
                            new_x_center = (
                                (clipped_left + clipped_right) / 2 - col_start
                            ) / tile_width
                            new_y_center = (
                                (clipped_top + clipped_bottom) / 2 - row_start
                            ) / tile_height
                            new_width = (
                                (clipped_right - clipped_left) / tile_width
                            )
                            new_height = (
                                (clipped_bottom - clipped_top) / tile_height
                            )

                            # Only include if bounding box has meaningful size
                            if new_width > 0.01 and new_height > 0.01:
                                # Clip to [0, 1]
                                new_x_center = np.clip(new_x_center, 0, 1)
                                new_y_center = np.clip(new_y_center, 0, 1)
                                new_width = np.clip(new_width, 0, 1)
                                new_height = np.clip(new_height, 0, 1)

                                tile_anns.append(
                                    f"{class_id} {new_x_center:.6f} {new_y_center:.6f} "
                                    f"{new_width:.6f} {new_height:.6f}"
                                )

                    # Store annotations for this tile
                    tile_path = f"extent_{extent_id:04d}_tile_{tile_id:03d}"
                    if tile_anns:
                        tile_annotations[tile_path] = tile_anns

                    tile_id += 1

            logger.info(
                f"Created annotations for {len(tile_annotations)} tiles in extent {extent_id}"
            )
            return tile_annotations

        except Exception as e:
            logger.error(f"Failed to tile annotations for extent {extent_id}: {e}")
            return {}

    def clip_feature_server_by_extent(
        self, extent, out_name, extent_id
    ):
        """
        Query and clip vector data from ArcGIS FeatureServer by extent.

        Parameters
        ----------
        extent : tuple
            (minx, miny, maxx, maxy) bounds in EPSG:3857
        out_name : str
            Name prefix for the output layer
        extent_id : int
            Unique identifier for the extent (used in layer naming)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self.feature_server_url or not self.carto_output_path:
            logger.warning(
                "FeatureServer URL or carto output path not configured. Skipping vector clipping."
            )
            return False

        try:
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:25830", always_xy=True)

            xmin, ymin, xmax, ymax = extent

            # Project bbox from EPSG:3857 to EPSG:25830
            xmin2, ymin2 = transformer.transform(xmin, ymin)
            xmax2, ymax2 = transformer.transform(xmax, ymax)

            geometry = {
                "xmin": xmin2,
                "ymin": ymin2,
                "xmax": xmax2,
                "ymax": ymax2,
                "spatialReference": {"wkid": 25830},
            }

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

            logger.info(f"Found {len(features)} features for extent {extent_id}")

            if len(features) == 0:
                logger.info(f"No features found in FeatureServer for extent {extent_id}")
                return False

            # Convert GeoJSON to GeoDataFrame
            rows = []
            for feat in features:
                attrs = feat["properties"]
                attrs["geometry"] = shape(feat["geometry"])
                rows.append(attrs)

            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")

            # Save to geopackage
            layer_name = f"{out_name}_tile_{extent_id}"
            outfile = os.path.join(self.carto_output_path, "karto.gpkg")

            # Ensure output directory exists
            os.makedirs(self.carto_output_path, exist_ok=True)

            gdf.to_file(outfile, layer=layer_name, driver="GPKG", mode="a")

            logger.info(f"Saved {len(features)} features to layer '{layer_name}' in {outfile}")
            return True

        except Exception as e:
            logger.error(f"Failed to clip FeatureServer data for extent {extent_id}: {e}")
            return False

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
        return self.buildings_gdf[
            self.buildings_gdf.geometry.intersects(extent_geometry)
        ]

    def buildings_to_yolo_annotations(self, geotiff_path, extent_geometry, extent_id, buildings_gdf=None):
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
        buildings_gdf : GeoDataFrame, optional
            Buildings GeoDataFrame for this extent. If None, uses self.buildings_gdf

        Returns
        -------
        str or None
            Path to annotation file, or None if failed
        """
        try:
            # Get buildings in this extent
            if buildings_gdf is not None:
                buildings = buildings_gdf[
                    buildings_gdf.geometry.intersects(extent_geometry)
                ]
            else:
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
                annotations.append(
                    f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            if not annotations:
                logger.warning(f"No valid annotations for extent {extent_id}")
                return None

            # Save annotations
            label_path = self.labels_dir / f"extent_{extent_id:04d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))

            logger.info(
                f"Created annotations for extent {extent_id}: {len(annotations)} buildings"
            )
            return str(label_path)

        except Exception as e:
            logger.error(f"Failed to create annotations for extent {extent_id}: {e}")
            return None

    def process_extents(self, limit=None):
        """
        Process all extents: download imagery, clip vector data, and create annotations.

        Parameters
        ----------
        limit : int, optional
            Limit number of extents to process (for testing)
        """
        if self.extents_gdf is None:
            logger.error("Extents data not loaded. Call load_data() first.")
            return

        extents = self.extents_gdf.head(limit) if limit else self.extents_gdf

        successful = 0
        for idx, (_, extent_row) in enumerate(extents.iterrows()):
            extent_id = idx
            extent_geom = extent_row.geometry
            extent_bounds = extent_geom.bounds

            # Clip vector data from FeatureServer and load buildings for this extent
            buildings_gdf_extent = None
            if self.feature_server_url:
                self.clip_feature_server_by_extent(
                    extent_bounds, "buildings", extent_id
                )
                # Load the clipped buildings for this extent
                try:
                    layer_name = f"buildings_tile_{extent_id}"
                    outfile = os.path.join(self.carto_output_path, "karto.gpkg")
                    buildings_gdf_extent = gpd.read_file(outfile, layer=layer_name)
                    logger.info(f"Loaded {len(buildings_gdf_extent)} buildings from clipped layer for extent {extent_id}")
                except Exception as e:
                    logger.warning(f"Could not load clipped buildings for extent {extent_id}: {e}")
                    buildings_gdf_extent = None
            else:
                # Use pre-loaded buildings data
                buildings_gdf_extent = self.buildings_gdf

            # Download imagery
            geotiff_path = self.download_imagery(extent_bounds, extent_id)
            if not geotiff_path:
                continue

            # Create annotations (only if buildings data available)
            if buildings_gdf_extent is not None and len(buildings_gdf_extent) > 0:
                label_path = self.buildings_to_yolo_annotations(
                    geotiff_path, extent_geom, extent_id, buildings_gdf_extent
                )

                if label_path:
                    # Handle tiling if enabled
                    if self.enable_tiling:
                        self._process_tiling(geotiff_path, label_path, extent_id)
                    successful += 1
            else:
                logger.info(f"No buildings data available for extent {extent_id}, skipping annotations")
                # Handle tiling even without annotations
                if self.enable_tiling:
                    self._process_tiling(geotiff_path, None, extent_id)
                successful += 1  # Count imagery download as successful

        logger.info(f"Successfully processed {successful}/{len(extents)} extents")

    def _process_tiling(self, geotiff_path, label_path, extent_id):
        """
        Process tiling for a single extent.

        Parameters
        ----------
        geotiff_path : str
            Path to the original GeoTIFF
        label_path : str or None
            Path to the original annotation file
        extent_id : int
            Unique identifier for the extent
        """
        try:
            # Tile the image
            tile_paths = self.tile_image(geotiff_path, extent_id)

            if not tile_paths:
                logger.warning(f"No tiles created for extent {extent_id}")
                return

            # Load annotations if available
            if label_path and Path(label_path).exists():
                with open(label_path, "r") as f:
                    annotations = [line.strip() for line in f.readlines()]

                # Tile annotations
                tile_annotations = self.tile_annotations(
                    annotations, geotiff_path, extent_id
                )

                # Save tiled annotations
                for tile_name, tile_anns in tile_annotations.items():
                    label_file = self.labels_dir / f"{tile_name}.txt"
                    with open(label_file, "w") as f:
                        f.write("\n".join(tile_anns))

            # Remove original image and label (keep only tiles)
            Path(geotiff_path).unlink()
            if label_path and Path(label_path).exists():
                Path(label_path).unlink()

            logger.info(f"Tiling completed for extent {extent_id}")

        except Exception as e:
            logger.error(f"Failed to process tiling for extent {extent_id}: {e}")

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
        val_files = image_files[train_count : train_count + val_count]
        test_files = image_files[train_count + val_count :]

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
                shutil.move(
                    str(label_path), str(self.labels_dir / "train" / label_name)
                )

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

        logger.info(
            f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test"
        )

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
        output_dir="../../dataset/yolo_buildings",
        feature_server_url="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/Kartografia_BTB_Cartografia_5000/FeatureServer/16/query",
        carto_output_path="../../data/vector/carto",
    )

    # Run with limit for testing
    builder.run_pipeline(limit=10, split=False)
