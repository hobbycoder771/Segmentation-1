# GeoBizkaia Building Detection Project - Complete Analysis

## Project Overview

This is a **YOLO-based Building Detection System** that creates training datasets from geospatial data for detecting buildings in aerial/orthographic imagery. The project integrates:

1. **Vector data** (building polygons from GeoPackages)
2. **Raster imagery** (downloaded from ArcGIS MapServer)
3. **Geospatial processing** (coordinate transformations, CRS handling)
4. **YOLO annotation format** (for object detection training)

### Project Goal
Build a machine learning pipeline to detect and annotate buildings in orthographic imagery across the GeoBizkaia region, producing a YOLO-compatible dataset for YOLOv8 model training.

---

## Project Structure

```
GeoBizkaia-Segmentation-1/
│
├── 📄 START_HERE.txt                    ← Entry point with quick start guide
├── 📄 README_YOLO.md                    ← Project overview
├── 📄 QUICKSTART.md                     ← 5-minute setup guide
├── 📄 USAGE_FLOW.txt                    ← Visual workflow
├── 📄 YOLO_SETUP_GUIDE.md               ← Detailed documentation
├── 📄 COORDINATE_TRANSFORMATION.md      ← Technical deep dive on coordinate conversions
├── 📄 DELIVERY_SUMMARY.txt              ← Inventory of all created files
│
├── requirements_yolo.txt                ← Python dependencies (geopandas, rasterio, ultralytics, etc.)
│
├── 📁 src/geobizkaia/                   ← CORE PIPELINE SCRIPTS
│   ├── yolo_data_builder.py             ← Main orchestration class (510 lines)
│   │   • YOLODataBuilder class
│   │   • Downloads imagery from ArcGIS MapServer
│   │   • Converts PNG to georeferenced GeoTIFF
│   │   • Intersects buildings with extents
│   │   • Generates YOLO annotations
│   │   • Splits dataset into train/val/test
│   │
│   ├── run_yolo_pipeline.py             ← CLI interface (178 lines)
│   │   • --test: process 3 extents
│   │   • --full: process all extents with splitting
│   │   • --limit N: custom number of extents
│   │   • Custom split ratios
│   │   • Custom layer names
│   │
│   ├── inspect_gpkg.py                  ← Geopackage inspector utility (77 lines)
│   │   • Lists all layers in geopackages
│   │   • Shows geometry types, CRS, feature counts
│   │   • Displays sample data
│   │
│   └── [Other utility scripts]
│       ├── batch_clip_extract.py
│       ├── clip_featureserver.py
│       ├── download_image.py
│       ├── extent2gpkg.py
│       └── gpk2extents.py
│
├── 📁 data/                             ← INPUT DATA
│   ├── vector/
│   │   ├── extents/extent.gpkg          ← Tile extent definitions (vectors)
│   │   └── carto/karto.gpkg             ← Building polygon data
│   │
│   └── imagery/
│       └── imagery_tile_*.tif           ← 10 sample georeferenced imagery tiles
│
├── 📁 dataset/                          ← GENERATED YOLO DATASET
│   └── yolo_buildings/
│       ├── images/
│       │   ├── train/     (7 images, 339MB)
│       │   ├── val/       (1 image, 52MB)
│       │   └── test/      (3 images, 102MB)
│       ├── labels/
│       │   ├── train/     (7 .txt annotation files)
│       │   ├── val/       (1 .txt annotation file)
│       │   └── test/      (3 .txt annotation files)
│       └── data.yaml      ← YOLO training configuration
│
├── 📁 model/                            ← EMPTY (Ready for trained models)
│   ├── original-models/
│   └── training-models/
│
└── 📁 dataset/extent-dataset/           ← Alternative dataset format
```

---

## Input Data

### Vector Data (GeoPackages)
- **`extent.gpkg`**: Contains tile extents (spatial boundaries)
  - Layer: "extent" (can be queried with `inspect_gpkg.py`)
  - Geometry: Polygon
  - CRS: EPSG:3857 (Web Mercator)

- **`karto.gpkg`**: Contains building data
  - Layer: "buildings" (or multiple building-related layers)
  - Geometry: Polygon
  - CRS: EPSG:3857 (Web Mercator)
  - Contains all building footprints in the region

### Raster Data (Imagery)
- **Source**: ArcGIS MapServer (GeoBizkaia ORTO_EJ_2024)
  - Orthographic aerial imagery
  - Downloaded on-demand for each extent
  - Resolution: Custom (default 4096×4096 pixels)
  - Format: PNG → Georeferenced GeoTIFF
  - CRS: EPSG:3857

---

## Data Processing Pipeline

### Step 1: Data Loading
```
Load extents (tile boundaries) from extent.gpkg
Load buildings (polygons) from karto.gpkg
Ensure both use same CRS (EPSG:3857)
```

### Step 2: Imagery Download
```
For each extent:
  1. Get extent bounds (minx, miny, maxx, maxy)
  2. Call ArcGIS MapServer export API
  3. Download PNG imagery
  4. Convert PNG to georeferenced GeoTIFF with transforms
  5. Store in dataset/yolo_buildings/images/
```

### Step 3: Building to YOLO Conversion
```
For each extent:
  1. Find all buildings intersecting the extent
  2. For each building polygon:
     a. Get bounding box bounds
     b. Transform from map coordinates → pixel coordinates
     c. Calculate center point and dimensions
     d. Normalize to [0, 1] range
     e. Format as YOLO annotation: "0 x_center y_center width height"
  3. Store annotations in dataset/yolo_buildings/labels/
```

### Step 4: Train/Val/Test Split
```
Default split (customizable):
- Training:   70% of images
- Validation: 15% of images
- Test:       15% of images

Files are randomly shuffled and moved to appropriate subdirectories
```

### Complete Data Flow
```
karto.gpkg (buildings)       extent.gpkg (tile extents)
      ↓                             ↓
  Buildings                      Extents
      ↓                             ↓
      └────────────┬────────────────┘
                   ↓
         Download Imagery from ArcGIS
                   ↓
    Georeferenced GeoTIFF (4096×4096)
                   ↓
    Intersect Buildings with Extent
                   ↓
  Polygon → Pixel Coordinates → Normalize [0,1]
                   ↓
      YOLO Annotations (.txt files)
                   ↓
        Train/Val/Test Split
                   ↓
     Ready for YOLOv8 Training
```

---

## Output Dataset Structure

### Format: YOLO Object Detection
- **Images**: GeoTIFF format (georeferenced, 4096×4096 pixels)
- **Annotations**: Text files, one per image
- **Configuration**: data.yaml for YOLOv8

### Dataset Statistics (Current)
```
Total: 11 images (492 MB)
├── Training:   7 images (339 MB) - 63.6%
├── Validation: 1 image  (52 MB)  - 9.1%
└── Testing:    3 images (102 MB) - 27.3%

Total Buildings Annotated: Varies per image
Average: ~75-100 buildings per image
```

### YOLO Annotation Format
Each `.txt` file contains building annotations:
```
<class_id> <x_center> <y_center> <width> <height>

Example:
0 0.512 0.623 0.128 0.156    ← Building 1
0 0.784 0.445 0.095 0.112    ← Building 2
0 0.234 0.567 0.089 0.102    ← Building 3
```

Where:
- `class_id`: 0 (buildings only - single class)
- `x_center`, `y_center`: Normalized center coordinates [0, 1]
- `width`, `height`: Normalized dimensions [0, 1]

### Configuration File (data.yaml)
```yaml
path: /path/to/yolo_buildings
train: images/train
val: images/val
test: images/test

nc: 1                    # Number of classes
names:
  0: building           # Class name

train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

---

## Coordinate Transformation (Technical Details)

### Transformation Chain
1. **Map Space** (EPSG:3857, meters):
   - Building polygons with coordinates in meters
   - Extent boundaries in meters

2. **Image/Pixel Space**:
   - Georeferenced imagery with affine transform
   - Pixels [0, image_width] × [0, image_height]

3. **Normalized YOLO Space**:
   - Coordinates normalized to [0, 1]
   - Used for model training

### Key Transformations
```python
# Map → Pixel using affine transform
col, row = ~transform * (x_map, y_map)

# Pixel → Normalized
x_norm = col / image_width
y_norm = row / image_height
```

For detailed mathematical explanation, see: `COORDINATE_TRANSFORMATION.md`

---

## Core Python Scripts

### 1. YOLODataBuilder (yolo_data_builder.py)
Main orchestration class with methods:

```python
class YOLODataBuilder:
    __init__(imagery_url, extents_gpkg_path, extents_layer_name, 
             buildings_gpkg_path, bui
