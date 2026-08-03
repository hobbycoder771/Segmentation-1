# YOLO Building Detection Training Environment

## Summary

I've created a complete pipeline for building YOLO training datasets from GeoBizkaia geospatial data. The system downloads building imagery from ArcGIS MapServer and converts building vector data into YOLO-format annotations.

## What Was Created

### Core Scripts

1. **`yolo_data_builder.py`** - Main class that orchestrates the entire pipeline:
   - Downloads imagery from ArcGIS MapServer export API
   - Converts PNG to georeferenced GeoTIFF
   - Intersects buildings with tile extents
   - Converts building polygons to YOLO annotations
   - Splits dataset into train/val/test sets
   - Creates `data.yaml` configuration

2. **`run_yolo_pipeline.py`** - Command-line interface for easy execution:
   - `--test`: Quick 3-extent test
   - `--full`: Process all extents with splitting
   - `--limit N`: Process N extents
   - `--split`: Enable train/val/test splitting
   - Custom train/val ratios

3. **`inspect_gpkg.py`** - Utility to inspect geopackages:
   - Lists all layers
   - Shows geometry types, CRS, feature counts
   - Displays sample data and column names

### Configuration Files

- **`requirements_yolo.txt`** - All Python dependencies
- **`QUICKSTART.md`** - 5-minute setup guide
- **`YOLO_SETUP_GUIDE.md`** - Detailed documentation
- **`README_YOLO.md`** - This file

## Quick Start

### 1. Install

```bash
pip install -r requirements_yolo.txt
```

### 2. Identify Your Layer Names

```bash
cd src/geobizkaia
python inspect_gpkg.py ../../data/vector/extents/extent.gpkg
python inspect_gpkg.py ../../data/vector/carto/karto.gpkg
```

### 3. Test (3 extents)

```bash
python run_yolo_pipeline.py --test
```

### 4. Full Pipeline

```bash
python run_yolo_pipeline.py --full
```

## Data Flow

```
Vector Data (karto.gpkg)           Tile Extents (extent.gpkg)
        ↓                                    ↓
    Buildings                         Extents/Tiles
        ↓                                    ↓
        └─────────────┬─────────────────────┘
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

## Output Structure

```
dataset/yolo_buildings/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## YOLO Format

Each building is represented as:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]:

- `class_id`: 0 (buildings only)
- `x_center`, `y_center`: normalized center of building
- `width`, `height`: normalized dimensions

Example:

```
0 0.512 0.623 0.128 0.156
0 0.784 0.445 0.095 0.112
```

## Configuration Options

Update layer names in `run_yolo_pipeline.py`:

```python
parser.add_argument("--extents-layer", default="extent")      # Your layer name
parser.add_argument("--buildings-layer", default="buildings") # Your layer name
```

Or pass via command line:

```bash
python run_yolo_pipeline.py --full --extents-layer my_extent --buildings-layer my_buildings
```

## Dataset Customization

```bash
# 80/10/10 split
python run_yolo_pipeline.py --full --train-ratio 0.8 --val-ratio 0.1

# Smaller images (2048×2048)
python run_yolo_pipeline.py --full --image-size 2048

# 5 extents for testing
python run_yolo_pipeline.py --limit 5 --split
```

## Training with YOLOv8

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano/small/medium/large/xlarge

results = model.train(
    data='dataset/yolo_buildings/data.yaml',
    epochs=100,
    imgsz=4096,
    device=0,
    patience=20,
    save=True,
    name='buildings_detection'
)
```

## Key Features

**Geospatial Integration**: Works with geopackages and coordinate systems
**ArcGIS Integration**: Automatic imagery download from MapServer
**Automatic Georeferencing**: PNG to GeoTIFF conversion with transforms
**CRS Handling**: Automatic reprojection if needed
**Coordinate Transformation**: Building polygons → pixel coordinates → normalized YOLO format
**Dataset Splitting**: Automatic train/val/test organization
**Easy CLI**: Simple command-line interface for all operations
**Logging**: Detailed progress tracking

## Troubleshooting

| Issue                  | Solution                                          |
| ---------------------- | ------------------------------------------------- |
| Layer not found        | Run `inspect_gpkg.py` to see actual layer names   |
| No buildings in extent | Check CRS match or extent boundaries              |
| Download failures      | Check internet, verify ArcGIS URL                 |
| Memory issues          | Use `--limit` or reduce `--image-size`            |
| API errors             | Verify coordinates are in correct CRS (EPSG:3857) |

## Files Overview

| File                    | Lines | Purpose                   |
| ----------------------- | ----- | ------------------------- |
| `yolo_data_builder.py`  | 350+  | Core YOLO dataset builder |
| `run_yolo_pipeline.py`  | 170+  | CLI runner                |
| `inspect_gpkg.py`       | 80+   | Geopackage inspector      |
| `requirements_yolo.txt` | 20+   | Dependencies              |

## Learning Resources

- [YOLO Format Docs](https://docs.ultralytics.com/datasets/detect/)
- [YOLOv8 Training](https://docs.ultralytics.com/modes/train/)
- [GeoPandas](https://geopandas.org/)
- [Rasterio](https://rasterio.readthedocs.io/)

## Next Steps

1. Install dependencies
2. Run `inspect_gpkg.py` to identify layers
3. Update layer names if needed
4. Test with `--test` flag
5. Run `--full` pipeline
6. Train YOLOv8 model

---

**Ready to build your building detection model!**
