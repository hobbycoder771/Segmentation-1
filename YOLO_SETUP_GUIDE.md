# YOLO Building Detection Training Environment Setup

## Overview

This project sets up a complete pipeline to download building imagery and create YOLO training data for building detection from GeoBizkaia geospatial data.

## Directory Structure

```
project/
├── data/
│   ├── imagery/          # Downloaded satellite/orthophoto images
│   └── vector/
│       ├── carto/        # karto.gpkg - building footprints
│       └── extents/      # extent.gpkg - tile extents
├── dataset/
│   └── yolo_buildings/   # YOLO format dataset (output)
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
└── src/
    └── geobizkaia/
        ├── yolo_data_builder.py    # Main YOLO data builder
        └── inspect_gpkg.py          # Geopackage inspection utility
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_yolo.txt
```

### 2. Inspect Your Geopackages

First, determine the correct layer names in your geopackages:

```bash
cd src/geobizkaia
python inspect_gpkg.py ../../data/vector/extents/extent.gpkg
python inspect_gpkg.py ../../data/vector/carto/karto.gpkg
```

This will show you:
- Available layers
- Geometry types
- CRS (coordinate reference systems)
- Feature counts
- Column names

**Important**: Note the layer names as you'll need them in the next step.

### 3. Configure and Run the Pipeline

Edit the `yolo_data_builder.py` script and update the layer names based on your inspection:

```python
builder = YOLODataBuilder(
    imagery_url="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2024/MapServer/export",
    extents_gpkg_path="../../data/vector/extents/extent.gpkg",
    extents_layer_name="extent",              # ← Update with actual layer name
    buildings_gpkg_path="../../data/vector/carto/karto.gpkg",
    buildings_layer_name="buildings",         # ← Update with actual layer name
    output_dir="../../dataset/yolo_buildings"
)
```

### 4. Test with Limited Data

Start by processing a few extents to validate the workflow:

```python
# In yolo_data_builder.py
builder.run_pipeline(limit=5, split=False)
```

### 5. Full Pipeline

Once you've validated the test run:

```python
# Remove the limit to process all extents
builder.run_pipeline(split=True)
```

## YOLO Format Overview

### Image Files

- Downloaded from ArcGIS MapServer export service
- Georeferenced GeoTIFF format (4096x4096 pixels by default)
- Located in `images/train`, `images/val`, `images/test`

### Annotation Files (`.txt`)

YOLO format: `<class_id> <x_center> <y_center> <width> <height>`

All values are **normalized** to [0, 1]:
- `class_id`: 0 = building
- `x_center`, `y_center`: normalized center coordinates
- `width`, `height`: normalized dimensions

Example:
```
0 0.512 0.623 0.128 0.156
0 0.784 0.445 0.095 0.112
```

### data.yaml

Configuration file for YOLO training:
```yaml
path: /path/to/dataset/yolo_buildings
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: building
```

## Data Flow

```
extent.gpkg (tile extents)
    ↓
ArcGIS MapServer API
    ↓
Download 4096×4096 imagery
    ↓
Georeferenced GeoTIFF
    ↓
karto.gpkg (building polygons)
    ↓
Intersect buildings with extent
    ↓
Convert polygon → pixel coordinates
    ↓
Normalize coordinates [0,1]
    ↓
YOLO annotations (.txt files)
    ↓
Train/Val/Test split
    ↓
Ready for YOLO training
```

## Training with YOLOv8

Once your dataset is ready, you can train a YOLO model:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano model

# Train
results = model.train(
    data='dataset/yolo_buildings/data.yaml',
    epochs=100,
    imgsz=4096,
    device=0,  # GPU device (0 for first GPU)
    patience=20,
    save=True,
    name='buildings_detection'
)
```

## Troubleshooting

### Common Issues

**Issue**: Layer not found in geopackage
- **Solution**: Run `inspect_gpkg.py` to see available layers and update layer names

**Issue**: No buildings found in extent
- **Solution**: Check if buildings CRS matches extent CRS. Script auto-reprojects if needed.

**Issue**: API errors downloading imagery
- **Solution**: Check internet connection and verify ArcGIS MapServer URL is accessible

**Issue**: Memory issues with large imagery
- **Solution**: Reduce `image_size` parameter or process fewer extents with `limit` parameter

## Performance Notes

- Default image size: 4096×4096 pixels (adjustable)
- Typical processing time per extent: 10-30 seconds (depending on download speed)
- Default train/val/test split: 70%/15%/15%

## Next Steps

1. Run `inspect_gpkg.py` to identify your layer names
2. Update the configuration in `yolo_data_builder.py`
3. Test with `limit=5`
4. Process full dataset with `split=True`
5. Train YOLOv8 model using the generated `data.yaml`

## References

- [YOLO Format Documentation](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
- [YOLOv8 Training](https://docs.ultralytics.com/modes/train/)
- [GeoPandas Documentation](https://geopandas.org/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
