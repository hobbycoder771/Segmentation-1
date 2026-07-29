# Quick Start Guide - YOLO Building Detection

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements_yolo.txt
```

### Step 2: Identify Layer Names
```bash
cd src/geobizkaia
python inspect_gpkg.py ../../data/vector/extents/extent.gpkg
python inspect_gpkg.py ../../data/vector/carto/karto.gpkg
```

Note the layer names you see (e.g., "extent", "buildings", "karto", etc.)

### Step 3: Update Configuration

Edit `src/geobizkaia/run_yolo_pipeline.py` and update these lines if your layer names differ:

```python
parser.add_argument(
    "--extents-layer",
    default="extent",           # ← Update if different
    help="Layer name for extents in geopackage"
)
parser.add_argument(
    "--buildings-layer",
    default="buildings",        # ← Update if different
    help="Layer name for buildings in geopackage"
)
```

### Step 4: Test with 3 Extents
```bash
cd src/geobizkaia
python run_yolo_pipeline.py --test
```

### Step 5: Process Full Dataset
```bash
python run_yolo_pipeline.py --full
```

## Command Examples

```bash
# Test mode (3 extents, no splitting)
python run_yolo_pipeline.py --test

# Full processing with splitting
python run_yolo_pipeline.py --full

# Custom: process 10 extents with 80/10/10 split
python run_yolo_pipeline.py --limit 10 --split --train-ratio 0.8 --val-ratio 0.1

# Custom image size and different layers
python run_yolo_pipeline.py --full --image-size 2048 --extents-layer my_extent --buildings-layer my_buildings

# Get help
python run_yolo_pipeline.py --help
```

## Output Structure

After running the pipeline:
```
dataset/
└── yolo_buildings/
    ├── images/
    │   ├── train/     (70% of images)
    │   ├── val/       (15% of images)
    │   └── test/      (15% of images)
    ├── labels/
    │   ├── train/     (corresponding labels)
    │   ├── val/
    │   └── test/
    └── data.yaml      (YOLO config file)
```

## Train YOLO Model

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano, small, medium, large, xlarge

results = model.train(
    data='dataset/yolo_buildings/data.yaml',
    epochs=100,
    imgsz=4096,
    device=0,
    patience=20
)
```

## Troubleshooting

**Q: "Layer not found" error**
- A: Run `inspect_gpkg.py` to see actual layer names and update the configuration

**Q: No buildings detected in extents**
- A: Verify buildings and extents use same CRS, or check if buildings are within extent boundaries

**Q: Download failures**
- A: Check internet connection, verify ArcGIS URL is accessible

**Q: Memory issues**
- A: Reduce `--image-size` or use `--limit` to process fewer extents

## Key Files

| File | Purpose |
|------|---------|
| `yolo_data_builder.py` | Core YOLO dataset builder class |
| `run_yolo_pipeline.py` | Command-line runner for pipeline |
| `inspect_gpkg.py` | Utility to inspect geopackage structure |
| `requirements_yolo.txt` | Python dependencies |
| `YOLO_SETUP_GUIDE.md` | Detailed documentation |

## Next Steps

1. ✅ Install dependencies
2. ✅ Identify layer names with `inspect_gpkg.py`
3. ✅ Test with `--test` flag
4. ✅ Run full pipeline with `--full` flag
5. 🔄 Train YOLOv8 model on generated dataset

Good luck! 
