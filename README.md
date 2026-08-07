# YOLO Building Segmentation Pipeline - Progressive Training

Convert building polygons to pixel-level segmentation masks and train YOLOv8-Seg models incrementally as new imagery becomes available.

## Quick Start - Progressive Training Workflow

### Complete Workflow (Repeat Each Cycle with Different Areas)

```bash
cd src/geobizkaia

# 1. Define new extents (geographic areas)
#    Edit data/vector/extents/extent.gpkg with new areas (replaces previous extents)

# 2. Generate dataset from those extents
#    Downloads fresh imagery and generates masks for all extents
python run_segmentation_pipeline.py --full

# 3. Retrain model on the new dataset
#    The model automatically loads from the previous best model and continues learning
python train_segmentation_model.py --model m --epochs 50
```

### What Happens at Each Step

**Step 1 - Define New Extents:**
- Replace contents of `data/vector/extents/extent.gpkg` with new geographic areas
- Extents are not accumulated, they are replaced each cycle
- These define which areas to download imagery for

**Step 2 - Generate Fresh Dataset:**
- Downloads fresh imagery from ArcGIS service for the defined extents
- Clips vector data (buildings) from `data/vector/carto/karto.gpkg`
- Generates segmentation masks
- **Completely overwrites** `dataset/yolo_buildings_seg/` with new data from new extents
- Splits data into train/val/test sets

**Step 3 - Retrain Model:**
- **Automatically loads** best model from previous training session
- Trains on the newly generated dataset (different geographic area)
- Applies knowledge learned from previous areas to new area
- **Updates** `model/best_yolov8m-seg.pt` with improved model
- **Tracks** all training sessions in `model/training_history_yolov8m-seg.json`

### Results After Each Cycle

- **Persistent Model**: `model/best_yolov8{n,s,m,l,x}-seg.pt` - Progressively improved across different areas
- **Training History**: `model/training_history_yolov8{size}-seg.json` - Shows all training runs and epochs
- **Dataset**: `dataset/yolo_buildings_seg/` - Fresh data from current extents (replaced each cycle)
- **Model Improvement**: Model learns from each new geographic area and generalizes better

## Pipeline Overview - Progressive Training Cycle

```
Cycle 1: Area A Data
1. Define Extents (Area A)
   ├─ Edit extent.gpkg with extents for Area A
   ↓
2. Generate Dataset
   ├─ Download imagery → Clip vectors → Generate masks
   ├─ Output: dataset/yolo_buildings_seg/ (Area A data)
   ↓
3. Train Model
   ├─ Load: Pre-trained YOLOv8-Seg
   ├─ Train: on Area A data
   └─ Save: model/best_yolov8m-seg.pt v1

Cycle 2: Area B Data (DIFFERENT from Area A)
1. Define New Extents (Area B - replace Area A)
   ├─ Edit extent.gpkg with extents for Area B
   ↓
2. Generate Fresh Dataset
   ├─ Download imagery → Clip vectors → Generate masks
   ├─ Output: dataset/yolo_buildings_seg/ (Area B data - overwrites Area A)
   ↓
3. Retrain Model
   ├─ Load: model/best_yolov8m-seg.pt v1 (from Area A)
   ├─ Train: on Area B data (applies knowledge from Area A)
   └─ Save: model/best_yolov8m-seg.pt v2 (improved)

Cycle 3+: Repeat with New Areas
   └─ Model progressively improves as it learns from different regions
```

## Directory Structure

```
src/geobizkaia/
├── mask_generator.py              # Polygon → PNG mask conversion
├── segmentation_metrics.py        # IoU, Dice, accuracy metrics
├── segmentation_data_builder.py   # Main data pipeline
├── train_segmentation_model.py    # Model training
└── run_segmentation_pipeline.py   # CLI interface

dataset/yolo_buildings_seg/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml

model/
├── best_yolov8n-seg.pt
├── best_yolov8m-seg.pt
└── training_history_yolov8*.json
```

## Command Examples

### Data Generation from Extents

```bash
# Generate dataset from ALL extents (recommended for retraining)
python run_segmentation_pipeline.py --full

# Test mode (first 3 extents, no train/val/test split)
python run_segmentation_pipeline.py --test

# Custom: Process specific number of extents with split
python run_segmentation_pipeline.py --limit 10 --split

# Custom image size (default: 4096)
python run_segmentation_pipeline.py --full --image-size 2048

# Without automatic vector clipping (if FeatureServer is unavailable)
python run_segmentation_pipeline.py --full --no-vector-clipping
```

**Important**: The script completely overwrites `dataset/yolo_buildings_seg/` - this is intentional. Each cycle you define new extents (replacing previous ones), and the pipeline generates a fresh dataset for those new areas.

### Model Training - Progressive Training Examples

```bash
# Initial training (first time with fresh dataset)
python train_segmentation_model.py --model m --epochs 50

# Progressive training (next time with new imagery - RECOMMENDED)
# Automatically loads the previous best model
python train_segmentation_model.py --model m --epochs 50

# Fresh training (ignore previous model, start over)
python train_segmentation_model.py --model m --epochs 50 --no-continuous

# Resume from specific checkpoint
python train_segmentation_model.py --model m --epochs 50 --resume /path/to/checkpoint.pt

# Show training history
python train_segmentation_model.py --show-history

# Validate existing model
python train_segmentation_model.py --validate-only

# Custom settings
python train_segmentation_model.py --model m --epochs 50 --batch 8 --imgsz 1024 --device 0
```

## Model Sizes

| Size | Parameters | Speed | Accuracy |
|------|-----------|-------|----------|
| nano (n) | 3.3M | Fast ⚡ | Lower |
| small (s) | 11.6M | Medium | Medium |
| medium (m) | 26.9M | Medium | High |
| large (l) | 43.7M | Slow | Very High |
| xlarge (x) | 68.2M | Slowest | Best |

## How Progressive Training Works

### Training Flow Across Cycles

Each cycle uses **new extent data** - extents are not accumulated, they are replaced.

1. **Cycle 1: Initial Training**
   - Define extents in `data/vector/extents/extent.gpkg` (e.g., Area A)
   - Run pipeline: Download imagery + generate masks for Area A
   - Create dataset: `dataset/yolo_buildings_seg/` (Area A data)
   - Train model: Load pre-trained YOLOv8-Seg
   - Save: `model/best_yolov8m-seg.pt` (v1)

2. **Cycle 2: New Extents, Retrain Model**
   - Replace extents in `data/vector/extents/extent.gpkg` (e.g., Area B - different from A)
   - Run pipeline: Download imagery + generate masks for Area B only
   - **Overwrites** `dataset/yolo_buildings_seg/` with Area B data
   - Train model: **Automatically loads** `model/best_yolov8m-seg.pt` (v1)
   - Trains on Area B using knowledge from Area A
   - Save: `model/best_yolov8m-seg.pt` (v2) - improved model

3. **Cycle 3+: Repeat with Different Areas**
   - Replace extents with new areas (e.g., Area C)
   - Run pipeline (overwrites dataset with Area C)
   - Run training (loads v2, trains on Area C, saves v3)
   - Model keeps improving even though datasets are different

### Example Timeline

```
Cycle 1: Area A (10 extents) → 75% accuracy → model v1 saved
Cycle 2: Area B (10 extents) → 82% accuracy → model v2 saved (loaded v1, trained on new area)
Cycle 3: Area C (10 extents) → 87% accuracy → model v3 saved (loaded v2, trained on another area)
```

### Key Points
- ✅ Dataset is **completely fresh** each cycle (different geographic areas)
- ✅ Extents are **replaced, not accumulated** (define new areas each time)
- ✅ Model **loads previous best** automatically (trains on new area with prior knowledge)
- ✅ Model **progressively improves** as it learns from different regions
- ✅ No manual model management needed

## Performance Tips

### Memory Issues?
```bash
# Reduce batch size
python train_segmentation_model.py --model m --batch 2

# Reduce image size
python train_segmentation_model.py --model m --imgsz 1024

# Use smaller model
python train_segmentation_model.py --model n
```

### Training Too Slow?
```bash
# Use smaller model
python train_segmentation_model.py --model n

# Reduce image size
python train_segmentation_model.py --imgsz 1024

# Increase batch size
python train_segmentation_model.py --batch 16
```

### Starting Fresh?
```bash
# Ignore previous model, start from pre-trained YOLOv8-Seg
python train_segmentation_model.py --model m --no-continuous --epochs 50
```

## Documentation

- **SEGMENTATION_GUIDE.md** - Comprehensive user guide with examples
- **SEGMENTATION_IMPLEMENTATION_SUMMARY.md** - Technical architecture details
- **IMPLEMENTATION_COMPLETE.txt** - Quick reference and next steps

## Progressive Training Features

| Feature | Description |
|---------|-------------|
| **Continuous Training** | Automatically loads previous best model for next training session |
| **Training History** | Tracks all training runs with timestamps and epochs |
| **Persistent Model** | Single best model file updated incrementally |
| **Automatic Checkpoint** | Best weights copied to persistent location after each training |
| **No Manual Steps** | Just run the script - it handles everything automatically |

## Troubleshooting

### Layer not found error
```bash
# Verify layer names
cd src/geobizkaia
python inspect_gpkg.py ../../data/vector/carto/karto.gpkg
```

### No masks generated
- Check that polygons are valid (non-empty, valid geometry)
- Verify imagery downloaded correctly
- Check mask statistics in logs

### Model training fails
- Check GPU memory: `nvidia-smi`
- Reduce batch size or image size
- Verify data.yaml paths are correct

## Next Steps

1. ✅ Generate initial dataset: `python run_segmentation_pipeline.py --test`
2. ✅ Start training: `python train_segmentation_model.py --model m --epochs 50`
3. ✅ When new imagery arrives, add it to `dataset/yolo_buildings_seg/`
4. ✅ Run training again: `python train_segmentation_model.py --model m --epochs 50`
5. ✅ Check training history: `python train_segmentation_model.py --show-history`

## Support

For issues:
1. Check SEGMENTATION_GUIDE.md troubleshooting section
2. Review training logs: `tail -f training_seg.log`
3. Inspect data: `python inspect_gpkg.py`

---

**Ready to segment!** Start with: `python run_segmentation_pipeline.py --test`
