# YOLO Building Segmentation Pipeline - Progressive Training

Convert building polygons to pixel-level segmentation masks and train YOLOv8-Seg models incrementally as new imagery becomes available.

## Quick Start - Progressive Training Workflow

### 1. Initial Training (First Time)

```bash
cd src/geobizkaia

# Generate dataset
python run_segmentation_pipeline.py --test

# Train model (automatically starts from pre-trained YOLOv8-Seg)
python train_segmentation_model.py --model m --epochs 50
```

### 2. Progressive Training (When New Imagery Arrives)

```bash
# Add new imagery to dataset/yolo_buildings_seg/images/ and dataset/yolo_buildings_seg/masks/

# Run training again - automatically continues from the best previous model
python train_segmentation_model.py --model m --epochs 50
```

The training will:
- **Automatically load** the best model from the previous training session
- **Continue training** on the new imagery
- **Update** `model/best_yolov8m-seg.pt` with the improved model
- **Track** all training sessions in `model/training_history_yolov8m-seg.json`

### 3. Results

- **Persistent Model**: `model/best_yolov8{n,s,m,l,x}-seg.pt` - Updated after each training
- **Training History**: `model/training_history_yolov8{size}-seg.json` - Tracks all training sessions
- **Dataset**: `dataset/yolo_buildings_seg/` - Add new data here for progressive training

## Pipeline Overview

```
Building Polygons (GeoPackage)
        ↓
  [MaskGenerator]
        ↓
Binary PNG Masks (0/255)
        ↓
   [Dataset Split]
   train/val/test
        ↓
[YOLOv8-Seg Training]
        ↓
Trained Segmentation Model
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

### Data Generation

```bash
# Test mode (3 extents, no split)
python run_segmentation_pipeline.py --test

# Full mode (all extents, with split)
python run_segmentation_pipeline.py --full

# Custom: 10 extents with split
python run_segmentation_pipeline.py --limit 10 --split

# Custom image size
python run_segmentation_pipeline.py --full --image-size 2048

# Without FeatureServer queries
python run_segmentation_pipeline.py --test --no-vector-clipping
```

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

### Workflow

1. **First Training Session**
   - Load pre-trained YOLOv8-Seg model
   - Train on initial dataset
   - Save best model to `model/best_yolov8m-seg.pt`
   - Record session in `model/training_history_yolov8m-seg.json`

2. **Next Training Session (New Imagery)**
   - **Automatically load** `model/best_yolov8m-seg.pt` (previous best model)
   - Train on old + new imagery combined
   - Improve on previous weights
   - Update persistent model with new best
   - Record new session in history

3. **Repeat as Needed**
   - Each run builds on the previous best model
   - Model improves incrementally with new data
   - No manual intervention needed

### Example Timeline

```
Training Run 1: 0% → 75% accuracy → Saved to model/best_yolov8m-seg.pt
Training Run 2: 75% → 82% accuracy → Updated model/best_yolov8m-seg.pt
Training Run 3: 82% → 87% accuracy → Updated model/best_yolov8m-seg.pt
```

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
