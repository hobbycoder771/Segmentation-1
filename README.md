# YOLO Building Segmentation Pipeline

Convert building polygons to pixel-level segmentation masks and train YOLOv8-Seg models.

## Quick Start

### 1. Generate Segmentation Dataset

```bash
cd src/geobizkaia

# Test with 3 extents
python run_segmentation_pipeline.py --test

# Full dataset with train/val/test split
python run_segmentation_pipeline.py --full
```

### 2. Train Model

```bash
# Nano model (fast, lower accuracy)
python train_segmentation_model.py --model n --epochs 100 --imgsz 4096

# Medium model (balanced)
python train_segmentation_model.py --model m --epochs 100 --imgsz 4096

# Large model (slower, higher accuracy)
python train_segmentation_model.py --model l --epochs 100 --imgsz 4096
```

### 3. Results

- **Model saved to**: `model/best_yolov8{n,s,m,l,x}-seg.pt`
- **Training log**: `training_seg.log`
- **Dataset**: `dataset/yolo_buildings_seg/`

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

### Model Training

```bash
# Quick test (nano model, 10 epochs)
python train_segmentation_model.py --model n --epochs 10

# Production (medium model, 100 epochs)
python train_segmentation_model.py --model m --epochs 100 --imgsz 4096

# Resume from checkpoint
python train_segmentation_model.py --resume /path/to/checkpoint.pt

# Validation only
python train_segmentation_model.py --validate-only

# Custom batch size and device
python train_segmentation_model.py --model m --batch 8 --device 0
```

## Model Sizes

| Size | Parameters | Speed | Accuracy |
|------|-----------|-------|----------|
| nano (n) | 3.3M | Fast ⚡ | Lower |
| small (s) | 11.6M | Medium | Medium |
| medium (m) | 26.9M | Medium | High |
| large (l) | 43.7M | Slow | Very High |
| xlarge (x) | 68.2M | Slowest | Best |

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

## Documentation

- **SEGMENTATION_GUIDE.md** - Comprehensive user guide with examples
- **SEGMENTATION_IMPLEMENTATION_SUMMARY.md** - Technical architecture details
- **IMPLEMENTATION_COMPLETE.txt** - Quick reference and next steps

## Key Differences: Detection vs Segmentation

| Aspect | Detection | Segmentation |
|--------|-----------|--------------|
| Output | Bounding boxes | Pixel-level masks |
| Format | `.txt` coordinates | `.png` binary images |
| Model | YOLOv8 detect | YOLOv8-Seg |
| Values | Center, width, height | 0/255 binary mask |

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

1. ✅ Read SEGMENTATION_GUIDE.md
2. ✅ Generate dataset: `python run_segmentation_pipeline.py --test`
3. ✅ Train model: `python train_segmentation_model.py --model n --epochs 10`
4. ✅ Review metrics and saved model

## Support

For issues:
1. Check SEGMENTATION_GUIDE.md troubleshooting section
2. Review training logs: `tail -f training_seg.log`
3. Inspect data: `python inspect_gpkg.py`

---

**Ready to segment!** Start with: `python run_segmentation_pipeline.py --test`
