# Continuous Training - Model Learning Over Time

## Overview

The updated `train_yolo_model.py` now supports **continuous training**, where a single persistent model improves with each training session as it learns from new imagery. This means:

- **First run**: Creates a new model and trains it
- **Second run**: Loads the best model from run 1 and fine-tunes it with new data
- **Third run**: Loads the improved model from run 2 and continues learning
- **And so on...**

The model grows in accuracy over time as you add more training data and run the training script.

## How It Works

### Architecture

```
model/
├── best_yolov8n.pt          ← Persistent best model (tracked in git)
├── training_history_yolov8n.json  ← Training history (tracked in git)
└── runs/                    ← All training runs (NOT tracked - too large)
    ├── train_yolov8n_20260804_100000/
    ├── train_yolov8n_20260804_110000/
    └── ... (temporary, can be deleted)
```

### Training Flow

```
1. First run: Load pre-trained yolov8n.pt
   ↓
2. Train for 100 epochs on new data
   ↓
3. Save best model to: model/best_yolov8n.pt
   ↓
4. Record training session in: training_history_yolov8n.json

---

5. Second run: Load model/best_yolov8n.pt
   ↓
6. Train for 100 more epochs on new data (150 total runs images)
   ↓
7. Update model/best_yolov8n.pt with new best weights
   ↓
8. Update training history
```

## Usage

### Basic Continuous Training (Default)

```bash
# First training session
python train_yolo_model.py --model n --epochs 100

# Model is saved to: model/best_yolov8n.pt
# Results in: model/runs/train_yolov8n_20260804_100000/

# Second training session (with more data)
python train_yolo_model.py --model n --epochs 100

# Automatically loads model/best_yolov8n.pt and fine-tunes it
# Results in: model/runs/train_yolov8n_20260804_110000/
# Updates: model/best_yolov8n.pt with new best weights
```

### Typical Workflow

```bash
# 1. Generate first batch of training data
python run_yolo_pipeline.py --full --enable-tiling --tile-size 640 --split

# 2. Train model (first run - creates best_yolov8n.pt)
python train_yolo_model.py --model n --epochs 100

# 3. Download more imagery / generate more training data
# (Update your source data and re-run pipeline with --full)
python run_yolo_pipeline.py --full --enable-tiling --tile-size 640 --split

# 4. Continue training (loads best_yolov8n.pt and fine-tunes)
python train_yolo_model.py --model n --epochs 100

# Model continues improving!
```

## Command Reference

### Continuous Training Mode (Default)

```bash
# Enable continuous training explicitly (same as default)
python train_yolo_model.py --continuous --model n --epochs 100

# View training history
python train_yolo_model.py --show-history

# Continue training with medium model
python train_yolo_model.py --model m --epochs 100
```

### Fresh Training (Start Over)

```bash
# Disable continuous training - always start fresh
python train_yolo_model.py --no-continuous --model n --epochs 100

# This WILL NOT load model/best_yolov8n.pt
# Instead loads fresh pre-trained yolov8n.pt
```

### Manual Resume

```bash
# Resume from a specific checkpoint (overrides continuous training)
python train_yolo_model.py --resume path/to/runs/train_yolov8n_20260804_100000/weights/last.pt --epochs 50

# Continue from a specific checkpoint for another 50 epochs
```

### Validation & Testing

```bash
# Validate the persistent best model
python train_yolo_model.py --validate-only --model n

# Test the persistent best model
python train_yolo_model.py --test-only model/best_yolov8n.pt

# Show complete training history
python train_yolo_model.py --show-history
```

## Training History

The system tracks all training sessions in JSON format:

### File: `model/training_history_yolov8n.json`

```json
{
  "total_training_runs": 3,
  "total_epochs_trained": 300,
  "training_sessions": [
    {
      "run_number": 1,
      "date": "2026-08-04T10:00:00",
      "epochs": 100,
      "results_dir": "../../model/runs/train_yolov8n_20260804_100000"
    },
    {
      "run_number": 2,
      "date": "2026-08-04T11:30:00",
      "epochs": 100,
      "results_dir": "../../model/runs/train_yolov8n_20260804_113000"
    },
    {
      "run_number": 3,
      "date": "2026-08-04T13:00:00",
      "epochs": 100,
      "results_dir": "../../model/runs/train_yolov8n_20260804_130000"
    }
  ]
}
```

View it with:
```bash
python train_yolo_model.py --show-history
```

## Model Selection

Each model size has its own persistent checkpoint:

```bash
# Nano model (smallest, fastest)
python train_yolo_model.py --model n --epochs 100
# Saved to: model/best_yolov8n.pt

# Small model
python train_yolo_model.py --model s --epochs 100
# Saved to: model/best_yolov8s.pt

# Medium model (recommended for building detection)
python train_yolo_model.py --model m --epochs 100
# Saved to: model/best_yolov8m.pt

# Large model (higher accuracy, slower)
python train_yolo_model.py --model l --epochs 100
# Saved to: model/best_yolov8l.pt

# Extra-large model (best accuracy, most resources)
python train_yolo_model.py --model x --epochs 100
# Saved to: model/best_yolov8x.pt
```

Each model maintains its own training history and persistent weights.

## Git Integration

### What's Tracked

```
model/
├── best_yolov8n.pt          ✓ TRACKED
├── best_yolov8m.pt          ✓ TRACKED
├── training_history_yolov8n.json  ✓ TRACKED
└── runs/                    ✗ NOT TRACKED
```

### Why?

- **Persistent models** (best_yolov8*.pt) are relatively small (~100-200MB) and essential
- **Training histories** are small JSON files that track progress
- **Training runs** are large and temporary (can be recreated by training again)

### Workflow

```bash
# Train and improve model
python train_yolo_model.py --model m --epochs 100

# See what changed
git status
# Output:
# modified: model/best_yolov8m.pt
# modified: model/training_history_yolov8m.json

# Commit improvements
git add model/best_yolov8m.pt model/training_history_yolov8m.json
git commit -m "Improved model after training with new Bizkaia dataset (300 epochs total)"

# Share with team
git push
```

## Tips for Effective Continuous Training

### 1. Expand Dataset Incrementally

```bash
# Week 1: 100 extents
python run_yolo_pipeline.py --full --enable-tiling
python train_yolo_model.py --model m --epochs 50

# Week 2: +100 new extents (200 total)
python run_yolo_pipeline.py --full --enable-tiling
python train_yolo_model.py --model m --epochs 50  # Fine-tunes with new data

# Week 3: +100 new extents (300 total)
python run_yolo_pipeline.py --full --enable-tiling
python train_yolo_model.py --model m --epochs 50  # Further improved
```

### 2. Monitor Progress

```bash
# Check training history
python train_yolo_model.py --show-history

# Compare performance across runs
# Look at results in model/runs/train_yolov8m_*/ directories
```

### 3. Use Appropriate Batch Sizes

```bash
# Smaller batch for fine-tuning on new data
python train_yolo_model.py --model m --batch 4 --epochs 50

# Or larger batch if you have more GPU memory
python train_yolo_model.py --model m --batch 16 --epochs 50
```

### 4. Early Stopping

The script has built-in early stopping (default patience: 20 epochs). Training stops if validation metrics don't improve for 20 consecutive epochs.

```bash
# Adjust patience if needed
python train_yolo_model.py --model m --patience 30 --epochs 100
```

## Comparing with Previous Approach

### Before (New training each time)
```bash
python train_yolo_model.py  # Always starts fresh
└─ Loads: yolov8n.pt (pre-trained)
└─ Result: model/runs/train_yolov8n_20260804_100000/ (isolated)
└─ Model never improves: Always fresh weights
```

### After (Continuous improvement)
```bash
python train_yolo_model.py  # Loads and improves
└─ Loads: model/best_yolov8n.pt (previous best)
└─ Trains: With new data
└─ Result: model/runs/train_yolov8n_20260804_110000/ (new results)
└─ Updates: model/best_yolov8n.pt (better weights)
└─ Model continuously improves
```

## Troubleshooting

### "No persistent model found"

**Symptoms:** Training always seems to start fresh

**Solution:**
```bash
# Check if best model exists
ls -lh model/best_yolov8n.pt

# If not found, first training creates it
python train_yolo_model.py --model n --epochs 50
```

### Want to start fresh with a model

```bash
# Disable continuous training for this run only
python train_yolo_model.py --no-continuous --model n --epochs 100

# Or manually delete and restart (CAUTION!)
rm model/best_yolov8n.pt model/training_history_yolov8n.json
python train_yolo_model.py --model n --epochs 100
```

### Model file is corrupted

```bash
# Delete and retrain
rm model/best_yolov8n.pt
python train_yolo_model.py --model n --epochs 100
```

## Performance Expectations

With continuous training and incremental data:

| Training Runs | Total Epochs | Total Data | Expected Improvement |
|---------------|-------------|-----------|----------------------|
| 1 | 100 | 1000 images | Baseline |
| 2 | 200 | 2000 images | 5-15% accuracy gain |
| 3 | 300 | 3000 images | 8-20% accuracy gain |
| 5 | 500 | 5000 images | 15-30% accuracy gain |

Results depend on data quality and diversity.
