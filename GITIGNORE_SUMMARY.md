# Git Ignore Configuration Summary

**File**: `.gitignore`  
**Date**: August 6, 2026  
**Purpose**: Prevent large/temporary files from being tracked in Git

## What is IGNORED (Not tracked in Git)

### 🖼️ Image Files
All image formats are excluded to avoid large repository size:
- `*.tif`, `*.tiff` (GeoTIFF raster data)
- `*.png`, `*.jpg`, `*.jpeg` (segmentation masks, outputs)
- `*.gif`, `*.bmp`, `*.webp` (other formats)

**Size saved**: ~50MB per 4096×4096 image

### 📦 Dataset Directories
- `dataset/` - All generated datasets (images, masks, labels)
- `data/vector/extents/` - GeoPackage extents
- `data/vector/carto/` - Building polygon data

**Size saved**: ~500MB+ per full dataset

### 🤖 Model Files
- `model/` - Trained weights (*.pt, *.pth, *.onnx)
- `runs/` - YOLOv8 training outputs

**Size saved**: ~50-500MB per trained model

### 📊 Training Outputs
- `*.log` - Training logs
- `training_history_*.json` - Metrics
- `logs/` - Debug logs

### 🐍 Python Cache
- `__pycache__/` - Compiled Python files
- `*.egg-info/` - Package info
- `.pytest_cache/` - Test cache

### 🔧 IDE & Tools
- `.vscode/`, `.idea/` - IDE settings
- `.DS_Store` - macOS files
- `Thumbs.db` - Windows thumbnails

## What IS Tracked (In Git)

### ✅ Source Code
- `*.py` - All Python scripts
- `src/` - Source directory

### ✅ Documentation
- `*.md` - Markdown guides and READMEs
- `.gitignore` - This file

### ✅ Configuration
- `*.yaml`, `*.yml` - Configuration files
- `.env` - Environment files (if committed)

### ✅ Small Data
- `data/` (small reference files only)
- Config files

## Size Impact

### Before .gitignore
- Dataset: ~500MB
- Models: ~200MB
- Images: ~150MB
- **Total**: ~850MB+

### After .gitignore
- Source code: ~2MB
- Docs: ~0.5MB
- Configs: ~0.1MB
- **Total**: ~2.6MB

**Reduction**: ~99% smaller repository! 🎉

## How to Use

### Check what will be ignored
```bash
git status
```

### Verify a specific file is ignored
```bash
git check-ignore -v <filename>
```

### Force track an ignored file (if needed)
```bash
git add -f <filename>
```

### Remove ignored files from Git history
```bash
git rm -r --cached dataset/
git commit -m "Remove large image files"
```

## Important Notes

1. **Large files will NOT be uploaded** to GitHub
2. **Generate datasets locally** after cloning
3. **Train models locally** - weights won't be tracked
4. **Safe for CI/CD** - no huge files to download

## If You Need to Share

Use these alternatives for large files:
- **Git LFS**: For binary files (images, models)
- **Cloud storage**: Google Drive, AWS S3
- **Shared folder**: Network drive

## Recommended for Production

If hosting on GPU server:
```bash
# Generate dataset once
python run_segmentation_pipeline.py --full

# Keep trained model
git add model/best_*.pt
git commit -m "Add trained model"
```

---

**Status**: ✅ Active  
**Repository Size**: ~2.6MB (after .gitignore)  
**Last Updated**: August 6, 2026
