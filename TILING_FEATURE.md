# Image Tiling Feature for YOLO Building Detection

## Overview

The tiling feature automatically splits large 4096×4096 downloaded images into smaller tiles (512×512, 640×640, etc.) for improved YOLO training performance. This increases the training dataset size and improves model generalization without requiring additional data downloads.

## How It Works

### Image Tiling

When tiling is enabled:
1. Each 4096×4096 image is divided into a grid of smaller tiles
2. For example, with 512×512 tiles: 4096/512 = 8×8 = 64 tiles per image
3. Each tile is saved as a separate GeoTIFF with proper geospatial georeferencing
4. Original full images are deleted to save space

### Annotation Tiling

Building annotations are automatically clipped and adjusted for each tile:
1. Bounding boxes that intersect a tile are included in that tile's annotations
2. Boxes are clipped to tile boundaries and normalized to the new tile coordinate system
3. Very small boxes (<1% of tile size) are filtered out to avoid noise
4. Empty tiles (no building annotations) are excluded from the dataset

## Usage

### Basic Usage: 512×512 tiles

```bash
python run_yolo_pipeline.py --test --enable-tiling --tile-size 512
```

### With 640×640 tiles (YOLO standard)

```bash
python run_yolo_pipeline.py --full --enable-tiling --tile-size 640 --split
```

### With overlapping tiles (64px overlap)

```bash
python run_yolo_pipeline.py --test --enable-tiling --tile-size 512 --tile-overlap 64
```

### Available Tile Sizes

- `256`: Smaller tiles, more data samples, higher computational overhead
- `512`: Balanced option (default)
- `640`: YOLO-8 standard size, good for transfer learning
- `768`: Larger tiles, more context
- `1024`: Very large tiles, less data samples

## Configuration Parameters

### `--enable-tiling`
Enable the tiling feature. Without this flag, images are processed as full 4096×4096 images (default: disabled).

### `--tile-size {256,512,640,768,1024}`
Size of tiles in pixels (must be square). Default is 512.

### `--tile-overlap`
Overlap between adjacent tiles in pixels. For example:
- `0` (default): No overlap, tiles are adjacent
- `64`: Each tile overlaps 64px with neighbors, creates more samples but increases computation

## Impact on Dataset

### Dataset Size Example

With a 4096×4096 image and different tiling strategies:

| Tile Size | Tiles per Image | With 10 Extents |
|-----------|-----------------|-----------------|
| Full 4096×4096 | 1 | 10 images |
| 512×512 (no overlap) | 64 | 640 images |
| 512×512 (64px overlap) | 81 | 810 images |
| 640×640 (no overlap) | 36 | 360 images |
| 640×640 (64px overlap) | 49 | 490 images |

### Training Benefits

- **More samples**: More training data from same source → better model convergence
- **Better generalization**: Local patterns from different regions of same extent
- **Memory efficiency**: Smaller images fit better in GPU memory with larger batch sizes
- **Standard sizes**: 640×640 aligns with YOLO-8 default training size

### Considerations

- **Context loss**: Smaller tiles may lose surrounding urban context
- **Boundary buildings**: Buildings at tile edges might have clipped annotations
- **Storage**: With overlap, storage requirements increase significantly
- **Training time**: More tiles = longer training epochs

## Implementation Details

### File Naming Convention

Original image: `extent_0001.tif`

Tiled images: `extent_0001_tile_000.tif`, `extent_0001_tile_001.tif`, etc.

### Tile Indexing

Tiles are numbered sequentially left-to-right, top-to-bottom:
- Tile 0: top-left
- Tile 1: top-left+1 column
- etc.

### Annotation Handling

- Annotations are clipped to tile boundaries
- Normalized coordinates are recalculated relative to tile size
- Boxes with <1% area are filtered out
- Tiles without annotations are not included in final dataset

### Georeferencing Preservation

Each tile maintains:
- Proper GeoTIFF transformation matrix
- CRS information (EPSG:3857)
- LZW compression for efficiency

## Workflow Example

```bash
# Test run with tiling
python run_yolo_pipeline.py --test --enable-tiling --tile-size 512

# Output structure:
# dataset/yolo_buildings/
# ├── images/
# │   ├── extent_0000_tile_000.tif
# │   ├── extent_0000_tile_001.tif
# │   ├── ... (64 tiles per extent)
# │   ├── extent_0001_tile_000.tif
# │   └── ... (etc)
# └── labels/
#     ├── extent_0000_tile_000.txt
#     ├── extent_0000_tile_001.txt
#     └── ... (etc)

# Full run with splitting and tiling
python run_yolo_pipeline.py --full --enable-tiling --tile-size 640 --split

# Output structure after splitting:
# dataset/yolo_buildings/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# ├── labels/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── data.yaml
```

## Performance Recommendations

### For Building Detection (Bizkaia Dataset)

1. **Standard setup**: `--enable-tiling --tile-size 640`
   - Aligns with YOLO-8 standard
   - Reasonable dataset size
   - Good context preservation

2. **Large dataset**: `--enable-tiling --tile-size 512`
   - Maximizes training samples
   - Better for smaller building detection
   - Requires more GPU memory

3. **High accuracy**: `--enable-tiling --tile-size 640 --tile-overlap 64`
   - Best for complex scenes
   - 1.36× more samples
   - Higher computational cost

## Troubleshooting

### Issue: "No tiles created for extent"
- Check if image is actually being downloaded
- Verify image size matches expected 4096×4096
- Check disk space

### Issue: Missing annotations in tiles
- This is normal - tiles with no buildings are excluded
- Check that buildings data is properly loaded
- Verify annotation clipping isn't filtering too aggressively

### Issue: Very small dataset after tiling
- Reduce tile size to get more tiles per image
- Check vector data quality and coverage
- Ensure building geometries are valid

## Technical Implementation

See `yolo_data_builder.py` for implementation details:
- `tile_image()`: Splits GeoTIFF into georeferenced tiles
- `tile_annotations()`: Clips and adjusts YOLO annotations
- `_process_tiling()`: Orchestrates tiling workflow
