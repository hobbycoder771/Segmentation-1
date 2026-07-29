# Coordinate Transformation Guide

This document explains how building geometries are transformed from vector coordinates to YOLO format.

## Step-by-Step Transformation

### Step 1: Input Data
```
Building Polygon (Shapefile/GeoPackage)
├── Coordinate System: EPSG:3857 (Web Mercator)
├── Geometry: Polygon with coordinates in meters
└── Example bounds: (minx, miny, maxx, maxy) in map units
```

### Step 2: Image Georeferencing
```
Downloaded Imagery (PNG)
├── Size: 4096 × 4096 pixels
├── Extent: (minx, miny, maxx, maxy) in EPSG:3857
└── Transform: Affine transformation mapping pixels to map coordinates

Affine Transform Matrix:
┌                              ┐
│ pixel_width    0    minx     │
│ 0    -pixel_height  maxy     │
│ 0              0     1       │
└                              ┘

pixel_width = (maxx - minx) / 4096
pixel_height = (maxy - miny) / 4096
```

### Step 3: Polygon to Pixel Conversion

For each building polygon:

```python
# Get bounding box in map coordinates
minx, miny, maxx, maxy = polygon.bounds

# Apply inverse transform (map → pixel)
col_min, row_max = ~transform * (minx, miny)  # Top-left corner
col_max, row_min = ~transform * (maxx, maxy)  # Bottom-right corner

# Calculate center and dimensions in pixels
x_pixel_center = (col_min + col_max) / 2
y_pixel_center = (row_min + row_max) / 2
width_pixels = col_max - col_min
height_pixels = row_max - row_min
```

### Step 4: Normalization to [0, 1]

```python
# Divide by image dimensions to normalize
x_center = x_pixel_center / image_width
y_center = y_pixel_center / image_height
width = width_pixels / image_width
height = height_pixels / image_height

# Clip values to valid range
x_center = np.clip(x_center, 0, 1)
y_center = np.clip(y_center, 0, 1)
width = np.clip(width, 0, 1)
height = np.clip(height, 0, 1)
```

### Step 5: YOLO Annotation

```
YOLO Format: <class_id> <x_center> <y_center> <width> <height>

Example output (one building per line):
0 0.512 0.623 0.128 0.156
0 0.784 0.445 0.095 0.112
0 0.234 0.567 0.089 0.102
```

## Visual Example

### Original Tile Extent
```
Map Coordinates (EPSG:3857):
(-319500, 5346000) ─────────────── (-318500, 5346000)
    │                                    │
    │  ┌─────┐   ┌────┐                 │
    │  │  B1 │   │ B2 │    ...          │
    │  └─────┘   └────┘                 │
    │                                    │
(-319500, 5347000) ─────────────── (-318500, 5347000)

Map extent: 1000m × 1000m
```

### Downloaded Image (4096×4096 pixels)
```
Pixel Space:
(0, 0) ──────────────────────────────────── (4096, 0)
   │                                          │
   │  ┌─────┐   ┌────┐                      │
   │  │ B1  │   │ B2 │    ...               │
   │  └─────┘   └────┘                      │
   │                                          │
(0, 4096) ────────────────────────────────── (4096, 4096)

Resolution: 0.244 m/pixel (1000m / 4096px)
```

### Transformation Formula
```
Map Coordinate → Pixel Coordinate:
col = (map_x - minx) / pixel_width
row = (maxy - map_y) / pixel_height

Pixel Coordinate → Normalized [0,1]:
x_norm = col / image_width
y_norm = row / image_height
```

### Example Calculation

Building B1 in map coordinates:
- Center: (319250, 5346750)
- Width: 50m
- Height: 40m

Transformation:
```
col_center = (319250 - (-319500)) / 0.244 = 2381.97 px ≈ 2382
row_center = (5346750 - 5346000) / 0.244 = 3074.59 px ≈ 3075

width_pixels = 50 / 0.244 = 204.92 ≈ 205 px
height_pixels = 40 / 0.244 = 163.93 ≈ 164 px

Normalized:
x_center = 2382 / 4096 = 0.581
y_center = 3075 / 4096 = 0.751
width = 205 / 4096 = 0.050
height = 164 / 4096 = 0.040

YOLO Annotation:
0 0.581 0.751 0.050 0.040
```

## Key Considerations

1. **CRS Consistency**: Both buildings and extent must use same CRS (EPSG:3857)
2. **Coordinate Order**: Map coordinates are (x, y), but image coordinates are (col, row)
3. **Y-Axis Direction**: Map Y increases upward; image row increases downward (hence inversion)
4. **Clipping**: Normalized values may exceed [0,1] if building extends beyond tile
5. **Pixel Resolution**: Changes based on extent size and image resolution

## Code Reference

See `yolo_data_builder.py` lines ~280-310 for the complete transformation logic:

```python
def buildings_to_yolo_annotations(self, geotiff_path, extent_geometry, extent_id):
    # ... 
    # Transform to pixel coordinates
    col_min, row_max = ~transform * (minx, miny)
    col_max, row_min = ~transform * (maxx, maxy)
    
    # Normalize to image size
    x_center = ((col_min + col_max) / 2) / image_width
    y_center = ((row_min + row_max) / 2) / image_height
    width = (col_max - col_min) / image_width
    height = (row_max - row_min) / image_height
    # ...
```

## Verification

To verify transformations are correct:

1. **Check bounds**: x_center, y_center, width, height should be in [0, 1]
2. **Check scale**: building width should be small fraction (0.01-0.3)
3. **Check count**: number of buildings per tile should be reasonable
4. **Visual check**: Open image + annotations with annotation viewer

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| All values 0 or 1 | Clipping applied | Check buildings intersect extent |
| Missing buildings | CRS mismatch | Verify buildings CRS matches extent |
| Wrong scale | Pixel resolution incorrect | Check image dimensions |
| Inverted Y axis | Y coordinate not inverted | Should use maxy - map_y, not miny |

