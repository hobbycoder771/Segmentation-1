# create a polygon feature class in a geopackage from a extent (xmin,ymin,xmax,ymax)

import geopandas as gpd
from shapely.geometry import box

# Extent (xmin, ymin, xmax, ymax)
extent = (-320500, 5347000, -319500, 5348000)

# Create polygon
polygon = box(*extent)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:3857")  # Web Mercator

# Save to GeoPackage
output_file = "../data/vector/extents/extent.gpkg"
layer_name = "extent"

gdf.to_file(output_file, layer=layer_name, driver="GPKG")

print(f"GeoPackage saved to: {output_file}")
