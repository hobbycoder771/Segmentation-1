"""
clip vector layer in feature server with given extent
create
outfile = f"data/tile_{i}.gpkg"
"""

import json
import requests
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import shape

# Get the parent directory (project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# FeatureServer URL
# ----------------------------------------------------

URL = "https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/Kartografia_BTB_Cartografia_5000/FeatureServer/16/query"

# ----------------------------------------------------
# Bounding boxes in EPSG:3857
# ----------------------------------------------------

bboxes = [
    (-320500, 5347000, -319500, 5348000)
]  # (-320500, 5349000, -319500, 5350000), (-321500, 5348000, -320500, 5349000)

# ----------------------------------------------------
# Transformer
# ----------------------------------------------------

transformer = Transformer.from_crs("EPSG:3857", "EPSG:25830", always_xy=True)

# ----------------------------------------------------
# Process every bbox
# ----------------------------------------------------
file_name = "arizgoiti_2_1000_2024"
for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes, start=1):

    print(f"\nTile {i}")

    # ------------------------------------
    # Project bbox
    # ------------------------------------

    xmin2, ymin2 = transformer.transform(xmin, ymin)
    xmax2, ymax2 = transformer.transform(xmax, ymax)

    geometry = {
        "xmin": xmin2,
        "ymin": ymin2,
        "xmax": xmax2,
        "ymax": ymax2,
        "spatialReference": {"wkid": 25830},
    }

    params = {
        "f": "geojson",
        "where": "1=1",
        "geometry": json.dumps(geometry),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 25830,
        "outSR": 3857,
        "returnGeometry": "true",
        "outFields": "*",
    }

    print("Querying FeatureServer...")

    r = requests.get(URL, params=params)

    r.raise_for_status()

    geojson = r.json()

    print("Features:", len(geojson["features"]))

    if len(geojson["features"]) == 0:
        continue

    # ------------------------------------
    # GeoDataFrame
    # ------------------------------------

    rows = []

    for feat in geojson["features"]:

        attrs = feat["properties"]

        attrs["geometry"] = shape(feat["geometry"])

        rows.append(attrs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")

    outfile = f"data/{file_name}_tile_{i}.gpkg"

    gdf.to_file(outfile, driver="GPKG")

    print("Saved:", outfile)

print("\nFinished.")
