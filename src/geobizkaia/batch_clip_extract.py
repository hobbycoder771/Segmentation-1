import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
import os
import json
import requests
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import shape


def list_extents(gpkg_path, layer_name):
    gdf = gpd.read_file(
        gpkg_path,  # r"..\..\data\vector\extents\extent.gpkg",
        layer=layer_name,  # "extent"
    )
    # Create a list of extent tuples
    extents = [geom.bounds for geom in gdf.geometry]
    print(extents)
    return extents


def clip_feature_server_by_extent(
    service_url, extent, output_path, out_name, layer_count
):

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:25830", always_xy=True)

    xmin, ymin, xmax, ymax = extent

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

    r = requests.get(service_url, params=params)

    r.raise_for_status()

    geojson = r.json()

    print("Features:", len(geojson["features"]))

    if len(geojson["features"]) == 0:
        print("no features founded in feature service")

    # ------------------------------------
    # GeoDataFrame
    # ------------------------------------

    rows = []

    for feat in geojson["features"]:

        attrs = feat["properties"]

        attrs["geometry"] = shape(feat["geometry"])

        rows.append(attrs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")

    layer = f"{out_name}_tile_{layer_count}"
    outfile = os.path.join(output_path, "karto.gpkg")

    gdf.to_file(outfile, layer=layer, driver="GPKG", mode="a")  # append a new layer

    print("Saved:", outfile)

    print(layer_count)


def png_to_geotiff(png_file, extent, output_tif):

    # Read PNG
    with rasterio.open(png_file) as src:

        image = src.read()

        width = src.width
        height = src.height
        profile = src.profile

    # Build affine transform
    transform = from_bounds(
        extent["xmin"],
        extent["ymin"],
        extent["xmax"],
        extent["ymax"],
        width,
        height,
    )

    # Update profile for GeoTIFF
    profile.update(
        driver="GTiff",
        crs=f"EPSG:{extent['spatialReference']['wkid']}",
        transform=transform,
        compress="lzw",
    )

    # Write GeoTIFF
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(image)

    print(f"Created: {output_tif}")


def extract_map_server_by_extent(
    mapserver_url, extent, output_path, out_name, layer_count
):
    # output_path data\imagery
    # extent is a tuple
    bbox_str = ", ".join(map(str, extent))
    params = {
        "bbox": bbox_str,  # "-319500, 5346000, -318500, 5347000", xmin, ymin, xmax, ymax
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": "4096,4096",
        "format": "png32",
        "transparent": "false",
        "f": "json",
    }
    print("downloading image....")
    r = requests.get(mapserver_url, params=params)

    result = r.json()
    image_url = result["href"]

    r = requests.get(image_url)
    r.raise_for_status()

    image_name = f"{out_name}_tile_{layer_count}"
    image_path = os.path.join(project_path, output_path)

    with open(f"{image_path}/{image_name}.png", "wb") as f:
        f.write(r.content)

    print("assigning projection...")

    png_to_geotiff(
        f"{image_path}\{image_name}.png",
        result["extent"],
        f"{image_path}\{image_name}.tif",
    )

    os.remove(f"{image_path}\{image_name}.png")

    print("extracted image")


# --------------------------------------------------------------------------------

project_path = r"C:\Users\gonzalo.echeverria\BILBOMATICA\TASK\GeoBizkaia-Segmentation-1"

carto_service_url = "https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/Kartografia_BTB_Cartografia_5000/FeatureServer/16/query"
mapserver_url = "https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2024/MapServer/export"


gpkg_path = os.path.join(project_path, r"data\vector\extents\extent.gpkg")
layer_name = "extent"

# list of extents
extents_list = list_extents(gpkg_path, layer_name)

for i, extent in enumerate(extents_list, start=1):
    clip_feature_server_by_extent(
        carto_service_url,
        extent,
        os.path.join(project_path, r"data\vector\carto"),
        "buildings",
        i,
    )

    extract_map_server_by_extent(
        mapserver_url,
        extent,
        r"data\imagery",
        "imagery",  # {out_name}_tile_{layer_count}
        i,
    )
