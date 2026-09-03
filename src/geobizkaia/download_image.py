"""
Convert a PNG downloaded from an ArcGIS MapServer export
into a georeferenced GeoTIFF.

Parameters
----------
png_file : str
    Path to the downloaded PNG.

extent : dict
    The 'extent' object returned by the ArcGIS REST export operation.

output_tif : str
    Output GeoTIFF.
"""

import requests
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime


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


now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start:", start_time)

image_name = "imagery_tile_1"
url = "https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2025/MapServer/export"

params = {
    "bbox": "-294310, 5337138, -293310, 5338138",  # xmin, ymin, xmax, ymax
    "bboxSR": "3857",
    "imageSR": "3857",
    "size": "4096,4096",
    "format": "png32",
    "transparent": "false",
    "f": "json",
}
print("downloading image....")
r = requests.get(url, params=params)

result = r.json()
image_url = result["href"]

r = requests.get(image_url)
r.raise_for_status()

with open(f"../../data/{image_name}.png", "wb") as f:
    f.write(r.content)

print("assigning projection...")
png_to_geotiff(
    f"../../data/{image_name}.png", result["extent"], f"../../data/{image_name}.tif"
)

now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("Start:", start_time)
print("Finish:", end_time)
