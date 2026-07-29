import geopandas as gpd


def list_extents(gpkg_path, layer_name):
    gdf = gpd.read_file(
        gpkg_path,  # r"..\..\data\vector\extents\extent.gpkg",
        layer=layer_name,  # "extent"
    )
    # Create a list of extent tuples
    extents = [geom.bounds for geom in gdf.geometry]
    return extents
