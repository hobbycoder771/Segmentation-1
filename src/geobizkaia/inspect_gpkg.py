"""
Utility script to inspect geopackage files and display available layers.
"""

import geopandas as gpd
import fiona
from pathlib import Path
import sys


def inspect_gpkg(gpkg_path):
    """
    Inspect a geopackage file and print all layers and their properties.
    
    Parameters
    ----------
    gpkg_path : str
        Path to the geopackage file
    """
    gpkg_path = Path(gpkg_path)
    
    if not gpkg_path.exists():
        print(f"Error: File not found: {gpkg_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {gpkg_path.name}")
    print(f"{'='*60}\n")
    
    try:
        # Get all layer names
        with fiona.collection(str(gpkg_path)) as src:
            layers = src.schema
            print("Layers in geopackage:")
            print("-" * 60)
    except Exception as e:
        print(f"Error reading layers: {e}")
        return
    
    # List all layers using fiona
    try:
        layers = fiona.listlayers(str(gpkg_path))
        print(f"Available layers: {layers}\n")
        
        # For each layer, show details
        for layer in layers:
            print(f"\nLayer: {layer}")
            print("-" * 60)
            
            try:
                gdf = gpd.read_file(gpkg_path, layer=layer)
                
                print(f"  Geometry type: {gdf.geometry.type.unique()}")
                print(f"  CRS: {gdf.crs}")
                print(f"  Number of features: {len(gdf)}")
                print(f"  Columns: {list(gdf.columns)}")
                print(f"  Bounds: {gdf.total_bounds}")
                print(f"  Sample row:")
                print(gdf.head(1).to_string(max_colwidth=50))
                
            except Exception as e:
                print(f"  Error reading layer: {e}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_gpkg.py <gpkg_path>")
        print("\nExample:")
        print("  python inspect_gpkg.py ../../data/vector/extents/extent.gpkg")
        print("  python inspect_gpkg.py ../../data/vector/carto/karto.gpkg")
        sys.exit(1)
    
    gpkg_path = sys.argv[1]
    inspect_gpkg(gpkg_path)
