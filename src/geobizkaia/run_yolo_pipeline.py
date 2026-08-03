"""
Simple runner script for the YOLO data building pipeline.

This script extracts imagery and clips vector data for YOLO building detection dataset.

Usage:
    python run_yolo_pipeline.py --help
    python run_yolo_pipeline.py --test
    python run_yolo_pipeline.py --full
    python run_yolo_pipeline.py --limit 5 --split
    python run_yolo_pipeline.py --test --no-vector-clipping
"""

import argparse
import sys
from pathlib import Path
from yolo_data_builder import YOLODataBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO training dataset for building detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_yolo_pipeline.py --test
    Test mode: 3 extents, imagery + vector clipping, no splitting
  
  python run_yolo_pipeline.py --full
    Full mode: all extents, imagery + vector clipping, train/val/test split
  
  python run_yolo_pipeline.py --limit 10 --split
    Process 10 extents with imagery + vector clipping and splitting
  
  python run_yolo_pipeline.py --test --no-vector-clipping
    Test mode without vector clipping (imagery only)
  
  python run_yolo_pipeline.py --limit 5 --carto-output-path ./custom_carto
    Custom output path for vector data
        """
    )
    
    parser.add_argument(
        "--extents-layer",
        default="extent",
        help="Layer name for extents in geopackage (default: extent)"
    )
    parser.add_argument(
        "--buildings-layer",
        default="buildings",
        help="Layer name for buildings in geopackage (default: buildings)"
    )
    parser.add_argument(
        "--imagery-url",
        default="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2024/MapServer/export",
        help="ArcGIS MapServer export URL"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=4096,
        help="Size of downloaded imagery (square, default: 4096)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../dataset/yolo_buildings",
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of extents to process (for testing)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split dataset into train/val/test"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process 3 extents without splitting"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full mode: process all extents with splitting"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--feature-server-url",
        default="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/Kartografia_BTB_Kartografia_5000/FeatureServer/16/query",
        help="ArcGIS FeatureServer URL for vector data clipping"
    )
    parser.add_argument(
        "--carto-output-path",
        default="../../data/vector/carto",
        help="Output path for clipped vector data (geopackage)"
    )
    parser.add_argument(
        "--no-vector-clipping",
        action="store_true",
        help="Disable vector data clipping from FeatureServer"
    )
    
    args = parser.parse_args()
    
    # Handle preset modes
    if args.test:
        args.limit = 3
        args.split = False
        print("Running in TEST mode (3 extents, no splitting)")
    elif args.full:
        args.limit = None
        args.split = True
        print("Running in FULL mode (all extents, with splitting)")
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print(f"Error: train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) must be < 1.0")
        sys.exit(1)
    
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    extents_gpkg = project_root / "data" / "vector" / "extents" / "extent.gpkg"
    buildings_gpkg = project_root / "data" / "vector" / "carto" / "karto.gpkg"
    
    # Check files exist
    if not extents_gpkg.exists():
        print(f"Error: Extents geopackage not found: {extents_gpkg}")
        sys.exit(1)
    
    if not buildings_gpkg.exists():
        print(f"Error: Buildings geopackage not found: {buildings_gpkg}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("YOLO Building Detection Dataset Builder")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Extents layer: {args.extents_layer}")
    print(f"  Buildings layer: {args.buildings_layer}")
    print(f"  Image size: {args.image_size}×{args.image_size}")
    print(f"  Limit: {args.limit if args.limit else 'All extents'}")
    print(f"  Split: {args.split}")
    if args.split:
        print(f"  Train/Val/Test ratio: {args.train_ratio:.1%}/{args.val_ratio:.1%}/{1-args.train_ratio-args.val_ratio:.1%}")
    print(f"  Vector clipping: {'Disabled' if args.no_vector_clipping else 'Enabled'}")
    if not args.no_vector_clipping:
        print(f"  Carto output: {args.carto_output_path}")
    print(f"  Output: {args.output_dir}")
    print()
    
    try:
        # Prepare feature server URL (only if vector clipping is enabled)
        feature_server_url = None if args.no_vector_clipping else args.feature_server_url
        carto_output_path = None if args.no_vector_clipping else args.carto_output_path
        
        builder = YOLODataBuilder(
            imagery_url=args.imagery_url,
            extents_gpkg_path=str(extents_gpkg),
            extents_layer_name=args.extents_layer,
            buildings_gpkg_path=str(buildings_gpkg),
            buildings_layer_name=args.buildings_layer,
            output_dir=args.output_dir,
            image_size=args.image_size,
            crs="EPSG:3857",
            feature_server_url=feature_server_url,
            carto_output_path=carto_output_path
        )
        
        # Run pipeline with custom split ratios if provided
        if args.split:
            builder.run_pipeline(limit=args.limit, split=False)
            # Now split with custom ratios
            builder.split_dataset(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
            builder.create_dataset_yaml(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        else:
            builder.run_pipeline(limit=args.limit, split=False)
        
        print("\n" + "="*70)
        print("Pipeline completed successfully!")
        print("="*70)
        
        if args.split:
            print(f"\nDataset ready for training at: {args.output_dir}")
            print(f"Use this data.yaml for YOLOv8 training:")
            print(f"  data: {Path(args.output_dir).absolute()}/data.yaml")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
