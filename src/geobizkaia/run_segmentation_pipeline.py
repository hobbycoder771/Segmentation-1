"""YOLO Segmentation Data Pipeline Runner - CLI interface."""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO segmentation dataset from geospatial data"
    )

    parser.add_argument("--test", action="store_true", help="Test mode: 3 extents")
    parser.add_argument("--full", action="store_true", help="Full mode: all extents")
    parser.add_argument("--limit", type=int, help="Limit extents to process")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--output-dir", default="../../dataset/yolo_buildings_seg")
    parser.add_argument("--image-size", type=int, default=4096)
    parser.add_argument("--no-vector-clipping", action="store_true")
    parser.add_argument("--mask-value", type=int, default=255)
    parser.add_argument("--extents-layer", default="extent")
    parser.add_argument("--objects-layer", default="buildings")
    parser.add_argument(
        "--imagery-url",
        default="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/ORTO_EJ_2024/MapServer/export",
    )
    parser.add_argument(
        "--feature-server-url",
        default="https://geo.bizkaia.eus/arcgisserverinspire/rest/services/Kartografia_Cartografia/Kartografia_BTB_Cartografia_5000/FeatureServer/16/query",
    )
    parser.add_argument("--carto-output-path", default="../../data/vector/carto")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("YOLO SEGMENTATION DATA PIPELINE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    
    # Determine mode
    if args.test:
        limit = 3
        split = False
        print("Mode: TEST (3 extents, no split)")
    elif args.full:
        limit = None
        split = True
        print("Mode: FULL (all extents, with split)")
    else:
        limit = args.limit
        split = args.split
        print(f"Mode: CUSTOM (limit={limit}, split={split})")
    
    print("="*60 + "\n")

    try:
        logger.info("Starting segmentation data pipeline...")
        
        # Import builder
        try:
            from segmentation_data_builder import SegmentationDataBuilder
        except ImportError as e:
            logger.error(f"Could not import SegmentationDataBuilder: {e}")
            logger.error("Make sure segmentation_data_builder.py is in the same directory")
            sys.exit(1)
        
        # Create and run builder
        builder = SegmentationDataBuilder(
            imagery_url=args.imagery_url,
            extents_gpkg_path="../../data/vector/extents/extent.gpkg",
            extents_layer_name=args.extents_layer,
            objects_gpkg_path="../../data/vector/carto/karto.gpkg",
            objects_layer_name=args.objects_layer,
            output_dir=args.output_dir,
            image_size=args.image_size,
            crs="EPSG:3857",
            feature_server_url=args.feature_server_url if not args.no_vector_clipping else None,
            carto_output_path=args.carto_output_path if not args.no_vector_clipping else None,
            mask_value=args.mask_value,
        )
        
        # Run pipeline
        builder.run_pipeline(limit=limit, split=split)
        
        print("\n" + "="*60)
        print("SEGMENTATION PIPELINE COMPLETED!")
        print("="*60)
        print(f"Dataset: {args.output_dir}")
        print("\nNext: python train_segmentation_model.py --model n --epochs 100 --imgsz 4096")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
