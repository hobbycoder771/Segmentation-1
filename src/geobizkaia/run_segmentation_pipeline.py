"""YOLO Segmentation Data Pipeline Runner - CLI interface."""

import argparse
import sys
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_output_directory(output_dir):
    """Clean up all files from the output directory before starting the pipeline.
    
    This ensures we start fresh and avoid cascading tile files like:
    test_test_test_extent_0006_tile_0089_tile_0002_tile_0000.tif
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Output directory does not exist yet: {output_path}")
        return
    
    logger.info(f"Cleaning up output directory: {output_path}")
    try:
        # Remove the entire output directory and recreate it to start fresh
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory cleaned successfully")
    except Exception as e:
        logger.warning(f"Failed to clean output directory: {e}")


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
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap ratio 0-1 (default: 0.2)")
    parser.add_argument("--no-tiling", action="store_true", help="Disable tiling")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("YOLO SEGMENTATION DATA PIPELINE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    
    # Clean up output directory before starting
    cleanup_output_directory(args.output_dir)
    
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

        # Apply tiling if enabled
        if not args.no_tiling:
            from tile_generator import TileGenerator
            tiler = TileGenerator(tile_size=args.tile_size, overlap=args.tile_overlap)
            
            for split_name in ["train", "val", "test"]:
                images_split = Path(args.output_dir) / "images" / split_name
                masks_split = Path(args.output_dir) / "masks" / split_name
                
                if images_split.exists() and list(images_split.glob("*.tif")):
                    logger.info(f"Tiling {split_name} split...")
                    tiler.tile_dataset(
                        str(images_split),
                        str(masks_split),
                        str(images_split),
                        str(masks_split),
                        prefix=split_name
                    )
        
        # Create YOLO segmentation labels from masks
        logger.info("Creating YOLO segmentation labels from masks...")
        try:
            from create_segmentation_labels import create_segmentation_labels
            created_labels = create_segmentation_labels(args.output_dir, args.output_dir)
            logger.info(f"Created {created_labels} label files for training")
        except Exception as e:
            logger.warning(f"Failed to create segmentation labels: {e}")
        
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
