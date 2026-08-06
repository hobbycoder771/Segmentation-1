"""Convert mask PNG files to YOLO segmentation label format (.txt files with normalized coordinates)."""

import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mask_to_yolo_label(mask_path, image_size):
    """
    Convert a binary mask PNG to YOLO segmentation format.
    
    Returns a list of (class_id, normalized_polygon_coords) tuples.
    For building segmentation, class_id is always 0 (building).
    """
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    
    # Find contours in the mask
    # OpenCV requires uint8 format
    if len(mask_array.shape) == 3:
        mask_array = np.max(mask_array, axis=2)
    
    mask_uint8 = mask_array.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    labels = []
    for contour in contours:
        # Approximate contour to reduce number of points
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:  # Skip very small contours
            continue
        
        # Normalize coordinates to [0, 1]
        normalized_coords = []
        for point in approx:
            x, y = point[0]
            norm_x = x / image_size
            norm_y = y / image_size
            normalized_coords.extend([norm_x, norm_y])
        
        if normalized_coords:
            # Format: class_id followed by normalized polygon coordinates
            label = f"0 {' '.join(map(str, normalized_coords))}"
            labels.append(label)
    
    return labels


def create_segmentation_labels(dataset_dir, output_dir="../../dataset/yolo_buildings_seg"):
    """
    Create YOLO segmentation label files from mask PNGs.
    
    Args:
        dataset_dir: Path to dataset directory with images and masks subdirectories
        output_dir: Output directory (will create labels subdirectory)
    """
    output_path = Path(output_dir)
    labels_dir = output_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path(dataset_dir)
    
    total_created = 0
    
    # Process each split (train, val, test)
    for split_name in ["train", "val", "test"]:
        images_split = dataset_path / "images" / split_name
        masks_split = dataset_path / "masks" / split_name
        labels_split = labels_dir / split_name
        
        if not images_split.exists():
            logger.warning(f"Images split not found: {images_split}")
            continue
        
        labels_split.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = sorted(list(images_split.glob("*.tif")) + list(images_split.glob("*.tiff")))
        
        for image_file in image_files:
            mask_file = masks_split / (image_file.stem + ".png")
            
            if not mask_file.exists():
                logger.warning(f"Mask not found for image: {image_file.name}")
                continue
            
            try:
                # Get image size (assume square)
                with Image.open(image_file) as img:
                    img_size = img.width
                
                # Convert mask to YOLO label format
                labels = mask_to_yolo_label(str(mask_file), img_size)
                
                # Write label file
                label_file = labels_split / (image_file.stem + ".txt")
                with open(label_file, "w") as f:
                    for label in labels:
                        f.write(label + "\n")
                
                total_created += 1
                logger.debug(f"Created label: {label_file.name} ({len(labels)} objects)")
                
            except Exception as e:
                logger.error(f"Failed to create label for {image_file.name}: {e}")
    
    logger.info(f"Successfully created {total_created} label files")
    return total_created


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create YOLO segmentation label files from mask PNGs"
    )
    parser.add_argument(
        "--dataset-dir",
        default="../../dataset/yolo_buildings_seg",
        help="Dataset directory with images and masks subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        default="../../dataset/yolo_buildings_seg",
        help="Output directory (labels will be created here)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("YOLO SEGMENTATION LABEL CONVERTER")
    print("="*60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    try:
        created = create_segmentation_labels(args.dataset_dir, args.output_dir)
        print("\n" + "="*60)
        print(f"Conversion completed successfully!")
        print(f"Created {created} label files")
        print("="*60 + "\n")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
