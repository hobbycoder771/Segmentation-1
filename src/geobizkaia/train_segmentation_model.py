"""YOLO Segmentation Model Training Script."""

import argparse
import logging
from pathlib import Path
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class YOLOSegmentationTrainer:
    """Manages YOLOv8-Seg training pipeline."""

    def __init__(
        self,
        data_yaml: str = "../../dataset/yolo_buildings_seg/data.yaml",
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 4,
        device = "auto",
        patience: int = 20,
        save_dir: str = "../../model/runs",
        model_dir: str = "../../model",
        continuous_training: bool = True,
    ):
        self.data_yaml = Path(data_yaml).resolve()
        self.model_size = model_size
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.save_dir = Path(save_dir).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.continuous_training = continuous_training

        self.best_model_path = self.model_dir / f"best_yolov8{model_size}-seg.pt"
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at {self.data_yaml}")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized YOLOSegmentationTrainer (model: yolov8{model_size}-seg)")

    def train(self, resume: str = None) -> str:
        """Train segmentation model using YOLOv8-Seg."""
        logger.info("Training configuration:")
        logger.info(f"  Model: yolov8{self.model_size}-seg")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Image size: {self.imgsz}")
        logger.info(f"  Device: {self.device}")
        
        try:
            from ultralytics import YOLO
            
            model_name = f"yolov8{self.model_size}-seg"
            logger.info(f"Loading model: {model_name}")
            model = YOLO(model_name)
            
            logger.info("Starting training...")
            results = model.train(
                data=str(self.data_yaml),
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                patience=self.patience,
                save=True,
                project=str(self.save_dir),
                name=f"train_yolov8{self.model_size}_seg",
                exist_ok=True,
            )
            
            logger.info(f"Training completed. Results saved to: {results.save_dir}")
            return str(self.best_model_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None

    def validate(self, weights: str = None) -> dict:
        """Validate segmentation model."""
        logger.info("Running validation...")
        try:
            from ultralytics import YOLO
            weights_path = weights or str(self.best_model_path)
            model = YOLO(weights_path)
            metrics = model.val(data=str(self.data_yaml), imgsz=self.imgsz)
            logger.info("Validation completed")
            return metrics
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg model")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or GPU id (0,1,2...)")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data", default="../../dataset/yolo_buildings_seg/data.yaml")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--no-continuous", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--test-only", help="Test inference only")

    args = parser.parse_args()

    # Auto-detect device if not specified
    import torch
    if args.device == "auto":
        args.device = 0 if torch.cuda.is_available() else "cpu"
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.info(f"Auto-detected device: {device_type} ({args.device})")
    elif isinstance(args.device, str) and args.device != "cpu":
        try:
            args.device = int(args.device)
        except ValueError:
            args.device = "cpu"

    trainer = YOLOSegmentationTrainer(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        patience=args.patience,
        continuous_training=not args.no_continuous,
    )

    if args.validate_only:
        trainer.validate()
    elif args.test_only:
        logger.info(f"Test inference with: {args.test_only}")
    else:
        best_model = trainer.train(resume=args.resume)
        if best_model:
            logger.info(f"Best model saved to: {best_model}")


if __name__ == "__main__":
    main()
