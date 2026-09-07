"""
train_segmentation_model.py

YOLO Segmentation Model Training Script.

Progressive training: Train incrementally as new imagery becomes available.
The best model from each training session is saved and used for the next session.

Usage:
    python train_segmentation_model.py --model m --epochs 50
    # Next time with new imagery:
    python train_segmentation_model.py --model m --epochs 50
    # The script will automatically load the previous best model and continue training
"""

import argparse
import logging
from pathlib import Path
import torch
import json
from datetime import datetime
import shutil
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train_segmentation_model.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


class YOLOSegmentationTrainer:
    """Manages YOLOv8-Seg training pipeline."""

    def __init__(
        self,
        data_yaml: str = "dataset/yolo_buildings_seg/data.yaml",
        model_size: str = "n",
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 4,
        device="auto",
        patience: int = 20,
        save_dir: str = "model/runs",
        model_dir: str = "model",
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

        self.best_model_path = self.model_dir / \
            f"best_yolov8{model_size}-seg.pt"
        self.training_history_path = (
            self.model_dir / f"training_history_yolov8{model_size}-seg.json"
        )

        if not self.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at {self.data_yaml}")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized YOLOSegmentationTrainer (model: yolov8{model_size}-seg)")

        if self.best_model_path.exists():
            logger.info(f"Found existing best model: {self.best_model_path}")

    def _get_training_history(self) -> dict:
        """Load training history from JSON file."""
        if self.training_history_path.exists():
            try:
                with open(self.training_history_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load training history: {e}")

        return {
            "total_training_runs": 0,
            "total_epochs_trained": 0,
            "training_sessions": [],
        }

    def _save_training_history(self, history: dict):
        """Save training history to JSON file."""
        try:
            with open(self.training_history_path, "w") as f:
                json.dump(history, f, indent=2)
            logger.info(
                f"Training history saved to {self.training_history_path}")
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def _update_training_history(self, epochs_trained: int, results_dir: str):
        """Update training history with new session."""
        history = self._get_training_history()
        history["total_training_runs"] += 1
        history["total_epochs_trained"] += epochs_trained
        history["training_sessions"].append(
            {
                "run_number": history["total_training_runs"],
                "date": datetime.now().isoformat(),
                "epochs": epochs_trained,
                "results_dir": results_dir,
            }
        )
        self._save_training_history(history)

    def train(self, resume: str = None) -> str:
        """Train segmentation model using YOLOv8-Seg with progressive training support."""
        logger.info("Training configuration:")
        logger.info(f"  Model: yolov8{self.model_size}-seg")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Image size: {self.imgsz}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Continuous training: {self.continuous_training}")

        try:
            from ultralytics import YOLO

            # Determine which model to load
            model_to_load = resume

            if (
                not resume
                and self.continuous_training
                and self.best_model_path.exists()
            ):
                # Progressive training: load the persistent best model
                model_to_load = str(self.best_model_path)
                logger.info(
                    f"Continuous training enabled - loading persistent best model: {model_to_load}"
                )
            else:
                if not resume:
                    # Fresh training: load pre-trained model
                    model_to_load = f"yolov8{self.model_size}-seg"
                    logger.info(
                        f"Fresh training - loading pre-trained model: {model_to_load}"
                    )

            logger.info(f"Loading model: {model_to_load}")
            model = YOLO(model_to_load)

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
                workers=6,        # tune to your physical core count, not logical
                cache="disk",      # or True for RAM cache if dataset fits in memory
            )

            logger.info(
                f"Training completed. Results saved to: {results.save_dir}")

            # Copy best model to persistent location (for progressive training)
            if self.continuous_training:
                training_best = Path(results.save_dir) / "weights" / "best.pt"
                if training_best.exists():
                    shutil.copy(training_best, self.best_model_path)
                    logger.info(
                        f"Copied best model to persistent location: {self.best_model_path}"
                    )

                    # Update training history
                    self._update_training_history(
                        self.epochs, str(results.save_dir))

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
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-Seg model for progressive segmentation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial training with default settings
  python train_segmentation_model.py --model m --epochs 50
  
  # Progressive training with new imagery (automatically continues from previous best model)
  python train_segmentation_model.py --model m --epochs 50
  
  # Fresh training (ignore previous model)
  python train_segmentation_model.py --model m --epochs 50 --no-continuous
  
  # Resume from specific checkpoint
  python train_segmentation_model.py --model m --epochs 50 --resume path/to/checkpoint.pt
  
  # Show training history
  python train_segmentation_model.py --show-history
  
  # Validate existing model
  python train_segmentation_model.py --validate-only
        """,
    )

    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: nano (n), small (s), medium (m), large (l), xlarge (x). Default: n")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs. Default: 100")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size. Default: 4")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size. Default: 640")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, or GPU id (0,1,2...). Default: auto")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement). Default: 20")
    parser.add_argument("--data", default="dataset/yolo_buildings_seg/data.yaml",
                        help="Path to data.yaml configuration")
    parser.add_argument("--resume", help="Resume from specific checkpoint")
    parser.add_argument("--no-continuous", action="store_true",
                        help="Disable continuous training (always start fresh)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate without training")
    parser.add_argument("--show-history", action="store_true",
                        help="Show training history and exit")

    args = parser.parse_args()

    # Auto-detect device if not specified
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

    # Show training history if requested
    if args.show_history:
        history = trainer._get_training_history()
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING HISTORY")
        logger.info("=" * 60)
        if history["total_training_runs"] > 0:
            logger.info(
                f"Total training runs: {history['total_training_runs']}")
            logger.info(
                f"Total epochs trained: {history['total_epochs_trained']}")
            logger.info("\nDetailed sessions:")
            for session in history["training_sessions"]:
                logger.info(
                    f"  Run {session['run_number']}: {session['epochs']} epochs on {session['date']}"
                )
        else:
            logger.info("No training history found")
        logger.info("=" * 60)
        return

    # Execute training or validation
    if args.validate_only:
        logger.info("Running validation-only mode...")
        trainer.validate()
    else:
        logger.info("Running progressive training pipeline...")
        best_model = trainer.train(resume=args.resume)
        if best_model:
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Best model saved to: {best_model}")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()
