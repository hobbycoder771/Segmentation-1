"""Segmentation Metrics Module - Calculates evaluation metrics for segmentation."""

import numpy as np
from typing import Dict, Optional


class SegmentationMetrics:
    """Computes segmentation evaluation metrics."""
    
    @staticmethod
    def iou(mask_pred: np.ndarray, mask_true: np.ndarray, class_id: int = 1) -> float:
        """Calculate Intersection over Union (IoU)."""
        if mask_pred.shape != mask_true.shape:
            return 0.0
        
        pred_binary = (mask_pred == class_id).astype(np.uint8)
        true_binary = (mask_true == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, true_binary).sum()
        union = np.logical_or(pred_binary, true_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(intersection) / float(union)
    
    @staticmethod
    def dice(mask_pred: np.ndarray, mask_true: np.ndarray, class_id: int = 1) -> float:
        """Calculate Dice coefficient."""
        if mask_pred.shape != mask_true.shape:
            return 0.0
        
        pred_binary = (mask_pred == class_id).astype(np.uint8)
        true_binary = (mask_true == class_id).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, true_binary).sum()
        denominator = pred_binary.sum() + true_binary.sum()
        
        if denominator == 0:
            return 1.0
        
        return float(2 * intersection) / float(denominator)
    
    @staticmethod
    def pixel_accuracy(mask_pred: np.ndarray, mask_true: np.ndarray) -> float:
        """Calculate pixel-level accuracy."""
        if mask_pred.shape != mask_true.shape:
            return 0.0
        
        correct = np.sum(mask_pred == mask_true)
        total = mask_pred.size
        
        return float(correct) / float(total)
    
    @staticmethod
    def all_metrics(
        mask_pred: np.ndarray,
        mask_true: np.ndarray,
        class_id: int = 1
    ) -> Dict[str, float]:
        """Calculate all available metrics."""
        return {
            "iou": SegmentationMetrics.iou(mask_pred, mask_true, class_id),
            "dice": SegmentationMetrics.dice(mask_pred, mask_true, class_id),
            "pixel_accuracy": SegmentationMetrics.pixel_accuracy(mask_pred, mask_true),
        }
