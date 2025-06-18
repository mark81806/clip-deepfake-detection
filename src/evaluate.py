# evaluate.py

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Configure a logger for this module
log = logging.getLogger(__name__)

def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) and its corresponding threshold.

    The EER is the point on the ROC curve where the False Positive Rate (FPR)
    equals the False Negative Rate (FNR).

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_score (np.ndarray): Target scores, can be probabilities or logits.

    Returns:
        A tuple containing:
        - The EER value.
        - The threshold at which the EER was calculated.
    """
    try:
        if len(np.unique(y_true)) != 2:
            log.warning("Cannot compute EER: only one class present in labels.")
            return 0.5, 0.5

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        
        # Find the index where the absolute difference between fpr and fnr is minimal
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        
        # EER is the average of fpr and fnr at that point
        eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
        eer_threshold = thresholds[eer_index]
        
        return float(eer), float(eer_threshold)
    except Exception as e:
        log.error(f"An unexpected error occurred during EER calculation: {e}")
        return 0.5, 0.5 # Return a default neutral value

def plot_roc_curve_to_file(y_true: np.ndarray, y_score: np.ndarray, file_path: Path):
    """
    Generates and saves a Receiver Operating Characteristic (ROC) curve plot.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Target scores.
        file_path (Path): The path where the plot image will be saved.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
    
    # Add EER point to the plot
    eer, _ = compute_eer(y_true, y_score)
    plt.plot(eer, 1-eer, 'o', markersize=8, label=f'EER = {eer:.4f}', fillstyle='full', c='red')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(file_path, dpi=300)
        log.info(f"ROC curve saved successfully to {file_path}")
    except Exception as e:
        log.error(f"Failed to save ROC curve plot: {e}")
    finally:
        plt.close()


class ModelEvaluator:
    """
    A class to manage the evaluation of a trained model on a test dataset.
    """
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: torch.device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def run_evaluation(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Executes the evaluation loop, calculates metrics, and returns the results.

        Returns:
            A tuple containing:
            - A dictionary of performance metrics.
            - A pandas DataFrame with detailed per-sample predictions.
        """
        log.info("Starting model evaluation...")
        self.model.to(self.device)
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_filepaths = []

        with torch.no_grad():
            for images, labels, paths in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                logits, _ = self.model(images)
                
                # Use softmax to get probabilities for the 'fake' class (class 1)
                probabilities = F.softmax(logits, dim=1)[:, 1]
                # Get predicted class indices
                predictions = torch.argmax(logits, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_filepaths.extend(paths)

        # Convert lists to numpy arrays for calculations
        true_labels = np.array(all_labels)
        pred_scores = np.array(all_probabilities)
        
        # Calculate standard binary classification metrics
        pred_labels_at_0_5 = (pred_scores > 0.5).astype(int)
        
        metrics = {
            'AUC': roc_auc_score(true_labels, pred_scores),
            'Accuracy': accuracy_score(true_labels, pred_labels_at_0_5),
            'F1_Score': f1_score(true_labels, pred_labels_at_0_5)
        }
        
        # Calculate EER and its threshold
        eer, eer_thresh = compute_eer(true_labels, pred_scores)
        metrics['EER'] = eer
        metrics['EER_Threshold'] = eer_thresh

        log.info("Evaluation complete.")
        for name, value in metrics.items():
            log.info(f"  - {name}: {value:.4f}")
            
        # Create a detailed results DataFrame
        results_df = pd.DataFrame({
            'filepath': all_filepaths,
            'true_label': true_labels,
            'predicted_label_at_0.5': pred_labels_at_0_5,
            'fake_probability_score': pred_scores
        })
        
        return metrics, results_df