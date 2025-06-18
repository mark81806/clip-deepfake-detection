# main.py

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import components from our other modules
from dataset import FaceForgeryDataset, CustomSampleDataset
from model import ForgeryDetector
from train import ModelTrainer
from evaluate import ModelEvaluator, plot_roc_curve_to_file

# --- Configuration ---
# These are classes used for training and testing.
# 'Real_youtube' is assumed to be the "real" class (label 0).
# 'FaceSwap' is one type of "fake" used for training.
# 'NeuralTextures' is another type of "fake" used for testing generalization.
TRAIN_CLASSES = ['Real_youtube', 'FaceSwap']
TEST_FAKE_CLASS = ['NeuralTextures']
REAL_CLASS = 'Real_youtube'
FAKE_LABEL = 1
REAL_LABEL = 0

def setup_environment(seed: int):
    """Configures random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def setup_logging(log_dir: Path):
    """Configures logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s (%(name)s)",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_data_loaders(config: argparse.Namespace) -> dict:
    """Creates and returns the data loaders for training and testing."""
    # Standard normalization for models pre-trained on ImageNet/CLIP
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                           std=(0.26862954, 0.26130258, 0.27577711))
    ])

    loaders = {}

    # --- Training Loader ---
    if config.mode in ['train', 'both']:
        try:
            train_dataset = FaceForgeryDataset(
                root_dir=config.data_dir,
                class_names=TRAIN_CLASSES,
                transform=transform
            )
            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True
            )
            logging.info(f"Training data loader created with {len(train_dataset)} samples.")
        except FileNotFoundError as e:
            logging.error(f"Could not create training dataset: {e}")
            # Exit if training is required but data is missing
            if config.mode == 'train':
                exit(1)
    
    # --- Testing Loader ---
    if config.mode in ['test', 'both']:
        try:
            # Create a balanced test set for robust evaluation
            # 1. Get a subset of real images
            real_test_subset = FaceForgeryDataset(
                root_dir=config.data_dir,
                class_names=[REAL_CLASS],
                transform=transform,
                max_samples_per_class=5000 # Limit real samples to avoid overwhelming imbalance
            )
            # 2. Get the "unseen" fake images
            fake_test_set = FaceForgeryDataset(
                root_dir=config.data_dir,
                class_names=TEST_FAKE_CLASS,
                transform=transform
            )
            
            # Combine them into a custom list of (path, label)
            test_samples = []
            test_samples.extend([(p, REAL_LABEL) for p, _ in zip(real_test_subset.image_paths, real_test_subset.labels)])
            test_samples.extend([(p, FAKE_LABEL) for p, _ in zip(fake_test_set.image_paths, fake_test_set.labels)])

            logging.info(f"Test set created with {len(real_test_subset)} real and {len(fake_test_set)} fake samples.")

            test_dataset = CustomSampleDataset(samples=test_samples, transform=transform)
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        except FileNotFoundError as e:
            logging.error(f"Could not create test dataset: {e}")
            # Exit if testing is required but data is missing
            if config.mode == 'test':
                exit(1)

    return loaders

def main(config: argparse.Namespace):
    """Main execution function."""
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    setup_environment(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Configuration: {config}")
    
    # --- Model Initialization ---
    model = ForgeryDetector(clip_model_name="ViT-B/32", num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized: {type(model).__name__}")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # --- Data Loading ---
    loaders = get_data_loaders(config)
    
    # --- Training Phase ---
    if config.mode in ['train', 'both']:
        if 'train' not in loaders:
            logging.error("Training mode selected, but training data loader could not be created. Exiting.")
            return

        trainer = ModelTrainer(
            model=model,
            train_loader=loaders['train'],
            device=device,
            learning_rate=config.lr,
            num_epochs=config.epochs,
            output_dir=output_dir
        )
        model = trainer.train()
        
        # Save the final model state
        final_model_path = output_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final trained model saved to {final_model_path}")
    
    # --- Testing Phase ---
    if config.mode in ['test', 'both']:
        if 'test' not in loaders:
            logging.error("Test mode selected, but test data loader could not be created. Exiting.")
            return
            
        # Load the trained model if we are in 'test' only mode
        if config.mode == 'test':
            model_path = output_dir / "final_model.pth"
            if not model_path.exists():
                logging.error(f"Model file not found at {model_path}. Please train a model first or provide the path.")
                return
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Loaded pre-trained model from {model_path}")
        
        evaluator = ModelEvaluator(
            model=model,
            test_loader=loaders['test'],
            device=device
        )
        metrics, results_df = evaluator.run_evaluation()

        # Save artifacts
        results_df.to_csv(output_dir / "test_predictions.csv", index=False)
        logging.info(f"Test predictions saved to {output_dir / 'test_predictions.csv'}")

        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Test metrics saved to {output_dir / 'test_metrics.json'}")

        plot_roc_curve_to_file(
            y_true=results_df['true_label'].values,
            y_score=results_df['fake_probability_score'].values,
            file_path=output_dir / "roc_curve.png"
        )
    
    logging.info("Process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CLIP-based deepfake detector.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save logs, models, and results.")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', help="Execution mode: 'train' only, 'test' only, or 'both'.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loaders.")
    
    args = parser.parse_args()
    main(args)