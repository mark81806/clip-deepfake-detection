# train.py

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configure a logger for this module
log = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the training loop, optimization, and checkpointing for the model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        num_epochs: int = 15,
        weight_decay: float = 1e-4,
        output_dir: Path = Path("./outputs"),
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            device (torch.device): The device (CPU or CUDA) to train on.
            learning_rate (float): The initial learning rate for the optimizer.
            num_epochs (int): The total number of epochs to train for.
            weight_decay (float): Weight decay for the AdamW optimizer.
            output_dir (Path): Directory to save model checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        # Identify and collect only the parameters that require gradients
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        log.info(f"Identified {len(self.trainable_params)} trainable parameter groups.")

        # Set up optimizer, loss function, and scheduler
        self.optimizer = AdamW(self.trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """
        Executes the main training loop.
        """
        log.info("Starting model training...")
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            total_epoch_loss = 0.0
            
            # Iterate over batches of data
            for batch_idx, (images, labels, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # --- Forward Pass ---
                self.optimizer.zero_grad()
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                # --- Backward Pass and Optimization ---
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                
                self.optimizer.step()
                
                total_epoch_loss += loss.item()

                if (batch_idx + 1) % 50 == 0:
                    log.info(
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # --- End of Epoch ---
            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            log.info(
                f"Completed Epoch {epoch + 1}/{self.num_epochs} | "
                f"Average Loss: {avg_epoch_loss:.4f} | "
                f"Learning Rate: {current_lr:.6f}"
            )
            
            # Update the learning rate
            self.scheduler.step()
            
            # Save a checkpoint periodically
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.num_epochs:
                self._save_checkpoint(epoch + 1)
        
        log.info("Training finished.")
        return self.model

    def _save_checkpoint(self, epoch: int):
        """Saves the model's state dictionary to a file."""
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.output_dir / checkpoint_name
        
        # We save the state_dict, which is the recommended way
        torch.save(self.model.state_dict(), checkpoint_path)
        log.info(f"Saved model checkpoint to {checkpoint_path}")