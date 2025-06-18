# dataset.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset

# Configure a logger for this module
log = logging.getLogger(__name__)

class FaceForgeryDataset(Dataset):
    """
    A PyTorch Dataset for loading real and forged face images.

    This dataset walks a directory structure where each subdirectory
    represents a class (e.g., 'Real_youtube', 'FaceSwap'). It collects
    image paths and their corresponding integer labels.

    Attributes:
        image_paths (List[Path]): A list of paths to the images.
        labels (List[int]): A list of corresponding integer labels.
        transform (callable, optional): A function/transform to apply to the images.
    """

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        transform: Optional[callable] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Initializes the dataset by scanning for image files.

        Args:
            root_dir (str): The path to the main data directory.
            class_names (List[str]): A list of subdirectory names, each representing a class.
                                     The order determines the label assignment (0, 1, ...).
            transform (callable, optional): A transform to be applied to each image.
            max_samples_per_class (int, optional): The maximum number of samples to load
                                                   for each class. Defaults to None (all samples).
        """
        self.root_path = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        self.image_paths = []
        self.labels = []

        log.info(f"Initializing dataset from root: {self.root_path}")
        self._find_samples(class_names, max_samples_per_class)

        if not self.image_paths:
            log.error(f"No images found in {self.root_path} for specified classes.")
            raise FileNotFoundError("Dataset is empty. Check data paths and class names.")

    def _find_samples(self, class_names: List[str], max_samples: Optional[int]):
        """Scans class subdirectories and populates the sample lists."""
        for class_name in class_names:
            class_dir = self.root_path / class_name
            class_idx = self.class_to_idx.get(class_name)

            if not class_dir.is_dir():
                log.warning(f"Class directory not found, skipping: {class_dir}")
                continue

            log.info(f"Loading images from '{class_name}' directory...")
            
            # Find all image files recursively, case-insensitively
            found_files = []
            for ext in self.SUPPORTED_EXTENSIONS:
                # Search for both lowercase and uppercase extensions
                found_files.extend(class_dir.rglob(f"*{ext}"))
                found_files.extend(class_dir.rglob(f"*{ext.upper()}"))
            
            # Remove duplicates that might arise from case-insensitive filesystems
            found_files = sorted(list(set(found_files)))

            if not found_files:
                log.warning(f"No images with supported extensions found in {class_dir}")
                continue

            if max_samples and max_samples > 0:
                log.info(f"Limiting to {max_samples} samples for class '{class_name}'.")
                found_files = found_files[:max_samples]

            self.image_paths.extend(found_files)
            self.labels.extend([class_idx] * len(found_files))
            log.info(f"Found {len(found_files)} images for class '{class_name}'.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - The transformed image as a PyTorch tensor.
            - The integer label.
            - The string path of the original image.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # Open image and ensure it's in RGB format
            image = Image.open(image_path).convert("RGB")
        except (IOError, OSError) as e:
            log.warning(f"Corrupt image file, replacing with dummy: {image_path}. Error: {e}")
            # Create a black placeholder image
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        # Apply transformations if they exist
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Basic conversion to tensor if no transform is provided
            image_tensor = torch.from_numpy(np.array(image))

        return image_tensor, label, str(image_path)

class CustomSampleDataset(Dataset):
    """A dataset created from an explicit list of (image_path, label) tuples."""
    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[callable] = None):
        """
        Args:
            samples (List[Tuple[str, int]]): A list of (image_path, label) tuples.
            transform (callable, optional): A transform to apply to images.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Retrieves a sample."""
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            # --- FIX IS HERE ---
            # Convert the path object to a string before returning
            return image, label, str(image_path) 
        except Exception as e:
            log.warning(f"Could not load image {image_path}: {e}. Returning a placeholder.")
            placeholder_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                placeholder_image = self.transform(placeholder_image)
            # --- AND FIX IS HERE ---
            # Also convert to string in the exception case
            return placeholder_image, label, str(image_path)