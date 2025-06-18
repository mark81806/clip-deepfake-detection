# model.py

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class ForgeryDetector(nn.Module):
    """
    A deepfake detection model leveraging a pre-trained CLIP vision encoder.

    This model uses a frozen CLIP model to extract powerful image features.
    It then enhances these features in two ways:
    1. A trainable "adapter" network (a small MLP) refines the visual features.
    2. It computes the image's similarity to a set of hand-crafted text prompts
       related to authenticity and forgery.

    These refined visual features and text-similarity scores are combined
    and fed into a final classification head to produce a real/fake prediction.
    """

    def __init__(self, clip_model_name: str = "ViT-B/32", num_classes: int = 2):
        """
        Initializes the model, loads CLIP, and sets up trainable layers.

        Args:
            clip_model_name (str): The name of the CLIP model to load (e.g., "ViT-B/32").
            num_classes (int): The number of output classes (typically 2 for real/fake).
        """
        super().__init__()

        # --- Constants ---
        CLIP_EMBED_DIM = 512
        ADAPTER_HIDDEN_DIM = 256
        ADAPTER_OUTPUT_DIM = 128
        
        # --- Load and Freeze CLIP Backbone ---
        # We load the model on the CPU first; it will be moved to the correct device later.
        self.vision_backbone, self.image_processor = clip.load(clip_model_name, device="cpu")
        
        # Freeze all parameters of the pre-trained CLIP model.
        # We only want to train our custom adapter and classifier.
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # --- Text Prompts for Zero-Shot Guidance ---
        # These prompts help the model reason about the image content.
        # They are structured in pairs: (real description, fake description).
        self.detection_prompts: List[str] = [
            "a genuine photograph of a face", "a computer-generated image of a face",
            "an authentic person's portrait", "a digitally manipulated face",
            "a real person", "a synthetic AI-generated face"
        ]
        
        # Tokenize prompts and register as a buffer to ensure it moves with the model.
        prompt_tokens = clip.tokenize(self.detection_prompts)
        self.register_buffer("prompt_tokens", prompt_tokens)

        # --- Trainable Layers ---
        # 1. Projection Adapter: A small network to adapt CLIP features for our task.
        self.projection_adapter = nn.Sequential(
            nn.Linear(CLIP_EMBED_DIM, ADAPTER_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(ADAPTER_HIDDEN_DIM, ADAPTER_OUTPUT_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 2. Classification Head: Combines adapted features and text similarity scores.
        # The input size is the adapter's output dim + 6 engineered text features.
        num_text_features = 6
        classifier_input_dim = ADAPTER_OUTPUT_DIM + num_text_features
        
        self.classification_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model.

        Args:
            image_batch (torch.Tensor): A batch of preprocessed images.

        Returns:
            A tuple containing:
            - logits (torch.Tensor): Raw output scores for each class.
            - projected_visuals (torch.Tensor): The output from the projection adapter.
        """
        # --- Feature Extraction (using frozen CLIP) ---
        with torch.no_grad():
            # Encode images and text prompts
            visual_embeddings = self.vision_backbone.encode_image(image_batch)
            textual_embeddings = self.vision_backbone.encode_text(self.prompt_tokens)
            
            # Normalize embeddings to unit vectors for cosine similarity
            visual_embeddings = F.normalize(visual_embeddings, p=2, dim=-1)
            textual_embeddings = F.normalize(textual_embeddings, p=2, dim=-1)

        # --- Text-Image Similarity Features ---
        # Calculate cosine similarity between each image and all text prompts.
        # Shape: [batch_size, num_prompts]
        prompt_similarities = visual_embeddings @ textual_embeddings.T

        # Engineer features from the similarity scores
        # Even indices (0, 2, 4) correspond to "real" prompts
        # Odd indices (1, 3, 5) correspond to "fake" prompts
        real_scores = prompt_similarities[:, 0::2].mean(dim=1, keepdim=True)
        fake_scores = prompt_similarities[:, 1::2].mean(dim=1, keepdim=True)

        text_derived_features = torch.cat([
            real_scores,                                  # 1. Avg similarity to "real" prompts
            fake_scores,                                  # 2. Avg similarity to "fake" prompts
            prompt_similarities.mean(dim=1, keepdim=True),# 3. Overall avg similarity
            (real_scores - fake_scores),                  # 4. Difference between real and fake scores
            prompt_similarities.std(dim=1, keepdim=True), # 5. Std dev of similarities (confidence)
            prompt_similarities.max(dim=1, keepdim=True)[0] # 6. Max similarity score
        ], dim=1)

        # --- Feature Adaptation and Classification ---
        # Pass visual features through the trainable adapter
        projected_visuals = self.projection_adapter(visual_embeddings)
        
        # Combine the adapted visual features with the text-derived features
        combined_features = torch.cat([projected_visuals, text_derived_features], dim=1)

        # Get final classification logits
        logits = self.classification_head(combined_features)
        
        return logits, projected_visuals