"""
General-purpose baseline models.

These models are evaluated separately to benchmark against the proposed
fused Re-ID pipeline. They do not participate in the calibration or fusion steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Helper function to convert images to PIL format
def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")


# DINOv2 model implementation - general-purpose baseline
class DINOv2Baseline(nn.Module):

    # Normalisation constants from DINOv2 pretraining
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)
    _INPUT_SIZE = 224

    # Embedding dimensions for each hub variant
    _DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, variant: str = "dinov2_vits14"):
        super().__init__()

        self.variant = variant
        self.backbone = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        self.embedding_dim = self._DIMS.get(variant, 384)

    # Preprocessing for inference
    def eval_transform(self):
        return transforms.Compose([
            transforms.Lambda(_to_pil),
            transforms.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # Preprocessing for training with augmentation
    def train_transform(self):
        return transforms.Compose([
            transforms.Lambda(_to_pil),
            transforms.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    # Returns the [CLS] token embedding
    def forward(self, x):
        return self.backbone(x)

    # Extract L2-normalised embeddings
    def embed(self, x):
        with torch.no_grad():
            return F.normalize(self.forward(x).float(), dim=1)

    # Save backbone weights to disk
    def save(self, path):
        torch.save(
            {"variant": self.variant, "backbone": self.backbone.state_dict()},
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(state["backbone"], strict=True)
