"""
Complementary visual representations for wildlife individual Re-ID.

Models implemented:
1. ComplementaryModel (EVA02) - general visual representation from large-scale image data
   - Not specifically trained for wildlife Re-ID
   - Provides complementary information alongside the global descriptor

Reference:
- EVA02 paper: Fang et al., Image and Vision Computing 149 (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
from timm.data import resolve_model_data_config


# Helper function to convert images to PIL format
def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")


# Helper to resolve device
def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# EVA02 model implementation - complementary representation
class ComplementaryModel(nn.Module):

    DEFAULT_TIMM_ID = "eva02_base_patch14_224.mim_in22k"

    def __init__(self, timm_id: str = DEFAULT_TIMM_ID):
        super().__init__()

        self.timm_id = timm_id

        # Load EVA02 without classification head, using average pooling for fixed-length output
        self.backbone = timm.create_model(
            timm_id,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Resolve preprocessing config from the model so mean, std and input size always match
        self.data_config = resolve_model_data_config(self.backbone)
        self.input_size = int(self.data_config["input_size"][-1])
        self.embedding_dim = self.backbone.num_features

    # Preprocessing for inference
    def eval_transform(self):
        return transforms.Compose([
            transforms.Lambda(_to_pil),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config["mean"],
                std=self.data_config["std"],
            ),
        ])

    # Preprocessing for training with augmentation
    def train_transform(self):
        return transforms.Compose([
            transforms.Lambda(_to_pil),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config["mean"],
                std=self.data_config["std"],
            ),
        ])

    def forward(self, x):
        return self.backbone(x)

    # Extract L2-normalised embeddings
    def embed(self, x):
        with torch.no_grad():
            return F.normalize(self.forward(x).float(), dim=1)

    # Save backbone weights to disk
    def save(self, path):
        torch.save(
            {"timm_id": self.timm_id, "backbone": self.backbone.state_dict()},
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(state["backbone"], strict=True)




