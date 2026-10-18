"""
General-Purpose Baseline Models

Provides general-purpose baseline models evaluated independently to benchmark against
the proposed fused Re-ID pipeline.

Note:
    These baseline models do not participate in candidate generation, score calibration,
    or multi-modal feature fusion.
"""

from typing import Any, Dict, Tuple, Union
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def convert_to_pil_image(image_input: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Converts input image arrays or PIL objects into standard RGB PIL Image format.

    Args:
        image_input: Input image as a PIL Image object or an HWC NumPy ndarray.

    Returns:
        RGB PIL Image object.

    Raises:
        TypeError: If image_input is not a PIL Image or NumPy ndarray.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input)
    raise TypeError(
        f"Unsupported image input type: {type(image_input)}. Expected PIL.Image.Image or np.ndarray."
    )


class DINOv2Baseline(nn.Module):
    """
    DINOv2 baseline model wrapper for feature extraction and benchmarking.

    Extracts [CLS] token representations from pretrained Vision Transformer backbones.
    """

    # ImageNet normalization statistics used during DINOv2 pretraining
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    STANDARD_INPUT_SIZE: int = 224

    # Embedding dimension dimensions mapped by Torch Hub model variant keys
    EMBEDDING_DIMENSIONS: Dict[str, int] = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_variant: str = "dinov2_vits14") -> None:
        """
        Args:
            model_variant: DINOv2 backbone variant name (e.g., 'dinov2_vits14', 'dinov2_vitb14').
        """
        super().__init__()

        self.model_variant = model_variant
        # Load pretrained DINOv2 backbone from PyTorch Hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_variant, verbose=False
        )
        self.embedding_dim = self.EMBEDDING_DIMENSIONS.get(model_variant, 384)

    def get_evaluation_transform(self) -> transforms.Compose:
        """Constructs image preprocessing pipeline for evaluation and inference."""
        return transforms.Compose([
            transforms.Lambda(convert_to_pil_image),
            transforms.Resize((self.STANDARD_INPUT_SIZE, self.STANDARD_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def get_training_transform(self) -> transforms.Compose:
        """Constructs image preprocessing pipeline with data augmentations for fine-tuning."""
        return transforms.Compose([
            transforms.Lambda(convert_to_pil_image),
            transforms.Resize((self.STANDARD_INPUT_SIZE, self.STANDARD_INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DINOv2 backbone.

        Args:
            input_tensors: Tensor batch of shape (B, C, H, W).

        Returns:
            Unnormalized [CLS] embedding tensor of shape (B, embedding_dim).
        """
        return self.backbone(input_tensors)

    def extract_l2_normalized_embeddings(
        self, input_tensors: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts L2-normalized feature embeddings for vector cosine similarity search.

        Args:
            input_tensors: Tensor batch of shape (B, C, H, W).

        Returns:
            L2-normalized embedding tensor of shape (B, embedding_dim) in float32 precision.
        """
        with torch.no_grad():
            raw_embeddings = self.forward(input_tensors).float()
            return F.normalize(raw_embeddings, p=2, dim=1)

    def save_checkpoint(self, checkpoint_filepath: str) -> None:
        """Saves backbone model weights to disk."""
        checkpoint_data = {
            "model_variant": self.model_variant,
            "backbone_state_dict": self.backbone.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath: str) -> None:
        """Loads backbone model weights from a saved checkpoint file."""
        checkpoint_state = torch.load(checkpoint_filepath, map_location="cpu")
        
        # Support both new and legacy key formats for backward compatibility
        backbone_weights = checkpoint_state.get(
            "backbone_state_dict", checkpoint_state.get("backbone")
        )
        self.backbone.load_state_dict(backbone_weights, strict=True)
        