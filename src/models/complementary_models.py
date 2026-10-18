"""
Complementary Visual Representations for Wildlife Individual Re-ID

Implements the EVA02 vision transformer backbone to extract complementary general-purpose
visual features for wildlife individual re-identification tasks.

Reference:
- EVA02 paper: Fang et al., Image and Vision Computing 149 (2024)
"""

from typing import Dict, Optional, Tuple, Union
import os
import numpy as np
from PIL import Image
import timm
from timm.data import resolve_model_data_config
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


def resolve_execution_device(
    user_device: Optional[Union[str, torch.device]] = None
) -> torch.device:
    """
    Resolves target PyTorch execution device, defaulting to CUDA if available.

    Args:
        user_device: Optional user-specified target device string or torch.device.

    Returns:
        Resolved torch.device object.
    """
    if user_device is not None:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComplementaryFeatureExtractor(nn.Module):
    """
    EVA02 vision transformer model wrapper for extracting complementary visual features.

    Extracts general-purpose visual representations pre-trained on large-scale image data
    to complement domain-specific global feature descriptors during multi-modal fusion.
    """

    DEFAULT_TIMM_MODEL_ID: str = "eva02_base_patch14_224.mim_in22k"

    def __init__(self, timm_model_id: str = DEFAULT_TIMM_MODEL_ID) -> None:
        """
        Args:
            timm_model_id: Architecture identifier string for timm model instantiation.
        """
        super().__init__()

        self.timm_model_id = timm_model_id

        # Instantiates EVA02 backbone without classification head, using global average pooling
        try:
            self.backbone = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )
        except Exception:
            # Fall back to offline HuggingFace local hub cache if network request fails
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.backbone = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )

        # Resolve model data configurations (mean, std, and input image resolution) dynamically
        self.data_config: Dict[str, object] = resolve_model_data_config(self.backbone)
        self.input_resolution: int = int(self.data_config["input_size"][-1])
        self.embedding_dim: int = self.backbone.num_features

    def get_evaluation_transform(self) -> transforms.Compose:
        """Constructs image preprocessing pipeline for evaluation and inference."""
        return transforms.Compose([
            transforms.Lambda(convert_to_pil_image),
            transforms.Resize((self.input_resolution, self.input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config["mean"],
                std=self.data_config["std"],
            ),
        ])

    def get_training_transform(self) -> transforms.Compose:
        """Constructs image preprocessing pipeline with data augmentations for fine-tuning."""
        return transforms.Compose([
            transforms.Lambda(convert_to_pil_image),
            transforms.Resize((self.input_resolution, self.input_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.data_config["mean"],
                std=self.data_config["std"],
            ),
        ])

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EVA02 vision transformer backbone.

        Args:
            input_tensors: Input tensor batch of shape (B, C, H, W).

        Returns:
            Unnormalized global feature embeddings of shape (B, embedding_dim).
        """
        return self.backbone(input_tensors)

    def extract_l2_normalized_embeddings(
        self, input_tensors: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts L2-normalized feature representations for cosine similarity evaluation.

        Args:
            input_tensors: Input tensor batch of shape (B, C, H, W).

        Returns:
            L2-normalized feature embedding tensor of shape (B, embedding_dim) in float32 precision.
        """
        with torch.no_grad():
            raw_embeddings = self.forward(input_tensors).float()
            return F.normalize(raw_embeddings, p=2, dim=1)

    def save_checkpoint(self, checkpoint_filepath: str) -> None:
        """Saves model configuration and backbone weights to disk."""
        checkpoint_data = {
            "timm_model_id": self.timm_model_id,
            "backbone_state_dict": self.backbone.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath: str) -> None:
        """Loads backbone state weights from a saved checkpoint file."""
        checkpoint_state = torch.load(checkpoint_filepath, map_location="cpu")

        # Supports both legacy and new checkpoint dictionary keys
        backbone_weights = checkpoint_state.get(
            "backbone_state_dict", checkpoint_state.get("backbone")
        )
        self.backbone.load_state_dict(backbone_weights, strict=True)