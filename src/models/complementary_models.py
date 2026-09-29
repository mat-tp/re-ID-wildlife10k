"""
Complementary visual representations for wildlife individual Re-ID.

Models implemented:
1. ComplementaryModel (EVA02) - general visual representation from large-scale image data
   - Not specifically trained for wildlife Re-ID
   - Provides complementary information alongside the global descriptor

2. DINOv2Baseline - self-supervised vision transformer
   - Used as a general-purpose baseline (not part of fusion)
   - Evaluated separately to compare against wildlife-specific representations

Reference:
- EVA02 paper: Fang et al., Image and Vision Computing 149 (2024)
- DINOv2 paper: Oquab et al., arXiv:2304.07193 (2023)
- AnimalCLEF 2025 fusion approach: Kim et al., CLEF Working Notes (2025)
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


# DINOv2 model implementation - general-purpose baseline (not part of fusion)
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


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("EVA02 smoke test  (ComplementaryModel)")
    print("=" * 60)
    eva02 = ComplementaryModel().eval()
    print(f"  timm_id       : {eva02.timm_id}")
    print(f"  input_size    : {eva02.input_size}")
    print(f"  embedding_dim : {eva02.embedding_dim}")

    dummy = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    t0 = time.time()
    z = eva02.embed(eva02.eval_transform()(dummy).unsqueeze(0))
    print(f"  output shape  : {tuple(z.shape)}")
    print(f"  L2 norm       : {float(z.norm()):.6f}  (expected ~1.0)")
    print(f"  inference time: {time.time() - t0:.3f}s")

    print()
    print("=" * 60)
    print("DINOv2 smoke test  (DINOv2Baseline)")
    print("=" * 60)
    dino = DINOv2Baseline().eval()
    print(f"  variant       : {dino.variant}")
    print(f"  embedding_dim : {dino.embedding_dim}")

    t0 = time.time()
    z2 = dino.embed(dino.eval_transform()(dummy).unsqueeze(0))
    print(f"  output shape  : {tuple(z2.shape)}")
    print(f"  L2 norm       : {float(z2.norm()):.6f}  (expected ~1.0)")
    print(f"  inference time: {time.time() - t0:.3f}s")
