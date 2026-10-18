"""
Global Model Representations and Fine-Tuning Pipeline

Provides whole-animal global feature embedding models for wildlife re-identification:
1. MegaDescriptor (Swin Transformer variants: L-384, B-224, T-224)
2. MiewID (EfficientNetV2 + GeM pooling + BatchNorm1d)

Includes ArcFace loss header implementation, chunked feature extraction, and full 
fine-tuning pipelines with cosine learning rate scheduling and state checkpointing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config

from src.data_processing.augmentations import eval_transform, train_transform

# MegaDescriptor model variants mapped to Hugging Face and ImageNet-initialized backbones
MEGADESCRIPTOR_VARIANTS: Dict[str, Dict[str, Union[str, int]]] = {
    "L-384": {
        "huggingface_id": "hf-hub:BVRA/MegaDescriptor-L-384",
        "embedding_dim": 1536,
        "imagenet_backbone": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    },
    "B-224": {
        "huggingface_id": "hf-hub:BVRA/MegaDescriptor-B-224",
        "embedding_dim": 1024,
        "imagenet_backbone": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    },
    "T-224": {
        "huggingface_id": "hf-hub:BVRA/MegaDescriptor-T-224",
        "embedding_dim": 768,
        "imagenet_backbone": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    },
}


class BaseEmbeddingModel(nn.Module):
    """Abstract base module for whole-animal embedding models."""

    embedding_dim: int
    input_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    def get_evaluation_transform(self) -> Callable:
        """Constructs image evaluation transformation pipeline."""
        return eval_transform(self.input_size, self.mean, self.std)

    def get_training_transform(self, augmentation_config: Dict[str, Any]) -> Callable:
        """Constructs image training transformation pipeline with data augmentations."""
        return train_transform(
            self.input_size, self.mean, self.std, augmentation_config
        )

    @torch.no_grad()
    def embed(self, input_tensors: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized feature representations for cosine similarity evaluation.

        Args:
            input_tensors: Image tensor batch of shape (B, C, H, W).

        Returns:
            L2-normalized feature embeddings of shape (B, embedding_dim) in float32.
        """
        raw_embeddings = self.forward(input_tensors)
        return F.normalize(raw_embeddings.float(), dim=1)


class GlobalModel(BaseEmbeddingModel):
    """
    MegaDescriptor Swin Transformer backbone for global image representation extraction.
    """

    def __init__(
        self,
        variant: str = "L-384",
        initialization: str = "megadescriptor",
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        """
        Args:
            variant: Architecture variant key ('L-384', 'B-224', or 'T-224').
            initialization: Pretrained weight source ('megadescriptor' or 'imagenet').
            enable_gradient_checkpointing: Whether to enable memory-efficient gradient checkpointing.
        """
        super().__init__()

        if variant not in MEGADESCRIPTOR_VARIANTS:
            raise ValueError(
                f"Invalid variant '{variant}'. Choose from: {list(MEGADESCRIPTOR_VARIANTS.keys())}"
            )

        variant_spec = MEGADESCRIPTOR_VARIANTS[variant]
        self.variant = variant
        self.initialization = initialization

        # Determine backbone identifier string based on requested initialization
        backbone_model_name = (
            variant_spec["huggingface_id"]
            if initialization == "megadescriptor"
            else variant_spec["imagenet_backbone"]
        )

        self.backbone = timm.create_model(
            backbone_model_name, pretrained=True, num_classes=0
        )

        if enable_gradient_checkpointing and hasattr(
            self.backbone, "set_grad_checkpointing"
        ):
            self.backbone.set_grad_checkpointing(True)

        self.embedding_dim = self.backbone.num_features

        # Extract data configurations (input resolution, mean, std) from timm model
        data_config = resolve_data_config({}, model=self.backbone)
        self.data_config = data_config
        self.input_size = int(data_config["input_size"][-1])
        self.mean = tuple(data_config["mean"])
        self.std = tuple(data_config["std"])

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Swin backbone."""
        return self.backbone(input_tensors)

    def save_checkpoint(self, checkpoint_filepath: str) -> None:
        """Saves model variant metadata and backbone state weights to disk."""
        directory_path = os.path.dirname(os.path.abspath(checkpoint_filepath))
        os.makedirs(directory_path, exist_ok=True)

        checkpoint_payload = {
            "variant": self.variant,
            "initialization": self.initialization,
            "backbone_state_dict": self.backbone.state_dict(),
        }
        torch.save(checkpoint_payload, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath: str) -> None:
        """Loads state weights from a saved checkpoint file."""
        checkpoint_data = torch.load(checkpoint_filepath, map_location="cpu")

        # Fallback support for legacy state dictionary key names
        state_dict_key = (
            "model_state"
            if "model_state" in checkpoint_data
            else "backbone_state_dict"
            if "backbone_state_dict" in checkpoint_data
            else "backbone"
        )
        self.backbone.load_state_dict(checkpoint_data[state_dict_key], strict=True)


class GeneralizedMeanPooling2d(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer for fine-grained feature pooling.
    
    Formula: f(X) = (1/N * sum(X^p))^(1/p)
    """

    def __init__(self, power: float = 3.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.power = nn.Parameter(torch.ones(1) * power)
        self.epsilon = epsilon

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        clamped_features = feature_map.clamp(min=self.epsilon)
        powered_features = clamped_features.pow(self.power)

        spatial_height, spatial_width = feature_map.size(-2), feature_map.size(-1)
        pooled_features = F.avg_pool2d(
            powered_features, (spatial_height, spatial_width)
        )

        return pooled_features.pow(1.0 / self.power)


class MiewIDModel(BaseEmbeddingModel):
    """
    MiewID Model (EfficientNetV2 + GeM Pooling + Batch Normalization).

    Loads official pretrained weights directly from Hugging Face (`conservationxlabs/miewid-msv3`).
    """

    DEFAULT_HUGGINGFACE_ID: str = "conservationxlabs/miewid-msv3"

    def __init__(
        self,
        huggingface_id: Optional[str] = None,
        input_resolution: int = 440,
    ) -> None:
        """
        Args:
            huggingface_id: Optional custom Hugging Face model repository ID.
            input_resolution: Expected input image resolution (width and height).
        """
        super().__init__()

        from huggingface_hub import hf_hub_download
        import safetensors.torch

        self.input_size = input_resolution
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.embedding_dim = 2152

        # Initialize EfficientNetV2 backbone and replace standard global pooling with GeM
        self.backbone = timm.create_model(
            "efficientnetv2_rw_m", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeneralizedMeanPooling2d()
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

        target_repo_id = huggingface_id or self.DEFAULT_HUGGINGFACE_ID
        weights_filepath = hf_hub_download(target_repo_id, "model.safetensors")
        state_dict = safetensors.torch.load_file(weights_filepath)

        # Separate backbone weights from batch normalization weights
        backbone_state_dict = {
            key.replace("backbone.", ""): tensor
            for key, tensor in state_dict.items()
            if key.startswith("backbone.")
        }
        batch_norm_state_dict = {
            key.replace("bn.", ""): tensor
            for key, tensor in state_dict.items()
            if key.startswith("bn.")
        }

        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        self.batch_norm.load_state_dict(batch_norm_state_dict, strict=True)

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_tensors)
        flattened_features = features.view(features.shape[0], -1)
        return self.batch_norm(flattened_features)


class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) Head.

    Computes decision logits: s * cos(theta + margin) for true target classes.
    Reference: Deng et al. (CVPR 2019)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of incoming feature vectors.
            num_classes: Total number of target identity classes.
            scale: Feature scaling factor s.
            margin: Angular margin m in radians.
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

        # Monotonicity boundary conditions when (theta + margin) > pi
        self.threshold = math.cos(math.pi - margin)
        self.margin_penalty = math.sin(math.pi - margin) * margin

    def forward(self, feature_embeddings: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_embeddings: Tensor batch of shape (B, embedding_dim).
            target_labels: Class index tensor batch of shape (B,).

        Returns:
            Scaled ArcFace logits of shape (B, num_classes).
        """
        # Normalize weights and feature vectors to compute cosine angles
        normalized_features = F.normalize(feature_embeddings.float())
        normalized_weights = F.normalize(self.weight.float())
        cosine_similarity = F.linear(normalized_features, normalized_weights)

        # Compute sin(theta) = sqrt(1 - cos(theta)^2)
        sine_similarity = torch.sqrt(
            (1.0 - cosine_similarity ** 2).clamp(1e-7, 1.0)
        )

        # Trigonometric expansion: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi_cosine = (
            cosine_similarity * self.cos_margin - sine_similarity * self.sin_margin
        )

        # Maintain monotonicity for extreme angles
        phi_cosine = torch.where(
            cosine_similarity > self.threshold,
            phi_cosine,
            cosine_similarity - self.margin_penalty,
        )

        # Construct one-hot target class indicator mask
        one_hot_mask = F.one_hot(
            target_labels, num_classes=cosine_similarity.size(1)
        ).bool()

        # Apply angular margin penalty exclusively to true target class logits
        output_logits = torch.where(one_hot_mask, phi_cosine, cosine_similarity)
        return self.scale * output_logits


def resolve_device(user_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolves computational device, prioritizing CUDA if available."""
    if user_device:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(
    model: BaseEmbeddingModel,
    data_loader: torch.utils.data.DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    use_automatic_mixed_precision: bool = True,
    progress_bar_description: str = "Extracting features",
) -> Tuple[torch.Tensor, Tuple[List[Any], ...]]:
    """
    Extracts L2-normalized feature embeddings and metadata across a data loader.

    Args:
        model: Embedding model instance.
        data_loader: PyTorch DataLoader yielding batches of (images, *metadata).
        device: Computational device target.
        use_automatic_mixed_precision: Whether to use CUDA float16 mixed precision.
        progress_bar_description: Description label for tqdm progress display.

    Returns:
        Tuple containing:
            - concatenated_features: CPU Tensor of shape (N_samples, embedding_dim).
            - extracted_metadata: Tuple of lists containing metadata fields in loader order.
    """
    from tqdm import tqdm

    target_device = resolve_device(device)
    model.eval().to(target_device)

    feature_batches: List[torch.Tensor] = []
    metadata_buckets: Optional[List[List[Any]]] = None

    for batch_items in tqdm(data_loader, desc=progress_bar_description, dynamic_ncols=True):
        image_tensors = batch_items[0]
        metadata_fields = batch_items[1:]

        is_cuda_enabled = use_automatic_mixed_precision and target_device.type == "cuda"
        with torch.autocast(
            device_type=target_device.type, dtype=torch.float16, enabled=is_cuda_enabled
        ):
            embeddings = model.embed(image_tensors.to(target_device, non_blocking=True))

        feature_batches.append(embeddings.float().cpu())

        if metadata_buckets is None:
            metadata_buckets = [[] for _ in metadata_fields]

        for bucket, field_data in zip(metadata_buckets, metadata_fields):
            if isinstance(field_data, torch.Tensor):
                bucket.extend(field_data.tolist())
            else:
                bucket.extend(list(field_data))

    concatenated_features = torch.cat(feature_batches, dim=0)
    tuple_metadata = tuple(metadata_buckets or [])

    return concatenated_features, tuple_metadata


def fine_tune_global_model(
    model: BaseEmbeddingModel,
    train_loader: torch.utils.data.DataLoader,
    num_classes: int,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    head_learning_rate_multiplier: float = 10.0,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 1,
    arcface_margin: float = 0.5,
    arcface_scale: float = 64.0,
    gradient_accumulation_steps: int = 8,
    max_gradient_norm: float = 5.0,
    use_automatic_mixed_precision: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    log_interval_steps: int = 50,
    evaluation_function: Optional[Callable[[BaseEmbeddingModel], float]] = None,
    evaluation_interval_epochs: int = 2,
    checkpoint_filepath: Optional[str] = None,
    resume_from_checkpoint: bool = True,
) -> ArcFaceHead:
    """
    Fine-tunes global embedding models using ArcFace classification loss.

    Features cosine decay with linear warmup, separate learning rates for fresh ArcFace heads,
    gradient scaling/accumulation, and checkpoint state restoration.

    Args:
        model: Global feature embedding model.
        train_loader: DataLoader yielding (images, target_labels, row_indices).
        num_classes: Total unique identity classes in training set.
        epochs: Total training epochs.
        learning_rate: Base learning rate for backbone parameters.
        head_learning_rate_multiplier: Multiplier for ArcFace head learning rate.
        weight_decay: Optimizer weight decay factor.
        warmup_epochs: Number of linear warmup epochs.
        arcface_margin: Angular margin penalty m.
        arcface_scale: Feature scaling factor s.
        gradient_accumulation_steps: Steps to accumulate gradients before optimizer update.
        max_gradient_norm: Maximum norm threshold for gradient clipping.
        use_automatic_mixed_precision: Enable mixed precision training.
        device: Target execution device.
        log_interval_steps: Step interval for printing training progress.
        evaluation_function: Function evaluating model score (higher is better).
        evaluation_interval_epochs: Epoch interval between validation calls.
        checkpoint_filepath: Filepath destination for saving best checkpoint weights.
        resume_from_checkpoint: Whether to resume training from an existing `_latest.pt` file.

    Returns:
        Trained ArcFaceHead instance.
    """
    target_device = resolve_device(device)
    model.to(target_device)

    # Initialize ArcFace head layer
    arcface_head = ArcFaceHead(
        embedding_dim=model.embedding_dim,
        num_classes=num_classes,
        scale=arcface_scale,
        margin=arcface_margin,
    ).to(target_device)

    # Parameter groups with differential learning rates
    optimizer = torch.optim.SGD(
        [
            {"params": model.parameters(), "lr": learning_rate},
            {
                "params": arcface_head.parameters(),
                "lr": learning_rate * head_learning_rate_multiplier,
            },
        ],
        momentum=0.9,
        weight_decay=weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // gradient_accumulation_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # Learning rate schedule: Linear warmup followed by Cosine Annealing Decay
    def compute_lr_factor(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=compute_lr_factor)

    is_amp_enabled = use_automatic_mixed_precision and target_device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler(
        target_device.type, enabled=is_amp_enabled
    )

    best_validation_score = -1.0
    starting_epoch = 0
    latest_checkpoint_path: Optional[str] = None

    if checkpoint_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_filepath)), exist_ok=True)
        latest_checkpoint_path = checkpoint_filepath.replace("_best.pt", "_latest.pt")

        if resume_from_checkpoint and os.path.isfile(latest_checkpoint_path):
            print(f"[Resume] Loading checkpoint from {latest_checkpoint_path}...")
            checkpoint = torch.load(latest_checkpoint_path, map_location=target_device)

            model.backbone.load_state_dict(checkpoint["model_state"])
            arcface_head.load_state_dict(checkpoint["head_state"])
            optimizer.load_state_dict(checkpoint["opt_state"])
            lr_scheduler.load_state_dict(checkpoint["sched_state"])
            gradient_scaler.load_state_dict(checkpoint["scaler_state"])

            best_validation_score = checkpoint.get("best_score", -1.0)
            starting_epoch = checkpoint["epoch"] + 1
            print(
                f"[Resume] Resuming training at epoch {starting_epoch + 1}/{epochs} "
                f"(Best validation score: {best_validation_score:.4f})"
            )

    for epoch in range(starting_epoch, epochs):
        model.train()
        arcface_head.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        step_count = 0

        for step_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(target_device, non_blocking=True)
            targets = targets.to(target_device, non_blocking=True)

            with torch.autocast(
                device_type=target_device.type, dtype=torch.float16, enabled=is_amp_enabled
            ):
                embeddings = model(images)
                logits = arcface_head(embeddings, targets)
                loss = F.cross_entropy(logits, targets)

            # Normalize loss for accumulated gradients
            normalized_loss = loss / gradient_accumulation_steps
            gradient_scaler.scale(normalized_loss).backward()

            if (step_idx + 1) % gradient_accumulation_steps == 0:
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(arcface_head.parameters()),
                    max_norm=max_gradient_norm,
                )

                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            step_count += 1

            if log_interval_steps and (step_idx + 1) % log_interval_steps == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                average_loss = running_loss / step_count
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step_idx + 1}/{len(train_loader)}]: "
                    f"Loss = {average_loss:.4f} | LR = {current_lr:.2e}"
                )

        mean_epoch_loss = running_loss / max(step_count, 1)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch + 1}/{epochs}] Finished - Mean Loss = {mean_epoch_loss:.4f} | "
            f"LR = {current_lr:.2e}"
        )

        # Save state to latest checkpoint after every epoch
        if latest_checkpoint_path:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.backbone.state_dict(),
                    "head_state": arcface_head.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "sched_state": lr_scheduler.state_dict(),
                    "scaler_state": gradient_scaler.state_dict(),
                    "best_score": best_validation_score,
                },
                latest_checkpoint_path,
            )

        # Periodic validation step
        is_final_epoch = (epoch + 1) == epochs
        is_eval_epoch = (epoch + 1) % evaluation_interval_epochs == 0

        if evaluation_function is not None and (is_eval_epoch or is_final_epoch):
            current_score = evaluation_function(model)
            print(
                f"  Validation Score: {current_score:.4f} "
                f"(Previous Best: {max(best_validation_score, current_score):.4f})"
            )

            if current_score > best_validation_score and checkpoint_filepath:
                best_validation_score = current_score
                if hasattr(model, "save_checkpoint"):
                    model.save_checkpoint(checkpoint_filepath)
                else:
                    torch.save(
                        {"model_state": model.backbone.state_dict()}, checkpoint_filepath
                    )

            best_validation_score = max(best_validation_score, current_score)

    return arcface_head