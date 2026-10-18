from typing import Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def convert_to_pil_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Converts input image data (PIL Image or NumPy array) into an RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def create_evaluation_transform(image_size: int, mean: list, std: list) -> T.Compose:
    """Creates the deterministic evaluation transformation pipeline for feature extraction."""
    return T.Compose([
        T.Lambda(convert_to_pil_image),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])


class SpeciesAwareFlip:
    """Applies horizontal flipping conditionally based on whether the sample's species is permitted."""
    def __init__(self, allowed_species: list, probability: float = 0.5):
        self.allowed_species = set(allowed_species)
        self.probability = probability

    def __call__(self, img: Image.Image, species: str) -> Image.Image:
        if species in self.allowed_species and torch.rand(1).item() < self.probability:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img


def create_training_transform(image_size: int, mean: list, std: list, config: dict) -> T.Compose:
    """
    Creates the stochastic training transformation pipeline incorporating 
    cropping, rotation, photometric jitter, blurring, and random erasing.
    """
    operations = [
        T.Lambda(convert_to_pil_image),
        T.RandomResizedCrop(
            (image_size, image_size),
            scale=tuple(config["random_resized_crop_scale"]),
            ratio=(0.85, 1.18),
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.RandomApply(
            [T.RandomRotation(config["rotation_deg"], interpolation=T.InterpolationMode.BILINEAR)],
            p=0.5
        ),
        T.ColorJitter(**config["color_jitter"])
    ]
    
    if config.get("gaussian_blur_p", 0) > 0:
        operations.append(
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=config["gaussian_blur_p"])
        )
        
    operations.extend([T.ToTensor(), T.Normalize(mean, std)])
    
    if config.get("random_erasing_p", 0) > 0:
        operations.append(
            T.RandomErasing(p=config["random_erasing_p"], scale=(0.02, 0.15), value="random")
        )
        
    return T.Compose(operations)


# Backwards compatibility aliases
to_pil = convert_to_pil_image
eval_transform = create_evaluation_transform
train_transform = create_training_transform