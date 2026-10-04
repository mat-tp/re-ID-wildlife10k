"""
Train / eval transforms shared by ALL embedding models (global, EVA02, DINOv2).
Only the input size and normalisation statistics differ per model, so they are parameters.

Choices (why in DECISIONS.md #4):
 * Eval: plain resize to a square + normalise. MegaDescriptor's published usage does exactly this,
   and the benchmark baseline was evaluated this way -> comparable numbers.  [PAPER]
 * Train: mild RandomResizedCrop / rotation / photometric jitter / blur / random erasing.
   Random erasing + jitter are standard, cheap wins in re-ID (Luo et al., "Bag of Tricks", 2019). [PRACTICE]
 * NO horizontal flip by default. In animal Re-ID the left and right flank of a zebra, giraffe,
   leopard, tiger, seal... carry DIFFERENT patterns; a mirrored image looks like another animal.
   Flips are only enabled for species listed in config.augmentation['hflip_species'].
 * Hue jitter is tiny: coat / skin colour is itself an identity cue.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def eval_transform(size, mean, std):
    return T.Compose([T.Lambda(to_pil), T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(mean, std)])


class SpeciesAwareFlip:
    """Flip only when the sample's species is in the allow-list (needs the species at call time)."""
    def __init__(self, species, p=0.5):
        self.species, self.p = set(species), p

    def __call__(self, img, species):
        if species in self.species and torch.rand(1).item() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def train_transform(size, mean, std, cfg):
    """`cfg` = config.augmentation. Returns a callable(image) for PIL/ndarray input."""
    ops = [T.Lambda(to_pil),
           T.RandomResizedCrop((size, size), scale=tuple(cfg["random_resized_crop_scale"]),
                               ratio=(0.85, 1.18), interpolation=T.InterpolationMode.BICUBIC),
           T.RandomApply([T.RandomRotation(cfg["rotation_deg"], interpolation=T.InterpolationMode.BILINEAR)], p=0.5),
           T.ColorJitter(**cfg["color_jitter"])]
    if cfg.get("gaussian_blur_p", 0) > 0:
        ops.append(T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=cfg["gaussian_blur_p"]))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if cfg.get("random_erasing_p", 0) > 0:
        ops.append(T.RandomErasing(p=cfg["random_erasing_p"], scale=(0.02, 0.15), value="random"))
    return T.Compose(ops)
