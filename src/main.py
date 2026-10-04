"""
Smoke-test entry point for the complementary features pipeline.

Run from the project root:
    python -m src.main

This script verifies that:
1. ComplementaryModel (EVA02) loads, preprocesses, and embeds a synthetic
   image correctly — shape (1, 768), L2 norm ≈ 1.0.
2. DINOv2Baseline loads and embeds correctly — shape (1, 384), norm ≈ 1.0.

No dataset is required; both tests use a randomly generated numpy image.
"""

import time
import numpy as np
import torch

from src.models.complementary_models import ComplementaryModel
from src.models.baseline_models import DINOv2Baseline


def _check_embedding(name: str, model, dummy_img: np.ndarray) -> None:
    """Run a single forward pass and print a summary."""
    model.eval()
    transform = model.eval_transform()
    tensor = transform(dummy_img).unsqueeze(0)  # (1, C, H, W)

    t0 = time.time()
    z = model.embed(tensor)                      # (1, embedding_dim)
    elapsed = time.time() - t0

    norm = float(z.norm())
    print(f"  output shape  : {tuple(z.shape)}")
    print(f"  L2 norm       : {norm:.6f}  (expected ~1.0)")
    print(f"  inference time: {elapsed:.3f}s")
    assert z.ndim == 2,                           "Expected 2-D output (B, D)"
    assert abs(norm - 1.0) < 1e-4,               f"Norm {norm:.6f} is not unit — check L2 normalisation"
    print(f"  [{name}] PASS\n")


def main() -> None:
    # Synthetic image — same approach used in global_models.py __main__
    dummy = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # EVA02 – complementary representation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("EVA02 smoke test  (ComplementaryModel)")
    print("=" * 60)
    eva02 = ComplementaryModel()  # uses DEFAULT_TIMM_ID
    print(f"  timm_id       : {eva02.timm_id}")
    print(f"  input_size    : {eva02.input_size}")
    print(f"  embedding_dim : {eva02.embedding_dim}")
    _check_embedding("EVA02", eva02, dummy)

    # ------------------------------------------------------------------
    # DINOv2 – general-purpose baseline
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DINOv2 smoke test  (DINOv2Baseline)")
    print("=" * 60)
    dino = DINOv2Baseline()       # uses dinov2_vits14 by default
    print(f"  variant       : {dino.variant}")
    print(f"  embedding_dim : {dino.embedding_dim}")
    _check_embedding("DINOv2", dino, dummy)

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()