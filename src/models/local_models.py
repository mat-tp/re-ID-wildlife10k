"""
This file implements local feature extraction and matching methods that capture
fine-grained spatial patterns, such as spots, stripes, scars, or other distinctive
markings on individual animals.

Models to be implemented:
1. ALIKED: A Lighter Keypoint and Descriptor extraction network
   - Deformable transformation based
   - Keypoint detection and descriptor extraction
   - Provides correspondence between images

Unlike global descriptors, ALIKED does not produce a fixed-length embedding.
Instead, it detects keypoints and produces matching scores between image pairs.

Key aspects:
- Local keypoint detection
- Descriptor extraction
- Matching between image pairs
- Provides complementary evidence to global descriptors

Reference: ALIKED paper, WildFusion methodology

TODO: Implement ALIKED keypoint detection
TODO: Implement keypoint descriptor extraction
TODO: Implement matching between image pairs
TODO: Implement score computation from keypoint matches
TODO: Add configurable keypoint parameters (num_features, threshold)
"""

# Fill in code for local models here

from typing import Dict, Any, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd


# Helper function to convert images to a float RGB tensor [3, H, W] in [0, 1]
def to_tensor(image):
   if isinstance(image, Image.Image):
      image = np.asarray(image.convert("RGB"))
   elif not isinstance(image, np.ndarray):
      raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")

   if image.ndim == 2:                                  # grayscale -> 3 channels
      image = np.stack([image] * 3, axis=-1)
   elif image.ndim == 3 and image.shape[2] == 4:        # RGBA -> RGB
      image = image[..., :3]
   elif image.ndim != 3 or image.shape[2] != 3:
      raise ValueError(f"Unsupported image shape: {image.shape}")

   image = np.ascontiguousarray(image)
   if image.dtype == np.uint8:
      image = image.astype(np.float32) / 255.0
   else:                                                # assume float already in [0, 1]
      image = image.astype(np.float32)
   return torch.from_numpy(image).permute(2, 0, 1)      # NOTE: expects RGB, not BGR (OpenCV)


# ALIKED keypoint extraction + LightGlue matching
class LocalModel:
   def __init__(self, model_name="aliked-n16", max_keypoints=2048,
                detection_threshold=0.2, match_threshold=0.1,
                resize=1024, device=None):

      self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
      self.resize = resize
      self.extractor = ALIKED(model_name=model_name,
                              max_num_keypoints=max_keypoints,
                              detection_threshold=detection_threshold).eval().to(self.device)
      self.matcher = LightGlue(features="aliked",
                               filter_threshold=match_threshold).eval().to(self.device)

   # Detect keypoints and compute descriptors for one image
   @torch.no_grad()
   def extract(self, image) -> Dict[str, torch.Tensor]:
      img = to_tensor(image).to(self.device)
      # keypoints [1,N,2] (in original image coords), descriptors [1,N,128], keypoint_scores [1,N]
      return self.extractor.extract(img, resize=self.resize)

   # Match keypoints between two already-extracted feature sets
   @torch.no_grad()
   def match(self, feats0, feats1) -> Dict[str, Any]:
      out = self.matcher({"image0": feats0, "image1": feats1})
      f0, f1, m = [rbd(x) for x in (feats0, feats1, out)]   # remove batch dimension
      idx = m["matches"]                                     # [K, 2] index pairs
      return {
         "kpts0": f0["keypoints"][idx[:, 0]].cpu().numpy(),
         "kpts1": f1["keypoints"][idx[:, 1]].cpu().numpy(),
         "scores": m["scores"].cpu().numpy(),
         "num_matches": int(idx.shape[0]),
      }

   # Geometric verification: returns the number of RANSAC inliers (needs >= 8 matches)
   @staticmethod
   def geometric_inliers(kpts0, kpts1, thresh=3.0) -> int:
      if kpts0 is None or kpts1 is None or len(kpts0) < 8 or len(kpts1) < 8:
         return 0
      try:
         _, mask = cv2.findFundamentalMat(kpts0, kpts1, cv2.USAC_MAGSAC, thresh, 0.999, 10000)
         return int(mask.sum()) if mask is not None else 0
      except Exception:
         try:
            _, mask = cv2.findFundamentalMat(kpts0, kpts1, cv2.FM_RANSAC, thresh, 0.999, 10000)
            return int(mask.sum()) if mask is not None else 0
         except Exception:
            return 0

   # Scalar score for a pair of already-extracted feature sets
   def score(self, feats0, feats1, use_ransac=True) -> float:
      m = self.match(feats0, feats1)
      if use_ransac:
         return float(self.geometric_inliers(m["kpts0"], m["kpts1"]))
      return float(m["num_matches"])

   # Convenience: score directly from two raw images
   def similarity(self, img0, img1, use_ransac=True) -> float:
      return self.score(self.extract(img0), self.extract(img1), use_ransac)


if __name__ == "__main__":
   model = LocalModel()
   a = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
   b = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
   fa = model.extract(a)
   print("keypoints:", tuple(fa["keypoints"].shape), "descriptors:", tuple(fa["descriptors"].shape))
   print("score (random noise, expect ~0):", model.similarity(a, b))