"""
Local Feature Extraction, Matching, and Geometric Verification

Captures fine-grained spatial patterns (such as animal spots, stripes, scars, or distinct visual markings)
for individual animal re-identification.

Pipeline Overview:
1. ALIKED: Deformable keypoint detector and descriptor extractor network.
2. LightGlue: Deep neural network feature matcher for reliable keypoint correspondences.
3. RANSAC / MAGSAC: Geometric verification returning the count of geometrically consistent inlier matches.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
import numpy as np
from PIL import Image
import torch


def convert_image_to_float_tensor(
    image_input: Union[Image.Image, np.ndarray]
) -> torch.Tensor:
    """
    Converts various input image representations (PIL Images, NumPy ndarrays) into a float32 RGB tensor
    formatted as [3, H, W] with pixel intensity values normalized to the range [0.0, 1.0].

    Args:
        image_input: Input image provided as a PIL Image or NumPy array.

    Returns:
        Float32 RGB PyTorch tensor of shape (3, Height, Width) in the range [0.0, 1.0].

    Raises:
        TypeError: If image_input is not a PIL Image or NumPy array.
        ValueError: If input image array dimensions do not conform to expected shapes.
    """
    if isinstance(image_input, Image.Image):
        image_array = np.asarray(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        image_array = image_input
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input)}. Expected PIL.Image.Image or np.ndarray."
        )

    # Convert 2D grayscale image to 3-channel RGB representation
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    # Convert 4-channel RGBA image to 3-channel RGB
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[..., :3]
    elif image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Unsupported image dimensions: {image_array.shape}. Expected (H, W, 3).")

    # Ensure contiguous C-ordered array storage before tensor conversion
    image_array = np.ascontiguousarray(image_array)

    # Normalize integer pixel intensities [0, 255] to float range [0.0, 1.0]
    if image_array.dtype == np.uint8:
        float_image_array = image_array.astype(np.float32) / 255.0
    else:
        float_image_array = image_array.astype(np.float32)

    # Convert from HWC array structure to PyTorch CHW tensor ordering
    tensor_representation = torch.from_numpy(float_image_array).permute(2, 0, 1)
    return tensor_representation


def resolve_execution_device(
    user_device: Optional[Union[str, torch.device]] = None
) -> torch.device:
    """
    Resolves execution device, prioritizing CUDA if available when unassigned.

    Args:
        user_device: Device string or torch.device instance.

    Returns:
        Resolved torch.device object.
    """
    if user_device is not None:
        return torch.device(user_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalFeatureMatcher:
    """
    ALIKED local keypoint extractor paired with LightGlue matching and geometric verification.

    Unlike global descriptors that condense an entire image into a static vector embedding,
    this class detects distinct local keypoints and evaluates pairwise correspondence matching
    scores between images.
    """

    def __init__(
        self,
        model_name: str = "aliked-n16",
        max_keypoints: int = 2048,
        detection_threshold: float = 0.2,
        match_threshold: float = 0.1,
        resize_resolution: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Args:
            model_name: Pretrained ALIKED model variant identifier (e.g., 'aliked-n16').
            max_keypoints: Maximum number of keypoints to extract per image.
            detection_threshold: Minimum keypoint confidence threshold.
            match_threshold: Filtering score threshold for LightGlue candidate matches.
            resize_resolution: Long-edge image resizing resolution before keypoint extraction.
            device: Execution target device.
        """
        self.device = resolve_execution_device(device)
        self.resize_resolution = resize_resolution

        # Initialize ALIKED keypoint detector and descriptor extractor
        self.extractor = (
            ALIKED(
                model_name=model_name,
                max_num_keypoints=max_keypoints,
                detection_threshold=detection_threshold,
            )
            .eval()
            .to(self.device)
        )

        # Initialize LightGlue matcher configured for ALIKED feature representations
        self.matcher = (
            LightGlue(
                features="aliked",
                filter_threshold=match_threshold,
            )
            .eval()
            .to(self.device)
        )

    @torch.no_grad()
    def extract_features(
        self, image_input: Union[Image.Image, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Detects keypoints and extracts local feature descriptors for a single image.

        Args:
            image_input: Input RGB image array or PIL object.

        Returns:
            Dictionary containing:
                - 'keypoints': Tensor of shape (1, N, 2) in image coordinate space.
                - 'descriptors': Tensor of shape (1, N, D) with feature descriptors.
                - 'keypoint_scores': Tensor of shape (1, N) with confidence scores.
        """
        image_tensor = convert_image_to_float_tensor(image_input).to(self.device)
        extracted_features = self.extractor.extract(
            image_tensor, resize=self.resize_resolution
        )
        return extracted_features

    @torch.no_grad()
    def match_features(
        self,
        features_query: Dict[str, torch.Tensor],
        features_gallery: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Matches keypoints between two pre-extracted feature sets using LightGlue.

        Args:
            features_query: Extracted feature dictionary for query image.
            features_gallery: Extracted feature dictionary for gallery image.

        Returns:
            Dictionary containing matched keypoint coordinates, confidence scores, and match count:
                - 'keypoints_query': Matched keypoint coordinates array of shape (K, 2).
                - 'keypoints_gallery': Matched keypoint coordinates array of shape (K, 2).
                - 'match_scores': Pairwise confidence scores array of shape (K,).
                - 'num_matches': Integer total count of matched keypoints K.
        """
        match_output = self.matcher(
            {"image0": features_query, "image1": features_gallery}
        )

        # Remove batch dimensions using LightGlue's `rbd` utility helper
        query_feats, gallery_feats, matched_result = [
            rbd(feature_dict)
            for feature_dict in (features_query, features_gallery, match_output)
        ]

        # Extract coordinate index pairs for matched keypoints: shape (K, 2)
        match_indices = matched_result["matches"]
        query_indices = match_indices[:, 0]
        gallery_indices = match_indices[:, 1]

        matched_query_keypoints = query_feats["keypoints"][query_indices].cpu().numpy()
        matched_gallery_keypoints = gallery_feats["keypoints"][gallery_indices].cpu().numpy()
        confidence_scores = matched_result["scores"].cpu().numpy()

        return {
            "keypoints_query": matched_query_keypoints,
            "keypoints_gallery": matched_gallery_keypoints,
            "match_scores": confidence_scores,
            "num_matches": int(match_indices.shape[0]),
        }

    @staticmethod
    def compute_geometric_inliers(
        keypoints_query: Optional[np.ndarray],
        keypoints_gallery: Optional[np.ndarray],
        ransac_reprojection_threshold: float = 3.0,
    ) -> int:
        """
        Performs geometric verification using USAC_MAGSAC / RANSAC fundamental matrix estimation.

        Determines the number of matched keypoints that conform to valid epipolar geometry.

        Args:
            keypoints_query: Matched query keypoint coordinates array of shape (K, 2).
            keypoints_gallery: Matched gallery keypoint coordinates array of shape (K, 2).
            ransac_reprojection_threshold: Maximum allowed reprojection error threshold in pixels.

        Returns:
            Integer count of geometrically verified inlier matches.
        """
        # Fundamental matrix estimation requires at least 8 point correspondences
        if (
            keypoints_query is None
            or keypoints_gallery is None
            or len(keypoints_query) < 8
            or len(keypoints_gallery) < 8
        ):
            return 0

        # Attempt robust MAGSAC geometric estimation first
        try:
            _, inlier_mask = cv2.findFundamentalMat(
                keypoints_query,
                keypoints_gallery,
                cv2.USAC_MAGSAC,
                ransac_reprojection_threshold,
                0.999,
                10000,
            )
            return int(inlier_mask.sum()) if inlier_mask is not None else 0
        except Exception:
            # Fall back to standard RANSAC estimation if MAGSAC fails or is unsupported
            try:
                _, inlier_mask = cv2.findFundamentalMat(
                    keypoints_query,
                    keypoints_gallery,
                    cv2.FM_RANSAC,
                    ransac_reprojection_threshold,
                    0.999,
                    10000,
                )
                return int(inlier_mask.sum()) if inlier_mask is not None else 0
            except Exception:
                return 0

    def compute_pair_score(
        self,
        features_query: Dict[str, torch.Tensor],
        features_gallery: Dict[str, torch.Tensor],
        use_ransac_verification: bool = True,
    ) -> float:
        """
        Computes a scalar similarity score for two pre-extracted feature sets.

        Args:
            features_query: Extracted feature dictionary for query image.
            features_gallery: Extracted feature dictionary for gallery image.
            use_ransac_verification: If True, returns the count of RANSAC geometric inliers;
                                     otherwise, returns raw LightGlue match count.

        Returns:
            Float scalar score representing pairwise match quality.
        """
        match_result = self.match_features(features_query, features_gallery)

        if use_ransac_verification:
            inlier_count = self.compute_geometric_inliers(
                match_result["keypoints_query"],
                match_result["keypoints_gallery"],
            )
            return float(inlier_count)

        return float(match_result["num_matches"])

    def compute_image_similarity(
        self,
        image_query: Union[Image.Image, np.ndarray],
        image_gallery: Union[Image.Image, np.ndarray],
        use_ransac_verification: bool = True,
    ) -> float:
        """
        Convenience wrapper to extract features and compute pairwise matching scores directly
        from raw query and gallery images.

        Args:
            image_query: Raw query RGB image.
            image_gallery: Raw gallery RGB image.
            use_ransac_verification: Whether to apply RANSAC geometric inlier filtering.

        Returns:
            Float scalar similarity score.
        """
        query_features = self.extract_features(image_query)
        gallery_features = self.extract_features(image_gallery)

        return self.compute_pair_score(
            query_features, gallery_features, use_ransac_verification
        )


if __name__ == "__main__":
    # Smoke test execution with random noise images
    local_matcher = LocalFeatureMatcher()

    random_query_image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    random_gallery_image = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)

    extracted_query_features = local_matcher.extract_features(random_query_image)

    print(
        "Keypoints tensor shape:",
        tuple(extracted_query_features["keypoints"].shape),
    )
    print(
        "Descriptors tensor shape:",
        tuple(extracted_query_features["descriptors"].shape),
    )

    similarity_score = local_matcher.compute_image_similarity(
        random_query_image, random_gallery_image
    )
    print(
        f"Pairwise similarity score for synthetic noise images (expected ~0): {similarity_score:.1f}"
    )