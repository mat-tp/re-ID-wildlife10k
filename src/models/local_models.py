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