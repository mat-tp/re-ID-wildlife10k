"""
This file wil implements the extraction of features from all models:
- Global features (MiewID or MegaDescriptor)
- Complementary features (EVA02)
- Local features (ALIKED keypoints and descriptors)
- Baseline features (DINOv2)

The extraction pipeline applies:
1. Animal detection and cropping using MegaDetector
2. Preprocessing specific to each model
3. Feature extraction using the appropriate model
4. Storage of extracted features

Features are extracted before calibration and fusion.

TODO: Implement unified feature extraction pipeline
TODO: Add MegaDetector integration for animal detection and cropping
TODO: Implement batch feature extraction for efficiency
TODO: Add feature caching and storage
TODO: Implement feature loading from disk
"""

