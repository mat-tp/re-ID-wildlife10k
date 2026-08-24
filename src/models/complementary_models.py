"""
This file implements general purpose visual representations that may contain 
information not fully captured by either global morphology or local correspondences.

Models implemented:
1. EVA02: A visual representation for neon genesis
   - General visual representation from large-scale image data
   - Not specifically trained for wildlife Re-ID
   - Provides complementary information

2. DINOv2: Self-supervised vision transformer
   - Used as a general purpose baseline
   - Strong visual features without wildlife-specific training

Key aspects:
- General visual representations
- May capture information missed by wildlife-specific models
- Provides additional discriminative information

Reference: 
- EVA02 paper
- DINOv2 paper
- AnimalCLEF 2025 multi-representation fusion approach

TODO: Implement EVA02 model loading and feature extraction
TODO: Implement DINOv2 baseline
TODO: Add feature extraction pipelines
TODO: Implement proper preprocessing for each model
TODO: Add ablation experiments to test complementary information
"""
