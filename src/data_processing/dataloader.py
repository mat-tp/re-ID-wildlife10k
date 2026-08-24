"""
This file handles:
- Loading the WildlifeReID-10k dataset using the wildlife package
- Accessing the provided Excel file containing dataset metadata
- Applying animal detection and cropping (MegaDetector)
- Creating train/test splits
- Data augmentation and transformations
- Batch loading for training and evaluation

The WildlifeReID-10k dataset can be accessed using:
    pip install wildlife
    from wildlife.datasets import WildlifeReID10k

In this file we will process the Excel file that is already provided about 
the dataset to create proper data loaders for the pipeline.

TODO: Process the provided Excel metadata file
TODO: Implement animal detection and cropping pipeline
TODO: Create data augmentation transforms
TODO: Implement batch sampling strategies (balanced sampling for BAKS/BAUS)
"""

# Fill in code for data loading 