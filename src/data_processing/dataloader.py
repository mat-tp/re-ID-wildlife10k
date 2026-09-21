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

TODO: Implement animal detection and cropping pipeline
TODO: Create data augmentation transforms
TODO: Implement batch sampling strategies (balanced sampling for BAKS/BAUS)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt


# Creating a custom dataset class for WildLifeRe10k


class WildlifeReID10kDataset(Dataset):

    def __init__(self, metadata_csv = "data/metadata.csv", root="data", transform=None, verbose=False):
        self.data = pd.read_csv(metadata_csv, usecols=['path', 'identity', 'species', 'split', 'cluster_id', 'dataset'])

        # dropping out the date column as it is not needed for the pipeline
        # self.data = self.data.drop(columns=['date', 'orientation'])

        # Visualize the data information
        # print("Dataset Information:")
        # print(self.data.info())
        # print(self.data.columns)
        # print(self.data.iloc[:, 2].head(20))
        # print(self.data.iloc[:, 2].dtype)

        self.root = root
        self.transform = transform

        self.paths = self.data['path'].tolist()
        self.labels = self.data['identity'].tolist()
        self.species = self.data['species'].tolist()

        if verbose:
            print(self.data.info())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Interested in the path, identity, and species.
        # img_path = self.data.iloc[idx]['path']
        # label = self.data.iloc[idx]['identity']
        # species = self.data.iloc[idx]['species']

        # Load image (assuming images are in a specific directory)
        image = self.load_image(self.paths[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], self.species[idx]

    def load_image(self, rel_path):

        full_path = os.path.join(self.root, rel_path)
        # loading the image using OpenCV
        image = cv2.imread(full_path) # uint8, BGR

        # Convert (default) BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        # isVisualize = False  # Set to True if you want to visualize the image
        # if isVisualize:
        #     plt.imshow(image)
        #     plt.show()

        return image

if __name__ == "__main__":

    ds = WildlifeReID10kDataset("data/metadata.csv", root="data", verbose=True)
    img = ds.load_image(ds.paths[0])
    print(img.shape, img.dtype)   # (H, W, 3) uint8

    