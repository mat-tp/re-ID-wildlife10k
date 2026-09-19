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

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt


# Creating a custom dataset class for WildLifeRe10k


class WildlifeReID10kDataset(Dataset):

    def __init__(self, metadata_csv = "data/metadata.csv", transform=None):
        self.data = pd.read_csv(metadata_csv, usecols=['path', 'identity', 'species', 'split', 'cluster_id', 'dataset'])

        # dropping out the date column as it is not needed for the pipeline
        # self.data = self.data.drop(columns=['date', 'orientation'])

        # Visualize the data information
        print("Dataset Information:")
        print(self.data.info())
        print(self.data.columns)
        print(self.data.iloc[:, 2].head(20))
        print(self.data.iloc[:, 2].dtype)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Interested in the path, identity, and species.
        img_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['identity']
        species = self.data.iloc[idx]['species']

        # Load image (assuming images are in a specific directory)
        image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label, species

    def load_image(self, img_path):

        # loading the image using OpenCV
        image = cv2.imread(img_path)

        # Data normalisation using a min-max normalisation technique to scale the pixel values to a range of [0, 1]
        # dividing the pixel values by 255.0 to achieve this scaling.
        
        image = image / 255.0

        # Convert (default) BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        isVisualize = False  # Set to True if you want to visualize the image
        if isVisualize:
            plt.imshow(image)
            plt.show()

        return image

if __name__ == "__main__":

    # Example usage of the WildlifeReID10kDataset class

    my_dataset = WildlifeReID10kDataset(metadata_csv="data/metadata.csv")

    image = my_dataset.load_image("data/images/AerialCattle2017/3owflku95bxsx24643cybxu3qh/0/0/23497_0e074235de2834f1.jpg")

    