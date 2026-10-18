"""
Dataset + DataLoader construction for WildlifeReID-10k.

What changed versus the skeleton (and why):
 * BUG FIX: cv2.imread returned None on a missing/corrupt file and cvtColor was called BEFORE the
   None check, so the error message never appeared. Check first, convert after.
 * Dataset now works on a *role-annotated* dataframe (see splits.py) so the same class serves
   train / val / test / gallery, and returns (image, label_index, row_index):
     - integer labels avoid a Python dict lookup per sample inside the training loop,
     - row_index lets feature caches be aligned with the metadata rows (needed by the
       calibration / fusion / evaluation people).
 * Balanced sampling (WeightedRandomSampler) for training - see splits.sampling_weights.
 * Animal detection: images are ALREADY cropped by the benchmark authors, so MegaDetector is
   optional (config.data['use_megadetector']); see DECISIONS.md #3.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data_processing.splits import load_metadata, assign_roles, sampling_weights, get_frames, TRAIN


class WildlifeReID10kDataset(Dataset):
    def __init__(self, df, root="data", transform=None, label_map=None):
        """
        df         : dataframe with columns path, identity, species, dataset (+ role). Row order = dataset order.
        label_map  : {identity: int}. Identities not in the map get -1 (val_unknown / test_unknown).
        """
        self.data = df.reset_index(drop=True)
        self.root, self.transform = root, transform
        self.paths = self.data["path"].tolist()
        self.identities = self.data["identity"].tolist()
        self.species = self.data["species"].tolist()
        self.label_map = label_map or {}
        self.labels = np.array([self.label_map.get(i, -1) for i in self.identities], dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def load_image(self, rel_path):
        full = os.path.abspath(os.path.join(self.root, rel_path))
        full_read = "\\\\?\\" + full if (os.name == "nt" and not full.startswith("\\\\?\\")) else full
        img = cv2.imread(full_read, cv2.IMREAD_COLOR)           # uint8 BGR (alpha, if any, is dropped)
        if img is None:
            raise FileNotFoundError(f"Image not found / unreadable: {full}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img = self.load_image(self.paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx]), idx


def build_label_map(train_df):
    return {lab: i for i, lab in enumerate(sorted(train_df["identity"].unique()))}


def make_train_loader(train_df, transform, label_map, root="data", batch_size=16,
                      id_alpha=0.5, dataset_beta=0.25, num_workers=4, seed=0):
    ds = WildlifeReID10kDataset(train_df, root, transform, label_map)
    w = sampling_weights(train_df["identity"], train_df["dataset"], id_alpha, dataset_beta)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)     # 1 "epoch" = len(ds) draws
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)


def make_eval_loader(df, transform, root="data", batch_size=64, num_workers=4):
    ds = WildlifeReID10kDataset(df, root, transform)      # labels not needed: identities live in df
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0)


if __name__ == "__main__":
    import src.config as C
    df = get_frames(C.data)
    print(df.role.value_counts())
    ds = WildlifeReID10kDataset(df[df.role == TRAIN], C.data["dataset_root"])
    img = ds.load_image(ds.paths[0])
    print(img.shape, img.dtype)
