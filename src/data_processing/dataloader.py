import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from src.data_processing.splits import load_metadata, assign_roles, sampling_weights, get_frames, TRAIN


def convert_to_rgb_image(relative_path: str, root_dir: str = "data") -> np.ndarray:
    """Loads an image using OpenCV, verifies file integrity, and converts BGR to RGB format with Windows long-path support."""
    full = os.path.abspath(os.path.join(root_dir, relative_path))
    full_read = "\\\\?\\" + full if (os.name == "nt" and not full.startswith("\\\\?\\")) else full
    image = cv2.imread(full_read, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable at path: {full}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class WildlifeReID10kDataset(Dataset):
    """PyTorch Dataset supporting role-annotated dataframes for WildlifeReID-10k."""

    def __init__(self, dataframe: pd.DataFrame = None, root_dir: str = "data", transform=None, label_map: dict = None,
                 df: pd.DataFrame = None, root: str = None):
        data_df = dataframe if dataframe is not None else df
        self.data = data_df.reset_index(drop=True)
        self.root_dir = root if root is not None else root_dir
        self.root = self.root_dir
        self.transform = transform
        self.paths = self.data["path"].tolist()
        self.identities = self.data["identity"].tolist()
        self.species = self.data["species"].tolist()
        self.label_map = label_map or {}
        self.labels = np.array([self.label_map.get(identity, -1) for identity in self.identities], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def load_image(self, rel_path: str) -> np.ndarray:
        return convert_to_rgb_image(rel_path, self.root_dir)

    def __getitem__(self, index: int):
        image = self.load_image(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index]), index


def build_label_map(train_dataframe: pd.DataFrame = None, train_df: pd.DataFrame = None) -> dict:
    """Constructs a mapping from unique animal identities to contiguous integer labels."""
    df = train_dataframe if train_dataframe is not None else train_df
    unique_identities = sorted(df["identity"].unique())
    return {identity: idx for idx, identity in enumerate(unique_identities)}


def make_train_loader(
    train_dataframe: pd.DataFrame = None,
    transform=None,
    label_map: dict = None,
    root_dir: str = "data",
    batch_size: int = 16,
    id_alpha: float = 0.5,
    dataset_beta: float = 0.25,
    num_workers: int = 4,
    seed: int = 0,
    train_df: pd.DataFrame = None,
    root: str = None
) -> DataLoader:
    """Constructs a DataLoader configured with balanced weighted random sampling for training."""
    df = train_dataframe if train_dataframe is not None else train_df
    r = root if root is not None else root_dir
    dataset = WildlifeReID10kDataset(df, r, transform, label_map)
    weights = sampling_weights(df["identity"], df["dataset"], id_alpha, dataset_beta)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )


def make_ddp_train_loader(
    train_dataframe: pd.DataFrame = None,
    transform=None,
    label_map: dict = None,
    root_dir: str = "data",
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 0
) -> tuple:
    """Constructs a DataLoader and its DistributedSampler for multi-GPU DDP training."""
    dataset = WildlifeReID10kDataset(train_dataframe, root_dir, transform, label_map)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    return loader, sampler


def make_eval_loader(
    dataframe: pd.DataFrame = None,
    transform=None,
    root_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 4,
    df: pd.DataFrame = None,
    root: str = None
) -> DataLoader:
    """Constructs a deterministic evaluation or gallery DataLoader."""
    data_df = dataframe if dataframe is not None else df
    r = root if root is not None else root_dir
    dataset = WildlifeReID10kDataset(data_df, r, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )


if __name__ == "__main__":
    import src.config as C
    
    # Example execution demonstrating DDP initialisation and training epoch loop.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        df = get_frames(C.data)
        train_df = df[df.role == TRAIN]
        label_map = build_label_map(train_df)
        
        # transformations placeholder for demonstration
        dummy_transform = lambda img: torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        train_loader, train_sampler = make_ddp_train_loader(
            train_dataframe=train_df,
            transform=dummy_transform,
            label_map=label_map,
            root_dir=C.data["dataset_root"]
        )
        
        num_epochs = 5
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            if dist.get_rank() == 0:
                print(f"Starting Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (images, labels, indices) in enumerate(train_loader):
                pass
                
        dist.destroy_process_group()
    else:
        # Fallback single-process execution for local inspection
        df = get_frames(C.data)
        print(df.role.value_counts())
        ds = WildlifeReID10kDataset(df[df.role == TRAIN], C.data["dataset_root"])
        img = convert_to_rgb_image(ds.paths[0], ds.root_dir)
        print(img.shape, img.dtype)