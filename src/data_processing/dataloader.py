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
    """Loads an image using OpenCV, verifies file integrity, and converts BGR to RGB format."""
    full_path = os.path.join(root_dir, relative_path)
    image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable at path: {full_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class WildlifeReID10kDataset(Dataset):
    """PyTorch Dataset supporting role-annotated dataframes for WildlifeReID-10k."""

    def __init__(self, dataframe: pd.DataFrame, root_dir: str = "data", transform=None, label_map: dict = None):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.paths = self.data["path"].tolist()
        self.identities = self.data["identity"].tolist()
        self.species = self.data["species"].tolist()
        self.label_map = label_map or {}
        self.labels = np.array([self.label_map.get(identity, -1) for identity in self.identities], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image = convert_to_rgb_image(self.paths[index], self.root_dir)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index]), index


def build_label_map(train_dataframe: pd.DataFrame) -> dict:
    """Constructs a mapping from unique animal identities to contiguous integer labels."""
    unique_identities = sorted(train_dataframe["identity"].unique())
    return {identity: idx for idx, identity in enumerate(unique_identities)}


def make_train_loader(
    train_dataframe: pd.DataFrame,
    transform,
    label_map: dict,
    root_dir: str = "data",
    batch_size: int = 16,
    id_alpha: float = 0.5,
    dataset_beta: float = 0.25,
    num_workers: int = 8,
    seed: int = 0
) -> DataLoader:
    """Constructs a DataLoader configured with balanced weighted random sampling for training."""
    dataset = WildlifeReID10kDataset(train_dataframe, root_dir, transform, label_map)
    weights = sampling_weights(train_dataframe["identity"], train_dataframe["dataset"], id_alpha, dataset_beta)
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
    train_dataframe: pd.DataFrame,
    transform,
    label_map: dict,
    root_dir: str = "data",
    batch_size: int = 16,
    num_workers: int = 8,
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
    dataframe: pd.DataFrame,
    transform,
    root_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 8
) -> DataLoader:
    """Constructs a deterministic evaluation or gallery DataLoader."""
    dataset = WildlifeReID10kDataset(dataframe, root_dir, transform)
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
            # Inform the distributed sampler about the current epoch 
            # to ensure different shuffle permutations across training cycles.
            train_sampler.set_epoch(epoch)
            
            if dist.get_rank() == 0:
                print(f"Starting Epoch {epoch + 1}/{num_epochs}")
                
            for batch_idx, (images, labels, indices) in enumerate(train_loader):
                # Model forward pass, loss calculation, and optimisation steps occur here
                pass
                
        dist.destroy_process_group()
    else:
        # Fallback single-process execution for local inspection this is for my version without GPU
        df = get_frames(C.data)
        print(df.role.value_counts())
        ds = WildlifeReID10kDataset(df[df.role == TRAIN], C.data["dataset_root"])
        img = convert_to_rgb_image(ds.paths[0], ds.root_dir)
        print(img.shape, img.dtype)