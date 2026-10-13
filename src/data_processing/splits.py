from typing import Dict, Union
import numpy as np
import pandas as pd

TRAIN, VAL_KNOWN, VAL_UNKNOWN = "train", "val_known", "val_unknown"
TEST_KNOWN, TEST_UNKNOWN = "test_known", "test_unknown"
NEW = "__new__"   # Label used for unknown individuals in open-set predictions


def load_metadata(path: str = "data/metadata.csv") -> pd.DataFrame:
    """Loads the benchmark metadata CSV containing paths, identities, species, and split information."""
    columns_to_load = ["path", "identity", "species", "split", "cluster_id", "dataset"]
    return pd.read_csv(path, usecols=columns_to_load, low_memory=False)


def assign_roles(
    dataframe: pd.DataFrame,
    unknown_id_fraction: float = 0.10,
    known_img_fraction: float = 0.10,
    min_ids_for_unknown: int = 10,
    min_groups_for_known: int = 3,
    seed: int = 0
) -> pd.DataFrame:
    """Assigns evaluation roles (train, val_known, val_unknown, test_known, test_unknown) with leakage control."""
    rng = np.random.default_rng(seed)
    df = dataframe.reset_index(drop=True).copy()
    is_train = (df["split"] == "train").to_numpy()
    train_identities = set(df.loc[is_train, "identity"])

    role = np.empty(len(df), dtype=object)
    role[is_train] = TRAIN
    test_mask = ~is_train
    role[test_mask] = np.where(df.loc[test_mask, "identity"].isin(train_identities), TEST_KNOWN, TEST_UNKNOWN)

    # (A) Allocate val_unknown by holding out whole identities per dataset to mimic open-set test distributions.
    train_subset = df[is_train]
    for _, group in train_subset.groupby("dataset"):
        unique_identities = np.sort(group["identity"].unique())
        if len(unique_identities) < min_ids_for_unknown:
            continue
        k_count = max(1, int(round(unknown_id_fraction * len(unique_identities))))
        chosen_identities = rng.choice(unique_identities, size=k_count, replace=False)
        role[group.index[group["identity"].isin(chosen_identities)]] = VAL_UNKNOWN

    # (B) Allocate val_known by holding out whole cluster groups per identity to prevent near-duplicate leakage.
    rest_subset = df[(role == TRAIN)]
    cluster_groups = rest_subset["cluster_id"].where(rest_subset["cluster_id"].notna(), "img_" + rest_subset.index.astype(str))
    for _, group in rest_subset.groupby("identity"):
        identity_clusters = cluster_groups.loc[group.index]
        unique_clusters = identity_clusters.unique()
        if len(unique_clusters) < min_groups_for_known:
            continue
        target_count = max(1, int(round(known_img_fraction * len(group))))
        held_count = 0
        shuffled_clusters = rng.permutation(unique_clusters)
        for cluster_id in shuffled_clusters[:-1]:  # Ensure at least one group remains in training
            cluster_members = cluster_groups.index[cluster_groups == cluster_id]
            if held_count >= target_count:
                break
            role[cluster_members] = VAL_KNOWN
            held_count += len(cluster_members)

    df["role"] = role
    return df


def cap_per_group(dataframe: pd.DataFrame, by: Union[str, list], n: int, seed: int = 0) -> pd.DataFrame:
    """Samples at most n rows per group to create lightweight monitoring subsets during training."""
    return dataframe.groupby(by, group_keys=False).apply(
        lambda g: g.sample(min(len(g), n), random_state=seed)
    )


def summarise(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarises dataset distribution statistics across assigned roles."""
    return dataframe.groupby("role").agg(
        images=("path", "size"),
        identities=("identity", "nunique"),
        datasets=("dataset", "nunique")
    )


def check_no_leakage(dataframe: pd.DataFrame) -> bool:
    """Performs rigorous sanity checks to guarantee zero data or cluster leakage across splits."""
    train_subset = dataframe[dataframe.role == TRAIN]
    val_known_subset = dataframe[dataframe.role == VAL_KNOWN]
    val_unknown_subset = dataframe[dataframe.role == VAL_UNKNOWN]
    
    assert set(val_known_subset.identity) <= set(train_subset.identity), "val_known identity missing from train"
    assert not (set(val_unknown_subset.identity) & set(train_subset.identity)), "val_unknown identity also present in train"
    assert not (set(val_unknown_subset.identity) & set(val_known_subset.identity)), "val_unknown identity overlaps with val_known"
    
    train_clusters = set(train_subset.cluster_id.dropna())
    val_clusters = set(val_known_subset.cluster_id.dropna())
    assert not (train_clusters & val_clusters), "a cluster is split across train and validation sets"
    assert (dataframe[dataframe.role.isin([TEST_KNOWN, TEST_UNKNOWN])].split == "test").all()
    return True


def sampling_weights(identities: np.ndarray, datasets: np.ndarray, id_alpha: float = 0.5, dataset_beta: float = 0.25) -> np.ndarray:
    """Computes normalized per-image sampling weights to balance dataset sizes and identity popularities."""
    ids_series = pd.Series(list(identities))
    ds_series = pd.Series(list(datasets))
    
    n_id = ids_series.map(ids_series.value_counts()).to_numpy(dtype=float)
    n_ds = ds_series.map(ds_series.value_counts()).to_numpy(dtype=float)
    
    weights = (n_id ** (-id_alpha)) * (n_ds ** (-dataset_beta))
    return weights / weights.mean()


def get_frames(cfg_data: dict) -> pd.DataFrame:
    """Loads metadata and assigns evaluation roles based on configuration parameters."""
    val_config = cfg_data["val"]
    metadata_df = load_metadata(cfg_data["metadata_csv"])
    return assign_roles(
        metadata_df,
        unknown_id_fraction=val_config["unknown_id_fraction"],
        known_img_fraction=val_config["known_img_fraction"],
        min_ids_for_unknown=val_config["min_ids_for_unknown"],
        seed=val_config["seed"]
    )