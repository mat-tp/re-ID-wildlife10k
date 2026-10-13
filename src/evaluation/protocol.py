"""
Protocol definition module for wildlife re-identification.
Establishes gallery and query set configurations for validation tuning and final testing phases.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from src.data_processing.splits import (
    TRAIN,
    VAL_KNOWN,
    VAL_UNKNOWN,
    TEST_KNOWN,
    TEST_UNKNOWN,
    NEW
)


@dataclass
class QuerySet:
    """Container holding metadata and ground truth labels for query evaluation batches."""
    rows: np.ndarray
    true: np.ndarray
    known: np.ndarray
    datasets: np.ndarray
    species: np.ndarray


def _create_query_set(dataframe: pd.DataFrame, mask: np.ndarray) -> QuerySet:
    """Constructs a QuerySet data container from filtered metadata subset masks."""
    query_subset = dataframe[mask]
    known_mask = query_subset.role.isin([VAL_KNOWN, TEST_KNOWN]).to_numpy()
    true_identities = np.where(known_mask, query_subset.identity.to_numpy(), NEW)
    
    return QuerySet(
        rows=query_subset.index.to_numpy(),
        true=true_identities,
        known=known_mask,
        datasets=query_subset.dataset.to_numpy(),
        species=query_subset.species.to_numpy()
    )


def define_evaluation_protocol(dataframe: pd.DataFrame, split: str = "val") -> Tuple[np.ndarray, QuerySet]:
    """
    Determines gallery indices and constructs the query set for validation or testing protocols.
    
    Validation uses the training split for the gallery and evaluation subsets for queries.
    Testing uses the full benchmark training split as the gallery to prevent open-set evaluation bias.
    """
    if split == "val":
        gallery_indices = dataframe.index[dataframe.role == TRAIN].to_numpy()
        query_mask = dataframe.role.isin([VAL_KNOWN, VAL_UNKNOWN])
        return gallery_indices, _create_query_set(dataframe, query_mask)
        
    if split == "test":
        gallery_indices = dataframe.index[dataframe.split == "train"].to_numpy()
        query_mask = dataframe.role.isin([TEST_KNOWN, TEST_UNKNOWN])
        return gallery_indices, _create_query_set(dataframe, query_mask)
        
    raise ValueError(f"Unsupported evaluation protocol split: {split}")
