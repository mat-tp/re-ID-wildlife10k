"""
Protocol Definition Module for Wildlife Re-Identification

Establishes gallery index sets and query metadata containers for validation tuning
and final benchmark testing phases.

Key Protocol Design:
- Validation Phase: Uses the training set as gallery; evaluates on validation known/unknown splits.
- Benchmark Testing Phase: Uses the full training split as gallery; evaluates on test known/unknown splits.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

from src.data_processing.splits import (
    NEW,
    TEST_KNOWN,
    TEST_UNKNOWN,
    TRAIN,
    VAL_KNOWN,
    VAL_UNKNOWN,
)


@dataclass
class QuerySet:
    """
    Data container holding query sample row indices, ground truth identities,
    dataset/species labels, and open-set known/unknown indicators.

    Attributes:
        sample_row_indices: Array of shape (N_queries,) containing index positions into the main DataFrame.
        ground_truth_identities: Array of shape (N_queries,) containing true identity labels 
                                  (or UNKNOWN_LABEL placeholder string for unknown queries).
        is_known_query_mask: Boolean array of shape (N_queries,) indicating if query identity exists in gallery.
        dataset_origin_names: Array of shape (N_queries,) containing origin dataset identifier strings.
        species_classification_names: Array of shape (N_queries,) containing species classification strings.
    """

    sample_row_indices: np.ndarray
    ground_truth_identities: np.ndarray
    is_known_query_mask: np.ndarray
    dataset_origin_names: np.ndarray
    species_classification_names: np.ndarray

    # Backward compatibility properties for legacy code consumption
    @property
    def rows(self) -> np.ndarray:
        return self.sample_row_indices

    @property
    def true(self) -> np.ndarray:
        return self.ground_truth_identities

    @property
    def known(self) -> np.ndarray:
        return self.is_known_query_mask

    @property
    def datasets(self) -> np.ndarray:
        return self.dataset_origin_names

    @property
    def species(self) -> np.ndarray:
        return self.species_classification_names


def construct_query_set(
    metadata_dataframe: pd.DataFrame, 
    selection_mask: pd.Series
) -> QuerySet:
    """
    Constructs a QuerySet container from a filtered metadata DataFrame mask.

    Args:
        metadata_dataframe: Pandas DataFrame containing dataset metadata columns.
        selection_mask: Boolean mask indicating query sample rows to select.

    Returns:
        Structured QuerySet instance.
    """
    query_subset_dataframe = metadata_dataframe[selection_mask]

    # Boolean mask checking if query identity is present in gallery (known vs unknown role)
    is_known_query_mask = query_subset_dataframe["role"].isin(
        [VAL_KNOWN, TEST_KNOWN]
    ).to_numpy()

    # Assign true label if query is known; assign UNKNOWN_LABEL placeholder ('__new__') if unknown
    ground_truth_identities = np.where(
        is_known_query_mask,
        query_subset_dataframe["identity"].to_numpy(),
        NEW,
    )

    return QuerySet(
        sample_row_indices=query_subset_dataframe.index.to_numpy(),
        ground_truth_identities=ground_truth_identities,
        is_known_query_mask=is_known_query_mask,
        dataset_origin_names=query_subset_dataframe["dataset"].to_numpy(),
        species_classification_names=query_subset_dataframe["species"].to_numpy(),
    )


def define_evaluation_protocol(
    metadata_dataframe: pd.DataFrame, 
    protocol_split_phase: str = "val"
) -> Tuple[np.ndarray, QuerySet]:
    """
    Determines gallery row indices and constructs the query evaluation set for a target phase.

    Protocol Splits:
    - Validation ('val'): Uses 'train' role rows as gallery; query set consists of 'val_known'
      and 'val_unknown' role rows.
    - Testing ('test'): Uses 'train' split rows as gallery; query set consists of 'test_known'
      and 'test_unknown' role rows.

    Args:
        metadata_dataframe: Main pandas DataFrame containing dataset metadata and split assignments.
        protocol_split_phase: Evaluation protocol phase string ('val' or 'test').

    Returns:
        Tuple containing:
            - gallery_row_indices: Array of shape (N_gallery,) with gallery row indices.
            - query_set_container: QuerySet instance containing query data and ground truths.

    Raises:
        ValueError: If protocol_split_phase is not 'val' or 'test'.
    """
    if protocol_split_phase == "val":
        gallery_row_indices = metadata_dataframe.index[
            metadata_dataframe["role"] == TRAIN
        ].to_numpy()
        query_selection_mask = metadata_dataframe["role"].isin([VAL_KNOWN, VAL_UNKNOWN])
        return gallery_row_indices, construct_query_set(metadata_dataframe, query_selection_mask)

    if protocol_split_phase == "test":
        gallery_row_indices = metadata_dataframe.index[
            metadata_dataframe["split"] == "train"
        ].to_numpy()
        query_selection_mask = metadata_dataframe["role"].isin([TEST_KNOWN, TEST_UNKNOWN])
        return gallery_row_indices, construct_query_set(metadata_dataframe, query_selection_mask)

    raise ValueError(
        f"Unsupported evaluation protocol split phase: '{protocol_split_phase}'. Expected 'val' or 'test'."
    )


# Backward compatibility aliases
protocol = define_evaluation_protocol