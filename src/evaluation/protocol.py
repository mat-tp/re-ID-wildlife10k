"""
    Who is gallery and who is query, for VALIDATION (tuning) and TEST (final, run once).

 validation : gallery = role 'train'                      queries = val_known + val_unknown
 test       : gallery = EVERY row with split == 'train'   queries = test_known + test_unknown
              (= train + val_known + val_unknown. The benchmark's gallery is its whole train split. If we left the
               validation rows out, a test image of a val_unknown animal would be called 'known' by the benchmark
               but have no gallery image -> unfairly counted wrong.)
Ground truth for an unknown query is NEW ('__new__').
"""
from dataclasses import dataclass
import numpy as np
from src.data_processing.splits import (TRAIN, VAL_KNOWN, VAL_UNKNOWN, TEST_KNOWN, TEST_UNKNOWN, NEW)


@dataclass
class QuerySet:
    rows: np.ndarray       # metadata row ids of the queries
    true: np.ndarray       # identity or NEW
    known: np.ndarray      # bool
    datasets: np.ndarray
    species: np.ndarray


def _qs(df, mask):
    q = df[mask]
    known = q.role.isin([VAL_KNOWN, TEST_KNOWN]).to_numpy()
    return QuerySet(q.index.to_numpy(), np.where(known, q.identity.to_numpy(), NEW), known,
                    q.dataset.to_numpy(), q.species.to_numpy())


def protocol(df, split="val"):
    """Returns (gallery_rows, QuerySet). df must be the role-annotated frame from get_frames()."""
    if split == "val":
        return df.index[df.role == TRAIN].to_numpy(), _qs(df, df.role.isin([VAL_KNOWN, VAL_UNKNOWN]))
    if split == "test":
        return df.index[df.split == "train"].to_numpy(), _qs(df, df.role.isin([TEST_KNOWN, TEST_UNKNOWN]))
    raise ValueError(split)
