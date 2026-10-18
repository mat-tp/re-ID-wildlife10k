"""
Validation split + known/unknown bookkeeping.

The benchmark CSV only has `train` / `test`. The proposal says validation data are used for
(i) global-model selection, (ii) calibration, (iii) fusion weights, (iv) the open-set threshold
tau, and that the test set stays untouched. So we must build a validation split ourselves,
from `train` only, and it has to mimic the two things the test set contains:

  * val_known   : new images of identities that ARE in the gallery/training data
  * val_unknown : identities that are NOT in the gallery/training data (open-set)

Leakage control follows WildlifeReID-10k's own argument (near-duplicate frames from one encounter
must not sit on both sides of a split): we move WHOLE `cluster_id` groups, never single images
of a cluster.

Roles (column `role`):
  train | val_known | val_unknown | test_known | test_unknown
`test_known/unknown` is derived exactly like the benchmark: a test identity is "unknown" iff it
has no image in `train`.
"""
import numpy as np
import pandas as pd

TRAIN, VAL_KNOWN, VAL_UNKNOWN = "train", "val_known", "val_unknown"
TEST_KNOWN, TEST_UNKNOWN = "test_known", "test_unknown"
NEW = "__new__"   # label used for "unknown individual" in open-set predictions


def load_metadata(path="data/metadata.csv"):
    cols = ["path", "identity", "species", "split", "cluster_id", "dataset"]
    return pd.read_csv(path, usecols=cols, low_memory=False)


def assign_roles(df, unknown_id_fraction=0.10, known_img_fraction=0.10,
                 min_ids_for_unknown=10, min_groups_for_known=3, seed=0):
    """Return a copy of df (reset index) with a `role` column. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True).copy()
    is_train = (df["split"] == "train").to_numpy()
    train_ids = set(df.loc[is_train, "identity"])

    role = np.empty(len(df), dtype=object)
    role[is_train] = TRAIN
    te = ~is_train
    role[te] = np.where(df.loc[te, "identity"].isin(train_ids), TEST_KNOWN, TEST_UNKNOWN)

    # (A) val_unknown: hold out whole identities, per dataset (test-unknown ids are spread over
    #     all 37 datasets, so the proxy must be too - we cannot use "train-only ids", which sit in 16).
    tr = df[is_train]
    for _, g in tr.groupby("dataset"):
        ids = np.sort(g["identity"].unique())
        if len(ids) < min_ids_for_unknown:
            continue
        k = max(1, int(round(unknown_id_fraction * len(ids))))
        chosen = rng.choice(ids, size=k, replace=False)
        role[g.index[g["identity"].isin(chosen)]] = VAL_UNKNOWN

    # (B) val_known: per remaining identity move whole cluster-groups (or lone images) to val,
    #     until ~known_img_fraction of its images are held out. Always keep >= 1 group in train.
    rest = df[(role == TRAIN)]
    grp = rest["cluster_id"].where(rest["cluster_id"].notna(), "img_" + rest.index.astype(str))
    for _, g in rest.groupby("identity"):
        groups = grp.loc[g.index]
        uniq = groups.unique()
        if len(uniq) < min_groups_for_known:     # too few groups: keep everything in train
            continue
        target = max(1, int(round(known_img_fraction * len(g))))
        held, order = 0, rng.permutation(uniq)
        for u in order[:-1]:                      # never take the last remaining group
            members = groups.index[groups == u]
            if held >= target:
                break
            role[members] = VAL_KNOWN
            held += len(members)

    df["role"] = role
    return df


def cap_per_group(df, by, n, seed=0):
    """Keep at most `n` rows per value of column(s) `by` (cheap monitoring subsets, e.g. during
    fine-tuning where extracting every validation image each epoch would be too slow)."""
    return df.sample(frac=1.0, random_state=seed).groupby(by, as_index=False).head(n)


def summarize(df):
    out = df.groupby("role").agg(images=("path", "size"), identities=("identity", "nunique"),
                                 datasets=("dataset", "nunique"))
    return out


def check_no_leakage(df):
    """Sanity checks used by the tests. Raises AssertionError on violation."""
    tr = df[df.role == TRAIN]
    vk = df[df.role == VAL_KNOWN]
    vu = df[df.role == VAL_UNKNOWN]
    assert set(vk.identity) <= set(tr.identity), "val_known identity missing from train"
    assert not (set(vu.identity) & set(tr.identity)), "val_unknown identity also in train"
    assert not (set(vu.identity) & set(vk.identity))
    c_tr = set(tr.cluster_id.dropna()); c_vk = set(vk.cluster_id.dropna())
    assert not (c_tr & c_vk), "a cluster is split across train and val"
    assert (df[df.role.isin([TEST_KNOWN, TEST_UNKNOWN])].split == "test").all()
    return True


def sampling_weights(identities, datasets, id_alpha=0.5, dataset_beta=0.25):
    """
    Per-image sampling weight  w_i = n_id(i)^-alpha * n_ds(i)^-beta   (normalised to mean 1).

    Motivation: BAKS averages over datasets, then identities, then images - so a plain shuffled
    epoch over-weights big datasets (CatIndividualImages 13k imgs vs ReunionTurtles 336) and
    'popular' identities (max 411 imgs, median 5). Full inverse-frequency (alpha=beta=1) would
    match BAKS exactly but repeats the rare images so often that the net memorises them, so we
    default to a square-root style compromise. These exponents are hyper-parameters to tune on val.
    """
    ids = pd.Series(list(identities)); ds = pd.Series(list(datasets))
    n_id = ids.map(ids.value_counts()).to_numpy(float)
    n_ds = ds.map(ds.value_counts()).to_numpy(float)
    w = n_id ** (-id_alpha) * n_ds ** (-dataset_beta)
    return w / w.mean()


def get_frames(cfg_data):
    """One call the whole team can use: role-annotated metadata (torch-free; also re-exported by dataloader.py)."""
    v = cfg_data["val"]
    return assign_roles(load_metadata(cfg_data["metadata_csv"]), v["unknown_id_fraction"],
                        v["known_img_fraction"], v["min_ids_for_unknown"], seed=v["seed"])
