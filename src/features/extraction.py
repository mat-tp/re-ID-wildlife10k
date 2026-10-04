"""
Feature extraction with resumable disk caching.

REWRITTEN: the skeleton version filtered on a `split` column of a dataset that no longer exists in that form and
expected (labels, species) from the loader, but the loader returns (image, label, row_index). It would have crashed.

New design: embed EVERY row of the metadata frame ONCE per model (140,488 images, order = metadata order) and store
float16 shards. Every experiment (val/test, any gallery) is then just numpy indexing on these arrays - nothing is
re-extracted when we change split, gallery or PCA size. Shards make a multi-hour run restartable.

    feats = extract_all(model, df, root, "features/megadescriptor_L384")     # (N, d) float32, aligned with df.index
"""
import os
import numpy as np


def extract_all(model, df, root, cache_dir, *, shard_size=10_000, batch_size=64, num_workers=8,
                amp=True, device=None, overwrite=False):
    from src.data_processing.dataloader import make_eval_loader      # lazy: keeps this module torch-free to import
    from src.models.global_models import extract_features
    os.makedirs(cache_dir, exist_ok=True)
    df = df.reset_index(drop=True)
    tf = model.eval_transform()
    parts = []
    for a in range(0, len(df), shard_size):
        f = os.path.join(cache_dir, f"shard_{a:07d}.npy")
        if overwrite or not os.path.isfile(f):
            print(f"[extraction] rows {a}-{min(a + shard_size, len(df))} / {len(df)} -> {f}")
            loader = make_eval_loader(df.iloc[a:a + shard_size], tf, root, batch_size, num_workers)
            feats, _ = extract_features(model, loader, device=device, amp=amp)
            np.save(f + ".tmp.npy", feats.numpy().astype(np.float16))
            os.replace(f + ".tmp.npy", f)                          # a half-written shard is never mistaken for a finished one
        parts.append(np.load(f))
    return np.concatenate(parts).astype(np.float32)


def load_features(cache_root, names):
    """names: {branch: sub-directory name}. Returns {branch: (N,d) float32}."""
    out = {}
    for branch, sub in names.items():
        d = os.path.join(cache_root, sub)
        shards = sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".npy") and "tmp" not in f)
        if not shards:
            raise FileNotFoundError(f"no shards in {d} - run extract_all first")
        out[branch] = np.concatenate([np.load(os.path.join(d, s)) for s in shards]).astype(np.float32)
    return out
