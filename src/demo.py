"""
Synthetic features + synthetic local scores aligned with the REAL metadata roles. Lets the whole pipeline
(calibration, fusion, ablation, UI) be exercised on a laptop without images or a GPU. NUMBERS FROM THIS ARE MEANINGLESS -
it only proves the code runs and behaves sensibly (e.g. more branches help when they carry independent evidence).
"""
import numpy as np


def make_demo_features(df, dims=None, strength=None, seed=0):
    dims = dims or {"global": 64, "complementary": 32, "baseline": 32}
    strength = strength or {"global": 1.0, "complementary": 0.6, "baseline": 0.4}
    rng = np.random.default_rng(seed)
    codes, uniq = np.unique(df.identity.to_numpy(), return_inverse=True)[1], None
    n_id = codes.max() + 1
    ds_codes = np.unique(df.dataset.to_numpy(), return_inverse=True)[1]
    difficulty = rng.uniform(0.8, 1.6, ds_codes.max() + 1)[ds_codes][:, None]     # some datasets are harder
    out = {}
    for b, d in dims.items():
        cen = rng.normal(size=(n_id, d)).astype(np.float32)
        noise = rng.normal(size=(len(df), d)).astype(np.float32)
        x = strength[b] * 2.0 * cen[codes] + difficulty * noise
        out[b] = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return out


class DemoLocalScorer:
    """Stands in for LocalScorer: inlier-count-like scores, higher for the same identity."""
    def __init__(self, df, seed=0):
        self.ids = df.identity.to_numpy(); self.rng = np.random.default_rng(seed)

    def score_pairs(self, q_rows, g_rows_per_query):
        same = self.ids[np.asarray(q_rows)][:, None] == self.ids[g_rows_per_query]
        return np.where(same, self.rng.poisson(14, same.shape), self.rng.poisson(3, same.shape)).astype(np.float32)
