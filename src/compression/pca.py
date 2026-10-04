"""
PCA compression of embeddings (proposal: "PCA is applied ... using training data, several target
dimensionalities are evaluated, unreduced representation is the reference").

Design decisions (details in DECISIONS.md #6):
 * PCA is fitted on TRAIN features only -> no test information leaks in.
 * It is applied PER embedding branch (global, EVA02): the proposal fuses SCORES, so the "fused
   representation" is not a single vector. We compress each branch, recompute cosine similarity
   on the compressed vectors, THEN calibrate + fuse. ALIKED keypoints are not a vector and are
   reported separately (storage only).
 * Optional whitening (+ L2 re-normalisation): known to help cosine retrieval on PCA-reduced
   descriptors (Jegou & Chum, ECCV 2012). We keep it as an ablation switch.
 * `concat_weighted` is provided for the alternative reading of the proposal (PCA on a single
   concatenated vector): cosine of [sqrt(w1)*e1, sqrt(w2)*e2] == w1*cos1 + w2*cos2.
"""
import numpy as np
from sklearn.decomposition import PCA


def l2n(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


class PCACompressor:
    def __init__(self, n_components=256, whiten=True, seed=0):
        self.k, self.whiten, self.seed = n_components, whiten, seed
        self.pca = None

    def fit(self, train_feats):
        x = l2n(train_feats)
        k = min(self.k, x.shape[1], x.shape[0] - 1)
        solver = "randomized" if k < 0.8 * min(x.shape) else "full"
        self.pca = PCA(n_components=k, whiten=self.whiten, svd_solver=solver, random_state=self.seed)
        self.pca.fit(x)
        return self

    def transform(self, feats):
        return l2n(self.pca.transform(l2n(feats)))

    def explained(self):
        return float(self.pca.explained_variance_ratio_.sum())

    def bytes_per_vector(self, dtype=np.float32):
        return int(self.pca.n_components_ * np.dtype(dtype).itemsize)


def sweep(train_feats, gallery_feats, query_feats, dims, evaluate, whiten=True):
    """
    Evaluate retrieval at several dimensionalities.
    `evaluate(gallery, queries)` -> dict of metrics (e.g. wraps nearest_neighbour + BAKS).
    dim=None (or 'full') = reference without compression.
    """
    rows = []
    for d in dims:
        if d in (None, "full"):
            g, q, dim, ev = l2n(gallery_feats), l2n(query_feats), gallery_feats.shape[1], 1.0
        else:
            c = PCACompressor(d, whiten).fit(train_feats)
            g, q, dim, ev = c.transform(gallery_feats), c.transform(query_feats), d, c.explained()
        rows.append({"dim": dim, "explained_var": ev, "bytes_per_img": dim * 4, **evaluate(g, q)})
    return rows


def concat_weighted(embeddings, weights):
    """[sqrt(w_i) * l2n(e_i)] -> one vector whose cosine equals the weighted sum of branch cosines."""
    w = np.asarray(weights, np.float64); w = w / w.sum()
    return np.concatenate([np.sqrt(wi) * l2n(e) for wi, e in zip(w, embeddings)], axis=1)
