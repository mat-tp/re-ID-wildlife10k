"""
Similarity primitives ( chunked so a 25k x 85k problem never builds the full matrix at once).

Three kinds of evidence:
  * global / complementary / baseline : L2-normalised vectors -> cosine similarity
  * local (ALIKED + LightGlue)        : a score for a PAIR of images (number of geometrically verified
                                         matches). It is NOT a vector, so it can only be computed for a
                                         short list of candidates, never for the whole gallery.
"""
from collections import OrderedDict
import numpy as np


def l2n(x, eps=1e-12):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def topk_cosine(q, g, k, chunk=1024):
    """Exact top-k gallery neighbours per query. Returns (idx[Nq,k], score[Nq,k]) sorted by score desc."""
    k = min(k, len(g))
    try:
        import torch
        if torch.cuda.is_available():
            q_norm = l2n(q)
            g_norm = l2n(g)
            with torch.no_grad():
                gt = torch.from_numpy(g_norm).cuda().t().contiguous()
                idx = np.empty((len(q), k), np.int64)
                sc = np.empty((len(q), k), np.float32)
                for a in range(0, len(q), chunk):
                    qc = torch.from_numpy(q_norm[a:a + chunk]).cuda()
                    s = torch.matmul(qc, gt)
                    val, top_i = torch.topk(s, k, dim=1)
                    idx[a:a + chunk] = top_i.cpu().numpy()
                    sc[a:a + chunk] = val.cpu().numpy()
                return idx, sc
    except Exception:
        pass

    q, g = l2n(q), l2n(g)
    gt = np.ascontiguousarray(g.T)
    idx = np.empty((len(q), k), np.int64); sc = np.empty((len(q), k), np.float32)
    for a in range(0, len(q), chunk):
        s = q[a:a + chunk] @ gt
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(s, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[a:a + chunk] = np.take_along_axis(part, o, 1)
        sc[a:a + chunk] = np.take_along_axis(ps, o, 1)
    return idx, sc


def candidate_cosine(q, g, cand, chunk=256):
    """cos(q_i, g[cand[i, j]]) for every candidate -> (Nq, K). Used to score ONE branch on a shortlist
    that was produced by another branch."""
    q, g = l2n(q), l2n(g)
    out = np.empty(cand.shape, np.float32)
    for a in range(0, len(q), chunk):
        out[a:a + chunk] = np.einsum("nd,nkd->nk", q[a:a + chunk], g[cand[a:a + chunk]])
    return out


class LocalScorer:
    """
    ALIKED keypoints + LightGlue matching -> RANSAC inlier count, for (query row, gallery row) pairs.
    Wraps src.models.local_models.LocalModel. UNTESTED here (needs torch + lightglue + images).

    image_loader(row) -> RGB uint8 ndarray.  Keypoints are cached per image row (LRU) and pair scores
    are memoised, so ablations that reuse the same pairs do not re-run the matcher.
    COST WARNING: every (query, candidate) pair is one LightGlue forward pass. Keep K_local small (10-20).
    """
    def __init__(self, image_loader, local_model=None, cache_images=3000):
        if local_model is None:
            from src.models.local_models import LocalModel
            local_model = LocalModel()
        self.m, self.load = local_model, image_loader
        self._kp, self._cap, self._pairs = OrderedDict(), cache_images, {}

    def _feats(self, row):
        if row in self._kp:
            self._kp.move_to_end(row); return self._kp[row]
        f = self.m.extract(self.load(row))
        self._kp[row] = f
        if len(self._kp) > self._cap:
            self._kp.popitem(last=False)
        return f

    def score_pairs(self, q_rows, g_rows_per_query):
        """q_rows: (Nq,) metadata rows; g_rows_per_query: (Nq,K) metadata rows -> (Nq,K) float32."""
        out = np.zeros(g_rows_per_query.shape, np.float32)
        for i, qr in enumerate(q_rows):
            for j, gr in enumerate(g_rows_per_query[i]):
                key = (int(qr), int(gr))
                if key not in self._pairs:
                    self._pairs[key] = self.m.score(self._feats(int(qr)), self._feats(int(gr)))
                out[i, j] = self._pairs[key]
        return out
