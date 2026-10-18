"""
Similarity primitives ( chunked so a 25k x 85k problem never builds the full matrix at once).

Three kinds of evidence:
  * global / complementary / baseline : L2-normalised vectors -> cosine similarity
  * local (ALIKED + LightGlue)        : a score for a PAIR of images (number of geometrically verified
                                         matches). It is NOT a vector, so it can only be computed for a
                                         short list of candidates, never for the whole gallery.
"""
import os
import pickle
import time
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
    ALIKED keypoints + LightGlue matching -> inlier / match score, for (query row, gallery row) pairs.
    Wraps src.models.local_models.LocalModel.

    image_loader(row) -> RGB uint8 ndarray. Keypoints are cached per image row (LRU) and pair scores
    are memoised, so ablations that reuse the same pairs do not re-run the matcher.
    """
    def __init__(self, image_loader, local_model=None, cache_images=3000, use_ransac=False, cache_file=None):
        if local_model is None:
            from src.models.local_models import LocalModel
            local_model = LocalModel()
        self.m, self.load = local_model, image_loader
        self._kp, self._cap = OrderedDict(), cache_images
        self.use_ransac = use_ransac
        self.cache_file = cache_file or "results/local_pairs_cache.pkl"
        self._pairs = {}
        if os.path.isfile(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._pairs = pickle.load(f)
                print(f"[LocalScorer] Loaded {len(self._pairs)} pre-computed pair scores from {self.cache_file}")
            except Exception as e:
                self._pairs = {}

    def save_cache(self):
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self._pairs, f)
            except Exception as e:
                pass

    def _feats(self, row):
        if row in self._kp:
            self._kp.move_to_end(row)
            return self._kp[row]
        img = self.load(row)
        f = self.m.extract(img)
        self._kp[row] = f
        if len(self._kp) > self._cap:
            self._kp.popitem(last=False)
        return f

    def extract_image(self, image):
        """Extracts ALIKED keypoints for a raw PIL or NumPy image."""
        return self.m.extract(image)

    def score_features_pair(self, feats0, feats1):
        return float(self.m.score(feats0, feats1, use_ransac=self.use_ransac))

    def score_query_against_candidate_rows(self, query_feats, cand_rows):
        """query_feats: pre-extracted ALIKED dict; cand_rows: 1D array of gallery rows."""
        scores = np.empty(len(cand_rows), dtype=np.float32)
        for idx, gr in enumerate(cand_rows):
            f_cand = self._feats(int(gr))
            scores[idx] = self.score_features_pair(query_feats, f_cand)
        return scores

    def score_pairs(self, q_rows, g_rows_per_query, log_interval=25):
        """q_rows: (Nq,) metadata rows; g_rows_per_query: (Nq,K) metadata rows -> (Nq,K) float32."""
        out = np.zeros(g_rows_per_query.shape, np.float32)
        total_q = len(q_rows)
        t_start = time.perf_counter()
        for i, qr in enumerate(q_rows):
            f_q = self._feats(int(qr))
            for j, gr in enumerate(g_rows_per_query[i]):
                key = (int(qr), int(gr))
                if key not in self._pairs:
                    self._pairs[key] = self.m.score(f_q, self._feats(int(gr)), use_ransac=self.use_ransac)
                out[i, j] = self._pairs[key]
            if log_interval and ((i + 1) % log_interval == 0 or (i + 1) == total_q):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (total_q - (i + 1)) / max(rate, 0.001)
                print(f"[local matching] {i + 1}/{total_q} queries processed ({((i + 1)/total_q)*100:.1f}%) | ETA: {eta:.0f}s", flush=True)
                self.save_cache()
        self.save_cache()
        return out

