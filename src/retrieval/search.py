"""
Gallery + search.

Retrieval is two-stage because ALIKED cannot be indexed:
  stage 1  cheap vector branch (cosine) over the WHOLE gallery  -> shortlist of K candidates
  stage 2  every requested branch is scored on the shortlist only, calibrated, fused, re-ranked
`Ranked` is the common output: for every query, K candidate gallery positions and scores, best first.
All metrics (accuracy, mAP, BAKS, BAUS) are computed from a `Ranked`.
"""
from dataclasses import dataclass
import numpy as np
from src.retrieval.similarity import l2n, topk_cosine


@dataclass
class Ranked:
    cand: np.ndarray    # (Nq, K) positions into the gallery arrays, best first
    score: np.ndarray   # (Nq, K) scores, descending

    def top1(self):
        return self.cand[:, 0], self.score[:, 0]

    def resort(self, new_score):
        """Re-rank the SAME candidates with a new score matrix (what fusion does)."""
        o = np.argsort(-new_score, axis=1, kind="stable")
        return Ranked(np.take_along_axis(self.cand, o, 1), np.take_along_axis(new_score, o, 1))


def rank_by_embedding(q, g, k=100, chunk=512):
    idx, sc = topk_cosine(q, g, k, chunk)
    return Ranked(idx, sc)


def distinct_identities(ranked, g_ids, k=5):
    """Collapse image hits into the top-k DISTINCT identities (what a user wants to see).
    Returns, per query, a list of (identity, best_score, gallery_position_of_best_image)."""
    out = []
    for c, s in zip(ranked.cand, ranked.score):
        seen, row = set(), []
        for pos, sc in zip(c, s):
            ident = g_ids[pos]
            if ident in seen:
                continue
            seen.add(ident); row.append((ident, float(sc), int(pos)))
            if len(row) == k:
                break
        out.append(row)
    return out


def decide(best_score, pred_identity, tau, new_label="__new__"):
    """Open-set rule from the proposal: known identity if S_max >= tau, otherwise 'new individual'."""
    return pred_identity if best_score >= tau else new_label


class LSHIndex:
    """
    Random-hyperplane LSH for cosine similarity (proposal: 'LSH may be used ... secondary measure').
    n_tables x n_bits hyperplanes; candidates = union of the buckets a query falls in; exact cosine on those.
    Compare with brute force using `recall_vs_exact` and record latency - that is the secondary result.
    """
    def __init__(self, g, n_tables=8, n_bits=12, seed=0):
        self.g = l2n(g); rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(n_tables, n_bits, self.g.shape[1])).astype(np.float32)
        self.pow = (1 << np.arange(n_bits)).astype(np.int64)
        self.tables = []
        for t in range(n_tables):
            codes = self._codes(self.g, t)
            order = np.argsort(codes, kind="stable"); cs = codes[order]
            starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
            self.tables.append({int(cs[s]): order[s:e] for s, e in zip(starts, np.r_[starts[1:], len(cs)])})

    def _codes(self, x, t):
        return ((x @ self.planes[t].T) > 0).astype(np.int64) @ self.pow

    def search(self, q, k=100):
        q = l2n(q); n = len(q)
        codes = [self._codes(q, t) for t in range(len(self.tables))]
        idx = np.full((n, k), -1, np.int64); sc = np.full((n, k), -1.0, np.float32)
        for i in range(n):
            c = np.unique(np.concatenate([self.tables[t].get(int(codes[t][i]), np.empty(0, np.int64))
                                          for t in range(len(self.tables))]))
            if len(c) == 0:
                continue
            s = self.g[c] @ q[i]
            o = np.argsort(-s)[:k]
            idx[i, :len(o)], sc[i, :len(o)] = c[o], s[o]
        return Ranked(idx, sc)

    @staticmethod
    def recall_vs_exact(approx, exact, at=1):
        return float(np.mean([len(set(a[:at]) & set(e[:at])) / at for a, e in zip(approx.cand, exact.cand)]))
