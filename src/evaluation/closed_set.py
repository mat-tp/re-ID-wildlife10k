"""
Closed-set evaluation: only KNOWN queries count (their identity is in the gallery); no threshold.
Reports BAKS (headline, benchmark metric), plain top-1/5/10 accuracy, and mAP@K.
"""
import numpy as np
from src.retrieval.metrics import (closed_set_baks, ranked_topk_accuracy, ranked_average_precision,
                                   dataset_balanced_mean)


def evaluate_closed_set(ranked, g_ids, q, ks=(1, 5, 10), per_dataset=False):
    g_ids = np.asarray(g_ids)
    ids = g_ids[ranked.cand]                                   # (Nq,K) identity per candidate
    pred = ids[:, 0]
    out = {"BAKS": closed_set_baks(ranked.score[:, 0], pred, q.true, q.datasets, q.known),
           "accuracy": float(np.mean(pred[q.known] == q.true[q.known])) if q.known.any() else float("nan")}
    out.update({f"top{k}": v for k, v in ranked_topk_accuracy(ids, q.true, q.known, ks).items()})
    counts = {i: c for i, c in zip(*np.unique(g_ids, return_counts=True))}
    n_pos = np.array([counts.get(t, 0) for t in q.true])
    ap = ranked_average_precision(ids, q.true, n_pos)
    out["mAP"] = float(np.nanmean(ap[q.known])) if q.known.any() else float("nan")
    out["mAP_ds"] = dataset_balanced_mean(ap, q.datasets, q.known)
    out["shortlist_recall"] = float(np.mean((ids[q.known] == q.true[q.known][:, None]).any(1))) if q.known.any() else float("nan")
    if per_dataset:
        corr = pred == q.true
        out["per_dataset"] = {d: float(np.mean(corr[(q.datasets == d) & q.known]))
                              for d in np.unique(q.datasets[q.known])}
    return out
