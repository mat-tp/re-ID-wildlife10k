"""
Metrics for closed-set and open-set Re-ID (numpy only).

Implemented here because the global-model comparison (MiewID vs fine-tuned MegaDescriptor,
proposal RQ-i) and the threshold tau must be judged with the SAME metric as the benchmark:

  BAKS = mean over datasets of ( mean over known identities of ( mean over images of 1[correct] ) )
  BAUS = mean over datasets of ( mean over unknown images of 1[predicted 'new'] )
  closed-set score   = BAKS on known queries only (unknown queries ignored)   [WildlifeReID-10k, Sec 5.2]
  open-set score     = (BAKS + BAUS) / 2  ("normalised balanced accuracy")     [WildlifeReID-10k, Sec 5.2]

Prediction rule (both papers): 1-nearest-neighbour by cosine similarity against the gallery;
open-set -> answer "new individual" when the best similarity < tau.

Note: the paper divides by D (all datasets). Some datasets have no unknown test images, so we
average over the datasets that actually contain the relevant queries - state this in the report.
"""
import numpy as np

NEW = "__new__"


def nearest_neighbour(q, g, g_labels, chunk=512):
    """q:(Nq,d) g:(Ng,d) L2-normalised float32. Returns best cosine sim and predicted label per query."""
    g_labels = np.asarray(g_labels)
    best_s = np.empty(len(q), np.float32); best_i = np.empty(len(q), np.int64)
    gt = np.ascontiguousarray(g.T)
    for a in range(0, len(q), chunk):
        s = q[a:a + chunk] @ gt
        best_i[a:a + chunk] = s.argmax(1)
        best_s[a:a + chunk] = s[np.arange(len(s)), best_i[a:a + chunk]]
    return best_s, g_labels[best_i]


def topk_identity_accuracy(q, g, g_labels, q_labels, ks=(1, 5, 10), chunk=256):
    """Plain top-k (k distinct identities). Not balanced - report next to BAKS, never instead."""
    g_labels = np.asarray(g_labels); q_labels = np.asarray(q_labels)
    hits = {k: 0 for k in ks}; gt = np.ascontiguousarray(g.T); kmax = max(ks)
    for a in range(0, len(q), chunk):
        s = q[a:a + chunk] @ gt
        order = np.argsort(-s, axis=1)[:, :min(400, s.shape[1])]
        for r, row in enumerate(order):
            ids = []
            for lab in g_labels[row]:
                if lab not in ids:
                    ids.append(lab)
                if len(ids) == kmax:
                    break
            for k in ks:
                hits[k] += q_labels[a + r] in ids[:k]
    return {k: hits[k] / len(q) for k in ks}


class _Groups:
    """Pre-computes (dataset, identity) cells so that BAKS at many thresholds is cheap."""
    def __init__(self, true, datasets, is_known):
        true = np.asarray(true); datasets = np.asarray(datasets)
        self.known = np.asarray(is_known, bool)
        k = np.flatnonzero(self.known)
        cell_key = np.char.add(datasets[k].astype(str), "||" + true[k].astype(str))
        self.cells, self.cell_of = np.unique(cell_key, return_inverse=True)
        self.cell_n = np.bincount(self.cell_of)
        ds_of_cell = np.array([c.split("||")[0] for c in self.cells])
        self.ds_names, self.ds_of_cell = np.unique(ds_of_cell, return_inverse=True)
        self.ds_cells = np.bincount(self.ds_of_cell)
        u = np.flatnonzero(~self.known)
        self.u_idx = u
        self.u_ds_names, self.u_ds = np.unique(datasets[u].astype(str), return_inverse=True)
        self.u_n = np.bincount(self.u_ds) if len(u) else np.array([])
        self.k_idx = k

    def baks(self, correct_known):
        if len(self.k_idx) == 0:
            return float("nan")
        cell = np.bincount(self.cell_of, weights=correct_known[self.k_idx].astype(float)) / self.cell_n
        ds = np.bincount(self.ds_of_cell, weights=cell) / self.ds_cells
        return float(ds.mean())

    def baus(self, correct_unknown):
        if len(self.u_idx) == 0:
            return float("nan")
        ds = np.bincount(self.u_ds, weights=correct_unknown[self.u_idx].astype(float)) / self.u_n
        return float(ds.mean())


def closed_set_baks(best_sim, pred, true, datasets, is_known):
    """Closed set: unknown queries ignored, no threshold."""
    G = _Groups(true, datasets, is_known)
    return G.baks(np.asarray(pred) == np.asarray(true))


def open_set_at(tau, best_sim, pred, true, datasets, is_known, _G=None):
    G = _G or _Groups(true, datasets, is_known)
    best_sim = np.asarray(best_sim); accepted = best_sim >= tau
    correct_known = (np.asarray(pred) == np.asarray(true)) & accepted
    correct_unknown = ~accepted
    b, u = G.baks(correct_known), G.baus(correct_unknown)
    return {"tau": float(tau), "BAKS": b, "BAUS": u, "NBA": float(np.nanmean([b, u]))}


def open_set_curve(best_sim, pred, true, datasets, is_known, n_thresholds=200):
    """BAKS/BAUS at many thresholds + AUC of BAUS-vs-BAKS (the curve drawn in WildlifeReID-10k Fig. 5)."""
    G = _Groups(true, datasets, is_known)
    taus = np.unique(np.quantile(best_sim, np.linspace(0, 1, n_thresholds)))
    taus = np.concatenate([[best_sim.min() - 1e-6], taus, [best_sim.max() + 1e-6]])
    rows = [open_set_at(t, best_sim, pred, true, datasets, is_known, G) for t in taus]
    b = np.array([r["BAKS"] for r in rows]); u = np.array([r["BAUS"] for r in rows])
    o = np.argsort(b)
    bo, uo = b[o], u[o]
    auc = float(np.sum((bo[1:] - bo[:-1]) * (uo[1:] + uo[:-1]) / 2))   # trapezoid rule (np.trapz removed in NumPy 2)
    return rows, auc


def best_threshold(best_sim, pred, true, datasets, is_known, n_thresholds=200):
    """tau maximising (BAKS+BAUS)/2 on VALIDATION data. Never call this on the test set."""
    rows, _ = open_set_curve(best_sim, pred, true, datasets, is_known, n_thresholds)
    return max(rows, key=lambda r: r["NBA"])


def best_threshold_per_group(best_sim, pred, true, datasets, is_known, groups, min_unknown=20, min_known=20):
    """Per-species (or per-dataset) tau [PROPOSAL: 'calibrated separately by species where supported'].
    Falls back to the global tau for groups with too few known/unknown validation queries."""
    best_sim = np.asarray(best_sim); groups = np.asarray(groups); is_known = np.asarray(is_known, bool)
    glob = best_threshold(best_sim, pred, true, datasets, is_known)["tau"]
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if (~is_known[m]).sum() < min_unknown or is_known[m].sum() < min_known:
            out[g] = glob; continue
        out[g] = best_threshold(best_sim[m], np.asarray(pred)[m], np.asarray(true)[m],
                                np.asarray(datasets)[m], is_known[m])["tau"]
    return out, glob


def evaluate_open_set_per_group(best_sim, pred, true, datasets, is_known, groups, taus, fallback):
    tau_q = np.array([taus.get(g, fallback) for g in np.asarray(groups)])
    G = _Groups(true, datasets, is_known)
    accepted = np.asarray(best_sim) >= tau_q
    b = G.baks((np.asarray(pred) == np.asarray(true)) & accepted); u = G.baus(~accepted)
    return {"BAKS": b, "BAUS": u, "NBA": float(np.nanmean([b, u]))}
