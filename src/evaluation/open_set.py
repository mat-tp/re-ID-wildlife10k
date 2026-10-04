"""
Open-set evaluation: known queries must be identified, unknown queries must be answered 'new individual'.
Rule: accept the best gallery match iff best similarity >= tau, else answer NEW.
  BAKS = known queries answered with the RIGHT identity AND accepted ; BAUS = unknown queries rejected
  open-set score (NBA) = (BAKS + BAUS) / 2

tau is chosen on VALIDATION (`select_tau`) and then FROZEN for the test set (`evaluate_open_set(..., tau=...)`).
Optionally one tau per species (proposal: 'calibrated separately by species where supported').
"""
import numpy as np
from src.retrieval.metrics import (open_set_at, open_set_curve, best_threshold, best_threshold_per_group,
                                   evaluate_open_set_per_group)


def _inputs(ranked, g_ids, q):
    return ranked.score[:, 0], np.asarray(g_ids)[ranked.cand[:, 0]], q.true, q.datasets, q.known


def select_tau(ranked, g_ids, q, per_species=False):
    """VALIDATION ONLY. Returns {'tau', 'BAKS', 'BAUS', 'NBA'} (+ 'species_tau' dict when per_species)."""
    s, p, t, d, k = _inputs(ranked, g_ids, q)
    best = best_threshold(s, p, t, d, k)
    if per_species:
        taus, glob = best_threshold_per_group(s, p, t, d, k, q.species)
        best = {**best, "species_tau": taus}
    return best


def evaluate_open_set(ranked, g_ids, q, tau, species_tau=None, with_curve=False):
    s, p, t, d, k = _inputs(ranked, g_ids, q)
    if species_tau is not None:
        r = evaluate_open_set_per_group(s, p, t, d, k, q.species, species_tau, tau); r["tau"] = float(tau)
    else:
        r = open_set_at(tau, s, p, t, d, k)
    if with_curve:
        rows, auc = open_set_curve(s, p, t, d, k)
        r["curve"], r["AUC"] = rows, auc
    return r
