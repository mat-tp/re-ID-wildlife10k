"""
Score-level fusion:  S = w_g*S_g + w_c*S_c + w_l*S_l   (w >= 0, sum = 1)  on CALIBRATED scores.
Weights are chosen on VALIDATION data by exhaustive grid search (3 branches, step 0.1 -> 66 combinations).
"""
import itertools
import numpy as np


def normalise_weights(w):
    s = float(sum(w.values()))
    return {k: v / s for k, v in w.items()} if s > 0 else {k: 1 / len(w) for k in w}


def fuse(calibrated, weights):
    """calibrated: {branch: (Nq,K) array}; weights: {branch: w}. Only branches present in `calibrated` count."""
    w = normalise_weights({k: weights.get(k, 0.0) for k in calibrated})
    return sum(w[k] * calibrated[k] for k in calibrated)


def grid_weights(names, step=0.1):
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=len(names) - 1):
        if sum(combo) <= n:
            yield dict(zip(names, [c / n for c in combo] + [(n - sum(combo)) / n]))


def tune_weights(calibrated, objective, step=0.1):
    """objective(fused_matrix) -> float (higher = better), evaluated on validation. Returns (best_w, table)."""
    names = list(calibrated)
    if len(names) == 1:
        return {names[0]: 1.0}, [({names[0]: 1.0}, objective(calibrated[names[0]]))]
    table = [(w, objective(fuse(calibrated, w))) for w in grid_weights(names, step)]
    return max(table, key=lambda t: t[1])[0], table
