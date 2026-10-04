"""
End-to-end experiment runner: shortlist -> score every branch -> calibrate -> fuse -> closed/open-set metrics.

Order of operations (all choices made on VALIDATION, then frozen):
  1. fit one ScoreCalibrator per branch on validation (query, candidate) pairs      (calibration)
  2. per configuration: tune fusion weights on validation (grid, objective = open-set NBA)  (fusion)
  3. per configuration: pick tau (and optional per-species taus) on validation              (open set)
  4. evaluate on validation (reported) and - only when asked - on the TEST protocol with 1-3 frozen.

A configuration is a tuple of branch names, e.g. ("global",), ("global","complementary"), ("global","local").
'local' is special: it is only ever computed on the K_local best candidates of a cheap branch.
"""
import time
import numpy as np
import pandas as pd
from src.retrieval.search import rank_by_embedding, Ranked
from src.retrieval.similarity import candidate_cosine
from src.retrieval.metrics import best_threshold
from src.features.calibration import ScoreCalibrator
from src.features.fusion import fuse, tune_weights, normalise_weights
from src.evaluation.protocol import protocol
from src.evaluation.closed_set import evaluate_closed_set
from src.evaluation.open_set import select_tau, evaluate_open_set

LOCAL = "local"


def config_name(branches):
    return "+".join(branches)


def compress_features(feats, df, dim, whiten=True, fit_rows=None):
    """PCA per branch, fitted on TRAIN rows only (RQ4). dim=None -> unchanged reference."""
    if dim is None:
        return feats
    from src.compression.pca import PCACompressor
    rows = fit_rows if fit_rows is not None else df.index[df.role == "train"].to_numpy()
    return {b: PCACompressor(dim, whiten).fit(x[rows]).transform(x) for b, x in feats.items()}


class Experiment:
    def __init__(self, df, feats, local_scorer=None, K=100, K_local=20, calibration="isotonic",
                 weight_step=0.1, max_cal_queries=4000, seed=0):
        self.df, self.feats, self.local = df, feats, local_scorer
        self.K, self.K_local, self.step = K, K_local, weight_step
        self.method, self.max_cal_q, self.seed = calibration, max_cal_queries, seed
        self.g_val, self.q_val = protocol(df, "val")
        self._prep, self.cal = {}, {}

    # stage 1 + 2 (raw scores)
    def _prepare(self, split, branches):
        g_rows, q = (self.g_val, self.q_val) if split == "val" else protocol(self.df, "test")
        key = (split, tuple(branches))
        if key in self._prep:
            return self._prep[key]
        cheap = [b for b in branches if b != LOCAL]
        sl = cheap[0] if cheap else ("global" if "global" in self.feats else next(iter(self.feats)))
        K = self.K_local if LOCAL in branches else self.K
        t0 = time.perf_counter()
        ranked = rank_by_embedding(self.feats[sl][q.rows], self.feats[sl][g_rows], K)
        raw = {}
        for b in branches:
            if b == sl:
                raw[b] = ranked.score
            elif b == LOCAL:
                if self.local is None:
                    raise RuntimeError("config uses 'local' but no LocalScorer was given")
                raw[b] = self.local.score_pairs(q.rows, g_rows[ranked.cand])
            else:
                raw[b] = candidate_cosine(self.feats[b][q.rows], self.feats[b][g_rows], ranked.cand)
        out = dict(g_rows=g_rows, q=q, cand=ranked.cand, raw=raw, g_ids=self.df.identity.to_numpy()[g_rows],
                   ms_per_query=1000 * (time.perf_counter() - t0) / max(len(q.rows), 1))
        self._prep[key] = out
        return out

    #  calibration
    def fit_calibrators(self, branches=None):
        branches = branches or sorted({b for b in self.feats} | ({LOCAL} if self.local else set()))
        rng = np.random.default_rng(self.seed)
        for b in branches:
            sl_cfg = (b,) if b != LOCAL else ("global", LOCAL) if "global" in self.feats else (next(iter(self.feats)), LOCAL)
            P = self._prepare("val", sl_cfg)
            same = (P["g_ids"][P["cand"]] == P["q"].true[:, None]) & P["q"].known[:, None]
            qi = np.arange(len(same))
            if len(qi) > self.max_cal_q:
                qi = rng.choice(qi, self.max_cal_q, replace=False)
            self.cal[b] = ScoreCalibrator(self.method).fit(P["raw"][b][qi], same[qi])
        return self

    # one configuration
    def _calibrated(self, P, branches):
        return {b: self.cal[b].transform(P["raw"][b]) for b in branches}

    def run_config(self, branches, weights=None, objective="NBA", per_species_tau=False, test=False):
        branches = tuple(branches)
        if not self.cal:
            self.fit_calibrators()
        P = self._prepare("val", branches)
        cal = self._calibrated(P, branches)

        def obj(fused):
            r = Ranked(P["cand"], fused).resort(fused)
            s, p = r.score[:, 0], P["g_ids"][r.cand[:, 0]]
            if objective == "BAKS":
                from src.retrieval.metrics import closed_set_baks
                return closed_set_baks(s, p, P["q"].true, P["q"].datasets, P["q"].known)
            return best_threshold(s, p, P["q"].true, P["q"].datasets, P["q"].known, n_thresholds=60)["NBA"]

        w = normalise_weights(weights) if weights else tune_weights(cal, obj, self.step)[0]
        ranked = Ranked(P["cand"], fuse(cal, w)); ranked = ranked.resort(ranked.score)
        best = select_tau(ranked, P["g_ids"], P["q"], per_species=per_species_tau)
        row = {"config": config_name(branches), "weights": {k: round(v, 3) for k, v in w.items()},
               "K": ranked.cand.shape[1], "tau": best["tau"], "ms_per_query": round(P["ms_per_query"], 2)}
        row.update({f"val_{k}": v for k, v in evaluate_closed_set(ranked, P["g_ids"], P["q"]).items()
                    if k != "per_dataset"})
        row.update({"val_open_BAKS": best["BAKS"], "val_open_BAUS": best["BAUS"], "val_open_NBA": best["NBA"]})
        if test:                                                  # frozen: weights, calibrators, tau all from validation
            T = self._prepare("test", branches)
            rt = Ranked(T["cand"], fuse(self._calibrated(T, branches), w)); rt = rt.resort(rt.score)
            row.update({f"test_{k}": v for k, v in evaluate_closed_set(rt, T["g_ids"], T["q"]).items()
                        if k != "per_dataset"})
            o = evaluate_open_set(rt, T["g_ids"], T["q"], best["tau"], best.get("species_tau"))
            row.update({"test_open_BAKS": o["BAKS"], "test_open_BAUS": o["BAUS"], "test_open_NBA": o["NBA"]})
        row["_species_tau"] = best.get("species_tau")
        return row

    def ablation(self, configs, fixed_weights=None, test=False, **kw):
        fixed_weights = fixed_weights or {}
        rows = [self.run_config(c, fixed_weights.get(tuple(c)), test=test, **kw) for c in configs]
        return pd.DataFrame(rows)


def default_ablation_configs(global_name="global", comp="complementary", local=LOCAL, baseline="baseline", with_local=True):
    g, c, l = global_name, comp, local
    cfgs = [(g,), (c,), (baseline,)]
    if with_local:
        cfgs += [(l,)]
    cfgs += [(g, c)]
    if with_local:
        cfgs += [(g, l), (c, l), (g, c, l)]
    return cfgs


def save_artifacts(ex, results, out_dir="results", name="ablation_val"):
    """Everything the UI / a later test run needs: table, per-config (weights, tau), fitted calibrators."""
    import json, os, pickle
    os.makedirs(out_dir, exist_ok=True)
    table = results.drop(columns=[c for c in results.columns if c.startswith("_")])
    table.assign(weights=table.weights.astype(str)).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    card = {r["config"]: {"weights": r["weights"], "tau": float(r["tau"]),
                          "species_tau": {str(k): float(v) for k, v in (r.get("_species_tau") or {}).items()}}
            for r in results.to_dict("records")}
    json.dump(card, open(os.path.join(out_dir, "thresholds.json"), "w"), indent=2)
    pickle.dump(ex.cal, open(os.path.join(out_dir, "calibrators.pkl"), "wb"))
