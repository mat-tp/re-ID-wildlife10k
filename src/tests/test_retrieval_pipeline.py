"""Torch-free end-to-end checks on the REAL metadata roles with synthetic features:  python -m src.tests.test_retrieval_pipeline"""
import time
import numpy as np
from src.data_processing.splits import get_frames
import src.config as C
from src.demo import make_demo_features, DemoLocalScorer
from src.evaluation.ablation import Experiment, default_ablation_configs, compress_features
from src.evaluation.protocol import protocol
from src.features.calibration import ScoreCalibrator, expected_calibration_error
from src.features.fusion import grid_weights, fuse
from src.retrieval.search import rank_by_embedding, LSHIndex, distinct_identities
from src.retrieval.similarity import topk_cosine, l2n
from src.retrieval.metrics import ranked_average_precision


def small_frame(frac=0.12, seed=0):
    df = get_frames(C.data)
    keep = df.sample(frac=frac, random_state=seed).index          # keeps original row ids as the index
    return df, df.loc[keep]


def test_protocol_gallery_rules():
    df = get_frames(C.data)
    gv, qv = protocol(df, "val"); gt, qt = protocol(df, "test")
    assert (df.role.iloc[gv] == "train").all() and (df.split.iloc[gt] == "train").all()
    assert len(gt) == len(gv) + (df.role.isin(["val_known", "val_unknown"])).sum()
    ids_gt = set(df.identity.iloc[gt])
    assert all(i in ids_gt for i in qt.true[qt.known]), "a 'known' test query has no gallery identity"
    assert not any(i in ids_gt for i in df.identity.iloc[qt.rows[~qt.known]]), "an 'unknown' test id is in the gallery"


def test_topk_and_ap():
    rng = np.random.default_rng(0); g = rng.normal(size=(500, 16)).astype(np.float32); q = g[:20] + .01
    idx, sc = topk_cosine(q, g, 5); assert (idx[:, 0] == np.arange(20)).all() and (np.diff(sc, axis=1) <= 1e-6).all()
    ap = ranked_average_precision(np.array([["a", "b", "a"]]), np.array(["a"]), [2])
    assert abs(ap[0] - (1 + 2 / 3) / 2) < 1e-9


def test_calibrator():
    rng = np.random.default_rng(0); y = rng.random(5000) < .3
    x = np.where(y, rng.normal(.7, .1, 5000), rng.normal(.4, .1, 5000))
    for m in ("isotonic", "pchip"):
        c = ScoreCalibrator(m).fit(x, y); o = np.argsort(x)
        assert (np.diff(c.transform(x[o])) >= -1e-6).all(), m
        p = c.transform(x); assert p.min() >= 0 and expected_calibration_error(p, y) < .05, m


def test_fusion_grid():
    ws = list(grid_weights(["a", "b", "c"], .1)); assert len(ws) == 66 and all(abs(sum(w.values()) - 1) < 1e-9 for w in ws)


def test_lsh():
    rng = np.random.default_rng(1); g = rng.normal(size=(5000, 32)).astype(np.float32); q = g[:200] + .3 * rng.normal(size=(200, 32)).astype(np.float32)
    ex = rank_by_embedding(q, g, 5); ap = LSHIndex(g, n_tables=12, n_bits=8).search(q, 5)
    r = LSHIndex.recall_vs_exact(ap, ex, 1); print("  LSH recall@1 vs exact:", round(r, 3)); assert r > .8


def test_ablation_end_to_end():
    full, df = small_frame()
    feats_all = make_demo_features(full)
    feats = {k: v[df.index.to_numpy()] for k, v in feats_all.items()}
    df = df.reset_index(drop=True)
    local = DemoLocalScorer(df)
    t = time.time(); ex = Experiment(df, feats, local, K=50, K_local=10, max_cal_queries=1500)
    res = ex.ablation(default_ablation_configs(), test=True)
    print(res[["config", "weights", "val_BAKS", "val_open_NBA", "test_BAKS", "test_open_NBA", "val_shortlist_recall"]].round(3).to_string())
    print(f"  ablation took {time.time()-t:.0f}s")
    r = res.set_index("config")
    assert r.loc["global+complementary+local", "val_open_NBA"] >= r.loc["global", "val_open_NBA"] - 1e-9, "fusion should not hurt on val (weights are tuned there)"
    assert r.loc["global", "val_BAKS"] > r.loc["baseline", "val_BAKS"], "stronger synthetic branch must win"
    assert (res.filter(like="test_").notna().all()).all()
    # compression sweep
    for d in (8, 32):
        ex2 = Experiment(df, compress_features(feats, df, d), local, K=50, K_local=10, max_cal_queries=1500)
        print("  PCA", d, round(ex2.run_config(("global",))["val_BAKS"], 3))


if __name__ == "__main__":
    for f in [test_protocol_gallery_rules, test_topk_and_ap, test_calibrator, test_fusion_grid, test_lsh, test_ablation_end_to_end]:
        f(); print("PASS", f.__name__)
