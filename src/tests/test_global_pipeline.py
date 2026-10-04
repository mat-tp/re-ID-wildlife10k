"""Torch-free checks: python -m src.tests.test_global_pipeline   (run from the project root)"""
import numpy as np, pandas as pd
from src.data_processing.splits import *
from src.retrieval.metrics import *
from src.compression.pca import PCACompressor, sweep, concat_weighted, l2n

def test_splits_real():
    r = assign_roles(load_metadata()); check_no_leakage(r)
    r2 = assign_roles(load_metadata()); assert (r.role == r2.role).all(), "not deterministic"
    print(summarize(r))

def test_metrics_toy():
    # 2 datasets; A: ids a1(2 imgs),a2(2 imgs); B: b1(1); one unknown query in A and in B
    true = np.array(["a1","a1","a2","a2","b1", NEW, NEW]); ds = np.array(list("AAAABAB"))
    known = true != NEW
    sim = np.array([.9,.8,.9,.2,.9,.3,.95]); pred = np.array(["a1","a1","a2","a1","b1","a1","b1"])
    # closed: A: a1=1.0, a2=0.5 ->0.75 ; B: b1=1 -> mean=0.875
    assert abs(closed_set_baks(sim, pred, true, ds, known) - 0.875) < 1e-9
    o = open_set_at(0.5, sim, pred, true, ds, known)
    # tau=.5: query4 (sim .2) rejected -> a2 = 0.5 -> A=.75, B=1 -> BAKS .875 ; unknown: A rejected(.3<.5)=1, B accepted=0 -> BAUS .5
    assert abs(o["BAKS"]-0.875) < 1e-9 and abs(o["BAUS"]-0.5) < 1e-9, o

def test_threshold_and_curve():
    rng = np.random.default_rng(0); n = 600
    known = rng.random(n) < .7; ds = rng.choice(list("ABC"), n); true = np.where(known, rng.choice(list("xyz"), n), NEW)
    sim = np.where(known, rng.normal(.8,.08,n), rng.normal(.5,.1,n)); pred = np.where(known, true, "x")
    b = best_threshold(sim, pred, true, ds, known); assert b["NBA"] > .9 and .55 < b["tau"] < .75, b
    _, auc = open_set_curve(sim, pred, true, ds, known); assert auc > .9

def test_pca():
    rng = np.random.default_rng(1); z = rng.normal(size=(2000, 20)); A = rng.normal(size=(20, 128))
    x = z @ A + .05 * rng.normal(size=(2000, 128))
    c = PCACompressor(20, whiten=True).fit(x[:1500]); y = c.transform(x[1500:])
    assert y.shape == (500, 20) and abs(np.linalg.norm(y, axis=1) - 1).max() < 1e-4 and c.explained() > .95
    a, b = rng.normal(size=(5, 8)), rng.normal(size=(5, 4)); w = [.6, .4]
    cat = concat_weighted([a, b], w); lhs = cat @ cat.T
    rhs = .6 * l2n(a) @ l2n(a).T + .4 * l2n(b) @ l2n(b).T
    assert np.allclose(lhs, rhs, atol=1e-5)

def test_sampling_weights():
    w = sampling_weights(["i1"]*100 + ["i2"], ["d"]*101); assert w[-1] > w[0] and abs(w.mean()-1) < 1e-9

if __name__ == "__main__":
    for f in [test_splits_real, test_metrics_toy, test_threshold_and_curve, test_pca, test_sampling_weights]:
        f(); print("PASS", f.__name__)
