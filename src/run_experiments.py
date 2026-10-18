"""
Command-line entry point for the real experiments.

  python -m src.run_experiments extract global|miewid|complementary|baseline     # hours on GPU, resumable (shards)
  python -m src.run_experiments ablation            [--test] [--no-local]         # val table (+ frozen test with --test)
  python -m src.run_experiments compression         [--test]                      # PCA sweep over config.compression
  python -m src.run_experiments ablation --demo                                   # synthetic features, checks the plumbing only

Rule: do NOT pass --test until every choice (global model, K, weights, tau) is final. The test split is used once.
"""
import argparse, os
import src.config as C
from src.data_processing.splits import get_frames
from src.evaluation.ablation import Experiment, default_ablation_configs, compress_features, save_artifacts

DIRS = {"global": "megadescriptor", "miewid": "miewid", "complementary": "eva02", "baseline": "dinov2"}


def build_model(which):
    if which == "global":
        from src.models.global_models import GlobalModel
        g = C.models["global"]
        m = GlobalModel(g["variant"], g["init"])
        ckpt = "checkpoints/megadescriptor_megadescriptor_latest.pt"
        if os.path.isfile(ckpt):
            print(f"[build_model] Loading fine-tuned weights from {ckpt}...")
            m.load(ckpt)
        return m
    if which == "miewid":
        from src.models.global_models import MiewIDModel
        return MiewIDModel(C.models["miewid"]["hf_id"])
    if which == "complementary":
        from src.models.complementary_models import ComplementaryModel
        return ComplementaryModel(C.models["complementary"]["timm_id"])
    from src.models.baseline_models import DINOv2Baseline
    return DINOv2Baseline()


def get_experiment(args, df):
    R = C.retrieval
    if args.demo:
        from src.demo import make_demo_features, DemoLocalScorer
        return Experiment(df, make_demo_features(df), None if args.no_local else DemoLocalScorer(df), R["K"], R["K_local"], weight_step=R["weight_step"])
    from src.features.extraction import load_features
    feats = load_features(R["feature_cache"], {b: d for b, d in DIRS.items() if os.path.isdir(os.path.join(R["feature_cache"], d))})
    local = None
    k_local = args.k_local if hasattr(args, "k_local") and args.k_local else R["K_local"]
    max_q = getattr(args, "max_queries", None)
    if not args.no_local:
        from src.retrieval.similarity import LocalScorer
        from src.data_processing.dataloader import convert_to_rgb_image
        cache_path = os.path.join(args.out, "local_pairs_cache.pkl")
        local = LocalScorer(lambda r: convert_to_rgb_image(df.path.iloc[r], C.data["dataset_root"]), use_ransac=getattr(args, "ransac", False), cache_file=cache_path)
    return Experiment(df, feats, local, R["K"], k_local, weight_step=R["weight_step"], max_queries=max_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "ablation", "compression"]); ap.add_argument("what", nargs="?")
    ap.add_argument("--test", action="store_true"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--out", default=C.retrieval["results_dir"])
    ap.add_argument("--max-queries", type=int, default=None, help="Limit validation/test queries for fast representative evaluation")
    ap.add_argument("--k-local", type=int, default=10, help="Candidate shortlist size for local verification")
    ap.add_argument("--ransac", action="store_true", help="Enable OpenCV MAGSAC/RANSAC geometric verification (slower)")
    ap.add_argument("--configs", type=str, default=None, help="Comma-separated configs to run, e.g. 'global,global+complementary,global+complementary+local'")
    a = ap.parse_args(); df = get_frames(C.data)
    if a.cmd == "extract":
        from src.features.extraction import extract_all
        extract_all(build_model(a.what), df, C.data["dataset_root"], os.path.join(C.retrieval["feature_cache"], DIRS[a.what]))
        return
    ex = get_experiment(a, df)
    if a.cmd == "ablation":
        names = [b for b in ex.feats]
        if a.configs:
            cfgs = [tuple(c.strip().split("+")) for c in a.configs.split(",")]
        else:
            cfgs = default_ablation_configs(with_local=not a.no_local) if {"global", "complementary", "baseline"} <= set(names) else [(b,) for b in names]
        res = ex.ablation(cfgs, test=a.test)
        print(res.drop(columns=[c for c in res.columns if c.startswith("_")]).round(4).to_string())
        save_artifacts(ex, res, a.out, "ablation_test" if a.test else "ablation_val")
    else:
        import pandas as pd
        rows = []
        for d in C.compression["target_dims"] + [None]:
            e = Experiment(df, compress_features(ex.feats, df, d, C.compression["whiten"]), ex.local, ex.K, ex.K_local, weight_step=ex.step)
            r = e.run_config(("global", "complementary") if "complementary" in ex.feats else ("global",), test=a.test)
            rows.append({"dim": d or "full", **{k: v for k, v in r.items() if not k.startswith("_")}})
        t = pd.DataFrame(rows); print(t.round(4).to_string()); os.makedirs(a.out, exist_ok=True); t.assign(weights=t.weights.astype(str)).to_csv(os.path.join(a.out, "compression.csv"), index=False)


if __name__ == "__main__":
    main()
