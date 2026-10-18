"""
Driver for the global-model study (proposal RQ-i). UNTESTED end-to-end: needs torch, a GPU and the
images, none of which were available where this was written. Read it, then run step by step.

  python -m src.models.run_global_experiments zero_shot  megadescriptor
  python -m src.models.run_global_experiments zero_shot  miewid
  python -m src.models.run_global_experiments fine_tune  megadescriptor

Protocol (identical for every model, validation data only until the very end):
  gallery = role 'train'; queries = 'val_known' + 'val_unknown'; 1-NN cosine; tau picked on val.
"""
import sys
import numpy as np
import src.config as C
from src.data_processing.dataloader import (get_frames, build_label_map, make_train_loader, make_eval_loader)
from src.data_processing.splits import TRAIN, VAL_KNOWN, VAL_UNKNOWN, NEW, cap_per_group
from src.models.global_models import GlobalModel, MiewIDModel, extract_features, fine_tune
from src.retrieval.metrics import nearest_neighbour, closed_set_baks, best_threshold


def validation_score(model, df, root, gallery_cap=None, query_cap=None, batch=32):
    """Returns dict(closed BAKS, open NBA, tau ...) on the validation queries."""
    gal = df[df.role == TRAIN]
    qry = df[df.role.isin([VAL_KNOWN, VAL_UNKNOWN])]
    if gallery_cap: gal = cap_per_group(gal, "identity", gallery_cap)
    if query_cap:   qry = cap_per_group(qry, ["dataset", "role"], query_cap)
    tf = model.eval_transform()
    print(f"\n[Validation] Extracting gallery features ({len(gal)} images)...")
    g, _ = extract_features(model, make_eval_loader(gal, tf, root, batch), amp=C.training["amp"], desc="Gallery features")
    print(f"\n[Validation] Extracting query features ({len(qry)} images)...")
    q, _ = extract_features(model, make_eval_loader(qry, tf, root, batch), amp=C.training["amp"], desc="Query features")
    print("\n[Validation] Computing 1-NN cosine similarity & open-set thresholds...")
    s, pred = nearest_neighbour(q.numpy(), g.numpy(), gal.identity.to_numpy())
    known = (qry.role == VAL_KNOWN).to_numpy()
    true = np.where(known, qry.identity.to_numpy(), NEW)
    ds = qry.dataset.to_numpy()
    out = {"closed_BAKS": closed_set_baks(s, pred, true, ds, known)}
    out.update({f"open_{k}": v for k, v in best_threshold(s, pred, true, ds, known).items()})
    return out


def main(mode, which):
    df = get_frames(C.data); root = C.data["dataset_root"]; g = C.models["global"]; T = C.training
    model = MiewIDModel(C.models["miewid"]["hf_id"]) if which == "miewid" else \
        GlobalModel(g["variant"], g["init"], g["grad_checkpointing"])
    if mode == "zero_shot":
        score = validation_score(model, df, root)
        print("\nFinal Validation Score:", score)
        import os, json
        os.makedirs("results", exist_ok=True)
        res_file = "results/global_model_comparison.json"
        existing = {}
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f: existing = json.load(f)
            except Exception: pass
        existing[f"{which}_zero_shot"] = score
        with open(res_file, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Results automatically logged to {res_file}")
        return

    train_df = df[df.role == TRAIN]                     # val_known / val_unknown are NOT trained on
    label_map = build_label_map(train_df)
    loader = make_train_loader(train_df, model.train_transform(C.augmentation), label_map, root,
                               T["micro_batch"], C.sampling["id_alpha"], C.sampling["dataset_beta"])
    eval_fn = lambda m: validation_score(m, df, root, gallery_cap=5, query_cap=60)["open_NBA"]   # cheap monitor
    fine_tune(model, loader, num_classes=len(label_map), epochs=T["epochs"], lr=T["lr"],
              head_lr_mult=T["head_lr_mult"], weight_decay=T["weight_decay"], warmup_epochs=T["warmup_epochs"],
            arcface_margin=T["arcface_margin"], arcface_scale=T["arcface_scale"],
              accum_steps=T["accum_steps"], grad_clip=T["grad_clip"], amp=T["amp"],
              eval_fn=eval_fn, eval_every=2, ckpt_path=f"checkpoints/{which}_{g['init']}_best.pt")
    model.load(f"checkpoints/{which}_{g['init']}_best.pt")
    full_score = validation_score(model, df, root)
    print("\nFULL validation:", full_score)
    import os, json
    os.makedirs("results", exist_ok=True)
    res_file = "results/global_model_comparison.json"
    existing = {}
    if os.path.exists(res_file):
        try:
            with open(res_file, "r") as f: existing = json.load(f)
        except Exception: pass
    existing[f"{which}_fine_tuned"] = full_score
    with open(res_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results automatically logged to {res_file}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
