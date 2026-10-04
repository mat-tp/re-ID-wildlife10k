"""
Central configuration for the whole pipeline (plain dictionaries, as in the skeleton).

Every value that was chosen on purpose carries a tag that says WHY:
  [PROPOSAL]  stated in docs/main.tex
  [PAPER]     taken from a referenced paper (MegaDescriptor / WildlifeReID-10k / WildFusion ...)
  [PRACTICE]  standard practice in the re-ID / metric-learning literature
  [JUDGEMENT] our own starting point - MUST be tuned/checked on the validation split
See DECISIONS.md for the long-form reasoning.
"""

data = {
    # CSV paths look like "images/<dataset>/...", so images live at <root>/images/...
    "dataset_root": "data",
    "metadata_csv": "data/metadata.csv",
    # Validation split is NOT provided by the benchmark, so we carve it out of `train`
    # (see src/data_processing/splits.py).                                 [PROPOSAL: val for tuning]
    "val": {
        "unknown_id_fraction": 0.10,   # share of identities per dataset held out as "new animals"
        "known_img_fraction": 0.10,    # share of images per identity held out as val queries (whole clusters)
        "min_ids_for_unknown": 10,     # datasets with fewer train identities keep all of them
        "seed": 0,
    },
    # Images in WildlifeReID-10k are already cropped to the animal by the authors. [PAPER]
    # So MegaDetector is OFF by default (see DECISIONS.md #3).
    "use_megadetector": False,
}

models = {
    "global": {
        "name": "megadescriptor",
        "variant": "L-384",             # L-384 -> 1536-d (config used to say 2048, which is wrong)
        "init": "megadescriptor",       # "imagenet" = leakage-free start (see global_models.py)
        "grad_checkpointing": True,
        "embedding_dim": 1536,
        "fine_tune": True
    },
    "miewid": {
        "hf_id": "conservationxlabs/miewid-msv3"
    },
    "complementary": {
        "name": "eva02",
        "timm_id": "eva02_base_patch14_224.mim_in22k",
        "embedding_dim": 768,
        "global_pool": "avg",
    },
    "local": {
        "name": "aliked",
        "num_features": 1000
    },
    "baseline": {
        "name": "dinov2_vits14",
        "embedding_dim": 384,
    },
}

augmentation = {
    "resize_mode": "squash",           # same as MegaDescriptor's own transform  [PAPER]
    "random_resized_crop_scale": (0.75, 1.0),   # mild: identity cues must survive [JUDGEMENT]
    "rotation_deg": 10,                                                          # [JUDGEMENT]
    "color_jitter": dict(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),  # hue tiny: colour is identity cue
    "gaussian_blur_p": 0.1,            # wild images are often soft / low-res     [JUDGEMENT]
    "random_erasing_p": 0.25,          # occlusion robustness                    [PRACTICE: Luo et al. 2019]
    "hflip_species": [],               # mirror flips create fake identities for asymmetric patterns -> off
}

sampling = {
    "id_alpha": 0.5,       # weight ~ n_images_of_identity ** -alpha             [JUDGEMENT, sqrt rule]
    "dataset_beta": 0.25,  # weight ~ n_images_of_dataset  ** -beta              [JUDGEMENT]
}

training = {
    "optimizer": "sgd", "momentum": 0.9,     # [PAPER: MegaDescriptor]
    "lr": 1e-3,                              # [PAPER: grid {0.01, 0.001}]
    "head_lr_mult": 10.0,                    # ArcFace head starts random          [PRACTICE]
    "weight_decay": 1e-4,                    # [JUDGEMENT] (skeleton used 0)
    "schedule": "cosine", "warmup_epochs": 1,   # cosine [PAPER]; warm-up [PRACTICE]
    "effective_batch": 128,                  # [PAPER]
    "micro_batch": 16, "accum_steps": 8,     # 16*8=128 fits Swin-L-384 in fp16
    "epochs": 12,                            # paper used 100 (~1 week/GPU); budget-limited [JUDGEMENT]
    "arcface_margin": 0.5, "arcface_scale": 64.0,   # [PAPER grid {0.25,.5,.75}x{32,64,128}]
    "grad_clip": 5.0,
    "amp": True,
}

# These should sum up to one.
fusion_weights = {
    "global_weight": 0.4,
    "complementary_weight": 0.3,
    "local_weight": 0.3,
    "method": "weighted_sum"
}

calibration = {
    "method": "isotonic",  # or "pchip"
    "use_validation": True
}

compression = {
    "method": "pca",
    "target_dims": [32, 64, 128, 256, 512],   # sweep; None = uncompressed reference [PROPOSAL]
    "target_dim": 256,
    "whiten": True,                            # [PRACTICE: Jegou & Chum 2012] - ablate on/off
    "fit_on": "train",                         # [PROPOSAL: PCA fitted on training data]
}

evaluation = {
    "protocols": ["closed_set", "open_set"],
    "metrics": ["accuracy", "mAP", "BAKS", "BAUS"],
    "top_k": [1, 5, 10]
}

# Retrieval / experiment runner (src/evaluation/ablation.py)
retrieval = {
    "K": 100,            # shortlist size for vector-only configs (also the depth of mAP@K)
    "K_local": 20,       # ALIKED/LightGlue is one matcher pass per pair, so only the best 20 candidates get it
    "weight_step": 0.1,  # fusion-weight grid resolution
    "feature_cache": "features",   # features/<branch dir>/shard_*.npy  (see features/extraction.py)
    "results_dir": "results",
}
