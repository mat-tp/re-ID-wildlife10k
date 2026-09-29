"""
This file handles all configuration settings for the entire pipeline including:
- Data paths and preprocessing parameters
- Model configurations (MiewID, MegaDescriptor, EVA02, ALIKED, DINOv2)
- Feature extraction parameters
- Calibration settings (isotonic regression, PCHIP interpolation)
- Fusion weights and methods
- Compression parameters (PCA)
- Evaluation protocols (closed-set, open-set)
- BAKS and BAUS metrics configuration

TODO: Implement configuration loading and saving
TODO: Add validation for configuration parameters
TODO: Create default configurations for different experiments
"""

# Fill in your configurations here
# Dictionary is used to represent the configurations (key-value) pairs.
data = {
    "dataset_root": "data/raw/WildlifeReID-10k", # The path which leads to the .excel file
}

models = {
    "global": {
        "name": "megadescriptor",
        "embedding_dim": 2048,
        "fine_tune": True
    },
    "complementary": {
        "name": "eva02",
        "timm_id": "eva02_base_patch14_224.mim_in22k",  # 87 M params, dim=768
        "embedding_dim": 768,
        "global_pool": "avg",
    },
    "local": {
        "name": "aliked",
        "num_features": 1000
    },
    "baseline": {
        "name": "dinov2_vits14",  # 21 M params, dim=384
        "embedding_dim": 384,
    }
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
    "target_dim": 256
}

evaluation = {
    "protocols": ["closed_set", "open_set"],
    "metrics": ["accuracy", "mAP", "BAKS", "BAUS"],
    "top_k": [1, 5, 10]
}