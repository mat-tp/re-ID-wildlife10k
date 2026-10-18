"""
Global Model Experiment Runner

Executes zero-shot evaluation and ArcFace fine-tuning workflows for global feature representation models
(MegaDescriptor Swin Transformer variants and MiewID EfficientNetV2).

Protocol Overview:
- Gallery: Dataset split defined by 'train' role.
- Queries: Dataset splits defined by 'val_known' and 'val_unknown' roles.
- Evaluation Metric: 1-Nearest-Neighbor cosine similarity search evaluated via closed-set BAKS
  (Balanced Accuracy on Known Sets) and open-set NBA (Normalized Balanced Accuracy).
"""

from typing import Any, Dict, Optional, Union
import json
import os
import sys
import numpy as np
import pandas as pd

import src.config as config
from src.data_processing.dataloader import (
    build_label_map,
    get_frames,
    make_eval_loader,
    make_train_loader,
)
from src.data_processing.splits import (
    NEW,
    TRAIN,
    VAL_KNOWN,
    VAL_UNKNOWN,
    cap_per_group,
)
from src.models.global_models import (
    GlobalModel,
    MiewIDModel,
    extract_features,
    fine_tune_global_model,
)
from src.retrieval.metrics import (
    compute_closed_set_baks,
    find_nearest_neighbors,
    find_optimal_threshold,
)


def evaluate_validation_performance(
    model: Union[GlobalModel, MiewIDModel],
    metadata_dataframe: pd.DataFrame,
    dataset_root_directory: str,
    gallery_sample_cap: Optional[int] = None,
    query_sample_cap: Optional[int] = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluates model performance on validation query splits (closed-set BAKS and open-set NBA).

    Args:
        model: Global model instance (GlobalModel or MiewIDModel).
        metadata_dataframe: Pandas DataFrame containing dataset metadata and split roles.
        dataset_root_directory: Root directory path storing image files.
        gallery_sample_cap: Optional upper limit on gallery images per identity for fast monitoring.
        query_sample_cap: Optional upper limit on query images per dataset role for fast monitoring.
        batch_size: Evaluation batch size for PyTorch DataLoader instances.

    Returns:
        Dictionary mapping metric names (e.g., 'closed_set_BAKS', 'open_NBA', 'open_tau') to score values.
    """
    # Slice metadata rows into gallery and query evaluation sets
    gallery_metadata = metadata_dataframe[metadata_dataframe["role"] == TRAIN]
    query_metadata = metadata_dataframe[
        metadata_dataframe["role"].isin([VAL_KNOWN, VAL_UNKNOWN])
    ]

    # Apply per-group sampling caps if specified (used for fast validation logging during fine-tuning)
    if gallery_sample_cap is not None:
        gallery_metadata = cap_per_group(
            gallery_metadata, group_columns="identity", max_samples=gallery_sample_cap
        )
    if query_sample_cap is not None:
        query_metadata = cap_per_group(
            query_metadata,
            group_columns=["dataset", "role"],
            max_samples=query_sample_cap,
        )

    # Resolve evaluation preprocessing pipeline
    if hasattr(model, "get_evaluation_transform"):
        evaluation_transform = model.get_evaluation_transform()
    else:
        evaluation_transform = model.eval_transform()

    print(f"\n[Validation] Extracting gallery features ({len(gallery_metadata)} images)...")
    gallery_loader = make_eval_loader(
        gallery_metadata,
        evaluation_transform,
        dataset_root_directory,
        batch_size,
    )
    gallery_features, _ = extract_features(
        model=model,
        data_loader=gallery_loader,
        use_automatic_mixed_precision=config.training["amp"],
        progress_bar_description="Gallery feature extraction",
    )

    print(f"\n[Validation] Extracting query features ({len(query_metadata)} images)...")
    query_loader = make_eval_loader(
        query_metadata,
        evaluation_transform,
        dataset_root_directory,
        batch_size,
    )
    query_features, _ = extract_features(
        model=model,
        data_loader=query_loader,
        use_automatic_mixed_precision=config.training["amp"],
        progress_bar_description="Query feature extraction",
    )

    print("\n[Validation] Computing 1-NN cosine similarity and open-set thresholds...")
    similarity_scores, predicted_identities = find_nearest_neighbors(
        query_features=query_features.numpy(),
        gallery_features=gallery_features.numpy(),
        gallery_labels=gallery_metadata["identity"].to_numpy(),
    )

    is_known_query_mask = (query_metadata["role"] == VAL_KNOWN).to_numpy()
    ground_truth_identities = np.where(
        is_known_query_mask,
        query_metadata["identity"].to_numpy(),
        NEW,
    )
    dataset_names = query_metadata["dataset"].to_numpy()

    # Compute closed-set BAKS score across known validation queries
    closed_set_baks_score = compute_closed_set_baks(
        best_similarities=similarity_scores,
        predicted_labels=predicted_identities,
        true_labels=ground_truth_identities,
        dataset_names=dataset_names,
        is_known_mask=is_known_query_mask,
    )

    # Calibrate open-set decision threshold tau and calculate open-set metrics
    optimal_threshold_results = find_optimal_threshold(
        best_similarities=similarity_scores,
        predicted_labels=predicted_identities,
        true_labels=ground_truth_identities,
        dataset_names=dataset_names,
        is_known_mask=is_known_query_mask,
    )

    performance_metrics: Dict[str, float] = {
        "closed_set_BAKS": closed_set_baks_score,
    }
    for metric_key, metric_value in optimal_threshold_results.items():
        performance_metrics[f"open_{metric_key}"] = metric_value

    return performance_metrics


def append_results_to_json_file(
    experiment_identifier: str, 
    results_dictionary: Dict[str, float], 
    results_filepath: str = "results/global_model_comparison.json"
) -> None:
    """Safely reads, updates, and persists experiment results to a JSON log file."""
    os.makedirs(os.path.dirname(os.path.abspath(results_filepath)), exist_ok=True)
    
    existing_results: Dict[str, Any] = {}
    if os.path.exists(results_filepath):
        try:
            with open(results_filepath, "r", encoding="utf-8") as file_handle:
                existing_results = json.load(file_handle)
        except Exception:
            existing_results = {}

    existing_results[experiment_identifier] = results_dictionary

    with open(results_filepath, "w", encoding="utf-8") as file_handle:
        json.dump(existing_results, file_handle, indent=2)

    print(f"Experiment results logged to '{results_filepath}'.")


def run_global_experiment(
    execution_mode: str, 
    model_name: str
) -> None:
    """
    Executes a global model experiment run.

    Args:
        execution_mode: Mode string ('zero_shot' or 'fine_tune').
        model_name: Target global model architecture ('megadescriptor' or 'miewid').
    """
    metadata_dataframe = get_frames(config.data)
    dataset_root_directory = config.data["dataset_root"]
    global_model_config = config.models["global"]
    training_config = config.training

    # Instantiate selected global embedding architecture
    if model_name.lower() == "miewid":
        model = MiewIDModel(
            huggingface_id=config.models["miewid"]["hf_id"]
        )
        initialization_tag = "miewid_pretrained"
    else:
        model = GlobalModel(
            variant=global_model_config["variant"],
            initialization=global_model_config["init"],
            enable_gradient_checkpointing=global_model_config["grad_checkpointing"],
        )
        initialization_tag = global_model_config["init"]

    # Execute zero-shot evaluation pipeline
    if execution_mode == "zero_shot":
        validation_scores = evaluate_validation_performance(
            model=model,
            metadata_dataframe=metadata_dataframe,
            dataset_root_directory=dataset_root_directory,
        )
        print(f"\nFinal Zero-Shot Validation Score ({model_name}):")
        print(validation_scores)

        append_results_to_json_file(
            experiment_identifier=f"{model_name}_zero_shot",
            results_dictionary=validation_scores,
        )
        return

    # Execute ArcFace fine-tuning pipeline
    if execution_mode == "fine_tune":
        # Fine-tune exclusively on the training split (never expose validation splits during optimization)
        train_metadata = metadata_dataframe[metadata_dataframe["role"] == TRAIN]
        identity_label_map = build_label_map(train_metadata)

        # Resolve training transform pipeline
        if hasattr(model, "get_training_transform"):
            training_transform = model.get_training_transform(config.augmentation)
        else:
            training_transform = model.train_transform(config.augmentation)

        train_data_loader = make_train_loader(
            metadata_df=train_metadata,
            transform=training_transform,
            label_map=identity_label_map,
            dataset_root=dataset_root_directory,
            micro_batch_size=training_config["micro_batch"],
            identity_alpha=config.sampling["id_alpha"],
            dataset_beta=config.sampling["dataset_beta"],
        )

        # Lightweight validation monitor callback evaluated during training iterations
        def monitoring_evaluation_callback(evaluation_model: Union[GlobalModel, MiewIDModel]) -> float:
            scores = evaluate_validation_performance(
                model=evaluation_model,
                metadata_dataframe=metadata_dataframe,
                dataset_root_directory=dataset_root_directory,
                gallery_sample_cap=5,
                query_sample_cap=60,
            )
            return scores.get("open_NBA", 0.0)

        checkpoint_filepath = f"checkpoints/{model_name}_{initialization_tag}_best.pt"

        # Fine-tune global model backbone and ArcFace classification header
        fine_tune_global_model(
            model=model,
            train_loader=train_data_loader,
            num_classes=len(identity_label_map),
            epochs=training_config["epochs"],
            learning_rate=training_config["lr"],
            head_learning_rate_multiplier=training_config["head_lr_mult"],
            weight_decay=training_config["weight_decay"],
            warmup_epochs=training_config["warmup_epochs"],
            arcface_margin=training_config["arcface_margin"],
            arcface_scale=training_config["arcface_scale"],
            gradient_accumulation_steps=training_config["accum_steps"],
            max_gradient_norm=training_config["grad_clip"],
            use_automatic_mixed_precision=training_config["amp"],
            evaluation_function=monitoring_evaluation_callback,
            evaluation_interval_epochs=2,
            checkpoint_filepath=checkpoint_filepath,
        )

        # Load best saved state checkpoint for full validation evaluation
        if hasattr(model, "load_checkpoint"):
            model.load_checkpoint(checkpoint_filepath)
        else:
            model.load(checkpoint_filepath)

        full_validation_scores = evaluate_validation_performance(
            model=model,
            metadata_dataframe=metadata_dataframe,
            dataset_root_directory=dataset_root_directory,
        )
        print(f"\nFinal Fine-Tuned Validation Score ({model_name}):")
        print(full_validation_scores)

        append_results_to_json_file(
            experiment_identifier=f"{model_name}_fine_tuned",
            results_dictionary=full_validation_scores,
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.models.run_global_experiments <mode> <model_name>")
        print("Modes: 'zero_shot', 'fine_tune'")
        print("Models: 'megadescriptor', 'miewid'")
        sys.exit(1)

    parsed_execution_mode = sys.argv[1]
    parsed_model_name = sys.argv[2]

    run_global_experiment(
        execution_mode=parsed_execution_mode, 
        model_name=parsed_model_name
    )