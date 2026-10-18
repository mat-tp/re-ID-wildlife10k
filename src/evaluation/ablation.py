"""
End-to-end experiment runner module for wildlife re-identification.

Executes shortlisting, branch scoring, calibration, fusion, and evaluation workflows across
closed-set and open-set scenarios.
"""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluation.closed_set import evaluate_closed_set
from src.evaluation.open_set import evaluate_open_set, select_tau
from src.evaluation.protocol import QuerySet, protocol
from src.features.calibration import ScoreCalibrator
from src.features.fusion import fuse, normalise_weights, tune_weights
from src.retrieval.metrics import best_threshold
from src.retrieval.search import Ranked, rank_by_embedding
from src.retrieval.similarity import candidate_cosine

LOCAL_BRANCH_NAME = "local"


def get_configuration_name(branches: Tuple[str, ...]) -> str:
    """Combines feature branch names into a unique configuration identifier string."""
    return "+".join(branches)


def compress_feature_branches(
    features: Dict[str, np.ndarray],
    dataframe: pd.DataFrame,
    target_dimensionality: Optional[int],
    whiten: bool = True,
    fit_rows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Applies PCA compression per feature branch, fitted exclusively on training rows to prevent data leakage.
    """
    if target_dimensionality is None:
        return features

    from src.compression.pca import PCACompressor

    training_rows = (
        fit_rows
        if fit_rows is not None
        else dataframe.index[dataframe["role"] == "train"].to_numpy()
    )

    return {
        branch_name: PCACompressor(target_dimensionality, whiten)
        .fit(feature_matrix[training_rows])
        .transform(feature_matrix)
        for branch_name, feature_matrix in features.items()
    }


class Experiment:
    """Manages end-to-end evaluation, calibration, weight tuning, and ablation studies."""

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
        local_scorer: Optional[Any] = None,
        k_candidates: int = 100,
        k_local_candidates: int = 20,
        calibration_method: str = "isotonic",
        weight_step: float = 0.1,
        max_calibration_queries: int = 4000,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # Resolve positional kwargs or alternative parameter keys for backward compatibility
        if dataframe is None and "df" in kwargs:
            dataframe = kwargs.pop("df")
        if features is None and "feats" in kwargs:
            features = kwargs.pop("feats")
        if "K" in kwargs:
            k_candidates = kwargs.pop("K")
        if "K_local" in kwargs:
            k_local_candidates = kwargs.pop("K_local")
        if "calibration" in kwargs:
            calibration_method = kwargs.pop("calibration")
        if "max_cal_queries" in kwargs:
            max_calibration_queries = kwargs.pop("max_cal_queries")
        if "step" in kwargs:
            weight_step = kwargs.pop("step")

        self.max_queries = kwargs.pop("max_queries", None)

        self.dataframe = dataframe
        self.features = features or {}
        self.local_scorer = local_scorer
        self.k_candidates = k_candidates
        self.k_local_candidates = k_local_candidates
        self.weight_step = weight_step
        self.calibration_method = calibration_method
        self.max_calibration_queries = max_calibration_queries
        self.seed = seed

        # Set up validation splits from protocol
        if self.dataframe is not None:
            self.gallery_val, self.query_val = protocol(self.dataframe, "val")
        else:
            self.gallery_val, self.query_val = np.array([]), None

        self._prepared_cache: Dict[Tuple[str, Tuple[str, ...], int], Dict[str, Any]] = {}
        self.calibrators: Dict[str, ScoreCalibrator] = {}

        # Backward compatibility attributes
        self.df = self.dataframe
        self.feats = self.features
        self.local = self.local_scorer
        self.K = self.k_candidates
        self.K_local = self.k_local_candidates
        self.step = self.weight_step
        self.cal = self.calibrators

    def _prepare_evaluation_splits(
        self, split: str, branches: Tuple[str, ...]
    ) -> Dict[str, Any]:
        """Prepares gallery rows, query sets, candidate shortlists, and raw branch scores."""
        if split == "val":
            gallery_rows, query_info = self.gallery_val, self.query_val
        else:
            gallery_rows, query_info = protocol(self.dataframe, "test")

        if self.max_queries and query_info is not None and len(query_info.rows) > self.max_queries:
            rng = np.random.default_rng(self.seed)
            selected_indices = np.sort(
                rng.choice(len(query_info.rows), self.max_queries, replace=False)
            )
            query_info = QuerySet(
                rows=query_info.rows[selected_indices],
                true=query_info.true[selected_indices],
                known=query_info.known[selected_indices],
                datasets=query_info.datasets[selected_indices],
                species=query_info.species[selected_indices],
            )

        cache_key = (split, tuple(branches), len(query_info.rows))
        if cache_key in self._prepared_cache:
            return self._prepared_cache[cache_key]

        non_local_branches = [branch for branch in branches if branch != LOCAL_BRANCH_NAME]
        primary_branch = (
            non_local_branches[0]
            if non_local_branches
            else ("global" if "global" in self.features else next(iter(self.features)))
        )
        candidate_limit = (
            self.k_local_candidates
            if LOCAL_BRANCH_NAME in branches
            else self.k_candidates
        )

        timer_start = time.perf_counter()
        ranked_candidates = rank_by_embedding(
            self.features[primary_branch][query_info.rows],
            self.features[primary_branch][gallery_rows],
            candidate_limit,
        )

        raw_branch_scores: Dict[str, np.ndarray] = {}
        for branch in branches:
            if branch == primary_branch:
                raw_branch_scores[branch] = ranked_candidates.score
            elif branch == LOCAL_BRANCH_NAME:
                if self.local_scorer is None:
                    raise RuntimeError(
                        "Configuration specifies 'local' branch, but no local scorer was provided."
                    )
                raw_branch_scores[branch] = self.local_scorer.score_pairs(
                    query_info.rows, gallery_rows[ranked_candidates.cand]
                )
            else:
                raw_branch_scores[branch] = candidate_cosine(
                    self.features[branch][query_info.rows],
                    self.features[branch][gallery_rows],
                    ranked_candidates.cand,
                )

        elapsed_time_ms = 1000.0 * (time.perf_counter() - timer_start)
        gallery_identities = self.dataframe.identity.to_numpy()[gallery_rows]

        prepared_data = {
            "gallery_rows": gallery_rows,
            "query_info": query_info,
            "candidate_indices": ranked_candidates.cand,
            "raw_scores": raw_branch_scores,
            "gallery_identities": gallery_identities,
            "ms_per_query": elapsed_time_ms / max(len(query_info.rows), 1),
        }

        self._prepared_cache[cache_key] = prepared_data
        return prepared_data

    def fit_score_calibrators(
        self, branches: Optional[List[str]] = None
    ) -> "Experiment":
        """Fits score calibrators for each feature branch using validation pairs."""
        target_branches = branches or sorted(
            set(self.features.keys()) | ({LOCAL_BRANCH_NAME} if self.local_scorer else set())
        )
        rng = np.random.default_rng(self.seed)

        for branch in target_branches:
            if branch == LOCAL_BRANCH_NAME:
                branch_configuration = (
                    ("global", LOCAL_BRANCH_NAME)
                    if "global" in self.features
                    else (next(iter(self.features)), LOCAL_BRANCH_NAME)
                )
            else:
                branch_configuration = (branch,)

            prepared_data = self._prepare_evaluation_splits("val", branch_configuration)

            matches = (
                prepared_data["gallery_identities"][prepared_data["candidate_indices"]]
                == prepared_data["query_info"].true[:, None]
            ) & prepared_data["query_info"].known[:, None]

            query_indices = np.arange(len(matches))
            if len(query_indices) > self.max_calibration_queries:
                query_indices = rng.choice(
                    query_indices, self.max_calibration_queries, replace=False
                )

            self.calibrators[branch] = ScoreCalibrator(self.calibration_method).fit(
                prepared_data["raw_scores"][branch][query_indices], matches[query_indices]
            )

        return self

    def _apply_calibrations(
        self, prepared_data: Dict[str, Any], branches: Tuple[str, ...]
    ) -> Dict[str, np.ndarray]:
        """Applies fitted score calibrators to raw branch scores."""
        return {
            branch: self.calibrators[branch].transform(prepared_data["raw_scores"][branch])
            for branch in branches
        }

    def run_configuration(
        self,
        branches: Union[List[str], Tuple[str, ...]],
        weights: Optional[Dict[str, float]] = None,
        objective: str = "NBA",
        per_species_tau: bool = False,
        test: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates a single feature branch configuration, tuning weights and decision thresholds on validation."""
        branches_tuple = tuple(branches)
        if not self.calibrators:
            self.fit_score_calibrators()

        prepared_data = self._prepare_evaluation_splits("val", branches_tuple)
        calibrated_scores = self._apply_calibrations(prepared_data, branches_tuple)

        def objective_function(fused_scores: np.ndarray) -> float:
            ranked_results = Ranked(prepared_data["candidate_indices"], fused_scores).resort(
                fused_scores
            )
            top_scores = ranked_results.score[:, 0]
            top_predictions = prepared_data["gallery_identities"][ranked_results.cand[:, 0]]

            if objective == "BAKS":
                from src.retrieval.metrics import closed_set_baks

                return closed_set_baks(
                    top_scores,
                    top_predictions,
                    prepared_data["query_info"].true,
                    prepared_data["query_info"].datasets,
                    prepared_data["query_info"].known,
                )

            return best_threshold(
                top_scores,
                top_predictions,
                prepared_data["query_info"].true,
                prepared_data["query_info"].datasets,
                prepared_data["query_info"].known,
                n_thresholds=60,
            )["NBA"]

        if weights:
            optimal_weights = normalise_weights(weights)
        else:
            optimal_weights = tune_weights(
                calibrated_scores, objective_function, self.weight_step
            )[0]

        fused_validation_scores = fuse(calibrated_scores, optimal_weights)
        ranked_val = Ranked(prepared_data["candidate_indices"], fused_validation_scores)
        ranked_val = ranked_val.resort(ranked_val.score)

        best_thresholds = select_tau(
            ranked_val,
            prepared_data["gallery_identities"],
            prepared_data["query_info"],
            per_species=per_species_tau,
        )

        row_results: Dict[str, Any] = {
            "config": get_configuration_name(branches_tuple),
            "weights": {
                key: round(val, 3) for key, val in optimal_weights.items()
            },
            "K": ranked_val.cand.shape[1],
            "tau": best_thresholds["tau"],
            "ms_per_query": round(prepared_data["ms_per_query"], 2),
        }

        closed_set_val_metrics = evaluate_closed_set(
            ranked_val, prepared_data["gallery_identities"], prepared_data["query_info"]
        )
        row_results.update({
            f"val_{key}": val
            for key, val in closed_set_val_metrics.items()
            if key != "per_dataset"
        })
        row_results.update({
            "val_open_BAKS": best_thresholds["BAKS"],
            "val_open_BAUS": best_thresholds["BAUS"],
            "val_open_NBA": best_thresholds["NBA"],
        })

        if test:
            test_data = self._prepare_evaluation_splits("test", branches_tuple)
            fused_test_scores = fuse(
                self._apply_calibrations(test_data, branches_tuple), optimal_weights
            )
            ranked_test = Ranked(test_data["candidate_indices"], fused_test_scores)
            ranked_test = ranked_test.resort(ranked_test.score)

            closed_set_test_metrics = evaluate_closed_set(
                ranked_test, test_data["gallery_identities"], test_data["query_info"]
            )
            row_results.update({
                f"test_{key}": val
                for key, val in closed_set_test_metrics.items()
                if key != "per_dataset"
            })

            open_set_test_results = evaluate_open_set(
                ranked_test,
                test_data["gallery_identities"],
                test_data["query_info"],
                best_thresholds["tau"],
                best_thresholds.get("species_tau"),
            )
            row_results.update({
                "test_open_BAKS": open_set_test_results["BAKS"],
                "test_open_BAUS": open_set_test_results["BAUS"],
                "test_open_NBA": open_set_test_results["NBA"],
            })

        row_results["_species_tau"] = best_thresholds.get("species_tau")
        return row_results

    def run_ablation_study(
        self,
        configurations: List[Union[List[str], Tuple[str, ...]]],
        fixed_weights: Optional[Dict[Tuple[str, ...], Dict[str, float]]] = None,
        test: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Executes ablation experiments across a list of branch configurations."""
        weights_map = fixed_weights or {}
        results_rows: List[Dict[str, Any]] = []

        for index, config_branches in enumerate(configurations, 1):
            config_tuple = tuple(config_branches)
            print(
                f"[ablation {index}/{len(configurations)}] Running config: "
                f"{' + '.join(config_tuple)} (test={test})...",
                flush=True,
            )
            start_time = time.perf_counter()

            config_weights = weights_map.get(config_tuple)
            result = self.run_configuration(
                list(config_tuple), weights=config_weights, test=test, **kwargs
            )

            elapsed_seconds = time.perf_counter() - start_time
            print(
                f"[ablation {index}/{len(configurations)}] Completed in {elapsed_seconds:.1f}s",
                flush=True,
            )
            results_rows.append(result)

        return pd.DataFrame(results_rows)

    # Aliases for backward compatibility
    ablation = run_ablation_study
    run_config = run_configuration
    fit_calibrators = fit_score_calibrators
    _prepare = _prepare_evaluation_splits
    _calibrated = _apply_calibrations


def get_default_ablation_configurations(
    global_name: str = "global",
    complementary_name: str = "complementary",
    local_name: str = LOCAL_BRANCH_NAME,
    baseline_name: str = "baseline",
    with_local: bool = True,
) -> List[Tuple[str, ...]]:
    """Generates standard feature branch combinations for ablation studies."""
    g, c, l, b = global_name, complementary_name, local_name, baseline_name
    configurations: List[Tuple[str, ...]] = [(g,), (c,), (b,)]

    if with_local:
        configurations.append((l,))

    configurations.append((g, c))

    if with_local:
        configurations.extend([(g, l), (c, l), (g, c, l)])

    return configurations


def save_experiment_artifacts(
    experiment: Experiment,
    results_dataframe: pd.DataFrame,
    output_directory: str = "results",
    name: str = "ablation_val",
) -> None:
    """Saves experiment evaluation tables, threshold lookup cards, and fitted calibrator pickles to disk."""
    os.makedirs(output_directory, exist_ok=True)

    clean_table = results_dataframe.drop(
        columns=[column for column in results_dataframe.columns if column.startswith("_")]
    )
    clean_table.assign(weights=clean_table["weights"].astype(str)).to_csv(
        os.path.join(output_directory, f"{name}.csv"), index=False
    )

    threshold_card = {
        row["config"]: {
            "weights": row["weights"],
            "tau": float(row["tau"]),
            "species_tau": {
                str(k): float(v)
                for k, v in (row.get("_species_tau") or {}).items()
            },
        }
        for row in results_dataframe.to_dict("records")
    }

    with open(os.path.join(output_directory, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_card, f, indent=2)

    with open(os.path.join(output_directory, "calibrators.pkl"), "wb") as f:
        pickle.dump(experiment.calibrators, f)


# Backward compatibility function aliases
config_name = get_configuration_name
compress_features = compress_feature_branches
default_ablation_configs = get_default_ablation_configurations
save_artifacts = save_experiment_artifacts
