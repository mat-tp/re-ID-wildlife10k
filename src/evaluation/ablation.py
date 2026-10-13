"""
End-to-end experiment runner module for wildlife re-identification.
Executes shortlisting, branch scoring, calibration, fusion, and evaluation workflows.
"""
import time
from typing import Dict, List, Optional, Tuple, Union
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


def get_configuration_name(branches: Tuple[str, ...]) -> str:
    """Combines branch names into a unique configuration identifier string."""
    return "+".join(branches)


def compress_feature_branches(
    features: Dict[str, np.ndarray],
    dataframe: pd.DataFrame,
    dimensionality: Optional[int],
    whiten: bool = True,
    fit_rows: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Applies PCA compression per feature branch, fitted exclusively on training rows to prevent leakage."""
    if dimensionality is None:
        return features
        
    from src.compression.pca import PCACompressor
    training_rows = fit_rows if fit_rows is not None else dataframe.index[dataframe.role == "train"].to_numpy()
    
    return {
        branch: PCACompressor(dimensionality, whiten).fit(feat_matrix[training_rows]).transform(feat_matrix)
        for branch, feat_matrix in features.items()
    }


class Experiment:
    """Manages end-to-end evaluation, calibration, weight tuning, and ablation studies."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: Dict[str, np.ndarray],
        local_scorer=None,
        k_candidates: int = 100,
        k_local_candidates: int = 20,
        calibration_method: str = "isotonic",
        weight_step: float = 0.1,
        max_calibration_queries: int = 4000,
        seed: int = 0
    ):
        self.dataframe = dataframe
        self.features = features
        self.local_scorer = local_scorer
        self.k_candidates = k_candidates
        self.k_local_candidates = k_local_candidates
        self.weight_step = weight_step
        self.calibration_method = calibration_method
        self.max_calibration_queries = max_calibration_queries
        self.seed = seed
        
        self.gallery_val, self.query_val = protocol(dataframe, "val")
        self._prepared_cache = {}
        self.calibrators = {}

    def _prepare_evaluation_splits(self, split: str, branches: Tuple[str, ...]) -> dict:
        """Prepares gallery rows, query sets, candidate shortlists, and raw branch scores."""
        gallery_rows, query_info = (self.gallery_val, self.query_val) if split == "val" else protocol(self.dataframe, "test")
        cache_key = (split, tuple(branches))
        
        if cache_key in self._prepared_cache:
            return self._prepared_cache[cache_key]
            
        cheap_branches = [branch for branch in branches if branch != LOCAL]
        shortest_branch = cheap_branches[0] if cheap_branches else ("global" if "global" in self.features else next(iter(self.features)))
        k_limit = self.k_local_candidates if LOCAL in branches else self.k_candidates
        
        timer_start = time.perf_counter()
        ranked = rank_by_embedding(self.features[shortest_branch][query_info.rows], self.features[shortest_branch][gallery_rows], k_limit)
        
        raw_scores = {}
        for branch in branches:
            if branch == shortest_branch:
                raw_scores[branch] = ranked.score
            elif branch == LOCAL:
                if self.local_scorer is None:
                    raise RuntimeError("Configuration specifies 'local' branch but no local scorer was provided.")
                raw_scores[branch] = self.local_scorer.score_pairs(query_info.rows, gallery_rows[ranked.cand])
            else:
                raw_scores[branch] = candidate_cosine(self.features[branch][query_info.rows], self.features[branch][gallery_rows], ranked.cand)
                
        elapsed_time_ms = 1000 * (time.perf_counter() - timer_start)
        gallery_identities = self.dataframe.identity.to_numpy()[gallery_rows]
        
        prepared_data = {
            "g_rows": gallery_rows,
            "q": query_info,
            "cand": ranked.cand,
            "raw": raw_scores,
            "g_ids": gallery_identities,
            "ms_per_query": elapsed_time_ms / max(len(query_info.rows), 1)
        }
        
        self._prepared_cache[cache_key] = prepared_data
        return prepared_data

    def fit_score_calibrators(self, branches: Optional[List[str]] = None) -> "Experiment":
        """Fits score calibrators for each feature branch using validation pairs."""
        branches = branches or sorted({b for b in self.features} | ({LOCAL} if self.local_scorer else set()))
        rng = np.random.default_rng(self.seed)
        
        for branch in branches:
            branch_cfg = (branch,) if branch != LOCAL else ("global", LOCAL) if "global" in self.features else (next(iter(self.features)), LOCAL)
            prepared_data = self._prepare_evaluation_splits("val", branch_cfg)
            
            matches = (prepared_data["g_ids"][prepared_data["cand"]] == prepared_data["q"].true[:, None]) & prepared_data["q"].known[:, None]
            query_indices = np.arange(len(matches))
            
            if len(query_indices) > self.max_calibration_queries:
                query_indices = rng.choice(query_indices, self.max_calibration_queries, replace=False)
                
            self.calibrators[branch] = ScoreCalibrator(self.calibration_method).fit(
                prepared_data["raw"][branch][query_indices], matches[query_indices]
            )
            
        return self

    def _apply_calibrations(self, prepared_data: dict, branches: Tuple[str, ...]) -> Dict[str, np.ndarray]:
        """Applies fitted score calibrators to raw branch scores."""
        return {branch: self.calibrators[branch].transform(prepared_data["raw"][branch]) for branch in branches}

    def run_configuration(
        self,
        branches: List[str],
        weights: Optional[dict] = None,
        objective: str = "NBA",
        per_species_tau: bool = False,
        test: bool = False
    ) -> dict:
        """Evaluates a single feature branch configuration, tuning weights and decision thresholds on validation."""
        branches = tuple(branches)
        if not self.calibrators:
            self.fit_score_calibrators()
            
        prepared_data = self._prepare_evaluation_splits("val", branches)
        calibrated_scores = self._apply_calibrations(prepared_data, branches)

        def objective_function(fused_scores: np.ndarray) -> float:
            ranked_res = Ranked(prepared_data["cand"], fused_scores).resort(fused_scores)
            scores, predictions = ranked_res.score[:, 0], prepared_data["g_ids"][ranked_res.cand[:, 0]]
            if objective == "BAKS":
                from src.retrieval.metrics import closed_set_baks
                return closed_set_baks(scores, predictions, prepared_data["q"].true, prepared_data["q"].datasets, prepared_data["q"].known)
            return best_threshold(scores, predictions, prepared_data["q"].true, prepared_data["q"].datasets, prepared_data["q"].known, n_thresholds=60)["NBA"]

        optimal_weights = normalise_weights(weights) if weights else tune_weights(calibrated_scores, objective_function, self.weight_step)[0]
        
        fused_validation = fuse(calibrated_scores, optimal_weights)
        ranked_val = Ranked(prepared_data["cand"], fused_validation)
        ranked_val = ranked_val.resort(ranked_val.score)
        
        best_thresholds = select_tau(ranked_val, prepared_data["g_ids"], prepared_data["q"], per_species=per_species_tau)
        
        row_results = {
            "config": get_configuration_name(branches),
            "weights": {key: round(val, 3) for key, val in optimal_weights.items()},
            "K": ranked_val.cand.shape[1],
            "tau": best_thresholds["tau"],
            "ms_per_query": round(prepared_data["ms_per_query"], 2)
        }
        
        row_results.update({
            f"val_{key}": val for key, val in evaluate_closed_set(ranked_val, prepared_data["g_ids"], prepared_data["q"]).items()
            if key != "per_dataset"
        })
        row_results.update({
            "val_open_BAKS": best_thresholds["BAKS"],
            "val_open_BAUS": best_thresholds["BAUS"],
            "val_open_NBA": best_thresholds["NBA"]
        })
        
        if test:
            test_data = self._prepare_evaluation_splits("test", branches)
            fused_test = fuse(self._apply_calibrations(test_data, branches), optimal_weights)
            ranked_test = Ranked(test_data["cand"], fused_test)
            ranked_test = ranked_test.resort(ranked_test.score)
            
            row_results.update({
                f"test_{key}": val for key, val in evaluate_closed_set(ranked_test, test_data["g_ids"], test_data["q"]).items()
                if key != "per_dataset"
            })
            open_results = evaluate_open_set(ranked_test, test_data["g_ids"], test_data["q"], best_thresholds["tau"], best_thresholds.get("species_tau"))
            row_results.update({
                "test_open_BAKS": open_results["BAKS"],
                "test_open_BAUS": open_results["BAUS"],
                "test_open_NBA": open_results["NBA"]
            })
            
        row_results["_species_tau"] = best_thresholds.get("species_tau")
        return row_results

    def run_ablation_study(self, configurations: List[List[str]], fixed_weights: Optional[dict] = None, test: bool = False, **kwargs) -> pd.DataFrame:
        """Executes ablation experiments across a list of branch configurations."""
        fixed_weights = fixed_weights or {}
        rows = [
            self.run_configuration(config, fixed_weights.get(tuple(config)), test=test, **kwargs)
            for config in configurations
        ]
        return pd.DataFrame(rows)


def get_default_ablation_configurations(
    global_name: str = "global",
    complementary_name: str = "complementary",
    local_name: str = LOCAL,
    baseline_name: str = "baseline",
    with_local: bool = True
) -> List[Tuple[str, ...]]:
    """Generates standard feature branch combinations for ablation studies."""
    g, c, l, b = global_name, complementary_name, local_name, baseline_name
    configurations = [(g,), (c,), (b,)]
    if with_local:
        configurations.append((l,))
    configurations.append((g, c))
    if with_local:
        configurations.extend([(g, l), (c, l), (g, c, l)])
    return configurations


def save_experiment_artifacts(experiment: Experiment, results_dataframe: pd.DataFrame, output_directory: str = "results", name: str = "ablation_val"):
    """Saves experiment evaluation tables, threshold lookup cards, and fitted calibrator pickles to disk."""
    import json
    import os
    import pickle
    
    os.makedirs(output_directory, exist_ok=True)
    table = results_dataframe.drop(columns=[col for col in results_dataframe.columns if col.startswith("_")])
    table.assign(weights=table.weights.astype(str)).to_csv(os.path.join(output_directory, f"{name}.csv"), index=False)
    
    threshold_card = {
        row["config"]: {
            "weights": row["weights"],
            "tau": float(row["tau"]),
            "species_tau": {str(k): float(v) for k, v in (row.get("_species_tau") or {}).items()}
        }
        for row in results_dataframe.to_dict("records")
    }
    
    json.dump(threshold_card, open(os.path.join(output_directory, "thresholds.json"), "w"), indent=2)
    pickle.dump(experiment.calibrators, open(os.path.join(output_directory, "calibrators.pkl"), "wb"))
    