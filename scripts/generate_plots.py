"""
Script to generate presentation-grade plots for the wildlife re-ID benchmark:
1. results/plot_baks_baus_vs_tau.png: Open-set decision curves (BAKS, BAUS, NBA vs threshold tau).
2. results/plot_pca_dimension_sweep.png: Dimension reduction tradeoff (32 to 512 dims vs uncompressed).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import src.config as C
from src.data_processing.splits import get_frames
from src.evaluation.ablation import Experiment
from src.features.extraction import load_features
from src.retrieval.similarity import LocalScorer
from src.data_processing.dataloader import convert_to_rgb_image
from src.retrieval.search import Ranked
from src.retrieval.metrics import compute_open_set_curve
from src.features.fusion import fuse


def generate_tau_curve_plot(output_dir: str = "results"):
    print("[1/2] Generating BAKS & BAUS vs tau plot...")
    df = get_frames(C.data)
    R = C.retrieval
    feats = load_features(
        R["feature_cache"],
        {"global": "megadescriptor", "complementary": "eva02", "baseline": "dinov2"},
    )
    cache_file = os.path.join(output_dir, "local_pairs_cache.pkl")
    local = LocalScorer(
        lambda r: convert_to_rgb_image(df.path.iloc[r], C.data["dataset_root"]),
        use_ransac=False,
        cache_file=cache_file,
    )
    ex = Experiment(df, feats, local, R["K"], 5, max_queries=200)
    ex.fit_score_calibrators()

    # Prepare fused tri-branch evaluation
    prep = ex._prepare_evaluation_splits("val", ("global", "complementary", "local"))
    cal = ex._apply_calibrations(prep, ("global", "complementary", "local"))
    weights = {"global": 0.3, "complementary": 0.6, "local": 0.1}
    fused_scores = fuse(cal, weights)

    cand_indices = prep.get("candidate_indices", prep.get("cand"))
    gallery_ids = prep.get("gallery_identities", prep.get("g_ids"))
    query_info = prep.get("query_info", prep.get("q"))

    ranked = Ranked(cand_indices, fused_scores).resort(fused_scores)
    top_scores = ranked.score[:, 0]
    top_preds = gallery_ids[ranked.cand[:, 0]]

    rows, auc_score = compute_open_set_curve(
        top_scores,
        top_preds,
        query_info.true,
        query_info.datasets,
        query_info.known,
        num_thresholds=200,
    )

    taus = np.array([r["tau"] for r in rows])
    baks = np.array([r["BAKS"] for r in rows])
    baus = np.array([r["BAUS"] for r in rows])
    nba = np.array([r["NBA"] for r in rows])

    # Filter out boundary sentinel values for cleaner visual presentation
    valid_mask = (taus >= 0.0) & (taus <= 1.0)
    taus, baks, baus, nba = (
        taus[valid_mask],
        baks[valid_mask],
        baus[valid_mask],
        nba[valid_mask],
    )

    best_idx = np.argmax(nba)
    best_tau = taus[best_idx]
    best_nba = nba[best_idx]
    best_baus = baus[best_idx]
    best_baks = baks[best_idx]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # Plot curves with distinct, high-contrast palette
    ax.plot(taus, baks, label="BAKS (Known Accuracy)", color="#1f77b4", linewidth=2.4)
    ax.plot(taus, baus, label="BAUS (Unknown Rejection)", color="#2ca02c", linewidth=2.4)
    ax.plot(taus, nba, label="NBA (Novelty-Balanced Accuracy)", color="#ff7f0e", linewidth=2.8, linestyle="-")

    # Mark optimal threshold
    ax.axvline(best_tau, color="#d62728", linestyle="--", linewidth=1.8, label=f"Optimal $\\tau^* = {best_tau:.3f}$")
    ax.scatter([best_tau], [best_nba], color="#d62728", s=80, zorder=5)

    # Annotate peak NBA
    ax.annotate(
        f"Peak NBA = {best_nba*100:.1f}%\nBAUS = {best_baus*100:.1f}%\nBAKS = {best_baks*100:.1f}%",
        xy=(best_tau, best_nba),
        xytext=(best_tau + 0.08, best_nba - 0.12),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2f2", edgecolor="#d62728", alpha=0.9),
    )

    ax.set_title("Open-Set Tradeoff: BAKS and BAUS vs Decision Threshold ($\\tau$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold ($\\tau$)", fontsize=11, labelpad=8)
    ax.set_ylabel("Metric Score", fontsize=11, labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plot_baks_baus_vs_tau.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot_path}")


def generate_pca_sweep_plot(output_dir: str = "results"):
    print("[2/2] Generating PCA dimension sweep plot...")
    csv_path = os.path.join(output_dir, "compression.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_comp = pd.read_csv(csv_path)

    # Standardize dim column
    dims = [str(d) for d in df_comp["dim"]]
    x_positions = np.arange(len(dims))

    val_baks = df_comp["val_BAKS"].to_numpy()
    val_nba = df_comp["val_open_NBA"].to_numpy()
    val_map = df_comp["val_mAP"].to_numpy()
    val_latency = df_comp["ms_per_query"].to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9.0, 5.4), dpi=300)

    # Primary axis: Accuracy metrics
    l1 = ax1.plot(x_positions, val_baks, marker="o", linewidth=2.4, color="#1f77b4", label="Closed BAKS")
    l2 = ax1.plot(x_positions, val_nba, marker="s", linewidth=2.4, color="#ff7f0e", label="Open NBA")
    l3 = ax1.plot(x_positions, val_map, marker="^", linewidth=2.2, color="#2ca02c", linestyle="--", label="mAP@100")

    ax1.set_xlabel("Feature Dimension (Whitened PCA vs Uncompressed)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Evaluation Metric Score", fontsize=11, labelpad=8)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{d}D" if d != "full" else "Uncompressed\n(Full)" for d in dims], fontsize=10)
    ax1.set_ylim(0.55, 0.90)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary axis: Search latency in ms/query
    ax2 = ax1.twinx()
    l4 = ax2.plot(x_positions, val_latency, marker="d", linewidth=2.0, color="#d62728", linestyle=":", label="Latency (ms/query)")
    ax2.set_ylabel("Search Latency (ms / query)", fontsize=11, color="#d62728", labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0.0, 1.8)
    ax2.grid(False)

    # Highlight 512D elbow point
    idx_512 = dims.index("512") if "512" in dims else -2
    ax1.annotate(
        f"Sweet Spot (512D):\nBAKS = {val_baks[idx_512]*100:.1f}%\nNBA = {val_nba[idx_512]*100:.1f}%\nLatency = {val_latency[idx_512]:.2f} ms",
        xy=(x_positions[idx_512], val_baks[idx_512]),
        xytext=(x_positions[idx_512] - 1.4, val_baks[idx_512] - 0.09),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.4, headwidth=5),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef6ff", edgecolor="#1f77b4", alpha=0.95),
    )

    # Combine legends
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, framealpha=0.9, fontsize=9.5)

    ax1.set_title("PCA Compression Sweep: Dimensionality vs Retrieval Accuracy & Latency", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plot_pca_dimension_sweep.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    generate_tau_curve_plot("results")
    generate_pca_sweep_plot("results")
