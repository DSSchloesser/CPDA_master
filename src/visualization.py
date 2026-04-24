"""
ednet_cpda/src/visualization.py
────────────────────────────────
Publication-quality figures for the CPDA report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    -1: "#aaaaaa",   # noise
    0:  "#2E86AB",   # Strategic Pacer
    1:  "#E84855",   # Speed Rusher
    2:  "#F9A03F",   # Passive Drifter
    3:  "#3BB273",   # Anxious Reworker
    4:  "#7B2D8B",
    5:  "#C97D4E",
}

CLUSTER_LABELS = {
    -1: "Noise",
    0:  "Strategic Pacer",
    1:  "Speed Rusher",
    2:  "Passive Drifter",
    3:  "Anxious Reworker",
}

FIG_DPI = 150
FONT_SIZE = 10

plt.rcParams.update({
    "figure.dpi":          FIG_DPI,
    "font.size":           FONT_SIZE,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "axes.labelsize":      10,
    "axes.titlesize":      11,
    "legend.fontsize":     9,
    "font.family":         "DejaVu Sans",
})


def _cluster_color(c: int) -> str:
    return PALETTE.get(int(c), "#555555")


def _cluster_label(c: int, feat_df: pd.DataFrame) -> str:
    if "cluster_name" in feat_df.columns:
        name = feat_df[feat_df["cluster_label"] == c]["cluster_name"].iloc[0]
        return f"C{c}: {name}" if c >= 0 else name
    return CLUSTER_LABELS.get(int(c), f"Cluster {c}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_umap_clusters(
    feat_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    clusters = sorted(feat_df["cluster_label"].unique())
    for c in clusters:
        sub = feat_df[feat_df["cluster_label"] == c]
        label = _cluster_label(c, feat_df)
        ax.scatter(
            sub["umap_x"], sub["umap_y"],
            c=_cluster_color(c),
            alpha=0.55 if c >= 0 else 0.15,
            s=18 if c >= 0 else 8,
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("Behavioral Latent Space: UMAP Projection with HDBSCAN Clusters")
    ax.legend(loc="best", markerscale=1.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_outcome_distributions(
    feat_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    outcomes = ["session_acc_mean", "retention_delta",
                "efficiency_index", "proficiency_trend"]
    outcomes = [o for o in outcomes if o in feat_df.columns]

    valid_df = feat_df[feat_df["cluster_label"] >= 0]
    clusters = sorted(valid_df["cluster_label"].unique())

    fig, axes = plt.subplots(1, len(outcomes), figsize=(4*len(outcomes), 4))
    if len(outcomes) == 1:
        axes = [axes]

    titles = {
        "session_acc_mean":  "Short-Term Task Completion\n(Session Accuracy)",
        "retention_delta":   "Long-Term Retention\n(Late − Early Accuracy)",
        "efficiency_index":  "Efficiency Index\n(Accuracy / log RT)",
        "proficiency_trend": "Proficiency Trend\n(Learning Curve Slope)",
    }

    for ax, col in zip(axes, outcomes):
        for c in clusters:
            vals = valid_df[valid_df["cluster_label"] == c][col].dropna().values
            if len(vals) < 5:
                continue
            label = _cluster_label(c, feat_df)
            ax.hist(
                vals, bins=25, alpha=0.5,
                color=_cluster_color(c), label=label,
                density=True, edgecolor="none",
            )
        ax.set_title(titles.get(col, col), fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if ax is axes[0] else "")

    handles = [
        mpatches.Patch(color=_cluster_color(c), label=_cluster_label(c, feat_df))
        for c in clusters
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(clusters),
               bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Outcome Distributions by Behavioral Cluster", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_efficiency_vs_retention(
    feat_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter: Efficiency Index (x) vs Retention Delta (y).
    Core visualisation of the research question.
    """
    valid_df = feat_df[
        (feat_df["cluster_label"] >= 0) &
        feat_df["efficiency_index"].notna() &
        feat_df["retention_delta"].notna()
    ]
    clusters = sorted(valid_df["cluster_label"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for c in clusters:
        sub = valid_df[valid_df["cluster_label"] == c]
        ax.scatter(
            sub["efficiency_index"], sub["retention_delta"],
            c=_cluster_color(c), alpha=0.55, s=22,
            label=_cluster_label(c, feat_df), edgecolors="none",
        )

    # overall trend line
    x = valid_df["efficiency_index"].values
    y = valid_df["retention_delta"].values
    if len(x) >= 10:
        coefs = np.polyfit(x, y, 1)
        xp = np.linspace(x.min(), x.max(), 100)
        ax.plot(xp, np.polyval(coefs, xp), "k--", lw=1.5,
                label=f"Linear fit (β={coefs[0]:.3f})", alpha=0.7)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Efficiency Index  (Accuracy / log RT)")
    ax.set_ylabel("Retention Delta  (Late Accuracy − Early Accuracy)")
    ax.set_title("Does Session Efficiency Predict Durable Knowledge Gain?")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_profiles(
    feat_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Radar / heatmap of cluster centroid profiles across key features."""
    profile_features = [
        "accuracy", "rt_log_mean", "rt_cv",
        "approx_entropy", "hurst_exponent",
        "sessions_per_day", "reattempt_fraction",
        "efficiency_index",
    ]
    available = [f for f in profile_features if f in feat_df.columns]
    valid_df  = feat_df[feat_df["cluster_label"] >= 0]

    centroids = valid_df.groupby("cluster_label")[available].median()
    # min-max normalise for radar comparability
    normed = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-9)

    clusters = sorted(centroids.index)
    n_feat   = len(available)
    angles   = np.linspace(0, 2*np.pi, n_feat, endpoint=False).tolist()
    angles  += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

    feat_labels = [
        f.replace("_", " ").replace("log mean", "log(RT)").title()
        for f in available
    ]

    for c in clusters:
        vals = normed.loc[c].values.tolist()
        vals += vals[:1]
        label = _cluster_label(c, feat_df)
        ax.plot(angles, vals, color=_cluster_color(c), linewidth=2, label=label)
        ax.fill(angles, vals, color=_cluster_color(c), alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.set_title("Cluster Behavioral Profiles (Normalized)", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_rt_distributions(
    feat_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Log-RT distribution by cluster: core kinematic signature."""
    valid_df = feat_df[feat_df["cluster_label"] >= 0]
    clusters = sorted(valid_df["cluster_label"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))

    for c in clusters:
        sub = valid_df[valid_df["cluster_label"] == c]
        vals = sub["rt_log_mean"].dropna().values
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.4)
        xr  = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 200)
        ax.plot(xr, kde(xr), color=_cluster_color(c), lw=2,
                label=_cluster_label(c, feat_df))
        ax.fill_between(xr, kde(xr), alpha=0.15, color=_cluster_color(c))

    ax.set_xlabel("Mean log(Response Time) per User")
    ax.set_ylabel("Density")
    ax.set_title("Response Time Distribution by Cluster")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def save_all_figures(feat_df: pd.DataFrame, output_dir: Path) -> dict:
    """Render and save all report figures; return dict of file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    fns = {
        "umap":         (plot_umap_clusters,           "fig1_umap_clusters.png"),
        "outcomes":     (plot_outcome_distributions,    "fig2_outcome_distributions.png"),
        "eff_ret":      (plot_efficiency_vs_retention,  "fig3_efficiency_vs_retention.png"),
        "profiles":     (plot_cluster_profiles,         "fig4_cluster_profiles.png"),
        "rt_dist":      (plot_rt_distributions,         "fig5_rt_distributions.png"),
    }
    for key, (fn, fname) in fns.items():
        path = output_dir / fname
        try:
            fig = fn(feat_df, save_path=path)
            plt.close(fig)
            figures[key] = path
            print(f"[viz] Saved {fname}")
        except Exception as e:
            print(f"[viz] WARNING: Could not generate {fname}: {e}")

    return figures
