"""
ednet_cpda/src/outcomes.py
──────────────────────────
Outcome operationalization and cluster-level statistical testing.

Research question:
  Does "efficient" session behavior (cluster membership) actually
  correlate with durable knowledge gain (long-term retention)?

Two outcome dimensions:
  1. Short-term task completion
       Operationalized as within-session accuracy on the LATTER half
       of interaction bundles (session_acc_mean).  This proxies the
       ability to navigate problem bundles immediately following
       content engagement.

  2. Long-term retention
       Operationalized as retention_delta: accuracy in the student's
       last 3 sessions minus accuracy in their first 3 sessions,
       measured across a multi-day temporal span. Positive values
       indicate durable knowledge gain; negative values indicate
       decay or fatigue.

Statistical strategy:
  - Kruskal–Wallis H-test (non-parametric ANOVA) for overall cluster
    differences; RT and accuracy distributions are skewed.
  - Post-hoc Dunn test with Holm–Bonferroni correction.
  - Effect size: η² (eta-squared) for each outcome.
  - Spearman rank correlation between cluster-level efficiency index
    and each outcome variable.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, spearmanr

# optional: statsmodels for Dunn test
try:
    from scikit_posthocs import posthoc_dunn
    _HAS_POSTHOCS = True
except ImportError:
    _HAS_POSTHOCS = False


def _eta_squared_kw(H: float, n: int, k: int) -> float:
    """η² from Kruskal–Wallis H statistic."""
    return (H - k + 1) / (n - k)


def cluster_outcome_summary(
    feat_df: pd.DataFrame,
    outcome_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per-cluster descriptive statistics for outcome variables.
    Returns a MultiIndex DataFrame: cluster × statistic.
    """
    if outcome_cols is None:
        outcome_cols = ["session_acc_mean", "retention_delta",
                        "proficiency_trend", "efficiency_index",
                        "accuracy", "span_days"]

    available = [c for c in outcome_cols if c in feat_df.columns]
    valid_df  = feat_df[feat_df["cluster_label"] >= 0].copy()

    summary = (
        valid_df
        .groupby("cluster_label")[available]
        .agg(["mean", "median", "std", "count"])
    )
    return summary


def kruskal_wallis_tests(
    feat_df: pd.DataFrame,
    outcome_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run Kruskal–Wallis H-test for each outcome variable across clusters.
    Returns a results DataFrame with H, p-value, and η².
    """
    if outcome_cols is None:
        outcome_cols = ["session_acc_mean", "retention_delta",
                        "proficiency_trend", "efficiency_index"]

    valid_df = feat_df[feat_df["cluster_label"] >= 0].dropna(subset=["cluster_label"])
    clusters = sorted(valid_df["cluster_label"].unique())
    results  = []

    for col in outcome_cols:
        if col not in valid_df.columns:
            continue
        groups = [
            valid_df.loc[valid_df["cluster_label"] == c, col].dropna().values
            for c in clusters
            if valid_df.loc[valid_df["cluster_label"] == c, col].dropna().shape[0] >= 5
        ]
        if len(groups) < 2:
            continue
        try:
            H, p = kruskal(*groups)
            n    = sum(len(g) for g in groups)
            eta2 = _eta_squared_kw(H, n, len(groups))
            results.append({
                "outcome":  col,
                "H_stat":   round(H, 3),
                "p_value":  round(p, 6),
                "eta2":     round(eta2, 4),
                "n":        n,
            })
        except Exception as e:
            results.append({"outcome": col, "H_stat": np.nan, "p_value": np.nan,
                            "eta2": np.nan, "n": 0})

    return pd.DataFrame(results)


def pairwise_mannwhitney(
    feat_df: pd.DataFrame,
    outcome_col: str,
    bonferroni: bool = True,
) -> pd.DataFrame:
    """
    Pairwise Mann–Whitney U tests between all cluster pairs for a given outcome.
    """
    valid_df = feat_df[feat_df["cluster_label"] >= 0].dropna(subset=[outcome_col])
    clusters = sorted(valid_df["cluster_label"].unique())
    pairs    = list(combinations(clusters, 2))
    n_tests  = len(pairs)

    rows = []
    for (c1, c2) in pairs:
        g1 = valid_df.loc[valid_df["cluster_label"] == c1, outcome_col].dropna().values
        g2 = valid_df.loc[valid_df["cluster_label"] == c2, outcome_col].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue
        U, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        r_effect = 1 - 2*U / (len(g1)*len(g2))   # rank-biserial correlation
        if bonferroni:
            p = min(1.0, p * n_tests)
        rows.append({
            "cluster_A": c1,
            "cluster_B": c2,
            "U_stat":    round(U, 1),
            "p_adj":     round(p, 6),
            "r_effect":  round(r_effect, 4),
            "mean_A":    round(g1.mean(), 4),
            "mean_B":    round(g2.mean(), 4),
        })

    return pd.DataFrame(rows)


def efficiency_retention_correlation(
    feat_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Spearman rank correlation between efficiency_index and
    both outcome dimensions across the full population.
    This is the core test of the research question:
    does session efficiency predict durable knowledge gain?
    """
    valid_df = feat_df[feat_df["cluster_label"] >= 0]

    results = {}
    for outcome in ["session_acc_mean", "retention_delta"]:
        if outcome not in valid_df.columns:
            continue
        sub = valid_df[["efficiency_index", outcome]].dropna()
        if len(sub) < 10:
            continue
        rho, p = spearmanr(sub["efficiency_index"], sub[outcome])
        results[f"spearman_{outcome}_rho"] = round(float(rho), 4)
        results[f"spearman_{outcome}_p"]   = round(float(p), 6)

    # efficiency vs retention_delta by cluster (within-cluster correlations)
    for c in sorted(valid_df["cluster_label"].unique()):
        sub = valid_df[valid_df["cluster_label"] == c][
            ["efficiency_index", "retention_delta"]
        ].dropna()
        if len(sub) < 8:
            continue
        rho, p = spearmanr(sub["efficiency_index"], sub["retention_delta"])
        results[f"cluster{c}_eff_ret_rho"] = round(float(rho), 4)
        results[f"cluster{c}_eff_ret_p"]   = round(float(p), 6)

    return results


def label_clusters(
    feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign interpretive labels to clusters based on centroid profiles.
    Labels map onto the Active Seeker / Passive Consumer typology.
    """
    valid_df = feat_df[feat_df["cluster_label"] >= 0]

    profile_cols = ["accuracy", "rt_log_mean", "approx_entropy",
                    "hurst_exponent", "sessions_per_day", "retention_delta",
                    "efficiency_index", "reattempt_fraction"]
    available = [c for c in profile_cols if c in valid_df.columns]

    centroids = valid_df.groupby("cluster_label")[available].median()

    # Rank clusters on three dimensions
    if "efficiency_index" in centroids and "retention_delta" in centroids:
        centroids["eff_rank"] = centroids["efficiency_index"].rank()
        centroids["ret_rank"] = centroids.get("retention_delta", pd.Series(0, index=centroids.index)).rank()
    if "sessions_per_day" in centroids:
        centroids["engagement_rank"] = centroids["sessions_per_day"].rank()

    label_map = {}
    archetype_names = {
        0: "Strategic Paced Learner",
        1: "Speed-Optimized Rusher",
        2: "Disengaged Drifter",
        3: "Anxious Reworker",
    }

    # Auto-assign interpretive labels by cross-referencing centroid ranks
    # (works for both synthetic and real data)
    if "eff_rank" in centroids.columns and "engagement_rank" in centroids.columns:
        eff_ranks = centroids["eff_rank"]
        eng_ranks = centroids["engagement_rank"]

        for c in centroids.index:
            eff = eff_ranks[c]
            eng = eng_ranks[c]
            n   = len(centroids)
            if eff >= 0.75 * n and eng >= 0.5 * n:
                label_map[c] = "Strategic Paced Learner (Active Seeker)"
            elif eff >= 0.5 * n and eng < 0.5 * n:
                label_map[c] = "Speed-Optimized Rusher"
            elif eff < 0.5 * n and eng < 0.25 * n:
                label_map[c] = "Disengaged Drifter (Passive Consumer)"
            else:
                label_map[c] = "Anxious Reworker"
    else:
        for c in centroids.index:
            label_map[c] = f"Cluster {c}"

    feat_df = feat_df.copy()
    feat_df["cluster_name"] = feat_df["cluster_label"].map(label_map)
    feat_df["cluster_name"] = feat_df["cluster_name"].fillna("Noise / Unassigned")
    return feat_df
