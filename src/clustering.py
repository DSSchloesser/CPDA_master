"""
ednet_cpda/src/clustering.py
─────────────────────────────
Latent behavioral cluster discovery.

Pipeline:
  1. Impute sparse features (median imputation; documented as limitation).
  2. Robust scale (RobustScaler) — preferred over StandardScaler for
     heavy-tailed RT distributions.
  3. UMAP dimensionality reduction to 2-D manifold embedding.
  4. HDBSCAN density clustering on the UMAP embedding.
  5. Fallback: if HDBSCAN is unavailable, KMeans k=4 is substituted
     (noted as limitation — assumes spherical clusters).

Cluster validation:
  - Silhouette score
  - Davies–Bouldin index
  - Calinski–Harabasz index
  - ARI against ground-truth archetypes (synthetic data only)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    warnings.warn("[clustering] umap-learn not available; PCA(n=2) used instead.")

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False
    warnings.warn("[clustering] hdbscan not available; KMeans(k=4) used as fallback.")

RANDOM_SEED = 42

# Features to use for clustering (exclude outcome proxies used for validation)
CLUSTER_FEATURES = [
    # kinematic
    "rt_log_mean", "rt_log_std", "rt_cv", "rt_skew", "rt_kurtosis",
    "rt_iqr", "rt_p10", "rt_p90",
    # accuracy/efficiency
    "accuracy", "efficiency_index", "sat_slope",
    "accuracy_drift",
    # complexity/dynamics
    "approx_entropy", "perm_entropy", "hurst_exponent",
    "rt_autocorr_lag1", "power_law_alpha",
    "proficiency_entropy", "proficiency_std",
    # temporal/longitudinal (session structure)
    "n_sessions", "interactions_per_session",
    "session_acc_std", "session_rt_cv",
    "sessions_per_day", "within_session_accel",
    "reattempt_fraction",
]

# Outcome features held out from clustering but used for post-hoc validation
OUTCOME_FEATURES = [
    "retention_delta",       # long-term retention proxy
    "proficiency_trend",     # direction of learning curve
    "session_acc_mean",      # short-term task completion proxy
]


def prepare_features(
    feat_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Tuple[np.ndarray, list, RobustScaler]:
    """
    Impute missing values and scale feature matrix.

    Returns
    -------
    X_scaled   : (n_users, n_features) float array
    cols_used  : list of column names actually used
    scaler     : fitted RobustScaler (for later inverse-transform)
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    # use only features present in the DataFrame
    cols_used = [c for c in feature_cols if c in feat_df.columns]
    missing   = set(feature_cols) - set(cols_used)
    if missing:
        warnings.warn(f"[clustering] Missing features (skipped): {missing}")

    X_raw = feat_df[cols_used].values.astype(float)

    # median imputation
    imputer  = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # robust scaling
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, cols_used, scaler


def embed_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> np.ndarray:
    """UMAP embedding with PCA fallback."""
    if _HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=RANDOM_SEED,
            verbose=False,
        )
        return reducer.fit_transform(X)
    else:
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        return pca.fit_transform(X)


def cluster_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 5,
) -> np.ndarray:
    """HDBSCAN clustering with KMeans fallback."""
    if _HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embedding)
        # if HDBSCAN collapses everything to noise, fall back
        n_real = (labels >= 0).sum()
        if n_real < 0.4 * len(labels):
            warnings.warn(
                "[clustering] HDBSCAN assigned >60% to noise. "
                "Falling back to KMeans(k=4). Consider tuning min_cluster_size."
            )
            km = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=20)
            labels = km.fit_predict(embedding)
    else:
        km = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=20)
        labels = km.fit_predict(embedding)

    return labels


def validate_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feat_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute internal and (optionally) external cluster validity indices."""
    valid_mask = labels >= 0
    X_v = X[valid_mask]
    l_v = labels[valid_mask]

    metrics: dict = {
        "n_clusters": int(labels.max() + 1),
        "noise_fraction": float((labels < 0).mean()),
        "n_valid": int(valid_mask.sum()),
    }

    if len(np.unique(l_v)) >= 2:
        metrics["silhouette"]         = float(silhouette_score(X_v, l_v, sample_size=min(500, len(X_v))))
        metrics["davies_bouldin"]     = float(davies_bouldin_score(X_v, l_v))
        metrics["calinski_harabasz"]  = float(calinski_harabasz_score(X_v, l_v))
    else:
        metrics["silhouette"] = metrics["davies_bouldin"] = metrics["calinski_harabasz"] = np.nan

    # external validation against ground-truth archetypes
    if feat_df is not None and "true_archetype" in feat_df.columns:
        true_labels = feat_df["true_archetype"].values[valid_mask]
        metrics["ari_vs_ground_truth"] = float(adjusted_rand_score(true_labels, l_v))

    return metrics


def run_clustering_pipeline(
    feat_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    End-to-end clustering pipeline.

    Returns
    -------
    feat_df_out : copy of feat_df with columns added:
                    cluster_label, umap_x, umap_y
    embedding   : (n_users, 2) UMAP coordinates
    metrics     : validation metrics dict
    """
    print("[clustering] Preparing features …")
    X_scaled, cols_used, scaler = prepare_features(feat_df)

    print("[clustering] Computing UMAP embedding …")
    embedding = embed_umap(X_scaled)

    print("[clustering] Running HDBSCAN …")
    labels = cluster_hdbscan(embedding)

    print("[clustering] Validating clusters …")
    metrics = validate_clusters(X_scaled, labels, feat_df)

    feat_df_out = feat_df.copy()
    feat_df_out["cluster_label"] = labels
    feat_df_out["umap_x"]        = embedding[:, 0]
    feat_df_out["umap_y"]        = embedding[:, 1]

    print(f"\n[clustering] Results:")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    return feat_df_out, embedding, metrics
