"""
ednet_cpda/run_analysis.py
───────────────────────────
Master orchestrator for the CPDA behavioral latent variable analysis.

Usage:
    python run_analysis.py                     # synthetic data (no download needed)
    python run_analysis.py --data-dir data/    # real EdNet-KT4 CSVs
    python run_analysis.py --n-users 500       # synthetic with custom population

Outputs (written to outputs/):
    feature_matrix.csv       — per-user behavioral features
    cluster_results.csv      — features + cluster labels + UMAP coords
    kruskal_wallis.csv        — cluster × outcome KW test results
    pairwise_tests.csv        — post-hoc pairwise comparisons
    efficiency_correlations.txt
    validation_metrics.json
    fig1_umap_clusters.png
    fig2_outcome_distributions.png
    fig3_efficiency_vs_retention.png
    fig4_cluster_profiles.png
    fig5_rt_distributions.png
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# ── sys.path for src imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import get_data, generate_synthetic_ednet
from src.features    import build_feature_matrix
from src.clustering  import run_clustering_pipeline
from src.outcomes    import (
    cluster_outcome_summary,
    kruskal_wallis_tests,
    pairwise_mannwhitney,
    efficiency_retention_correlation,
    label_clusters,
)
from src.visualization import save_all_figures


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CPDA EdNet behavioral cluster analysis")
    p.add_argument("--data-dir",  type=str, default="data",
                   help="Directory containing EdNet CSVs (default: data/)")
    p.add_argument("--n-users",   type=int, default=800,
                   help="Users for synthetic generation (default: 800)")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Output directory (default: outputs/)")
    p.add_argument("--min-interactions", type=int, default=40,
                   help="Minimum interactions per user (default: 40)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CPDA | Behavioral Latent Variable Analysis — EdNet-KT4")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    df = get_data(args.data_dir)
    print(f"      {len(df):,} interactions | {df['user_id'].nunique():,} users")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/6] Engineering behavioral features …")
    feat_df = build_feature_matrix(df, min_interactions=args.min_interactions)
    feat_df.to_csv(output_dir / "feature_matrix.csv")
    print(f"      Feature matrix saved → {output_dir}/feature_matrix.csv")

    # ── 3. Clustering ─────────────────────────────────────────────────────────
    print("\n[3/6] Running clustering pipeline …")
    feat_df, embedding, cluster_metrics = run_clustering_pipeline(feat_df)

    # ── 4. Cluster labeling ───────────────────────────────────────────────────
    print("\n[4/6] Labeling clusters …")
    feat_df = label_clusters(feat_df)

    cluster_results_path = output_dir / "cluster_results.csv"
    feat_df.to_csv(cluster_results_path)
    print(f"      Cluster results saved → {cluster_results_path}")

    # ── 5. Outcome analysis ───────────────────────────────────────────────────
    print("\n[5/6] Running outcome analyses …")

    summary = cluster_outcome_summary(feat_df)
    print("\n  Cluster × Outcome Summary:")
    print(summary.to_string())
    summary.to_csv(output_dir / "outcome_summary.csv")

    kw = kruskal_wallis_tests(feat_df)
    print("\n  Kruskal–Wallis Tests:")
    print(kw.to_string(index=False))
    kw.to_csv(output_dir / "kruskal_wallis.csv", index=False)

    pw_stc = pairwise_mannwhitney(feat_df, "session_acc_mean")
    pw_ret = pairwise_mannwhitney(feat_df, "retention_delta")
    pd_all = pd.concat([
        pw_stc.assign(outcome="session_acc_mean"),
        pw_ret.assign(outcome="retention_delta"),
    ], ignore_index=True)
    pd_all.to_csv(output_dir / "pairwise_tests.csv", index=False)
    print(f"\n  Pairwise tests saved → {output_dir}/pairwise_tests.csv")

    corrs = efficiency_retention_correlation(feat_df)
    print("\n  Efficiency × Retention Correlations:")
    for k, v in corrs.items():
        print(f"    {k}: {v}")
    with open(output_dir / "efficiency_correlations.json", "w") as f:
        json.dump(corrs, f, indent=2)

    # save cluster metrics
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump({k: (float(v) if hasattr(v, "item") else v)
                   for k, v in cluster_metrics.items()}, f, indent=2)

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    print("\n[6/6] Generating figures …")
    figs = save_all_figures(feat_df, output_dir)

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print(f"  All outputs written to: {output_dir.resolve()}")
    print("=" * 65)

    # ── brief interpretive summary ────────────────────────────────────────────
    _print_interpretive_summary(feat_df, kw, corrs, cluster_metrics)

    return feat_df, cluster_metrics


import pandas as pd

def _print_interpretive_summary(feat_df, kw, corrs, metrics):
    print("\n" + "─" * 65)
    print("  INTERPRETIVE SUMMARY")
    print("─" * 65)

    valid = feat_df[feat_df["cluster_label"] >= 0]
    cluster_counts = valid["cluster_label"].value_counts().sort_index()
    for c, n in cluster_counts.items():
        name = valid[valid["cluster_label"]==c]["cluster_name"].iloc[0] \
               if "cluster_name" in valid.columns else f"Cluster {c}"
        pct = 100 * n / len(valid)
        print(f"  [{c}] {name:<42} n={n:4d} ({pct:.1f}%)")

    # short-term vs long-term finding
    for _, row in kw.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else
              ("*" if row["p_value"] < 0.05 else "ns"))
        print(f"\n  KW {row['outcome']:25s}  H={row['H_stat']:.2f}  "
              f"p={row['p_value']:.4f} {sig}  η²={row['eta2']:.3f}")

    # core research question result
    rho_ret = corrs.get("spearman_retention_delta_rho", None)
    rho_stc = corrs.get("spearman_session_acc_mean_rho", None)
    if rho_ret is not None and rho_stc is not None:
        print(f"\n  Efficiency → Short-term completion  ρ = {rho_stc:+.3f}")
        print(f"  Efficiency → Long-term retention     ρ = {rho_ret:+.3f}")
        if abs(rho_ret) < 0.15 and abs(rho_stc) > 0.25:
            print("\n  → Finding: Session efficiency predicts short-term task completion")
            print("    but shows weak association with long-term retention.")
            print("    'Efficient' behavior (fast + accurate) may not equal durable learning.")
        elif rho_ret > 0.25:
            print("\n  → Finding: Efficiency positively associated with both outcomes.")
        else:
            print("\n  → Mixed or context-dependent association detected.")

    if "ari_vs_ground_truth" in metrics:
        print(f"\n  ARI vs ground-truth archetypes: {metrics['ari_vs_ground_truth']:.3f}")
        print("  (Synthetic data only — confirms cluster recovery quality)")
    print("─" * 65)


if __name__ == "__main__":
    feat_df, metrics = main()
