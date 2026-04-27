# CPDA — Behavioral Latent Variable Analysis

## Output Report

---

> **Core Finding:** Session efficiency (accuracy / log RT) strongly predicts short-term task completion (Spearman ρ = +0.8324, p ≈ 0) but is only **very weakly** associated with long-term retention (ρ = +0.0393, p ≈ 0). Clusters significantly differentiated STC (KW H = 22.539, p = 1.3e-05) but **not** LTR (H = 1.449, p = 0.2287). Three clusters were recovered, with 53 users (0.1068%) labeled as noise by HDBSCAN.

---

## Report Metadata


| Field                | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| **Analyst**          | `[ Name ]`                                                         |
| **Institution**      | Pearson TalentLens                                                 |
| **Dataset**          | EdNet-KT4 (Riiid, 2020) — 297,915 users | 131,441,538 interactions |
| **Pipeline Version** | ednet_cpda v1.0                                                    |
| **Run Date**         | `[ YYYY-MM-DD ]`                                                   |
| **Data Directory**   | `[ data/KT4/ ]`                                                    |


---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Run Configuration](#2-run-configuration)
3. [Feature Matrix Summary](#3-feature-matrix-summary)
4. [Clustering Results](#4-clustering-results)
5. [Cluster Profiles](#5-cluster-profiles)
6. [Outcome Analysis](#6-outcome-analysis)
7. [Limitations](#7-limitations)
8. [Pedagogical Implications](#8-pedagogical-implications)
9. [Analyst Notes & Annotations](#10-analyst-notes--annotations)

---

## 1. Executive Summary

This report documents the outputs of the CPDA Behavioral Latent Variable Analysis pipeline applied to the EdNet-KT4 dataset. The analysis recovered **three** latent behavioral clusters (plus a small noise fraction) from interaction-sequence telemetry and evaluated their differential predictive validity across two learning outcome dimensions: short-term task completion (STC) and long-term retention (LTR).

### 1.1 Research Question

> Do latent behavioral clusters recoverable from interaction-sequence telemetry differentially predict short-term task completion versus long-term knowledge retention? Does "efficient" session behavior correlate with durable learning?

### 1.2 Key Outcomes


| Dimension                   | Short-Term Task Completion (STC)        | Long-Term Retention (LTR)                                          |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| **Operationalization**      | Mean accuracy — latter half of sessions | Late session accuracy − early session accuracy (`retention_delta`) |
| **KW H-statistic**          | 22.539 (p = 1.3e-05) η² = 0.0004        | 1.449 (p = 0.2287) η² = 0.0000                                     |
| **Efficiency ρ (Spearman)** | ρ = +0.8324 (p ≈ 0)                     | ρ = +0.0393 (p ≈ 0)                                                |
| **Conclusion**              | Clusters significantly differentiated   | Clusters **NOT** differentiated                                    |


**Note on p-values:** values reported as `0.0` in outputs indicate numerical underflow (i.e., p is extremely small, not exactly zero).

---

## 2. Run Configuration

### 2.1 Dataset Parameters


| Parameter                                  | Value                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Dataset**                                | EdNet-KT4 (Riiid, 2020)                                                            |
| **Total interactions loaded**              | 19,453,330 *(sum of `n_interactions` over retained users in `feature_matrix.csv`)* |
| **Total users loaded**                     | 49,645 *(retained users in `feature_matrix.csv`)*                                  |
| **Data directory**                         | `[ Insert path ]`                                                                  |
| **Schema detected**                        | `lt_header` (action-trace format)                                                  |
| **questions.csv merged**                   | Yes — correctness resolved via `user_answer == correct_answer`                     |
| **Min. interactions threshold**            | 40                                                                                 |
| **Users retained after threshold**         | 49,645                                                                             |
| **Users assigned to clusters (non-noise)** | 49,592                                                                             |
| **Noise / unassigned (HDBSCAN)**           | 53 (0.1068%)                                                                       |
| **Pipeline version**                       | ednet_cpda v1.0                                                                    |
| **Run date / time**                        | `[ 2026-04-24 ]`                                                                   |
| **Python version**                         | `[ e.g. 3.12.x ]`                                                                  |


### 2.2 CLI Command Used

```bash
python run_analysis.py --data-dir data/KT4/ --min-interactions 40 --output-dir outputs/
```

### 2.3 Optional Package Status


| Package      | Role                               | Status                                                                                   |
| ------------ | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| `umap-learn` | Manifold embedding (2D projection) | Unknown *(not recorded in outputs; 2D embedding columns `umap_x`, `umap_y` are present)* |
| `hdbscan`    | Density clustering                 | **Installed** *(noise points labeled `cluster_label=-1` exist; n=53)*                    |
| `antropy`    | ApEn / permutation entropy         | Unknown *(entropy features present; backend not recorded)*                               |
| `nolds`      | Hurst exponent via DFA             | Unknown *(hurst exponent feature present; backend not recorded)*                         |


---

## 3. Feature Matrix Summary

42 behavioral features were extracted per user across four measurement layers. Outcome variables (`retention_delta`, `session_acc_mean`, `proficiency_trend`) were withheld from clustering inputs.


| Layer                           | N Features | Key Features                                                                   | Theoretical Basis                             |
| ------------------------------- | ---------- | ------------------------------------------------------------------------------ | --------------------------------------------- |
| **1 — Kinematic / Temporal**    | 9          | `rt_log_mean`, `rt_cv`, `rt_skew`, `power_law_alpha`                           | Response time modeling (van der Linden, 2006) |
| **2 — Accuracy / Efficiency**   | 5          | `accuracy`, `efficiency_index`, `sat_slope`, `accuracy_drift`                  | Speed-accuracy tradeoff (Heitz, 2014)         |
| **3 — Complexity / Dynamics**   | 6          | `approx_entropy`, `hurst_exponent`, `perm_entropy`, `rt_autocorr_lag1`         | Complexity science; ecological psychology     |
| **4 — Temporal / Longitudinal** | 8          | `n_sessions`, `sessions_per_day`, `reattempt_fraction`, `within_session_accel` | Spacing effect; forgetting curve              |


### 3.1 Feature Matrix File

Output: `outputs/feature_matrix.csv`

**Missingness notes (from `feature_matrix.csv`):**


| Feature                                 | % Missing | Reason                                                             | Treatment                                                     |
| --------------------------------------- | --------- | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| `retention_delta` *(outcome; withheld)* | 37.06%    | Insufficient longitudinal signal (cannot compute early→late delta) | Excluded from clustering; outcome analyses use available-case |
| `hurst_exponent`                        | 10.99%    | Users near the interaction threshold yield unstable estimates      | Median imputed                                                |
| `session_acc_std`                       | 4.19%     | Single-session users lack within-user session variability          | Median imputed                                                |
| `session_rt_cv`                         | 4.19%     | Single-session users lack within-user session variability          | Median imputed                                                |


---

## 4. Clustering Results

### 4.1 Pipeline Summary


| Stage                | Method Applied            | Notes                                                           |
| -------------------- | ------------------------- | --------------------------------------------------------------- |
| **Imputation**       | Median imputation         | Applied to features with > 0% missingness                       |
| **Scaling**          | RobustScaler (IQR)        | Preferred over StandardScaler for heavy-tailed RT distributions |
| **Dim. Reduction**   | `[ UMAP / PCA fallback ]` | n_neighbors=15, Euclidean metric, random_state=42               |
| **Clustering**       | **HDBSCAN**               | Noise fraction = 0.1068% (53 users)                             |
| **Outcome hold-out** | 3 features withheld       | `retention_delta`, `session_acc_mean`, `proficiency_trend`      |


### 4.2 Cluster Validity Indices


| Index                | Benchmark¹ | **Your Value**            | Interpretation                                  |
| -------------------- | ---------- | ------------------------- | ----------------------------------------------- |
| Silhouette Score     | 0.292      | **-0.0128**               | > 0.25 = interpretable cluster structure        |
| Davies-Bouldin Index | 1.811      | **2.0193**                | < 2.0 = acceptable compactness; lower is better |
| Calinski-Harabasz    | 1720.8     | **723.3730**              | Higher is better                                |
| ARI vs. Ground Truth | 0.267      | **N/A (real data)**       | 0 = random; 1 = perfect recovery                |
| Noise Fraction       | 0.0%       | **0.1068% (53 / 49,645)** | % users unassigned by HDBSCAN                   |
| N Clusters           | 4          | **3**                     | HDBSCAN-determined                              |


*¹ Benchmark from synthetic validation run (n=800, RANDOM_SEED=42)*

### 4.3 Figures

- `fig1_umap_clusters.png` — UMAP 2D projection with cluster coloring
- `fig2_outcome_distributions.png` — Outcome histograms by cluster
- `fig3_efficiency_vs_retention.png` — Efficiency vs. retention scatter
- `fig4_cluster_profiles.png` — Radar chart of cluster centroids
- `fig5_rt_distributions.png` — Log-RT KDE by cluster

---

## 5. Cluster Profiles

Three clusters were recovered (plus a small noise fraction). Values below are **medians** from `outputs/cluster_results.csv` (noise label `-1` excluded from the table).


| C#  | Label                            | N      | Accuracy | log(RT) | Efficiency | Sessions | STC    | LTR Δ   |
| --- | -------------------------------- | ------ | -------- | ------- | ---------- | -------- | ------ | ------- |
| C0  | Anxious Reworker                 | 1,288  | 0.5669   | 10.0807 | 0.05637    | 1.0      | 0.5667 | N/A     |
| C1  | Anxious Reworker (variant; rare) | 29     | 0.3208   | 9.2010  | 0.04433    | 4.0      | 0.5014 | -0.0583 |
| C2  | Speed-Optimized Rusher           | 48,275 | 0.5830   | 10.2910 | 0.05668    | 8.0      | 0.5792 | +0.0181 |


*Retention is not available for C0 (retention_n = 0). Among cluster-assigned users (N = 49,592), `retention_delta` is available for 31,229 users (62.97%). (An additional 17 noise-labeled users have `retention_delta`, but are excluded from cluster-based outcome tests.)*

---

## 6. Outcome Analysis

### 6.1 Kruskal-Wallis Results

All tests use non-parametric Kruskal-Wallis H. RT and accuracy distributions violate normality (Shapiro-Wilk p < 0.001). Effect size: η² = (H − k + 1) / (N − k).


| Outcome Variable         | H Statistic | p-value  | η²     | N      | Significant? |
| ------------------------ | ----------- | -------- | ------ | ------ | ------------ |
| `session_acc_mean` (STC) | 22.539      | 1.3e-05  | 0.0004 | 49,592 | **Yes**      |
| `retention_delta` (LTR)  | 1.449       | 0.228722 | 0.0000 | 31,229 | No (ns)      |
| `proficiency_trend`      | 109.599     | ≈ 0      | 0.0022 | 49,592 | **Yes**      |
| `efficiency_index`       | 20.477      | 3.6e-05  | 0.0004 | 49,592 | **Yes**      |


*(Populated from `outputs/kruskal_wallis.csv`; p-values shown as “≈ 0” indicate numerical underflow in the saved output.)*

### 6.2 Spearman Rank Correlations — Efficiency vs. Outcomes


| Relationship                  | Spearman ρ | p-value  | Interpretation                      |
| ----------------------------- | ---------- | -------- | ----------------------------------- |
| Efficiency → STC (population) | +0.8324    | ≈ 0      | Strong positive                     |
| Efficiency → LTR (population) | +0.0393    | ≈ 0      | Very weak positive                  |
| C0: Efficiency → LTR          | N/A        | —        | `retention_delta` unavailable in C0 |
| C1: Efficiency → LTR          | +0.2000    | 0.579584 | ns (small N)                        |
| C2: Efficiency → LTR          | +0.0390    | ≈ 0      | Very weak positive                  |


*(Populated from `outputs/efficiency_correlations.json`.)*

### 6.3 Post-Hoc Pairwise Tests

Mann-Whitney U with Bonferroni correction. Full results: `outputs/pairwise_tests.csv`.


| Cluster Pair | Outcome            | U Statistic | p (adj.) | r (effect) | Mean A  | Mean B  |
| ------------ | ------------------ | ----------- | -------- | ---------- | ------- | ------- |
| C0 vs C1     | `session_acc_mean` | 25,853      | 0.001186 | -0.3843    | 0.5658  | 0.4423  |
| C0 vs C2     | `session_acc_mean` | 29,727,266  | 0.021611 | +0.0438    | 0.5658  | 0.5711  |
| C1 vs C2     | `session_acc_mean` | 405,512     | 0.000263 | +0.4207    | 0.4423  | 0.5711  |
| C1 vs C2     | `retention_delta`  | 121,786     | 0.228728 | +0.2198    | -0.0464 | +0.0155 |


### 6.4 Core Finding Statement

> **This run shows a strong dissociation in magnitude between efficiency→STC and efficiency→LTR (ρ = +0.8324 vs. ρ = +0.0393). Clusters significantly differentiated STC, but did not significantly differentiate LTR (`retention_delta`). Practically, efficiency is a strong proxy for within-session performance, while its association with retention is trivial in size (ρ < 0.04) despite statistical significance at population scale.**

---

## 7. Limitations

Check each limitation that applies to this run.


| #   | Limitation                       | Description                                                                                                                                                                                            | Applies? |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| L1  | **Retention operationalization** | `retention_delta` measures cross-temporal performance stability, not memory retention. Without IRT-calibrated items or pre/post assessments, causal inference about durable encoding is not warranted. | `[ ✓ ]`  |
| L2  | **Median imputation bias**       | Users near the 40-interaction threshold receive imputed entropy and Hurst values converging on population medians. Bayesian partial pooling would be more principled.                                  | `[ ✓ ]`  |
| L3  | **Platform latency confound**    | `elapsed_time` derived from timestamp diffs reflects inter-event gaps, not true response latency. Interaction type, device, and network latency contribute noise.                                      | `[ ✓ ]`  |
| L4  | **UMAP stochasticity**           | Cluster boundaries are sensitive to hyperparameters and random seed. Embedding stability analysis (perturbation) was not performed.                                                                    | `[ ✓ ]`  |
| L5  | **No IRT calibration**           | Item difficulty is not modeled. Accuracy differences between clusters partially reflect item routing rather than pure behavioral style.                                                                | `[ ✓ ]`  |
| L6  | **Ecological validity**          | Aggregate session-level telemetry cannot capture within-session strategy shifts, external help-seeking, exam proximity, or fatigue.                                                                    | `[ ✓ ]`  |
| L7  | **Causal inference boundary**    | The typology is descriptive. Intervention recommendations require experimental validation and are not derivable from observational clustering alone.                                                   | `[ ✓ ]`  |


---

## 8. Pedagogical Implications

### 8.1 Fluency ≠ Learning

The strong efficiency-STC association (ρ = +0.8324) could tempt platform designers to treat fast, accurate session completion as a primary learning KPI. The very weak efficiency-LTR association (ρ = +0.0393) indicates this would be a category error: fluency metrics track performance state far more than learning trajectory.

### 8.2 Desirable Difficulties

Results are consistent with Bjork's desirable difficulties framework (1994): conditions that reduce immediate fluency (spaced practice, interleaving, retrieval under load) tend to produce better long-term retention. Platforms optimizing for massed, hint-rich, blocked practice may be optimizing for STC at the expense of LTR.

### 8.3 Cluster-Specific Instructional Targets

The small low-accuracy cluster (C1, accuracy median ≈ 0.321; n=29) shows a distinct behavioral signature and negative median retention delta (on the limited subset with retention available). The dominant cluster (C2; n=48,275) exhibits higher median STC and a small positive median retention delta (where available).

### 8.4 Measurement Recommendations

Platforms should track STC and LTR as separate KPIs with distinct temporal windows. Behavioral telemetry features most predictive of STC (efficiency index, within-session accuracy trajectory) may differ from those predictive of LTR (spacing regularity, re-attempt patterns), and LTR may be missing for users without sufficient longitudinal coverage.

---

## 9. Analyst Notes & Annotations

### 9.1 Comparison to Synthetic Benchmark

The synthetic benchmark run (n=800, RANDOM_SEED=42) produced the reference values used in this template. 


| Metric             | Synthetic Benchmark | Real-Data Value   | Direction |
| ------------------ | ------------------- | ----------------- | --------- |
| Silhouette Score   | 0.292               | -0.0128           | ↓         |
| ARI (ground truth) | 0.267               | N/A for real data | —         |
| ρ Efficiency → STC | +0.906              | +0.8324           | ↓         |
| ρ Efficiency → LTR | +0.031              | +0.0393           | ↑         |
| KW H — STC         | 138.16              | 22.539            | ↓         |
| KW H — LTR         | 0.59                | 1.449             | ↑         |
| Davies-Bouldin     | 1.811               | 2.0193            | ↑         |
| Calinski-Harabasz  | 1720.8              | 723.3730          | ↓         |


---

## References

Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. Shimamura (Eds.), *Metacognition*. MIT Press.

Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way. *Psychology and the Real World*, 2, 59–68.

Choi, Y., et al. (2020). EdNet: A large-scale hierarchical dataset in education. *AIED 2020*. arXiv:1912.03072.

Heitz, R. P. (2014). The speed-accuracy tradeoff: history, physiology, methodology, and behavior. *Frontiers in Neuroscience*, 8, 150.

van der Linden, W. J. (2006). A lognormal model for response times on test items. *Journal of Educational and Behavioral Statistics, 31*(2), 181–204.

van Orden, G. C., Holden, J. G., & Turvey, M. T. (2003). Self-organization of cognitive performance. *Journal of Experimental Psychology: General, 132*(3), 331–350.

---

