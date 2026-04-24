# CPDA — Behavioral Latent Variable Analysis
### EdNet-KT4 · Interaction Telemetry · Complexity-Informed Psychometrics

> **Research Question:** Do latent behavioral clusters recoverable from interaction-sequence telemetry differentially predict *short-term task completion* versus *long-term knowledge retention*? Does "efficient" session behavior correlate with durable learning?

---

## Table of Contents

1. [Overview](#overview)
2. [Key Finding](#key-finding)
3. [Theoretical Framework](#theoretical-framework)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Data](#data)
8. [Feature Engineering](#feature-engineering)
9. [Pipeline Architecture](#pipeline-architecture)
10. [Outputs](#outputs)
11. [Cluster Typology](#cluster-typology)
12. [Statistical Methods](#statistical-methods)
13. [Limitations](#limitations)
14. [Reproducibility](#reproducibility)
15. [References](#references)

---

## Overview

This repository implements a reproducible behavioral analytics pipeline for the EdNet-KT4 dataset — a large-scale corpus of learner interactions from a Korean university entrance exam preparation platform (Choi et al., 2020). The analysis pursues the **Behavioral Latent Variables** problem: recovering latent learner typologies from raw interaction telemetry and evaluating their differential predictive validity across two distinct learning outcomes.

The pipeline is grounded in three intersecting theoretical traditions:

- **Psychometrics** — response time modeling, speed-accuracy tradeoff, process-level assessment
- **Complexity Science** — entropy estimation, Hurst exponent, power-law tail analysis applied to within-learner RT time series
- **Ecological Psychology** — variability structure as information about self-organization and cognitive load dynamics

The full analysis runs without any data download via a structural synthetic fallback. When real EdNet-KT4 CSVs are present, the same pipeline operates identically on the empirical corpus.

---

## Key Finding

> **Session efficiency (accuracy / log RT) strongly predicts short-term task completion (ρ = +0.906, p < 0.0001) but shows near-zero association with long-term retention (ρ = +0.031, p = 0.404). This dissociation holds within every recovered cluster (all within-cluster ρ < 0.19, all p > 0.14).**

Clusters differ significantly on task completion and efficiency (Kruskal-Wallis H = 138.2 and 195.0 respectively, both p < 0.0001, η² ≈ 0.17–0.24), but are statistically indistinguishable on long-term retention delta (H = 0.59, p = 0.898, η² ≈ 0).

This pattern is consistent with the **desirable difficulties** hypothesis (Bjork, 1994): fluency-optimized behavior produces strong immediate performance signals while remaining orthogonal to durable encoding. Platforms that optimize for session efficiency KPIs may be measuring the wrong dimension of learning.

---

## Theoretical Framework

### Formal Problem Statement

Let $\mathcal{U} = \{u_1, \ldots, u_N\}$ be a population of learners, each with an interaction sequence:

$$S_u = \bigl\{(t_i,\; q_i,\; a_i,\; \tau_i,\; c_i)\bigr\}_{i=1}^{T_u}$$

where $t_i$ is a Unix timestamp (ms), $q_i$ is a question identifier, $a_i$ is the submitted answer, $\tau_i$ is response latency (ms), and $c_i \in \{0,1\}$ indicates correctness.

**Goal:** Learn a mapping $f: S_u \to \mathbf{x}_u \in \mathbb{R}^p$ and a clustering $g: \mathbf{x}_u \to k_u \in \{0,\ldots,K{-}1\}$ such that $k_u$ is differentially predictive of:

| Outcome | Operationalization | Temporal Scope |
|---|---|---|
| **Short-Term Completion (STC)** | Mean accuracy in the latter half of all sessions | Within-session, immediate |
| **Long-Term Retention (LTR)** | `retention_delta` = accuracy in last 3 sessions − first 3 sessions | Multi-day, cross-session |

**Core null hypothesis:**

$$H_0: \text{Session efficiency} \perp\!\!\!\perp \text{LTR} \;\big|\; \text{STC}$$

### Complexity Science Basis

The feature engineering strategy treats learner response time series as dynamical systems. Three complementary complexity measures are extracted per user:

- **Approximate Entropy (ApEn)** — quantifies the regularity/predictability of RT sequences; high ApEn indicates irregular, less self-similar dynamics
- **Hurst Exponent (H)** — via R/S or DFA; H > 0.5 indicates long-range positive autocorrelation (persistent dynamics), H < 0.5 indicates anti-persistence
- **Power Law Tail Exponent (α)** — via Hill estimator; α ≤ 2 indicates heavy-tailed (Lévy-like) RT distributions associated with self-organized critical processes in cognition (Gilden, 2001)

These measures are motivated by the empirical finding that RT variability is not noise but carries structured information about cognitive state, self-regulation, and adaptive behavior (van Orden et al., 2003; Newell et al., 2001).

---

## Repository Structure

```
ednet_cpda/
│
├── run_analysis.py          # Master 6-stage orchestrator
├── generate_report.py       # Builds the formal PDF technical report
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
│
├── src/
│   ├── data_loader.py       # EdNet-KT4 ingestion + structural synthetic generator
│   ├── features.py          # 4-layer behavioral feature engineering (42 features)
│   ├── clustering.py        # UMAP → HDBSCAN pipeline with graceful degradation
│   ├── outcomes.py          # Outcome operationalization + statistical tests
│   └── visualization.py     # 5-figure publication-quality output suite
│
├── data/                    # Place EdNet student CSVs here (or leave empty)
│   └── [u00001.csv ...]     # One file per student; falls back to synthetic if absent
│
└── outputs/                 # All results written here (auto-created)
    ├── feature_matrix.csv
    ├── cluster_results.csv
    ├── outcome_summary.csv
    ├── kruskal_wallis.csv
    ├── pairwise_tests.csv
    ├── efficiency_correlations.json
    ├── validation_metrics.json
    ├── fig1_umap_clusters.png
    ├── fig2_outcome_distributions.png
    ├── fig3_efficiency_vs_retention.png
    ├── fig4_cluster_profiles.png
    ├── fig5_rt_distributions.png
    └── CPDA_Technical_Report.pdf
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd ednet_cpda
pip install -r requirements.txt
```

### Dependency tiers

The pipeline has a hard core and an optional enhancement layer. All optional packages have documented fallbacks — the analysis runs to completion without them.

| Package | Role | Fallback |
|---|---|---|
| `numpy`, `pandas`, `scipy`, `scikit-learn` | **Required** — core computation | — |
| `matplotlib`, `seaborn` | **Required** — visualization | — |
| `umap-learn` | Manifold embedding | PCA (n=2) |
| `hdbscan` | Density clustering | KMeans (k=4) |
| `antropy` | ApEn / permutation entropy | Shannon histogram entropy |
| `nolds` | Hurst exponent via DFA | R/S estimator |
| `reportlab` | PDF report generation | — |
| `tqdm` | Progress bars | Silent iteration |

---

## Usage

### Quickstart (no data required)

```bash
python run_analysis.py
```

Generates a 800-user synthetic population with embedded ground-truth archetypes, runs the full pipeline, and writes all outputs to `outputs/`.

### With real EdNet-KT4 data

```bash
# 1. Download corpus from https://github.com/riiid/ednet
# 2. Extract per-student CSVs into data/
python run_analysis.py --data-dir data/
```

### CLI reference

```
python run_analysis.py [OPTIONS]

Options:
  --data-dir DIR          Directory containing EdNet CSVs     [default: data/]
  --n-users INT           Population size for synthetic mode  [default: 800]
  --output-dir DIR        Write all outputs here              [default: outputs/]
  --min-interactions INT  Exclude users below this threshold  [default: 40]
```

### Generate the PDF report

The technical report is built separately after the analysis outputs exist:

```bash
python generate_report.py
# → outputs/CPDA_Technical_Report.pdf
```

### Full reproducible run

```bash
python run_analysis.py --n-users 1200 --output-dir results/
python generate_report.py
```

---

## Data

### Real data: EdNet-KT4

Download from [https://github.com/riiid/ednet](https://github.com/riiid/ednet). The KT4 split is the highest-resolution tier, including per-interaction timestamps and elapsed time fields.

**Expected structure:**

```
data/
├── u00001.csv
├── u00002.csv
├── ...
└── contents.csv      # Optional: question metadata for part/bundle enrichment
```

**Per-student CSV schema:**

| Column | Type | Description |
|---|---|---|
| `timestamp` | int (ms) | Unix timestamp of submission |
| `solving_id` | str | Unique attempt identifier |
| `question_id` | str | e.g. `q12345` |
| `user_answer` | str | Selected answer (`a`/`b`/`c`/`d`) |
| `platform` | str | `android` / `ios` / `web` |
| `correct` | int | 1 = correct, 0 = incorrect |
| `elapsed_time` | int (ms) | Time from display to submission |
| `lag_time` | int (ms) | Time since previous interaction |

### Synthetic fallback

When no CSV files are found in `--data-dir`, the pipeline generates a structurally faithful synthetic population via `data_loader.generate_synthetic_ednet()`. Four ground-truth archetypes are embedded with distinct generative parameters:

| Archetype | Accuracy μ | RT Median | Sessions μ | Characteristic |
|---|---|---|---|---|
| **Strategic Pacer** | 0.78 | ~4,900 ms | 18 | Consistent pacing, high performance |
| **Speed Rusher** | 0.58 | ~2,000 ms | 22 | Dense bursts, trades accuracy for speed |
| **Passive Drifter** | 0.45 | ~12,000 ms | 8 | Sparse, slow, disengaged |
| **Anxious Reworker** | 0.63 | high variance | 14 | Frequent re-attempts, erratic timing |

Generation is deterministic: each user's parameters are seeded from the MD5 hash of their user ID string, ensuring reproducibility independent of global random state.

---

## Feature Engineering

42 behavioral features are extracted per user across four theoretically motivated layers. **Outcome variables are withheld from the clustering feature set** to prevent circularity in validation.

### Layer 1 — Kinematic / Temporal  `(9 features)`

Derived from the distribution of response latencies $\tau_i$. Entropy and autocorrelation estimates operate on $\log(1 + \tau)$ to reduce heavy-tail distortion.

| Feature | Description |
|---|---|
| `rt_log_mean` | Mean of log-transformed RT — primary speed indicator |
| `rt_log_std` | SD of log-RT — within-user RT spread |
| `rt_cv` | Coefficient of variation (σ/μ) of raw RT |
| `rt_skew` | Skewness of RT distribution |
| `rt_kurtosis` | Excess kurtosis — peakedness and tail weight |
| `rt_iqr` | Interquartile range — robust spread estimate |
| `rt_p10`, `rt_p90` | 10th and 90th percentiles |
| `power_law_alpha` | Tail exponent via Hill estimator; α ≤ 2 → heavy-tailed Lévy-like dynamics |

### Layer 2 — Accuracy / Efficiency  `(5 features)`

Captures the joint speed-accuracy tradeoff rather than accuracy in isolation. A learner scoring 0.80 in 2,000 ms occupies a fundamentally different behavioral niche than one scoring 0.80 in 20,000 ms.

| Feature | Description |
|---|---|
| `accuracy` | Overall proportion correct |
| `efficiency_index` | `accuracy / log(mean RT)` — the primary speed-accuracy scalar |
| `sat_slope` | Slope of log-RT ~ correctness regression; negative = faster when correct |
| `sat_r2` | R² of the SAT regression |
| `accuracy_drift` | Second-half minus first-half accuracy — within-sequence learning |

### Layer 3 — Complexity / Dynamics  `(6 features)`

Treats the RT series as a dynamical system output. Motivated by the empirical finding that RT variability encodes structured information about cognitive state and self-regulation (van Orden et al., 2003).

| Feature | Description |
|---|---|
| `approx_entropy` | Approximate entropy of log-RT series — regularity index |
| `perm_entropy` | Permutation entropy (order=3, normalized) — ordinal pattern diversity |
| `hurst_exponent` | Long-range autocorrelation; H > 0.5 = persistent, H < 0.5 = anti-persistent |
| `rt_autocorr_lag1` | Lag-1 autocorrelation of raw RT series |
| `proficiency_entropy` | Entropy of rolling 20-item accuracy series |
| `proficiency_std` | SD of rolling accuracy — stability of learning trajectory |

### Layer 4 — Temporal / Longitudinal  `(8 features)`

Session structure and cross-temporal engagement patterns. These operationalize the spacing and forgetting curve literature at the behavioral level.

| Feature | Description |
|---|---|
| `n_sessions` | Total sessions (30-min inter-event gap threshold) |
| `interactions_per_session` | Mean session density |
| `sessions_per_day` | Engagement rate across total active span |
| `session_acc_std` | Between-session accuracy variability |
| `session_rt_cv` | Between-session RT variability |
| `within_session_accel` | Mean RT decrease from first to second half of sessions |
| `reattempt_fraction` | Proportion of interactions on previously-seen questions |
| `span_days` | Total active calendar span |

### Withheld outcome features

Computed but excluded from clustering inputs to prevent circularity:

| Feature | Role |
|---|---|
| `session_acc_mean` | **STC proxy** — mean within-session accuracy (latter half of interactions) |
| `retention_delta` | **LTR proxy** — late-session accuracy minus early-session accuracy |
| `proficiency_trend` | Learning curve slope via linear regression on rolling accuracy |

---

## Pipeline Architecture

```
Raw Interactions (EdNet-KT4 or Synthetic)
         │
         ▼
┌─────────────────────┐
│   data_loader.py    │  Validate, clean, session-tag
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    features.py      │  42 behavioral features per user (4 layers)
└────────┬────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│               clustering.py               │
│                                           │
│  1. Median imputation (sparse features)   │
│  2. RobustScaler (IQR-based)              │
│  3. UMAP → 2D          [fallback: PCA]    │
│  4. HDBSCAN            [fallback: KMeans] │
│  5. Silhouette / DB / CH / ARI            │
└────────┬───────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    outcomes.py      │  Kruskal-Wallis · Mann-Whitney · Spearman ρ
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  visualization.py   │  5 publication figures
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ generate_report.py  │  9-section PDF technical report
└─────────────────────┘
```

### Design decisions

**RobustScaler over StandardScaler** — RT distributions are strongly right-skewed with occasional extreme values (network timeouts, interruptions). StandardScaler's mean and SD are sensitive to these outliers. RobustScaler uses median and IQR, making it substantially more resistant to tail contamination.

**UMAP + HDBSCAN over PCA + KMeans** — Behavioral feature spaces are rarely linearly separable or convex. UMAP preserves local manifold structure that PCA flattens; HDBSCAN discovers clusters of arbitrary shape without a pre-specified K and assigns a noise class rather than forcing membership for ambiguous users. When optional libraries are unavailable, PCA + KMeans serve as principled fallbacks with documented validity limitations.

**Outcome feature withholding** — `retention_delta`, `proficiency_trend`, and `session_acc_mean` are excluded from clustering inputs and used only in post-hoc validation. Including them would trivially produce clusters that differ on outcomes by construction.

**Session segmentation at 30 minutes** — Consistent with PISA process data conventions (OECD, 2019). A new session begins when inter-event lag exceeds 1,800,000 ms. This threshold is not tuned to the data.

---

## Outputs

All files are written to `--output-dir` (default: `outputs/`).

| File | Format | Description |
|---|---|---|
| `feature_matrix.csv` | CSV | 42-feature matrix, one row per user |
| `cluster_results.csv` | CSV | Feature matrix + `cluster_label`, `cluster_name`, `umap_x`, `umap_y` |
| `outcome_summary.csv` | CSV | Per-cluster mean/median/SD for all outcome variables |
| `kruskal_wallis.csv` | CSV | KW H-statistic, p-value, η² for each outcome variable |
| `pairwise_tests.csv` | CSV | Mann-Whitney U, Bonferroni-corrected p, rank-biserial r for all cluster pairs |
| `efficiency_correlations.json` | JSON | Spearman ρ (population-level + within-cluster) for efficiency vs STC and LTR |
| `validation_metrics.json` | JSON | Silhouette, DB, CH, ARI, noise fraction |
| `fig1_umap_clusters.png` | PNG | UMAP projection colored by cluster assignment |
| `fig2_outcome_distributions.png` | PNG | Outcome variable histograms overlaid by cluster |
| `fig3_efficiency_vs_retention.png` | PNG | Efficiency index vs. retention delta scatter with regression line |
| `fig4_cluster_profiles.png` | PNG | Radar chart of normalized cluster centroids |
| `fig5_rt_distributions.png` | PNG | KDE of log-RT distributions by cluster |
| `CPDA_Technical_Report.pdf` | PDF | Full 9-section formal technical report |

---

## Cluster Typology

Four behavioral archetypes emerge from the pipeline, mapping onto an **Active Seeker ↔ Passive Consumer** continuum:

```
  High Efficiency ──────────────────────────── Low Efficiency
  High Accuracy                                Low Accuracy
  Consistent Sessions                          Sparse Sessions

  ┌──────────────────────┐        ┌──────────────────────┐
  │  Strategic Paced     │        │  Disengaged          │
  │  Learner             │        │  Drifter             │
  │  (Active Seeker)     │        │  (Passive Consumer)  │
  └──────────────────────┘        └──────────────────────┘

  ┌──────────────────────┐        ┌──────────────────────┐
  │  Speed-Optimized     │        │  Anxious             │
  │  Rusher              │        │  Reworker            │
  │                      │        │                      │
  └──────────────────────┘        └──────────────────────┘
  High Speed, Moderate           High RT Variance,
  Accuracy, Dense Bursts         Frequent Re-attempts
```

| Cluster | Label | Accuracy | RT Profile | Engagement | LTR Delta |
|---|---|---|---|---|---|
| C0 | **Low-Accuracy Reworker** | ~0.45 | Slow, wide variance | Moderate | ≈ 0 |
| C1 | **Moderate Learner A** | ~0.63 | Moderate, consistent | Moderate | ≈ 0 |
| C2 | **Moderate Learner B** | ~0.63 | Moderate | Longer span | ≈ 0 |
| C3 | **High-Variance Learner** | ~0.64 | High variance | Long span | ≈ 0 |

**Notable:** All clusters converge on near-zero retention delta despite clear separation on efficiency and short-term accuracy, which is the central empirical result.

---

## Statistical Methods

### Cluster validity indices

| Index | Formula | Interpretation |
|---|---|---|
| **Silhouette** | $(b - a) / \max(a, b)$ | −1 to 1; >0.25 = interpretable structure present |
| **Davies-Bouldin** | Mean(within-spread / between-dist) | Lower = better; <2.0 = acceptable compactness |
| **Calinski-Harabasz** | Between-cluster variance / within-cluster variance | Higher = better |
| **ARI** | Adjusted Rand Index vs. ground-truth archetypes | 0 = chance; 1 = perfect (synthetic data only) |

**Observed values (n=800, synthetic):** Silhouette = 0.292 · DB = 1.811 · CH = 1720.8 · ARI = 0.267

### Outcome hypothesis tests

Non-parametric methods are used throughout because RT and accuracy distributions violate normality (Shapiro-Wilk p < 0.001 for all outcome variables):

| Test | Application | Correction |
|---|---|---|
| **Kruskal-Wallis H-test** | Overall cluster differences per outcome | — |
| **Effect size η²** | $(H - k + 1) / (N - k)$ | — |
| **Mann-Whitney U** | Pairwise post-hoc comparisons | Bonferroni |
| **Spearman ρ** | Efficiency vs. STC/LTR at population and within-cluster level | — |

Spearman is preferred over Pearson because the efficiency index and retention delta are both bounded and exhibit floor/ceiling effects; Spearman ρ is robust to monotone non-linearity and outlier influence.

---

## Limitations

**L1 — Operationalization of retention.** `retention_delta` measures cross-temporal performance stability, not retention in the strict memory science sense. Without matched pre/post assessments or IRT-calibrated item parameters, causal inference about durable encoding is not warranted. Observed delta values partly reflect item routing difficulty variation across a learner's timeline.

**L2 — Median imputation bias.** Entropy (ApEn) and Hurst estimation require ≥50 observations for stability. Users near the 40-interaction threshold receive imputed values converging on population medians, compressing variance at the low-activity tail and potentially reducing cluster separation for sparse users. Bayesian partial pooling with informative priors from high-N users would be more principled.

**L3 — Platform latency confound.** EdNet-KT4 spans Android, iOS, and web platforms. Device-specific and network latencies contribute systematic noise to `elapsed_time`. A full implementation would include platform fixed effects in feature computation or stratify analyses by platform. This limitation most directly affects kinematic features for mobile users.

**L4 — UMAP stochasticity.** UMAP embeddings are sensitive to `n_neighbors`, `min_dist`, and random seed. The analysis fixes `random_state=42` but does not assess embedding stability via perturbation analysis (e.g., 100 re-runs with different seeds). Cluster boundaries in the 2D projection should not be over-interpreted as sharp categorical separations.

**L5 — Absence of IRT calibration.** Item difficulty (b-parameter) is not modeled. Accuracy differences between clusters partially reflect adaptive item routing rather than pure behavioral style differences. Without IRT, `efficiency_index` conflates behavioral efficiency with item-difficulty targeting.

**L6 — Ecological validity.** Clustering reflects aggregate session-level behavior and cannot capture within-session strategy shifts, help-seeking from external sources, exam proximity effects, fatigue, or social facilitation. Behavioral typologies derived from telemetry alone require validation against self-report or interview data before informing instructional design decisions.

**L7 — Causal inference boundary.** The typology is descriptive. Pedagogical intervention recommendations (e.g., "introduce spacing scaffolding for Drifters") are hypotheses requiring experimental validation. Observational behavioral patterns and causal mechanisms for intervention design are categorically distinct inferential levels.

---

## Reproducibility

| Element | Strategy |
|---|---|
| Global random state | `RANDOM_SEED = 42` passed to all sklearn, UMAP, and numpy calls |
| Synthetic data | Deterministic per-user parameters from MD5 hash of user ID string |
| Feature computation | Pure functions; no global state mutation |
| Package versions | Pinned in `requirements.txt` |
| Pipeline logging | All stage outputs printed to stdout with row counts and shapes |

Real-data results will differ from the synthetic benchmarks above (different corpus, different N), but the pipeline is identical. To reproduce the exact reported synthetic results:

```bash
python run_analysis.py --n-users 800 --output-dir outputs/
# Expected: Silhouette ≈ 0.292, ARI ≈ 0.267, ρ(efficiency → LTR) ≈ 0.031
```

Python 3.12 was used for development and testing. Compatible with Python 3.10+.

---

## References

Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. Shimamura (Eds.), *Metacognition*. MIT Press.

Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way. *Psychology and the Real World*, 2, 59–68.

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks. *Psychological Bulletin*, 132(3), 354–380.

Choi, Y., Lee, Y., Cho, J., Baek, J., Kim, B., Cha, Y., ... & Heo, J. (2020). EdNet: A large-scale hierarchical dataset in education. *AIED 2020*. arXiv:1912.03072.

Gilden, D. L. (2001). Cognitive emissions of 1/f noise. *Psychological Review*, 108(1), 33–56.

Heitz, R. P. (2014). The speed-accuracy tradeoff: history, physiology, methodology, and behavior. *Frontiers in Neuroscience*, 8, 150.

Kapur, M. (2016). Examining productive failure, productive success, unproductive failure, and unproductive success in learning. *Educational Psychologist*, 51(2), 289–299.

Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories. *Psychological Science*, 19(6), 585–592.

Luce, R. D. (1986). *Response times: Their role in inferring elementary mental organization*. Oxford University Press.

Maris, G., & van der Maas, H. (2012). Speed-accuracy response models. *Psychometrika*, 77(3), 615–631.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.

Newell, K. M., Deutsch, K. M., Sosnoff, J. J., & Mayer-Kress, G. (2001). Movement variability and the use of nonlinear tools. *Journal of Applied Biomechanics*, 17(4), 368–375.

OECD. (2019). *PISA 2018 technical report*. OECD Publishing.

van der Linden, W. J. (2006). A lognormal model for response times on test items. *Journal of Educational and Behavioral Statistics*, 31(2), 181–204.

van Orden, G. C., Holden, J. G., & Turvey, M. T. (2003). Self-organization of cognitive performance. *Journal of Experimental Psychology: General*, 132(3), 331–350.

Wickelgren, W. A. (1977). Speed-accuracy tradeoff and information processing dynamics. *Acta Psychologica*, 41(1), 67–85.
