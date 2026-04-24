"""
ednet_cpda/src/features.py
──────────────────────────
Behavioral feature engineering for CPDA analysis.

Feature philosophy:
───────────────────
We extract features across four measurement layers that map directly onto
the theoretical constructs of interest:

  Layer 1 – KINEMATIC / TEMPORAL
    Raw time-domain statistics of response latency and inter-event timing.
    These capture the "metabolic rate" of a learning session.

  Layer 2 – ACCURACY / EFFICIENCY
    Performance metrics that separate raw accuracy from the effort cost
    at which it is achieved. A student who scores 80% in 3 seconds is
    behaviorally distinct from one who scores 80% in 30 seconds.

  Layer 3 – COMPLEXITY / DYNAMICS
    Entropy, autocorrelation, and distributional shape statistics derived
    from the time-series of response latencies. These index cognitive
    load variability and self-regulation consistency, drawing on
    complexity science (approximate entropy, power-law tails) and
    ecological psychology (within-session variability structure).

  Layer 4 – TEMPORAL / LONGITUDINAL
    Inter-session gap distributions, return rate, and proficiency drift
    across calendar time. These are the direct operationalizations of
    short-term task completion vs. long-term retention.

All features are computed per user and returned in a single flat DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(f"  {desc} …")
        return iterable

# optional heavy imports — graceful degradation
try:
    import antropy as ant
    _HAS_ANTROPY = True
except ImportError:
    _HAS_ANTROPY = False
    warnings.warn("[features] antropy not available; entropy features will use numpy fallback.")

try:
    import nolds
    _HAS_NOLDS = True
except ImportError:
    _HAS_NOLDS = False
    warnings.warn("[features] nolds not available; DFA/Hurst features will be skipped.")

EPSILON = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_entropy_approx(ts: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Approximate entropy using antropy if available, else Shannon fallback."""
    if len(ts) < 20:
        return np.nan
    ts = np.asarray(ts, dtype=float)
    if _HAS_ANTROPY:
        try:
            r = r_factor * ts.std()
            return ant.app_entropy(ts, order=m, metric='chebyshev')
        except Exception:
            pass
    # Shannon entropy of 20-bin histogram as fallback
    counts, _ = np.histogram(ts, bins=20)
    p = counts / (counts.sum() + EPSILON)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + EPSILON)))


def _safe_permutation_entropy(ts: np.ndarray, order: int = 3) -> float:
    if len(ts) < order + 2:
        return np.nan
    if _HAS_ANTROPY:
        try:
            return float(ant.perm_entropy(ts, order=order, normalize=True))
        except Exception:
            pass
    return np.nan


def _safe_hurst(ts: np.ndarray) -> float:
    """Hurst exponent via DFA (nolds) or R/S fallback."""
    if len(ts) < 50:
        return np.nan
    ts = np.asarray(ts, dtype=float)
    if _HAS_NOLDS:
        try:
            return float(nolds.hurst_rs(ts))
        except Exception:
            pass
    # Simple R/S estimate
    try:
        n = len(ts)
        mean_ts = ts.mean()
        deviations = np.cumsum(ts - mean_ts)
        R = deviations.max() - deviations.min()
        S = ts.std() + EPSILON
        return float(np.log(R / S) / np.log(n / 2))
    except Exception:
        return np.nan


def _power_law_alpha(ts: np.ndarray) -> float:
    """
    Estimate tail exponent α of response time distribution via Hill estimator.
    α > 2 → finite variance; α ≤ 2 → heavy-tailed (Lévy-like) dynamics.
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts[ts > 0]
    if len(ts) < 20:
        return np.nan
    k = max(10, len(ts) // 5)   # top-20% tail
    ts_sorted = np.sort(ts)[-k:]
    x_min = ts_sorted[0]
    if x_min <= 0:
        return np.nan
    return float(1 + k / np.sum(np.log(ts_sorted / x_min + EPSILON)))


def _autocorrelation_lag1(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 3:
        return np.nan
    ts = (ts - ts.mean()) / (ts.std() + EPSILON)
    return float(np.corrcoef(ts[:-1], ts[1:])[0, 1])


def _compute_proficiency_series(df_user: pd.DataFrame, window: int = 20) -> np.ndarray:
    """Rolling accuracy in windows of `window` interactions."""
    correct = df_user["correct"].values.astype(float)
    if len(correct) < window:
        return correct
    return np.convolve(correct, np.ones(window) / window, mode='valid')


def _session_boundaries(timestamps_ms: np.ndarray, gap_threshold_ms: int = 1_800_000) -> np.ndarray:
    """
    Returns session index per interaction.
    Default threshold = 30 minutes (1.8M ms).
    """
    diffs = np.diff(timestamps_ms, prepend=timestamps_ms[0])
    return np.cumsum(diffs > gap_threshold_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Per-user feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_user_features(df_user: pd.DataFrame) -> dict:
    """
    Extract the full behavioral feature vector for a single user.
    Returns a flat dict of scalar features.
    """
    uid = df_user["user_id"].iloc[0]
    feat: dict = {"user_id": uid, "n_interactions": len(df_user)}

    rt = df_user["elapsed_time"].values.astype(float)
    ts = df_user["timestamp"].values.astype(float)
    correct = df_user["correct"].values.astype(float)
    lag = df_user["lag_time"].values.astype(float)
    lag = lag[lag >= 0]

    # ── LAYER 1: Kinematic / Temporal ────────────────────────────────────────
    rt_log = np.log1p(rt)
    feat["rt_mean"]        = float(rt.mean())
    feat["rt_median"]      = float(np.median(rt))
    feat["rt_std"]         = float(rt.std())
    feat["rt_cv"]          = float(rt.std() / (rt.mean() + EPSILON))   # coefficient of variation
    feat["rt_log_mean"]    = float(rt_log.mean())
    feat["rt_log_std"]     = float(rt_log.std())
    feat["rt_skew"]        = float(skew(rt))
    feat["rt_kurtosis"]    = float(kurtosis(rt))
    feat["rt_iqr"]         = float(np.percentile(rt, 75) - np.percentile(rt, 25))
    feat["rt_p10"]         = float(np.percentile(rt, 10))
    feat["rt_p90"]         = float(np.percentile(rt, 90))

    if len(lag) > 1:
        feat["lag_mean"]   = float(lag.mean())
        feat["lag_median"] = float(np.median(lag))
        feat["lag_cv"]     = float(lag.std() / (lag.mean() + EPSILON))
    else:
        feat["lag_mean"] = feat["lag_median"] = feat["lag_cv"] = np.nan

    # ── LAYER 2: Accuracy / Efficiency ───────────────────────────────────────
    feat["accuracy"]           = float(correct.mean())
    feat["accuracy_first_half"] = float(correct[:len(correct)//2].mean()) if len(correct) >= 4 else np.nan
    feat["accuracy_second_half"]= float(correct[len(correct)//2:].mean()) if len(correct) >= 4 else np.nan
    feat["accuracy_drift"]     = feat["accuracy_second_half"] - feat["accuracy_first_half"] \
                                   if not np.isnan(feat["accuracy_first_half"]) else np.nan

    # Speed-Accuracy Tradeoff index: log(RT) predicted by accuracy
    if len(rt) >= 10 and correct.std() > 0:
        try:
            slope, _, r, _, _ = stats.linregress(correct, rt_log)
            feat["sat_slope"]  = float(slope)   # negative = faster when correct
            feat["sat_r2"]     = float(r**2)
        except Exception:
            feat["sat_slope"] = feat["sat_r2"] = np.nan
    else:
        feat["sat_slope"] = feat["sat_r2"] = np.nan

    # Efficiency index: accuracy per unit log-RT
    feat["efficiency_index"] = float(correct.mean() / (rt_log.mean() + EPSILON))

    # ── LAYER 3: Complexity / Dynamics ───────────────────────────────────────
    feat["approx_entropy"]      = _safe_entropy_approx(rt_log)
    feat["perm_entropy"]        = _safe_permutation_entropy(rt_log)
    feat["hurst_exponent"]      = _safe_hurst(rt_log)
    feat["rt_autocorr_lag1"]    = _autocorrelation_lag1(rt)
    feat["power_law_alpha"]     = _power_law_alpha(rt)

    # proficiency series complexity
    prof = _compute_proficiency_series(df_user)
    feat["proficiency_entropy"] = _safe_entropy_approx(prof) if len(prof) > 10 else np.nan
    feat["proficiency_trend"]   = float(np.polyfit(np.arange(len(prof)), prof, 1)[0]) \
                                    if len(prof) >= 3 else np.nan
    feat["proficiency_std"]     = float(prof.std()) if len(prof) >= 3 else np.nan

    # ── LAYER 4: Temporal / Longitudinal ─────────────────────────────────────
    sessions = _session_boundaries(ts.astype(int))
    n_sessions = int(sessions[-1]) + 1
    feat["n_sessions"] = n_sessions
    feat["interactions_per_session"] = float(len(rt) / max(1, n_sessions))

    session_accs = []
    session_rts  = []
    for s in range(n_sessions):
        mask = sessions == s
        if mask.sum() > 0:
            session_accs.append(correct[mask].mean())
            session_rts.append(rt[mask].mean())
    session_accs = np.array(session_accs)
    session_rts  = np.array(session_rts)

    feat["session_acc_mean"]   = float(session_accs.mean()) if len(session_accs) > 0 else np.nan
    feat["session_acc_std"]    = float(session_accs.std())  if len(session_accs) > 1 else np.nan
    feat["session_rt_cv"]      = float(session_rts.std() / (session_rts.mean() + EPSILON)) \
                                   if len(session_rts) > 1 else np.nan

    # inter-session gap statistics (proxy for return behaviour)
    span_days = (ts.max() - ts.min()) / 86_400_000
    feat["span_days"]          = float(span_days)
    feat["sessions_per_day"]   = float(n_sessions / max(1, span_days))
    feat["total_time_ms"]      = float(rt.sum())

    # retention proxy: compare accuracy of sessions 1–3 vs last 3
    if n_sessions >= 6:
        early_acc = session_accs[:3].mean()
        late_acc  = session_accs[-3:].mean()
        feat["retention_delta"] = float(late_acc - early_acc)
    else:
        feat["retention_delta"] = np.nan

    # within-session acceleration index: are items getting faster over time in a session?
    accel_vals = []
    for s in range(n_sessions):
        mask = sessions == s
        rt_s = rt[mask]
        if len(rt_s) >= 6:
            mid = len(rt_s) // 2
            accel_vals.append(rt_s[:mid].mean() - rt_s[mid:].mean())
    feat["within_session_accel"] = float(np.mean(accel_vals)) if accel_vals else np.nan

    # re-attempt fraction (question_id repeats within user)
    if "question_id" in df_user.columns:
        total = len(df_user)
        unique_q = df_user["question_id"].nunique()
        feat["reattempt_fraction"] = float(1 - unique_q / max(1, total))
    else:
        feat["reattempt_fraction"] = np.nan

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Population-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    min_interactions: int = 40,
) -> pd.DataFrame:
    """
    Compute per-user feature vectors across the full population.

    Parameters
    ----------
    df               : Raw interaction DataFrame from data_loader.get_data()
    min_interactions : Users with fewer interactions are excluded (insufficient
                       time series for reliable entropy / Hurst estimation).

    Returns
    -------
    pd.DataFrame  with one row per user and all behavioral features.
    """
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df_valid    = df[df["user_id"].isin(valid_users)]

    n_dropped = df["user_id"].nunique() - len(valid_users)
    if n_dropped > 0:
        print(f"[features] Excluded {n_dropped} users with < {min_interactions} interactions.")

    grouped  = df_valid.groupby("user_id")
    features = []

    for uid, grp in tqdm(grouped, desc="Extracting features", total=len(valid_users)):
        feat = extract_user_features(grp.copy())
        # carry ground-truth archetype if present (synthetic data)
        if "archetype" in grp.columns:
            feat["true_archetype"] = int(grp["archetype"].iloc[0])
        features.append(feat)

    feat_df = pd.DataFrame(features).set_index("user_id")
    print(f"[features] Feature matrix: {feat_df.shape[0]} users × {feat_df.shape[1]} features")
    return feat_df
