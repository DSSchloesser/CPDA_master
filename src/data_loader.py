"""
ednet_cpda/src/data_loader.py
─────────────────────────────
EdNet-KT4 data loading and preprocessing.

EdNet-KT4 schema (per-student CSV files):
  timestamp       : unix milliseconds
  solving_id      : unique attempt identifier
  question_id     : e.g. "q12345"
  user_answer     : student's selected answer (a/b/c/d)
  platform        : android / ios / web
  first_correct   : 1 if correct on first attempt, else 0
  correct         : 1 if final answer is correct
  elapsed_time    : ms from question display to submission
  lag_time        : ms since previous interaction (0 for first)

Content table (contents.csv):
  question_id, bundle_id, explanation_id, correct_answer,
  part, tags, deployed_at

We simulate a realistic synthetic sample when the raw data
is not locally present, preserving all structural properties
needed for the analysis pipeline.
"""

from __future__ import annotations

import os
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(f"  {desc} …")
        return iterable

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_SYNTHETIC_USERS = 800          # enough for stable cluster recovery
N_INTERACTIONS_MEAN = 420        # median EdNet-KT4 session depth
SYNTHETIC_TAG = "__synthetic__"


# ── helpers ───────────────────────────────────────────────────────────────────
def _seed(user_id: str) -> int:
    """Deterministic per-user seed from hashed user_id string."""
    return int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16) % (2**31)


# ── synthetic data generation ─────────────────────────────────────────────────
def generate_synthetic_ednet(
    n_users: int = N_SYNTHETIC_USERS,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic EdNet-KT4–compatible dataset that reproduces
    the distributional and temporal structure of the real corpus while
    embedding four ground-truth behavioral archetypes:

        Archetype 0 – Strategic Pacer
            High accuracy, moderate speed, consistent inter-session gaps.
        Archetype 1 – Speed Rusher
            Low-to-moderate accuracy, very fast response times, dense
            within-session bursts with irregular return intervals.
        Archetype 2 – Passive Drifter
            Low accuracy, long response latencies, sparse usage, many
            multi-day gaps between sessions.
        Archetype 3 – Anxious Reworker
            Moderate accuracy, highly variable response times, frequent
            re-attempts of previously seen content.

    These archetypes map directly onto the "Active Seeker / Passive
    Consumer" continuum described in the CPDA prompt and allow
    quantitative validation of recovered clusters.
    """
    rng = np.random.default_rng(seed)

    # archetype mixture proportions
    archetype_probs = [0.30, 0.25, 0.25, 0.20]
    archetypes = rng.choice(4, size=n_users, p=archetype_probs)

    records = []
    user_ids = [f"u{i:05d}" for i in range(n_users)]

    for uid, arch in zip(tqdm(user_ids, desc="Generating synthetic users"), archetypes):
        urng = np.random.default_rng(_seed(uid))

        # archetype-specific hyperparameters
        if arch == 0:   # Strategic Pacer
            n_int       = int(urng.normal(500, 80))
            acc_mu      = 0.78; acc_sd  = 0.08
            rt_mu_log   = 8.5;  rt_sd_log = 0.4   # ~4900 ms median
            lag_shape   = 2.0;  lag_scale = 3.0    # days between sessions
            n_sessions  = int(urng.normal(18, 3))
        elif arch == 1: # Speed Rusher
            n_int       = int(urng.normal(600, 120))
            acc_mu      = 0.58; acc_sd  = 0.12
            rt_mu_log   = 7.6;  rt_sd_log = 0.5   # ~2000 ms median
            lag_shape   = 1.2;  lag_scale = 1.5
            n_sessions  = int(urng.normal(22, 4))
        elif arch == 2: # Passive Drifter
            n_int       = int(urng.normal(200, 60))
            acc_mu      = 0.45; acc_sd  = 0.14
            rt_mu_log   = 9.4;  rt_sd_log = 0.7   # ~12000 ms median
            lag_shape   = 0.8;  lag_scale = 7.0
            n_sessions  = int(urng.normal(8, 3))
        else:           # Anxious Reworker
            n_int       = int(urng.normal(450, 100))
            acc_mu      = 0.63; acc_sd  = 0.15
            rt_mu_log   = 8.9;  rt_sd_log = 0.9   # high variance
            lag_shape   = 1.5;  lag_scale = 2.5
            n_sessions  = int(urng.normal(14, 4))

        n_int      = max(40, n_int)
        n_sessions = max(3, n_sessions)

        # build session structure
        session_sizes = urng.multinomial(n_int, [1/n_sessions]*n_sessions)
        session_gaps_days = urng.gamma(lag_shape, lag_scale, size=n_sessions)
        session_start_ms  = np.cumsum(session_gaps_days * 86_400_000).astype(int)

        parts      = urng.choice([1,2,3,4,5,6,7], size=n_int, p=[.10,.10,.15,.15,.20,.15,.15])
        bundle_ids = urng.integers(1000, 9999, size=n_int)
        q_ids      = urng.integers(10000, 99999, size=n_int)
        answers    = urng.choice(['a','b','c','d'], size=n_int)
        correct    = urng.random(n_int) < np.clip(urng.normal(acc_mu, acc_sd, n_int), 0.1, 0.98)
        elapsed    = np.exp(urng.normal(rt_mu_log, rt_sd_log, n_int)).astype(int)
        elapsed    = np.clip(elapsed, 500, 300_000)

        # flatten sessions to timestamps
        ts_list = []
        idx = 0
        for s_idx, sz in enumerate(session_sizes):
            t = int(session_start_ms[s_idx])
            for _ in range(sz):
                ts_list.append(t)
                t += int(elapsed[idx]) + urng.integers(500, 3000)
                idx += 1
        timestamps = np.array(ts_list[:n_int])

        lag_times = np.concatenate([[0], np.diff(timestamps)])

        # re-attempt flag for Anxious Reworker (repeated q_ids)
        if arch == 3:
            repeat_mask = urng.random(n_int) < 0.18
            q_ids[repeat_mask] = urng.choice(q_ids[~repeat_mask][:max(1, (~repeat_mask).sum())],
                                             size=repeat_mask.sum())

        df_user = pd.DataFrame({
            "user_id":      uid,
            "timestamp":    timestamps,
            "question_id":  [f"q{q}" for q in q_ids],
            "bundle_id":    [f"b{b}" for b in bundle_ids],
            "part":         parts,
            "user_answer":  answers,
            "correct":      correct.astype(int),
            "elapsed_time": elapsed,
            "lag_time":     lag_times.astype(int),
            "archetype":    arch,   # ground-truth label (held-out during clustering)
        })
        records.append(df_user)

    df = pd.concat(records, ignore_index=True)
    df[SYNTHETIC_TAG] = True

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / "ednet_synthetic.csv"
        df.to_csv(out, index=False)
        print(f"[data_loader] Saved synthetic data → {out}  ({len(df):,} rows)")

    return df


# ── EdNet schema definitions ──────────────────────────────────────────────────
#
# EdNet has multiple KT splits with different schemas. We sniff the first file
# to detect which variant is present, then apply the correct column mapping.
#
# KT1 (2 cols, no header):   timestamp, question_id
# KT2 (3 cols, no header):   timestamp, question_id, user_answer
# KT3 (with header):         timestamp, solving_id, question_id, user_answer,
#                             elapsed_time, is_correct
# KT4 (with header):         timestamp, solving_id, question_id, user_answer,
#                             platform, first_correct, correct,
#                             elapsed_time, lag_time
#
# Some distributions of KT4 omit the header row entirely (headerless KT4).
# The pipeline requires at minimum: timestamp, question_id, correct, elapsed_time.
# lag_time is derived from timestamp diffs when absent.

_KT4_HEADER_COLS = [
    "timestamp", "solving_id", "question_id", "user_answer",
    "platform", "first_correct", "correct", "elapsed_time", "lag_time",
]
_KT3_HEADER_COLS = [
    "timestamp", "solving_id", "question_id", "user_answer",
    "elapsed_time", "is_correct",
]
# KT1: timestamp, question_id
_KT1_POSITIONAL = ["timestamp", "question_id"]
# KT2: timestamp, question_id, user_answer
_KT2_POSITIONAL = ["timestamp", "question_id", "user_answer"]
# LT / KT2-KT4 action-trace format (has action_type column)
_LT_HEADER_COLS = ["timestamp", "action_type", "item_id", "cursor_time", "source", "user_answer", "platform"]


def _sniff_schema(fp: Path) -> str:
    """
    Read the first line of a CSV and return the detected schema variant:
        'kt4_header'     — has 'elapsed_time' in header row (KT1-style preprocessed)
        'kt3_header'     — has 'is_correct' in header row
        'lt_header'      — has 'action_type' in header row (KT2/KT3/KT4 raw format)
        'kt4_positional' — headerless, >=7 numeric columns
        'kt3_positional' — headerless, 6 columns
        'kt2_positional' — headerless, 3 columns
        'kt1_positional' — headerless, 2 columns
        'unknown'        — cannot determine
    """
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
        tokens = [t.strip().lower() for t in first_line.split(",")]

        if "elapsed_time" in tokens:
            return "kt4_header"
        if "is_correct" in tokens:
            return "kt3_header"
        # KT2/KT3/KT4 all use action_type as the event discriminator
        if "action_type" in tokens:
            return "lt_header"

        # headerless — decide by column count and whether first token is numeric
        try:
            int(tokens[0])
            is_data_row = True
        except ValueError:
            is_data_row = False

        if is_data_row:
            n_cols = len(tokens)
            if n_cols >= 7:
                return "kt4_positional"
            if n_cols == 6:
                return "kt3_positional"
            if n_cols == 3:
                return "kt2_positional"
            if n_cols == 2:
                return "kt1_positional"

    except Exception:
        pass
    return "unknown"


def _parse_user_file(fp: Path, schema: str) -> Optional[pd.DataFrame]:
    """
    Parse a single user CSV into a normalized DataFrame with canonical columns:
        user_id, timestamp, question_id, user_answer,
        correct, elapsed_time, lag_time
    Returns None on unrecoverable failure.
    """
    try:
        if schema == "lt_header":
            # ── EdNet KT2/KT3/KT4 action-trace format ────────────────────────
            df = pd.read_csv(fp, header=0, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]

            # Rename item_id → question_id for pipeline compatibility
            if "item_id" in df.columns and "question_id" not in df.columns:
                df = df.rename(columns={"item_id": "question_id"})

            # Keep only 'respond' rows — these are the actual answer submissions
            # 'enter' and 'submit' rows carry no answer and no correctness signal
            if "action_type" in df.columns:
                df = df[df["action_type"].str.strip().str.lower() == "respond"].copy()

            if len(df) == 0:
                return None  # file had no respond actions

            df = df.sort_values("timestamp").reset_index(drop=True)

            # elapsed_time: compute from enter→respond gap within each bundle visit.
            # Since enter rows are dropped, derive from timestamp diffs as proxy.
            if "elapsed_time" not in df.columns:
                df["elapsed_time"] = df["timestamp"].diff().abs().fillna(0).astype(int)
                nz = df.loc[df["elapsed_time"] > 0, "elapsed_time"]
                med = int(nz.median()) if len(nz) > 0 else 5000
                med = max(med, 1000)
                df.loc[df["elapsed_time"] == 0, "elapsed_time"] = med

            # lag_time: time since previous interaction
            if "lag_time" not in df.columns:
                df["lag_time"] = df["timestamp"].diff().fillna(0).abs().astype(int)

            # correct: not in LT files — resolved later via questions.csv join
            if "correct" not in df.columns:
                df["correct"] = np.nan

            df["user_id"] = fp.stem
            return df

        elif schema in ("kt4_header", "unknown"):
            df = pd.read_csv(fp, header=0, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            # if first column name is numeric-looking, file is actually headerless
            try:
                int(str(df.columns[0]).replace(".", ""))
                n = len(df.columns)
                df = pd.read_csv(fp, header=None, low_memory=False)
                if n >= 7:
                    df.columns = _KT4_HEADER_COLS[:len(df.columns)]
                elif n == 3:
                    df.columns = _KT2_POSITIONAL[:len(df.columns)]
                else:
                    df.columns = _KT1_POSITIONAL[:len(df.columns)]
            except (ValueError, IndexError):
                pass
        elif schema == "kt4_positional":
            df = pd.read_csv(fp, header=None, low_memory=False)
            df.columns = _KT4_HEADER_COLS[:len(df.columns)]
        elif schema == "kt3_header":
            df = pd.read_csv(fp, header=0, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
        elif schema == "kt3_positional":
            df = pd.read_csv(fp, header=None, low_memory=False)
            df.columns = _KT3_HEADER_COLS[:len(df.columns)]
        elif schema == "kt2_positional":
            df = pd.read_csv(fp, header=None, low_memory=False)
            df.columns = _KT2_POSITIONAL[:len(df.columns)]
        elif schema == "kt1_positional":
            df = pd.read_csv(fp, header=None, low_memory=False)
            df.columns = _KT1_POSITIONAL[:len(df.columns)]
        else:
            df = pd.read_csv(fp, header=0, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]

        # ── column normalisations ─────────────────────────────────────────────
        # is_correct → correct
        if "is_correct" in df.columns and "correct" not in df.columns:
            df = df.rename(columns={"is_correct": "correct"})
        # first_correct → correct
        if "correct" not in df.columns and "first_correct" in df.columns:
            df = df.rename(columns={"first_correct": "correct"})
        # KT1/KT2 have no correctness signal — synthesise as NaN (filtered downstream)
        if "correct" not in df.columns:
            df["correct"] = np.nan

        # ── require timestamp ─────────────────────────────────────────────────
        if "timestamp" not in df.columns:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── derive elapsed_time from timestamp diffs if absent ─────────────────
        if "elapsed_time" not in df.columns:
            df["elapsed_time"] = df["timestamp"].diff().abs().fillna(0).astype(int)
            # replace 0 (first row, no prior) with median of non-zero values
            nonzero_med = int(df.loc[df["elapsed_time"] > 0, "elapsed_time"].median())
            nonzero_med = max(nonzero_med, 1000)  # floor at 1 second
            df.loc[df["elapsed_time"] == 0, "elapsed_time"] = nonzero_med
            warnings.warn(
                f"[data_loader] {fp.name}: elapsed_time absent; "
                "derived from timestamp diffs (proxy — interpret with caution)."
            )

        # ── derive lag_time if absent ─────────────────────────────────────────
        if "lag_time" not in df.columns:
            df["lag_time"] = df["timestamp"].diff().fillna(0).abs().astype(int)

        df["user_id"] = fp.stem
        return df

    except Exception as exc:
        warnings.warn(f"[data_loader] Failed to parse {fp.name}: {exc}")
        return None


# ── real-data loader ───────────────────────────────────────────────────────────
def load_ednet_kt4(data_dir: str | Path) -> pd.DataFrame:
    """
    Load EdNet (KT1–KT4) from a directory of per-user CSV files.
    Automatically detects the schema variant by sniffing the first file.
    Falls back to synthetic generation if the directory is absent or empty.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("u*.csv"))

    if not csv_files:
        warnings.warn(
            f"[data_loader] No EdNet CSV files found in '{data_dir}'. "
            "Falling back to synthetic data. Download the real corpus from "
            "https://github.com/riiid/ednet and place student CSVs in data/.",
            UserWarning,
            stacklevel=2,
        )
        return generate_synthetic_ednet(output_dir=data_dir)

    schema = _sniff_schema(csv_files[0])
    print(f"[data_loader] Detected schema variant: '{schema}' "
          f"(sniffed from {csv_files[0].name})")
    print(f"[data_loader] Loading {len(csv_files):,} user files from {data_dir} …")

    dfs = []
    n_failed = 0
    for fp in tqdm(csv_files, desc="Reading CSVs"):
        parsed = _parse_user_file(fp, schema)
        if parsed is not None:
            dfs.append(parsed)
        else:
            n_failed += 1

    if not dfs:
        raise RuntimeError(
            f"[data_loader] No user files could be parsed (schema='{schema}'). "
            "Check that data/ contains valid EdNet CSV files."
        )
    if n_failed:
        warnings.warn(f"[data_loader] {n_failed:,} files skipped due to parse errors.")

    df_all = pd.concat(dfs, ignore_index=True)

    # ── resolve correctness from questions.csv ───────────────────────────────
    # KT2/KT3/KT4 action-trace files have no correct column — it lives in
    # questions.csv as correct_answer. We join on question_id and compare
    # user_answer == correct_answer to produce the binary correct column.
    questions_candidates = [
        data_dir / "questions.csv",
        data_dir / "contents.csv",
        data_dir / "Questions.csv",
        data_dir / "Contents.csv",
    ]
    questions_path = next((p for p in questions_candidates if p.exists()), None)

    if questions_path is not None:
        try:
            questions = pd.read_csv(questions_path)
            questions.columns = [c.strip().lower() for c in questions.columns]

            # normalise correct_answer column name
            if "correct_answer" in questions.columns and "correct" not in questions.columns:
                questions = questions.rename(columns={"correct_answer": "correct_answer"})

            q_merge_cols = ["question_id"]
            for col in ["correct_answer", "bundle_id", "part"]:
                if col in questions.columns:
                    q_merge_cols.append(col)

            if "question_id" in questions.columns and "question_id" in df_all.columns:
                df_all = df_all.merge(questions[q_merge_cols], on="question_id", how="left")
                print(f"[data_loader] Merged questions metadata ({len(questions):,} questions).")

                # derive correct from user_answer == correct_answer
                if "correct_answer" in df_all.columns and "user_answer" in df_all.columns:
                    if df_all["correct"].isna().all():
                        df_all["correct"] = (
                            df_all["user_answer"].str.strip().str.lower() ==
                            df_all["correct_answer"].str.strip().str.lower()
                        ).astype(float)
                        n_resolved = df_all["correct"].notna().sum()
                        print(f"[data_loader] Resolved correctness for {n_resolved:,} interactions "
                              f"(accuracy: {df_all['correct'].mean():.3f}).")
        except Exception as e:
            warnings.warn(f"[data_loader] Could not merge questions metadata: {e}")
    else:
        warnings.warn(
            "[data_loader] questions.csv not found in data directory. "
            "Correctness cannot be resolved for action-trace format. "
            "Copy questions.csv from the EdNet Contents download into your data folder."
        )

    # ── merge bundle/part metadata if not already present ────────────────────
    contents_path = data_dir / "contents.csv" if not (data_dir / "questions.csv").exists() else None
    if contents_path and contents_path.exists():
        try:
            content = pd.read_csv(contents_path)
            content.columns = [c.strip().lower() for c in content.columns]
            merge_cols = [c for c in ["question_id", "bundle_id", "part"] if c in content.columns]
            if "question_id" in merge_cols and "question_id" in df_all.columns:
                df_all = df_all.merge(content[merge_cols], on="question_id", how="left")
        except Exception as e:
            warnings.warn(f"[data_loader] Could not merge contents.csv: {e}")

    if "bundle_id" not in df_all.columns:
        df_all["bundle_id"] = df_all.get(
            "question_id", pd.Series(["q0"] * len(df_all), index=df_all.index)
        ).astype(str).str.replace("q", "b", regex=False)
    if "part" not in df_all.columns:
        df_all["part"] = 1

    df_all[SYNTHETIC_TAG] = False
    print(f"[data_loader] Loaded {len(df_all):,} interactions from "
          f"{df_all['user_id'].nunique():,} users.")
    return df_all


# ── unified entry point ────────────────────────────────────────────────────────
def get_data(data_dir: str | Path = "data") -> pd.DataFrame:
    """
    Primary entry point.  Returns a cleaned interaction DataFrame
    regardless of whether real or synthetic data is available.
    """
    df = load_ednet_kt4(data_dir)

    # ── minimal cleaning ──────────────────────────────────────────────────────
    df["timestamp"]    = pd.to_numeric(df["timestamp"],    errors="coerce")
    df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
    df["lag_time"]     = pd.to_numeric(df["lag_time"],     errors="coerce")
    df["correct"]      = pd.to_numeric(df["correct"],      errors="coerce")

    before = len(df)
    df = df.dropna(subset=["timestamp", "elapsed_time", "correct"])
    df = df[df["elapsed_time"].between(100, 600_000)]   # 0.1s – 10min
    df = df[df["elapsed_time"] > 0]
    if len(df) == 0:
        raise RuntimeError(
            "[data_loader] All rows were dropped during cleaning. "
            "Most likely cause: questions.csv is missing so correctness could not "
            "be resolved. Ensure questions.csv is in your data directory."
        )
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    print(f"[data_loader] After cleaning: {len(df):,} rows "
          f"(dropped {before - len(df):,} invalid rows)")
    return df
