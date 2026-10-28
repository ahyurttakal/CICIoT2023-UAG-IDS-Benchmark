# 08_threshold_operational_analysis.py
"""
Threshold calibration and operational-metric analysis for the
CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script performs threshold calibration and operational analysis while
preserving the primary benchmark design.

For each held-out attack family and seed:

1. The held-out family is completely excluded from model development.
2. The remaining training data are split into an inner training subset and
   an inner validation subset.
3. Models are trained only on the inner training subset.
4. The classification threshold is selected ONLY on the inner validation
   subset.
5. The calibrated threshold is then applied once to the untouched LOAFO
   test set containing:
       - held-out attack-family samples
       - an equally sized benign subset

This avoids selecting thresholds on the held-out attack-family test set.

Metrics
-------
For both the default threshold (0.5) and calibrated threshold:

- Accuracy
- Macro-F1
- Balanced accuracy
- MCC
- Attack precision
- Attack recall
- Attack F1
- Benign recall
- False-positive rate
- False-negative rate
- False alarms per 1,000 benign flows

Probability-based metrics (threshold-independent):

- ROC-AUC
- PR-AUC / Average Precision

Calibration strategies
----------------------
Two threshold strategies are evaluated:

A. validation_macro_f1
   Select threshold maximizing binary Macro-F1 on the inner validation set.

B. validation_fpr_constrained
   Among thresholds whose validation FPR is <= MAX_VALIDATION_FPR,
   select the threshold with the highest attack recall.
   If no threshold satisfies the constraint, select the threshold with
   the lowest validation FPR, breaking ties by higher attack recall.

The original five supervised classifiers are retained:
- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM

No anomaly model is added here; anomaly-oriented evaluation is handled
separately in script 09.

Author note
-----------
This analysis is kept separate from the primary benchmark pipeline so that
threshold calibration remains an explicitly defined post-training evaluation step.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional dependencies
HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("mergeddata") / "ciciot2023_working_with_source.csv"

OUT_DIR = Path("CICIoT2023_THRESHOLD_OPERATIONAL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

SEEDS = [42, 123, 2024, 2025, 2026]

# Matches the primary benchmark configuration.
MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 150_000

LOAFO_TEST_PER_CLASS_CAP = 25_000
MIN_TEST_PER_CLASS = 1_000
MIN_TRAIN_BENIGN_REMAINING = 10_000

SAFE_N_JOBS = 1

# Inner validation is created only from the non-held-out training pool.
INNER_VALIDATION_SIZE = 0.20

# Threshold grid.
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)

# Operational constrained strategy.
MAX_VALIDATION_FPR = 0.10


# ============================================================
# LABEL MAPPING
# ============================================================

def clean_label(x):
    return str(x).strip().upper()


def is_benign(label):
    label = str(label).strip().upper()
    return label in ["BENIGN", "BENIGNTRAFFIC"] or "BENIGN" in label


def map_binary(label):
    return "BENIGN" if is_benign(label) else "ATTACK"


def map_attack_family(label):
    label = str(label).strip().upper()

    if is_benign(label):
        return "BENIGN"

    if label.startswith("DDOS"):
        return "DDOS"

    if label.startswith("DOS"):
        return "DOS"

    if label.startswith("MIRAI"):
        return "MIRAI"

    if label.startswith("RECON"):
        return "RECON"

    if label in ["DNS_SPOOFING", "MITM-ARPSPOOFING"]:
        return "SPOOFING"

    if label in [
        "BROWSERHIJACKING",
        "COMMANDINJECTION",
        "SQLINJECTION",
        "XSS",
        "UPLOADING_ATTACK",
        "BACKDOOR_MALWARE",
    ]:
        return "WEB_MALWARE"

    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTE_FORCE"

    if label == "VULNERABILITYSCAN":
        return "VULNERABILITY_SCAN"

    return "OTHER"


# ============================================================
# DATA / FEATURE HELPERS
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Update DATA_PATH if the working dataset is elsewhere."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if LABEL_COL not in df.columns:
        raise ValueError(f"Required label column not found: {LABEL_COL}")

    if SOURCE_COL not in df.columns:
        raise ValueError(f"Required provenance column not found: {SOURCE_COL}")

    df[LABEL_COL] = df[LABEL_COL].apply(clean_label)

    df = df[
        (df[LABEL_COL] != "NAN")
        & (df[LABEL_COL] != "")
        & (df[LABEL_COL].notna())
    ].copy()

    df = df.reset_index(drop=True)

    # Keep deterministic row IDs only for leakage checks.
    df[ROW_ID_COL] = np.arange(len(df))

    df["binary_label"] = df[LABEL_COL].apply(map_binary)
    df["attack_family"] = df[LABEL_COL].apply(map_attack_family)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matches the primary benchmark feature construction.
    """
    drop_cols = [
        LABEL_COL,
        SOURCE_COL,
        ROW_ID_COL,
        "binary_label",
        "attack_family",
        "coarse_label",
        "fine_label",
    ]

    X = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    ).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def safe_stratified_sample(
    df: pd.DataFrame,
    n: int | None,
    seed: int,
    stratify_col: str,
) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.copy()

    idx = np.arange(len(df))

    try:
        _, sampled_idx = train_test_split(
            idx,
            test_size=n,
            random_state=seed,
            stratify=df[stratify_col],
        )
    except Exception:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(
            idx,
            size=n,
            replace=False,
        )

    return df.iloc[sampled_idx].copy()


def compute_balanced_loafo_test_size(
    family_count: int,
    benign_count: int,
) -> int:
    max_benign_test = benign_count - MIN_TRAIN_BENIGN_REMAINING

    if max_benign_test < MIN_TEST_PER_CLASS:
        max_benign_test = int(benign_count * 0.5)

    n_test = min(
        family_count,
        max_benign_test,
        LOAFO_TEST_PER_CLASS_CAP,
    )

    n_test = int(n_test)

    if n_test < MIN_TEST_PER_CLASS:
        return 0

    return n_test


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def make_models(seed: int) -> Dict[str, object]:
    """
    Matches the fixed model configurations of the primary benchmark.
    """
    models: Dict[str, object] = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=SAFE_N_JOBS,
                random_state=seed,
            ),
        ),
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=150,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=SAFE_N_JOBS,
        random_state=seed,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=SAFE_N_JOBS,
        random_state=seed,
    )

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=SAFE_N_JOBS,
            random_state=seed,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=64,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary",
            class_weight="balanced",
            n_jobs=SAFE_N_JOBS,
            random_state=seed,
            verbose=-1,
        )

    return models


# ============================================================
# METRICS
# ============================================================

def labels_to_binary_int(labels: pd.Series) -> np.ndarray:
    """
    ATTACK = 1, BENIGN = 0
    """
    s = pd.Series(labels).astype(str).str.upper()
    return (s == "ATTACK").astype(int).to_numpy()


def threshold_predictions(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def binary_metrics_from_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = threshold_predictions(y_prob, threshold)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    attack_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    attack_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    attack_f1 = (
        2 * attack_precision * attack_recall
        / (attack_precision + attack_recall)
        if (attack_precision + attack_recall) > 0
        else 0.0
    )

    benign_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),
        "benign_recall": float(benign_recall),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_alarms_per_1000_benign": float(
            false_positive_rate * 1000.0
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    try:
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )
    except Exception:
        pr_auc = np.nan

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def validation_threshold_table(
    y_val: np.ndarray,
    p_val: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for threshold in THRESHOLDS:
        m = binary_metrics_from_scores(
            y_val,
            p_val,
            threshold,
        )

        rows.append(m)

    return pd.DataFrame(rows)


def choose_macro_f1_threshold(
    threshold_table: pd.DataFrame,
) -> Tuple[float, pd.Series]:
    """
    Maximize Macro-F1.
    Ties:
      1. lower FPR
      2. higher attack recall
      3. threshold closest to 0.5
    """
    t = threshold_table.copy()

    t["distance_to_05"] = (
        t["threshold"] - 0.5
    ).abs()

    t = t.sort_values(
        by=[
            "macro_f1",
            "false_positive_rate",
            "attack_recall",
            "distance_to_05",
        ],
        ascending=[
            False,
            True,
            False,
            True,
        ],
    )

    best = t.iloc[0]

    return float(best["threshold"]), best


def choose_fpr_constrained_threshold(
    threshold_table: pd.DataFrame,
) -> Tuple[float, pd.Series, bool]:
    """
    Primary rule:
      among thresholds with validation FPR <= MAX_VALIDATION_FPR,
      maximize attack recall.

    Tie-breakers:
      1. higher attack F1
      2. lower FPR
      3. threshold closest to 0.5

    Fallback:
      if no threshold satisfies the FPR constraint, minimize FPR,
      then maximize attack recall.
    """
    t = threshold_table.copy()

    t["distance_to_05"] = (
        t["threshold"] - 0.5
    ).abs()

    feasible = t[
        t["false_positive_rate"]
        <= MAX_VALIDATION_FPR
    ].copy()

    if not feasible.empty:
        feasible = feasible.sort_values(
            by=[
                "attack_recall",
                "attack_f1",
                "false_positive_rate",
                "distance_to_05",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

        best = feasible.iloc[0]

        return float(best["threshold"]), best, True

    fallback = t.sort_values(
        by=[
            "false_positive_rate",
            "attack_recall",
            "attack_f1",
            "distance_to_05",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    best = fallback.iloc[0]

    return float(best["threshold"]), best, False


# ============================================================
# LOAFO PREPARATION
# ============================================================

def prepare_loafo_data(
    df: pd.DataFrame,
    held_out_family: str,
    seed: int,
):
    family_df_all = df[
        df["attack_family"] == held_out_family
    ].copy()

    benign_df_all = df[
        df["binary_label"] == "BENIGN"
    ].copy()

    n_test_per_class = compute_balanced_loafo_test_size(
        len(family_df_all),
        len(benign_df_all),
    )

    if n_test_per_class == 0:
        return None

    test_attack_df = family_df_all.sample(
        n=n_test_per_class,
        random_state=seed,
    ).copy()

    test_benign_df = benign_df_all.sample(
        n=n_test_per_class,
        random_state=seed,
    ).copy()

    test_df = pd.concat(
        [
            test_attack_df,
            test_benign_df,
        ],
        axis=0,
    ).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    test_ids = set(
        test_df[ROW_ID_COL].tolist()
    )

    train_pool = df[
        (df["attack_family"] != held_out_family)
        & (~df[ROW_ID_COL].isin(test_ids))
    ].copy()

    # Leakage checks.
    if held_out_family in set(
        train_pool["attack_family"].unique()
    ):
        raise RuntimeError(
            f"LOAFO leakage detected: "
            f"{held_out_family} remains in training pool."
        )

    overlap = set(
        train_pool[ROW_ID_COL]
    ).intersection(
        set(test_df[ROW_ID_COL])
    )

    if overlap:
        raise RuntimeError(
            f"Row overlap detected for {held_out_family}."
        )

    # Stratified cap matches the primary benchmark.
    train_pool = safe_stratified_sample(
        train_pool,
        MAX_TRAIN_ROWS,
        seed,
        stratify_col="binary_label",
    )

    test_df = safe_stratified_sample(
        test_df,
        MAX_TEST_ROWS,
        seed,
        stratify_col="binary_label",
    )

    # Inner development split: only from non-held-out training pool.
    dev_train_df, val_df = train_test_split(
        train_pool,
        test_size=INNER_VALIDATION_SIZE,
        random_state=seed,
        stratify=train_pool["binary_label"],
    )

    # Strong leakage check:
    # held-out family must be absent from BOTH inner subsets.
    for name, part in [
        ("dev_train", dev_train_df),
        ("validation", val_df),
    ]:
        if held_out_family in set(
            part["attack_family"].unique()
        ):
            raise RuntimeError(
                f"Held-out family leakage into {name}: "
                f"{held_out_family}"
            )

    X_dev = get_feature_matrix(dev_train_df)
    X_val = get_feature_matrix(val_df)
    X_test = get_feature_matrix(test_df)

    y_dev = labels_to_binary_int(
        dev_train_df["binary_label"]
    )
    y_val = labels_to_binary_int(
        val_df["binary_label"]
    )
    y_test = labels_to_binary_int(
        test_df["binary_label"]
    )

    meta = {
        "held_out_family": held_out_family,
        "seed": seed,
        "loafo_test_per_class": int(n_test_per_class),
        "dev_train_rows": int(len(dev_train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "test_attack_rows": int(
            (test_df["binary_label"] == "ATTACK").sum()
        ),
        "test_benign_rows": int(
            (test_df["binary_label"] == "BENIGN").sum()
        ),
        "validation_attack_rows": int(
            (val_df["binary_label"] == "ATTACK").sum()
        ),
        "validation_benign_rows": int(
            (val_df["binary_label"] == "BENIGN").sum()
        ),
        "held_out_family_absent_from_dev_train": True,
        "held_out_family_absent_from_validation": True,
        "train_test_row_overlap": 0,
    }

    return (
        X_dev,
        X_val,
        X_test,
        y_dev,
        y_val,
        y_test,
        meta,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def positive_class_probability(
    model,
    X,
) -> np.ndarray:
    """
    Return P(ATTACK=1).
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{type(model).__name__} does not support predict_proba()."
        )

    probs = model.predict_proba(X)

    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(
            "Expected binary predict_proba output with 2 columns."
        )

    return probs[:, 1].astype(float)


def evaluate_model(
    model_name: str,
    model,
    X_dev,
    X_val,
    X_test,
    y_dev,
    y_val,
    y_test,
    meta: Dict,
):
    # Training time is recorded because this rerun is already needed.
    train_start = time.perf_counter()
    model.fit(X_dev, y_dev)
    training_seconds = time.perf_counter() - train_start

    # Validation scores.
    val_start = time.perf_counter()
    p_val = positive_class_probability(
        model,
        X_val,
    )
    validation_inference_seconds = (
        time.perf_counter() - val_start
    )

    threshold_table = validation_threshold_table(
        y_val,
        p_val,
    )

    macro_threshold, macro_val_row = (
        choose_macro_f1_threshold(
            threshold_table
        )
    )

    (
        fpr_threshold,
        fpr_val_row,
        constraint_satisfied,
    ) = choose_fpr_constrained_threshold(
        threshold_table
    )

    # Untouched held-out family test set.
    test_start = time.perf_counter()
    p_test = positive_class_probability(
        model,
        X_test,
    )
    test_inference_seconds = (
        time.perf_counter() - test_start
    )

    prob_metrics = probability_metrics(
        y_test,
        p_test,
    )

    strategies = {
        "default_0.5": 0.5,
        "validation_macro_f1": macro_threshold,
        "validation_fpr_constrained": fpr_threshold,
    }

    rows = []

    for strategy, threshold in strategies.items():
        metrics = binary_metrics_from_scores(
            y_test,
            p_test,
            threshold,
        )

        row = {
            **meta,
            "model": model_name,
            "threshold_strategy": strategy,
            "selected_threshold": float(threshold),
            **metrics,
            **prob_metrics,
            "training_seconds": float(training_seconds),
            "validation_inference_seconds": float(
                validation_inference_seconds
            ),
            "test_inference_seconds": float(
                test_inference_seconds
            ),
            "test_inference_ms_per_flow": float(
                (test_inference_seconds / len(X_test))
                * 1000.0
            ),
            "test_throughput_flows_per_second": float(
                len(X_test) / test_inference_seconds
                if test_inference_seconds > 0
                else np.nan
            ),
        }

        if strategy == "validation_macro_f1":
            row.update({
                "validation_selected_macro_f1": float(
                    macro_val_row["macro_f1"]
                ),
                "validation_selected_fpr": float(
                    macro_val_row["false_positive_rate"]
                ),
                "validation_selected_attack_recall": float(
                    macro_val_row["attack_recall"]
                ),
                "validation_fpr_constraint_satisfied": np.nan,
            })

        elif strategy == "validation_fpr_constrained":
            row.update({
                "validation_selected_macro_f1": float(
                    fpr_val_row["macro_f1"]
                ),
                "validation_selected_fpr": float(
                    fpr_val_row["false_positive_rate"]
                ),
                "validation_selected_attack_recall": float(
                    fpr_val_row["attack_recall"]
                ),
                "validation_fpr_constraint_satisfied": bool(
                    constraint_satisfied
                ),
            })

        else:
            default_val = binary_metrics_from_scores(
                y_val,
                p_val,
                0.5,
            )

            row.update({
                "validation_selected_macro_f1": float(
                    default_val["macro_f1"]
                ),
                "validation_selected_fpr": float(
                    default_val["false_positive_rate"]
                ),
                "validation_selected_attack_recall": float(
                    default_val["attack_recall"]
                ),
                "validation_fpr_constraint_satisfied": np.nan,
            })

        rows.append(row)

    # Add identifiers to threshold table for optional diagnostics.
    threshold_table = threshold_table.copy()
    threshold_table["held_out_family"] = meta[
        "held_out_family"
    ]
    threshold_table["seed"] = meta["seed"]
    threshold_table["model"] = model_name

    return rows, threshold_table


# ============================================================
# SUMMARY TABLES
# ============================================================

def create_summary_tables(
    results_df: pd.DataFrame,
):
    metric_cols = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "benign_recall",
        "false_positive_rate",
        "false_negative_rate",
        "false_alarms_per_1000_benign",
        "roc_auc",
        "pr_auc",
        "selected_threshold",
        "training_seconds",
        "test_inference_ms_per_flow",
        "test_throughput_flows_per_second",
    ]

    summary = (
        results_df.groupby(
            ["model", "threshold_strategy"]
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(v) for v in col if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    summary.to_csv(
        OUT_DIR / "operational_metrics_model_summary.csv",
        index=False,
    )

    family_summary = (
        results_df.groupby(
            [
                "held_out_family",
                "model",
                "threshold_strategy",
            ]
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    family_summary.columns = [
        "_".join(
            str(v) for v in col if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in family_summary.columns
    ]

    family_summary.to_csv(
        OUT_DIR / "operational_metrics_by_family.csv",
        index=False,
    )

    # Direct comparison: default vs calibrated.
    key_cols = [
        "held_out_family",
        "seed",
        "model",
    ]

    selected_metrics = [
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "false_positive_rate",
        "false_alarms_per_1000_benign",
        "macro_f1",
        "mcc",
    ]

    comparison_parts = []

    for strategy in [
        "validation_macro_f1",
        "validation_fpr_constrained",
    ]:
        default_df = results_df[
            results_df["threshold_strategy"]
            == "default_0.5"
        ][key_cols + selected_metrics].copy()

        calibrated_df = results_df[
            results_df["threshold_strategy"]
            == strategy
        ][
            key_cols
            + selected_metrics
            + ["selected_threshold"]
        ].copy()

        merged = default_df.merge(
            calibrated_df,
            on=key_cols,
            suffixes=("_default", "_calibrated"),
        )

        merged["calibration_strategy"] = strategy

        for metric in selected_metrics:
            merged[f"{metric}_change"] = (
                merged[f"{metric}_calibrated"]
                - merged[f"{metric}_default"]
            )

        comparison_parts.append(merged)

    comparison = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    comparison.to_csv(
        OUT_DIR / "default_vs_calibrated_threshold_comparison.csv",
        index=False,
    )

    comparison_summary = (
        comparison.groupby(
            ["model", "calibration_strategy"]
        )[
            [
                "selected_threshold",
                *[
                    f"{m}_change"
                    for m in selected_metrics
                ],
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    comparison_summary.columns = [
        "_".join(
            str(v) for v in col if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in comparison_summary.columns
    ]

    comparison_summary.to_csv(
        OUT_DIR / "threshold_calibration_change_summary.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    family_counts = df[
        "attack_family"
    ].value_counts()

    attack_families = [
        fam
        for fam in family_counts.index
        if fam not in ["BENIGN", "OTHER"]
        and family_counts[fam] >= MIN_TEST_PER_CLASS
    ]

    print("Attack families:")
    for fam in attack_families:
        print(
            f"  {fam}: {family_counts[fam]:,}"
        )

    available_models = list(
        make_models(SEEDS[0]).keys()
    )

    print("\nModels:")
    for model_name in available_models:
        print(f"  {model_name}")

    if not HAS_XGBOOST:
        print(
            "\nWARNING: xgboost is not installed; "
            "XGBoost will be skipped."
        )

    if not HAS_LIGHTGBM:
        print(
            "\nWARNING: lightgbm is not installed; "
            "LightGBM will be skipped."
        )

    all_rows = []
    all_threshold_rows = []

    for seed in SEEDS:
        for held_out_family in attack_families:
            print()
            print("=" * 78)
            print(
                f"LOAFO THRESHOLD ANALYSIS | "
                f"family={held_out_family} | seed={seed}"
            )
            print("=" * 78)

            prepared = prepare_loafo_data(
                df,
                held_out_family,
                seed,
            )

            if prepared is None:
                print("Skipped: insufficient samples.")
                continue

            (
                X_dev,
                X_val,
                X_test,
                y_dev,
                y_val,
                y_test,
                meta,
            ) = prepared

            models = make_models(seed)

            for model_name, model in models.items():
                print(
                    f"Training {model_name}..."
                )

                try:
                    rows, threshold_table = evaluate_model(
                        model_name=model_name,
                        model=model,
                        X_dev=X_dev,
                        X_val=X_val,
                        X_test=X_test,
                        y_dev=y_dev,
                        y_val=y_val,
                        y_test=y_test,
                        meta=meta,
                    )

                    all_rows.extend(rows)
                    all_threshold_rows.append(
                        threshold_table
                    )

                    for row in rows:
                        print(
                            f"  {row['threshold_strategy']}: "
                            f"t={row['selected_threshold']:.2f}, "
                            f"Recall={row['attack_recall']:.4f}, "
                            f"FPR={row['false_positive_rate']:.4f}, "
                            f"AttackF1={row['attack_f1']:.4f}"
                        )

                except Exception as exc:
                    print(
                        f"ERROR | {model_name} | "
                        f"{held_out_family} | seed={seed}: "
                        f"{exc}"
                    )

                gc.collect()

            # Progressive save.
            pd.DataFrame(all_rows).to_csv(
                OUT_DIR / "threshold_operational_results_partial.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_rows)

    if results_df.empty:
        raise RuntimeError(
            "No model results were produced."
        )

    results_df.to_csv(
        OUT_DIR / "threshold_operational_results_all.csv",
        index=False,
    )

    if all_threshold_rows:
        threshold_df = pd.concat(
            all_threshold_rows,
            ignore_index=True,
        )

        threshold_df.to_csv(
            OUT_DIR / "validation_threshold_grid_results.csv",
            index=False,
        )

    create_summary_tables(
        results_df
    )

    # Manifest.
    manifest = {
        "dataset_path": str(DATA_PATH),
        "held_out_attack_families": attack_families,
        "seeds": SEEDS,
        "models_available": sorted(
            results_df["model"]
            .dropna()
            .unique()
            .tolist()
        ),
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "threshold_grid_min": float(
            THRESHOLDS.min()
        ),
        "threshold_grid_max": float(
            THRESHOLDS.max()
        ),
        "threshold_grid_step": 0.01,
        "fpr_constraint": MAX_VALIDATION_FPR,
        "threshold_selection_rule": (
            "Thresholds are selected only on an inner validation "
            "subset drawn from the non-held-out training pool. "
            "The held-out attack family is absent from model fitting "
            "and threshold calibration."
        ),
        "probability_metrics": [
            "ROC-AUC",
            "PR-AUC / Average Precision",
        ],
        "operational_metrics": [
            "attack precision",
            "attack recall",
            "attack F1",
            "false-positive rate",
            "false-negative rate",
            "false alarms per 1,000 benign flows",
        ],
    }

    with open(
        OUT_DIR / "threshold_operational_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Human-readable summary.
    model_summary_path = (
        OUT_DIR
        / "operational_metrics_model_summary.csv"
    )

    summary_lines = [
        "=" * 80,
        "CICIoT2023 THRESHOLD CALIBRATION AND OPERATIONAL ANALYSIS",
        "=" * 80,
        f"Dataset                     : {DATA_PATH}",
        f"Attack families             : {len(attack_families)}",
        f"Seeds                       : {len(SEEDS)}",
        (
            "Models evaluated            : "
            + ", ".join(
                sorted(
                    results_df["model"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )
        ),
        f"Inner validation fraction   : {INNER_VALIDATION_SIZE:.2f}",
        (
            "Threshold grid              : "
            f"{THRESHOLDS.min():.2f} to "
            f"{THRESHOLDS.max():.2f} "
            f"(step=0.01)"
        ),
        (
            "FPR constraint              : "
            f"{MAX_VALIDATION_FPR:.2f}"
        ),
        "",
        "LEAKAGE CONTROL",
        "-" * 80,
        (
            "Threshold selection is performed only on an inner validation "
            "subset drawn from the non-held-out training pool."
        ),
        (
            "The held-out attack family is absent from both model training "
            "and threshold calibration."
        ),
        (
            "The untouched balanced LOAFO test set is used only for final "
            "evaluation."
        ),
        "",
        "THRESHOLD STRATEGIES",
        "-" * 80,
        "1. default_0.5",
        "2. validation_macro_f1",
        "3. validation_fpr_constrained",
        "",
        "OUTPUTS",
        "-" * 80,
        "- threshold_operational_results_all.csv",
        "- validation_threshold_grid_results.csv",
        "- operational_metrics_model_summary.csv",
        "- operational_metrics_by_family.csv",
        "- default_vs_calibrated_threshold_comparison.csv",
        "- threshold_calibration_change_summary.csv",
        "- threshold_operational_manifest.json",
        "- threshold_operational_summary.txt",
        "",
        f"Primary summary table: {model_summary_path}",
        "=" * 80,
    ]

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUT_DIR / "threshold_operational_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary_text)

    print()
    print(summary_text)
    print()
    print(
        f"Output directory: "
        f"{OUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
