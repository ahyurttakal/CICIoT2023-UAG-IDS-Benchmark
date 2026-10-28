# 09_anomaly_baseline.py
"""
Anomaly-oriented baseline analysis for the CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script adds a one-class / anomaly-detection reference requested during
the study, while keeping the supervised benchmark unchanged.

Method
------
For each held-out attack family and seed:

1. The held-out attack family is completely excluded from model development.
2. Only BENIGN samples from the non-held-out development pool are used to fit
   the anomaly detector.
3. An inner validation set is drawn from the non-held-out data and is used
   only to select an anomaly-score threshold.
4. The untouched balanced LOAFO test set contains:
       - samples from the held-out attack family
       - an equally sized benign subset
5. Threshold selection never uses the held-out test set.

Implemented anomaly baseline
----------------------------
Isolation Forest

Why Isolation Forest?
---------------------
It is computationally practical for the current dataset size and provides a
clear unsupervised / one-class reference without broadening the manuscript
into a deep-anomaly-detection study.

Threshold strategies
--------------------
A. default_iforest
   Uses the detector's native decision boundary.

B. validation_macro_f1
   Selects a threshold on validation anomaly scores that maximizes binary
   Macro-F1.

C. validation_fpr_constrained
   Selects a threshold on validation data with FPR <= MAX_VALIDATION_FPR,
   maximizing attack recall within that feasible set.

Metrics
-------
- Accuracy
- Macro-F1
- Balanced accuracy
- MCC
- Attack precision
- Attack recall
- Attack F1
- Benign recall
- FPR / FNR
- False alarms per 1,000 benign flows
- ROC-AUC
- PR-AUC

No supervised classifier is retrained here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("mergeddata") / "ciciot2023_working_with_source.csv"

OUT_DIR = Path("CICIoT2023_ANOMALY_BASELINE")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

SEEDS = [42, 123, 2024, 2025, 2026]

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 150_000

LOAFO_TEST_PER_CLASS_CAP = 25_000
MIN_TEST_PER_CLASS = 1_000
MIN_TRAIN_BENIGN_REMAINING = 10_000

INNER_VALIDATION_SIZE = 0.20

# Isolation Forest settings
N_ESTIMATORS = 300
MAX_SAMPLES = "auto"
CONTAMINATION = "auto"
N_JOBS = 1

# Threshold grid is generated from validation score quantiles.
N_THRESHOLD_CANDIDATES = 181

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
# DATA HELPERS
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Update DATA_PATH if needed."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if SOURCE_COL not in df.columns:
        raise ValueError(f"Missing provenance column: {SOURCE_COL}")

    df[LABEL_COL] = df[LABEL_COL].apply(clean_label)

    df = df[
        (df[LABEL_COL] != "")
        & (df[LABEL_COL] != "NAN")
        & (df[LABEL_COL].notna())
    ].copy()

    df = df.reset_index(drop=True)
    df[ROW_ID_COL] = np.arange(len(df))

    df["binary_label"] = df[LABEL_COL].apply(map_binary)
    df["attack_family"] = df[LABEL_COL].apply(map_attack_family)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
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
# METRICS
# ============================================================

def labels_to_binary_int(labels: pd.Series) -> np.ndarray:
    s = pd.Series(labels).astype(str).str.upper()
    return (s == "ATTACK").astype(int).to_numpy()


def binary_metrics_from_scores(
    y_true: np.ndarray,
    anomaly_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Higher anomaly score => more attack-like.
    Predict ATTACK when score >= threshold.
    """
    y_pred = (anomaly_score >= threshold).astype(int)

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

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "false_alarms_per_1000_benign": float(fpr * 1000.0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def probability_metrics(
    y_true: np.ndarray,
    anomaly_score: np.ndarray,
) -> Dict[str, float]:
    try:
        roc_auc = float(
            roc_auc_score(
                y_true,
                anomaly_score,
            )
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = float(
            average_precision_score(
                y_true,
                anomaly_score,
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

def make_threshold_candidates(
    scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No finite validation scores.")

    quantiles = np.linspace(
        0.0,
        1.0,
        N_THRESHOLD_CANDIDATES,
    )

    thresholds = np.quantile(
        scores,
        quantiles,
    )

    thresholds = np.unique(
        thresholds.astype(float)
    )

    return thresholds


def threshold_table(
    y_val: np.ndarray,
    scores_val: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for threshold in make_threshold_candidates(
        scores_val
    ):
        row = binary_metrics_from_scores(
            y_val,
            scores_val,
            threshold,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def choose_macro_f1_threshold(
    table: pd.DataFrame,
) -> Tuple[float, pd.Series]:
    t = table.copy()

    # Prefer threshold nearest zero if exact metric ties occur.
    t["distance_to_zero"] = t["threshold"].abs()

    t = t.sort_values(
        by=[
            "macro_f1",
            "false_positive_rate",
            "attack_recall",
            "distance_to_zero",
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
    table: pd.DataFrame,
) -> Tuple[float, pd.Series, bool]:
    t = table.copy()
    t["distance_to_zero"] = t["threshold"].abs()

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
                "distance_to_zero",
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
            "distance_to_zero",
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

    development_pool = df[
        (df["attack_family"] != held_out_family)
        & (~df[ROW_ID_COL].isin(test_ids))
    ].copy()

    if held_out_family in set(
        development_pool["attack_family"].unique()
    ):
        raise RuntimeError(
            f"Held-out family leakage: {held_out_family}"
        )

    overlap = set(
        development_pool[ROW_ID_COL]
    ).intersection(
        set(test_df[ROW_ID_COL])
    )

    if overlap:
        raise RuntimeError(
            f"Train/test row overlap detected: {held_out_family}"
        )

    # Keep the balanced LOAFO test set unchanged.
    test_df = safe_stratified_sample(
        test_df,
        MAX_TEST_ROWS,
        seed,
        stratify_col="binary_label",
    )

    # --------------------------------------------------------
    # Anomaly-specific development split
    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT apply the supervised benchmark's global 300k stratified
    # development cap here. In a one-class detector this can unnecessarily
    # remove benign rows before the benign-only training set is formed.
    #
    # Instead:
    #   1) preserve the available benign development pool,
    #   2) reserve a small benign validation subset,
    #   3) sample an equally sized seen-attack validation subset,
    #   4) fit Isolation Forest on the remaining benign rows only.
    #
    # This preserves >= MIN_TRAIN_BENIGN_REMAINING benign training samples
    # while keeping threshold calibration balanced and computationally light.

    benign_dev_pool = development_pool[
        development_pool["binary_label"] == "BENIGN"
    ].copy()

    seen_attack_pool = development_pool[
        development_pool["binary_label"] == "ATTACK"
    ].copy()

    max_val_benign = (
        len(benign_dev_pool) - MIN_TRAIN_BENIGN_REMAINING
    )

    if max_val_benign < MIN_TEST_PER_CLASS:
        raise RuntimeError(
            "Insufficient benign development samples after reserving "
            f"{MIN_TRAIN_BENIGN_REMAINING} benign rows for one-class training. "
            f"Available benign development rows: {len(benign_dev_pool)}"
        )

    desired_val_benign = max(
        MIN_TEST_PER_CLASS,
        int(round(len(benign_dev_pool) * INNER_VALIDATION_SIZE)),
    )

    n_val_benign = min(
        desired_val_benign,
        max_val_benign,
        len(seen_attack_pool),
    )

    val_benign_df = benign_dev_pool.sample(
        n=n_val_benign,
        random_state=seed,
    ).copy()

    val_attack_df = seen_attack_pool.sample(
        n=n_val_benign,
        random_state=seed,
    ).copy()

    val_df = pd.concat(
        [val_benign_df, val_attack_df],
        axis=0,
    ).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    val_benign_ids = set(
        val_benign_df[ROW_ID_COL].tolist()
    )

    # One-class training: BENIGN ONLY, excluding validation benign rows.
    benign_train_df = benign_dev_pool[
        ~benign_dev_pool[ROW_ID_COL].isin(val_benign_ids)
    ].copy()

    if len(benign_train_df) < MIN_TRAIN_BENIGN_REMAINING:
        raise RuntimeError(
            f"Too few benign training rows: {len(benign_train_df)}"
        )

    if held_out_family in set(
        val_df["attack_family"].unique()
    ):
        raise RuntimeError(
            "Held-out family leaked into anomaly validation set."
        )

    X_benign_train = get_feature_matrix(
        benign_train_df
    )

    X_val = get_feature_matrix(
        val_df
    )

    X_test = get_feature_matrix(
        test_df
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
        "one_class_training": "BENIGN_ONLY",
        "benign_train_rows": int(
            len(benign_train_df)
        ),
        "validation_rows": int(
            len(val_df)
        ),
        "validation_attack_rows": int(
            (val_df["binary_label"] == "ATTACK").sum()
        ),
        "validation_benign_rows": int(
            (val_df["binary_label"] == "BENIGN").sum()
        ),
        "validation_sampling": "BALANCED_SEEN_ATTACK_VS_BENIGN",
        "benign_development_rows_before_validation": int(
            len(benign_dev_pool)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "test_attack_rows": int(
            (test_df["binary_label"] == "ATTACK").sum()
        ),
        "test_benign_rows": int(
            (test_df["binary_label"] == "BENIGN").sum()
        ),
        "held_out_family_absent_from_training": True,
        "held_out_family_absent_from_validation": True,
        "train_test_row_overlap": 0,
    }

    return (
        X_benign_train,
        X_val,
        X_test,
        y_val,
        y_test,
        meta,
    )


# ============================================================
# MODEL
# ============================================================

def make_isolation_forest(
    seed: int,
) -> IsolationForest:
    return IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        random_state=seed,
        n_jobs=N_JOBS,
    )


def anomaly_score(
    model: IsolationForest,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    sklearn IsolationForest decision_function:
        higher = more normal
        lower = more anomalous

    We negate it so:
        higher = more attack-like / anomalous
    """
    return -model.decision_function(
        X_scaled
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_one_family_seed(
    X_benign_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    meta: Dict,
):
    scaler = StandardScaler()

    fit_scale_start = time.perf_counter()
    X_train_scaled = scaler.fit_transform(
        X_benign_train
    )
    X_val_scaled = scaler.transform(
        X_val
    )
    X_test_scaled = scaler.transform(
        X_test
    )
    scaling_seconds = (
        time.perf_counter()
        - fit_scale_start
    )

    model = make_isolation_forest(
        meta["seed"]
    )

    train_start = time.perf_counter()
    model.fit(
        X_train_scaled
    )
    training_seconds = (
        time.perf_counter()
        - train_start
    )

    val_start = time.perf_counter()
    val_scores = anomaly_score(
        model,
        X_val_scaled,
    )
    validation_inference_seconds = (
        time.perf_counter()
        - val_start
    )

    table = threshold_table(
        y_val,
        val_scores,
    )

    macro_threshold, macro_row = (
        choose_macro_f1_threshold(
            table
        )
    )

    (
        fpr_threshold,
        fpr_row,
        fpr_constraint_satisfied,
    ) = choose_fpr_constrained_threshold(
        table
    )

    test_start = time.perf_counter()
    test_scores = anomaly_score(
        model,
        X_test_scaled,
    )
    test_inference_seconds = (
        time.perf_counter()
        - test_start
    )

    prob_metrics = probability_metrics(
        y_test,
        test_scores,
    )

    # Native IF boundary:
    # decision_function < 0 => anomaly
    # because anomaly_score = -decision_function,
    # equivalent threshold is 0.
    strategies = {
        "default_iforest": 0.0,
        "validation_macro_f1": macro_threshold,
        "validation_fpr_constrained": fpr_threshold,
    }

    rows = []

    for strategy, threshold in strategies.items():
        metrics = binary_metrics_from_scores(
            y_test,
            test_scores,
            threshold,
        )

        row = {
            **meta,
            "model": "IsolationForest",
            "threshold_strategy": strategy,
            "selected_threshold": float(
                threshold
            ),
            **metrics,
            **prob_metrics,
            "scaling_seconds": float(
                scaling_seconds
            ),
            "training_seconds": float(
                training_seconds
            ),
            "validation_inference_seconds": float(
                validation_inference_seconds
            ),
            "test_inference_seconds": float(
                test_inference_seconds
            ),
            "test_inference_ms_per_flow": float(
                test_inference_seconds
                / len(X_test)
                * 1000.0
            ),
            "test_throughput_flows_per_second": float(
                len(X_test)
                / test_inference_seconds
                if test_inference_seconds > 0
                else np.nan
            ),
        }

        if strategy == "validation_macro_f1":
            row.update({
                "validation_selected_macro_f1": float(
                    macro_row["macro_f1"]
                ),
                "validation_selected_fpr": float(
                    macro_row["false_positive_rate"]
                ),
                "validation_selected_attack_recall": float(
                    macro_row["attack_recall"]
                ),
                "validation_fpr_constraint_satisfied": np.nan,
            })

        elif strategy == "validation_fpr_constrained":
            row.update({
                "validation_selected_macro_f1": float(
                    fpr_row["macro_f1"]
                ),
                "validation_selected_fpr": float(
                    fpr_row["false_positive_rate"]
                ),
                "validation_selected_attack_recall": float(
                    fpr_row["attack_recall"]
                ),
                "validation_fpr_constraint_satisfied": bool(
                    fpr_constraint_satisfied
                ),
            })

        else:
            default_val = binary_metrics_from_scores(
                y_val,
                val_scores,
                0.0,
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

    table = table.copy()
    table["held_out_family"] = meta[
        "held_out_family"
    ]
    table["seed"] = meta[
        "seed"
    ]
    table["model"] = "IsolationForest"

    return rows, table


# ============================================================
# SUMMARY
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
            "threshold_strategy"
        )[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(v) for v in col
            if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    summary.to_csv(
        OUT_DIR / "anomaly_baseline_summary.csv",
        index=False,
    )

    by_family = (
        results_df.groupby(
            [
                "held_out_family",
                "threshold_strategy",
            ]
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    by_family.columns = [
        "_".join(
            str(v) for v in col
            if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in by_family.columns
    ]

    by_family.to_csv(
        OUT_DIR / "anomaly_baseline_by_family.csv",
        index=False,
    )

    # Compare default against calibrated thresholds.
    key_cols = [
        "held_out_family",
        "seed",
    ]

    selected_metrics = [
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "false_positive_rate",
        "false_alarms_per_1000_benign",
    ]

    default = results_df[
        results_df["threshold_strategy"]
        == "default_iforest"
    ][
        key_cols + selected_metrics
    ].copy()

    comparison_parts = []

    for strategy in [
        "validation_macro_f1",
        "validation_fpr_constrained",
    ]:
        calibrated = results_df[
            results_df["threshold_strategy"]
            == strategy
        ][
            key_cols
            + selected_metrics
            + ["selected_threshold"]
        ].copy()

        merged = default.merge(
            calibrated,
            on=key_cols,
            suffixes=(
                "_default",
                "_calibrated",
            ),
        )

        merged["calibration_strategy"] = strategy

        for metric in selected_metrics:
            merged[
                f"{metric}_change"
            ] = (
                merged[
                    f"{metric}_calibrated"
                ]
                - merged[
                    f"{metric}_default"
                ]
            )

        comparison_parts.append(
            merged
        )

    comparison = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    comparison.to_csv(
        OUT_DIR
        / "anomaly_default_vs_calibrated.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading dataset...")
    df = load_dataset(
        DATA_PATH
    )

    family_counts = df[
        "attack_family"
    ].value_counts()

    attack_families = [
        fam
        for fam in family_counts.index
        if fam not in [
            "BENIGN",
            "OTHER",
        ]
        and family_counts[fam]
        >= MIN_TEST_PER_CLASS
    ]

    print("Attack families:")
    for fam in attack_families:
        print(
            f"  {fam}: "
            f"{family_counts[fam]:,}"
        )

    all_rows = []
    all_threshold_rows = []

    for seed in SEEDS:
        for held_out_family in attack_families:
            print()
            print("=" * 78)
            print(
                f"ISOLATION FOREST LOAFO | "
                f"family={held_out_family} | "
                f"seed={seed}"
            )
            print("=" * 78)

            prepared = prepare_loafo_data(
                df,
                held_out_family,
                seed,
            )

            if prepared is None:
                print(
                    "Skipped: insufficient samples."
                )
                continue

            (
                X_benign_train,
                X_val,
                X_test,
                y_val,
                y_test,
                meta,
            ) = prepared

            try:
                rows, threshold_table_df = (
                    evaluate_one_family_seed(
                        X_benign_train,
                        X_val,
                        X_test,
                        y_val,
                        y_test,
                        meta,
                    )
                )

                all_rows.extend(
                    rows
                )

                all_threshold_rows.append(
                    threshold_table_df
                )

                for row in rows:
                    print(
                        f"  {row['threshold_strategy']}: "
                        f"t={row['selected_threshold']:.6f}, "
                        f"Recall={row['attack_recall']:.4f}, "
                        f"FPR={row['false_positive_rate']:.4f}, "
                        f"AttackF1={row['attack_f1']:.4f}"
                    )

            except Exception as exc:
                print(
                    f"ERROR | "
                    f"{held_out_family} | "
                    f"seed={seed}: "
                    f"{exc}"
                )

            pd.DataFrame(
                all_rows
            ).to_csv(
                OUT_DIR
                / "anomaly_baseline_results_partial.csv",
                index=False,
            )

    results_df = pd.DataFrame(
        all_rows
    )

    if results_df.empty:
        raise RuntimeError(
            "No Isolation Forest results were produced."
        )

    results_df.to_csv(
        OUT_DIR
        / "anomaly_baseline_results_all.csv",
        index=False,
    )

    if all_threshold_rows:
        threshold_df = pd.concat(
            all_threshold_rows,
            ignore_index=True,
        )

        threshold_df.to_csv(
            OUT_DIR
            / "anomaly_validation_threshold_grid.csv",
            index=False,
        )

    create_summary_tables(
        results_df
    )

    manifest = {
        "dataset_path": str(
            DATA_PATH
        ),
        "model": "IsolationForest",
        "training_mode": (
            "One-class training using BENIGN samples only."
        ),
        "held_out_attack_families": attack_families,
        "seeds": SEEDS,
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "contamination": CONTAMINATION,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "fpr_constraint": MAX_VALIDATION_FPR,
        "threshold_selection_rule": (
            "Thresholds are selected only on a validation "
            "subset from the non-held-out development pool. "
            "The held-out attack family is absent from "
            "training and threshold calibration."
        ),
        "threshold_strategies": [
            "default_iforest",
            "validation_macro_f1",
            "validation_fpr_constrained",
        ],
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
        OUT_DIR
        / "anomaly_baseline_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_lines = [
        "=" * 80,
        "CICIoT2023 ANOMALY-ORIENTED BASELINE",
        "=" * 80,
        f"Dataset                   : {DATA_PATH}",
        f"Model                     : IsolationForest",
        f"Attack families           : {len(attack_families)}",
        f"Seeds                     : {len(SEEDS)}",
        f"Training mode             : BENIGN ONLY",
        f"Inner validation fraction : {INNER_VALIDATION_SIZE:.2f}",
        f"Validation FPR constraint : {MAX_VALIDATION_FPR:.2f}",
        "",
        "LEAKAGE CONTROL",
        "-" * 80,
        (
            "The held-out attack family is excluded from anomaly-detector "
            "training and threshold calibration."
        ),
        (
            "Isolation Forest is fitted only on benign development samples."
        ),
        (
            "The balanced LOAFO test set is untouched until final evaluation."
        ),
        "",
        "THRESHOLD STRATEGIES",
        "-" * 80,
        "1. default_iforest",
        "2. validation_macro_f1",
        "3. validation_fpr_constrained",
        "",
        "OUTPUTS",
        "-" * 80,
        "- anomaly_baseline_results_all.csv",
        "- anomaly_baseline_summary.csv",
        "- anomaly_baseline_by_family.csv",
        "- anomaly_default_vs_calibrated.csv",
        "- anomaly_validation_threshold_grid.csv",
        "- anomaly_baseline_manifest.json",
        "- anomaly_baseline_summary.txt",
        "=" * 80,
    ]

    summary_text = "\n".join(
        summary_lines
    )

    with open(
        OUT_DIR
        / "anomaly_baseline_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            summary_text
        )

    print()
    print(
        summary_text
    )
    print()
    print(
        f"Output directory: "
        f"{OUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
