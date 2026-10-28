# 12_prevalence_preserving_loafo.py
"""
Prevalence-preserving LOAFO sensitivity analysis for the CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script evaluates whether the primary balanced LOAFO test
set (1:1 held-out attack vs benign) may alter the evaluation distribution.

The original balanced LOAFO protocol remains the primary prevalence-controlled
diagnostic benchmark. This script adds a sensitivity analysis in which the
held-out test composition preserves the attack-vs-benign prevalence observed
in the constructed working dataset, subject to the existing test-size cap.

IMPORTANT TERMINOLOGY
---------------------
This is called "prevalence-preserving within the working dataset", NOT
"real-world prevalence". The working dataset itself was created by controlled
sampling from CICIoT2023, so this analysis does not claim to reproduce an
external deployment prevalence.

Design
------
For each held-out family and seed:

1. The held-out attack family is completely excluded from training.
2. A benign evaluation reserve is selected exactly as in the original balanced
   LOAFO protocol.
3. The SAME trained model is evaluated on:
      A. balanced_loafo
         - n held-out attacks + n benign flows
      B. prevalence_preserving_loafo
         - attack/benign counts chosen to preserve the working-dataset
           family-vs-benign prevalence, subject to MAX_TEST_ROWS
4. The full benign reserve is excluded from training for BOTH evaluations.
   Therefore, differences between A and B reflect test-composition changes,
   not different fitted models.
5. All held-out-family rows are excluded from training regardless of which
   held-out attack rows are used for evaluation.

Focused models
--------------
- Random Forest
- LightGBM

Seeds
-----
42, 123, 2026

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
- PR-AUC / Average Precision
- Normalized robustness:
      MCC_01 = (MCC + 1) / 2
      R = (Macro-F1 + Balanced Accuracy + MCC_01) / 3

No threshold calibration is performed here; default classifier decisions are
used so that this analysis isolates evaluation-prevalence sensitivity.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HAS_LIGHTGBM = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("mergeddata") / "ciciot2023_working_with_source.csv"

OUT_DIR = Path("CICIoT2023_PREVALENCE_PRESERVING_LOAFO")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

SEEDS = [42, 123, 2026]

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 150_000

LOAFO_TEST_PER_CLASS_CAP = 25_000
MIN_TEST_PER_CLASS = 1_000
MIN_TRAIN_BENIGN_REMAINING = 10_000

SAFE_N_JOBS = 1

EVALUATION_MODES = [
    "balanced_loafo",
    "prevalence_preserving_loafo",
]


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


def balanced_test_per_class(
    family_count: int,
    benign_count: int,
) -> int:
    max_benign_test = benign_count - MIN_TRAIN_BENIGN_REMAINING

    if max_benign_test < MIN_TEST_PER_CLASS:
        max_benign_test = int(benign_count * 0.5)

    n = min(
        family_count,
        max_benign_test,
        LOAFO_TEST_PER_CLASS_CAP,
    )

    n = int(n)

    if n < MIN_TEST_PER_CLASS:
        return 0

    return n


def prevalence_preserving_counts(
    family_count: int,
    benign_count: int,
    benign_reserve_count: int,
) -> Tuple[int, int, float]:
    """
    Determine attack/benign test counts that preserve the family-vs-benign
    prevalence observed in the working dataset.

    The benign count cannot exceed the benign evaluation reserve selected
    for the balanced LOAFO protocol. The total test size cannot exceed
    MAX_TEST_ROWS.

    Returns
    -------
    n_attack, n_benign, target_attack_prevalence
    """
    total_reference = family_count + benign_count

    if total_reference <= 0:
        raise ValueError("Invalid family/benign counts.")

    p_attack = family_count / total_reference
    p_benign = 1.0 - p_attack

    if p_attack <= 0 or p_benign <= 0:
        raise ValueError(
            "Prevalence-preserving test requires both attack and benign rows."
        )

    # Maximum total test size allowed by:
    # 1) global test cap,
    # 2) available benign evaluation reserve,
    # 3) available held-out attack rows.
    total_by_benign = int(
        np.floor(
            benign_reserve_count / p_benign
        )
    )

    total_by_attack = int(
        np.floor(
            family_count / p_attack
        )
    )

    total_test = min(
        MAX_TEST_ROWS,
        total_by_benign,
        total_by_attack,
    )

    if total_test < 2:
        raise ValueError(
            "Unable to construct prevalence-preserving test set."
        )

    n_attack = int(
        round(total_test * p_attack)
    )
    n_benign = total_test - n_attack

    # Safety bounds.
    n_attack = min(
        max(n_attack, 1),
        family_count,
    )

    n_benign = min(
        max(n_benign, 1),
        benign_reserve_count,
    )

    # Recalculate attack count if benign clipping changed total.
    max_attack_given_total = max(
        1,
        MAX_TEST_ROWS - n_benign,
    )
    n_attack = min(
        n_attack,
        max_attack_given_total,
        family_count,
    )

    return (
        int(n_attack),
        int(n_benign),
        float(p_attack),
    )


# ============================================================
# MODELS
# ============================================================

def make_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=150,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
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
    s = pd.Series(labels).astype(str).str.upper()
    return (s == "ATTACK").astype(int).to_numpy()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

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

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    balanced_acc = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    mcc = float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )

    mcc_01 = (mcc + 1.0) / 2.0

    robustness = (
        macro_f1
        + balanced_acc
        + mcc_01
    ) / 3.0

    roc_auc = np.nan
    pr_auc = np.nan

    if y_prob is not None:
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
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "mcc_01": float(mcc_01),
        "robustness_equal_normalized": float(
            robustness
        ),
        "attack_precision": float(
            attack_precision
        ),
        "attack_recall": float(
            attack_recall
        ),
        "attack_f1": float(
            attack_f1
        ),
        "benign_recall": float(
            benign_recall
        ),
        "false_positive_rate": float(
            fpr
        ),
        "false_negative_rate": float(
            fnr
        ),
        "false_alarms_per_1000_benign": float(
            fpr * 1000.0
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# LOAFO CONSTRUCTION
# ============================================================

def prepare_paired_evaluations(
    df: pd.DataFrame,
    held_out_family: str,
    seed: int,
):
    family_df = df[
        df["attack_family"] == held_out_family
    ].copy()

    benign_df = df[
        df["binary_label"] == "BENIGN"
    ].copy()

    family_count = len(family_df)
    benign_count = len(benign_df)

    n_balanced = balanced_test_per_class(
        family_count=family_count,
        benign_count=benign_count,
    )

    if n_balanced == 0:
        return None

    # This benign reserve is exactly the one used by the balanced LOAFO
    # evaluation. The entire reserve is withheld from training for BOTH
    # test-composition modes.
    # --------------------------------------------------------
    # Deterministic nested test-sample construction
    # --------------------------------------------------------
    # Create one deterministic ordering for the held-out-family rows and one
    # deterministic ordering for benign rows. Both evaluation modes are built
    # from prefixes of these same orderings.
    #
    # This ensures that when one evaluation requires fewer rows than the other,
    # the smaller set is a strict subset of the larger set whenever feasible.
    # Consequently, paired differences are driven by test composition rather
    # than by independently resampling attack/benign examples.

    attack_order = family_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    benign_order = benign_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    # Full benign reserve used for BOTH evaluation modes and excluded from
    # training in both cases.
    benign_reserve = benign_order.iloc[
        :n_balanced
    ].copy()

    balanced_attack = attack_order.iloc[
        :n_balanced
    ].copy()

    balanced_benign = benign_reserve.copy()

    balanced_test = pd.concat(
        [
            balanced_attack,
            balanced_benign,
        ],
        axis=0,
    ).sample(
        frac=1.0,
        random_state=seed + 1,
    ).reset_index(drop=True)

    (
        n_prev_attack,
        n_prev_benign,
        target_attack_prevalence,
    ) = prevalence_preserving_counts(
        family_count=family_count,
        benign_count=benign_count,
        benign_reserve_count=n_balanced,
    )

    # Prevalence-preserving rows come from the SAME deterministic orderings.
    # If fewer attack/benign rows are required, they are nested subsets of the
    # corresponding balanced-evaluation pools wherever feasible.
    prev_attack = attack_order.iloc[
        :n_prev_attack
    ].copy()

    prev_benign = benign_reserve.iloc[
        :n_prev_benign
    ].copy()

    prevalence_test = pd.concat(
        [
            prev_attack,
            prev_benign,
        ],
        axis=0,
    ).sample(
        frac=1.0,
        random_state=seed + 2,
    ).reset_index(drop=True)

    # SAME training data for both evaluation modes:
    # - exclude entire held-out family
    # - exclude full benign evaluation reserve
    reserve_ids = set(
        benign_reserve[ROW_ID_COL].tolist()
    )

    train_pool = df[
        (df["attack_family"] != held_out_family)
        & (~df[ROW_ID_COL].isin(reserve_ids))
    ].copy()

    if held_out_family in set(
        train_pool["attack_family"].unique()
    ):
        raise RuntimeError(
            f"Held-out family leakage: {held_out_family}"
        )

    train_df = safe_stratified_sample(
        train_pool,
        MAX_TRAIN_ROWS,
        seed,
        stratify_col="binary_label",
    )

    train_ids = set(
        train_df[ROW_ID_COL].tolist()
    )

    balanced_ids = set(
        balanced_test[ROW_ID_COL].tolist()
    )

    prevalence_ids = set(
        prevalence_test[ROW_ID_COL].tolist()
    )

    if train_ids.intersection(balanced_ids):
        raise RuntimeError(
            f"Train/balanced-test overlap: {held_out_family}"
        )

    if train_ids.intersection(prevalence_ids):
        raise RuntimeError(
            f"Train/prevalence-test overlap: {held_out_family}"
        )

    X_train = get_feature_matrix(
        train_df
    )

    y_train = labels_to_binary_int(
        train_df["binary_label"]
    )

    X_balanced = get_feature_matrix(
        balanced_test
    )

    y_balanced = labels_to_binary_int(
        balanced_test["binary_label"]
    )

    X_prevalence = get_feature_matrix(
        prevalence_test
    )

    y_prevalence = labels_to_binary_int(
        prevalence_test["binary_label"]
    )

    meta = {
        "held_out_family": held_out_family,
        "seed": seed,
        "working_family_count": int(
            family_count
        ),
        "working_benign_count": int(
            benign_count
        ),
        "target_working_attack_prevalence": float(
            target_attack_prevalence
        ),
        "benign_evaluation_reserve_count": int(
            n_balanced
        ),
        "train_rows": int(
            len(train_df)
        ),
        "train_attack_rows": int(
            (train_df["binary_label"] == "ATTACK").sum()
        ),
        "train_benign_rows": int(
            (train_df["binary_label"] == "BENIGN").sum()
        ),
        "balanced_test_rows": int(
            len(balanced_test)
        ),
        "balanced_attack_rows": int(
            (balanced_test["binary_label"] == "ATTACK").sum()
        ),
        "balanced_benign_rows": int(
            (balanced_test["binary_label"] == "BENIGN").sum()
        ),
        "prevalence_test_rows": int(
            len(prevalence_test)
        ),
        "prevalence_attack_rows": int(
            (prevalence_test["binary_label"] == "ATTACK").sum()
        ),
        "prevalence_benign_rows": int(
            (prevalence_test["binary_label"] == "BENIGN").sum()
        ),
        "actual_prevalence_test_attack_fraction": float(
            (prevalence_test["binary_label"] == "ATTACK").mean()
        ),
        "same_training_data_for_both_evaluations": True,
        "held_out_family_absent_from_training": True,
        "train_balanced_test_overlap": 0,
        "train_prevalence_test_overlap": 0,
    }

    return (
        X_train,
        y_train,
        X_balanced,
        y_balanced,
        X_prevalence,
        y_prevalence,
        meta,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def positive_class_probability(
    model,
    X,
) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(X)

    if probs.ndim != 2 or probs.shape[1] != 2:
        return None

    return probs[:, 1].astype(float)


def evaluate_fitted_model(
    model,
    X_test,
    y_test,
) -> Dict[str, float]:
    t0 = time.perf_counter()

    y_pred = model.predict(
        X_test
    ).astype(int)

    inference_seconds = (
        time.perf_counter()
        - t0
    )

    y_prob = positive_class_probability(
        model,
        X_test,
    )

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    metrics.update({
        "inference_seconds": float(
            inference_seconds
        ),
        "inference_ms_per_flow": float(
            inference_seconds / len(X_test) * 1000.0
        ),
        "throughput_flows_per_second": float(
            len(X_test) / inference_seconds
            if inference_seconds > 0
            else np.nan
        ),
    })

    return metrics


# ============================================================
# SUMMARIES
# ============================================================

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = [
        "_".join(
            str(v) for v in col
            if str(v) != ""
        )
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]

    return out


def create_summary_tables(
    results_df: pd.DataFrame,
):
    metric_cols = [
        "test_attack_prevalence",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "robustness_equal_normalized",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "benign_recall",
        "false_positive_rate",
        "false_alarms_per_1000_benign",
        "roc_auc",
        "pr_auc",
    ]

    summary = (
        results_df.groupby(
            ["evaluation_mode", "model"]
        )[metric_cols]
        .agg(
            ["mean", "std", "min", "max"]
        )
        .reset_index()
    )

    summary = flatten_columns(
        summary
    )

    summary.to_csv(
        OUT_DIR / "prevalence_sensitivity_summary.csv",
        index=False,
    )

    by_family = (
        results_df.groupby(
            [
                "evaluation_mode",
                "held_out_family",
                "model",
            ]
        )[metric_cols]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    by_family = flatten_columns(
        by_family
    )

    by_family.to_csv(
        OUT_DIR / "prevalence_sensitivity_by_family.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired differences using the SAME fitted model
    # --------------------------------------------------------

    keys = [
        "held_out_family",
        "seed",
        "model",
    ]

    balanced = results_df[
        results_df["evaluation_mode"]
        == "balanced_loafo"
    ][
        keys + metric_cols
    ].copy()

    prevalence = results_df[
        results_df["evaluation_mode"]
        == "prevalence_preserving_loafo"
    ][
        keys + metric_cols
    ].copy()

    paired = balanced.merge(
        prevalence,
        on=keys,
        suffixes=(
            "_balanced",
            "_prevalence",
        ),
    )

    for metric in metric_cols:
        paired[
            f"{metric}_change_prevalence_minus_balanced"
        ] = (
            paired[
                f"{metric}_prevalence"
            ]
            - paired[
                f"{metric}_balanced"
            ]
        )

    paired.to_csv(
        OUT_DIR
        / "prevalence_vs_balanced_runlevel.csv",
        index=False,
    )

    change_cols = [
        c for c in paired.columns
        if c.endswith(
            "_change_prevalence_minus_balanced"
        )
    ]

    change_summary = (
        paired.groupby(
            "model"
        )[change_cols]
        .agg(
            ["mean", "std", "min", "max"]
        )
        .reset_index()
    )

    change_summary = flatten_columns(
        change_summary
    )

    change_summary.to_csv(
        OUT_DIR
        / "prevalence_vs_balanced_summary.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not HAS_LIGHTGBM:
        raise ImportError(
            "lightgbm is required for this sensitivity analysis."
        )

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

    print("\nAttack families:")
    for fam in attack_families:
        count = int(
            family_counts[fam]
        )

        benign_count = int(
            family_counts.get(
                "BENIGN",
                0,
            )
        )

        prevalence = (
            count / (count + benign_count)
            if (count + benign_count) > 0
            else np.nan
        )

        print(
            f"  {fam}: {count:,} | "
            f"working-set attack prevalence vs benign="
            f"{prevalence:.4f}"
        )

    all_rows = []
    design_rows = []

    for seed in SEEDS:
        for held_out_family in attack_families:
            print()
            print("=" * 88)
            print(
                f"PREVALENCE SENSITIVITY | "
                f"family={held_out_family} | "
                f"seed={seed}"
            )
            print("=" * 88)

            prepared = prepare_paired_evaluations(
                df=df,
                held_out_family=held_out_family,
                seed=seed,
            )

            if prepared is None:
                print(
                    "Skipped: insufficient samples."
                )
                continue

            (
                X_train,
                y_train,
                X_balanced,
                y_balanced,
                X_prevalence,
                y_prevalence,
                meta,
            ) = prepared

            design_rows.append(
                meta.copy()
            )

            print(
                f"  Balanced test: "
                f"{len(y_balanced):,} rows | "
                f"attack prevalence={np.mean(y_balanced):.4f}"
            )

            print(
                f"  Prevalence-preserving test: "
                f"{len(y_prevalence):,} rows | "
                f"attack prevalence={np.mean(y_prevalence):.4f} | "
                f"target={meta['target_working_attack_prevalence']:.4f}"
            )

            models = make_models(
                seed
            )

            for model_name, model in models.items():
                print(
                    f"Training {model_name} once..."
                )

                try:
                    t0 = time.perf_counter()

                    model.fit(
                        X_train,
                        y_train,
                    )

                    training_seconds = (
                        time.perf_counter()
                        - t0
                    )

                    # Balanced evaluation.
                    balanced_metrics = (
                        evaluate_fitted_model(
                            model,
                            X_balanced,
                            y_balanced,
                        )
                    )

                    balanced_row = {
                        **meta,
                        "model": model_name,
                        "evaluation_mode": "balanced_loafo",
                        "test_rows": int(
                            len(y_balanced)
                        ),
                        "test_attack_rows": int(
                            y_balanced.sum()
                        ),
                        "test_benign_rows": int(
                            len(y_balanced)
                            - y_balanced.sum()
                        ),
                        "test_attack_prevalence": float(
                            np.mean(y_balanced)
                        ),
                        "training_seconds": float(
                            training_seconds
                        ),
                        **balanced_metrics,
                    }

                    all_rows.append(
                        balanced_row
                    )

                    # Prevalence-preserving evaluation on the SAME fitted model.
                    prevalence_metrics = (
                        evaluate_fitted_model(
                            model,
                            X_prevalence,
                            y_prevalence,
                        )
                    )

                    prevalence_row = {
                        **meta,
                        "model": model_name,
                        "evaluation_mode": (
                            "prevalence_preserving_loafo"
                        ),
                        "test_rows": int(
                            len(y_prevalence)
                        ),
                        "test_attack_rows": int(
                            y_prevalence.sum()
                        ),
                        "test_benign_rows": int(
                            len(y_prevalence)
                            - y_prevalence.sum()
                        ),
                        "test_attack_prevalence": float(
                            np.mean(y_prevalence)
                        ),
                        "training_seconds": float(
                            training_seconds
                        ),
                        **prevalence_metrics,
                    }

                    all_rows.append(
                        prevalence_row
                    )

                    print(
                        f"  {model_name} | balanced: "
                        f"MacroF1={balanced_row['macro_f1']:.4f}, "
                        f"Precision={balanced_row['attack_precision']:.4f}, "
                        f"Recall={balanced_row['attack_recall']:.4f}, "
                        f"FPR={balanced_row['false_positive_rate']:.4f}"
                    )

                    print(
                        f"  {model_name} | prevalence: "
                        f"MacroF1={prevalence_row['macro_f1']:.4f}, "
                        f"Precision={prevalence_row['attack_precision']:.4f}, "
                        f"Recall={prevalence_row['attack_recall']:.4f}, "
                        f"FPR={prevalence_row['false_positive_rate']:.4f}"
                    )

                except Exception as exc:
                    print(
                        f"ERROR | {model_name} | "
                        f"{held_out_family} | "
                        f"seed={seed}: {exc}"
                    )

                gc.collect()

            # Progressive save.
            pd.DataFrame(
                all_rows
            ).to_csv(
                OUT_DIR
                / "prevalence_sensitivity_results_partial.csv",
                index=False,
            )

            pd.DataFrame(
                design_rows
            ).to_csv(
                OUT_DIR
                / "prevalence_test_design_partial.csv",
                index=False,
            )

    results_df = pd.DataFrame(
        all_rows
    )

    design_df = pd.DataFrame(
        design_rows
    )

    if results_df.empty:
        raise RuntimeError(
            "No prevalence-sensitivity results were produced."
        )

    results_df.to_csv(
        OUT_DIR
        / "prevalence_sensitivity_results_all.csv",
        index=False,
    )

    design_df.to_csv(
        OUT_DIR
        / "prevalence_test_design.csv",
        index=False,
    )

    create_summary_tables(
        results_df
    )

    manifest = {
        "dataset_path": str(
            DATA_PATH
        ),
        "analysis_name": (
            "Prevalence-preserving LOAFO sensitivity analysis"
        ),
        "terminology_note": (
            "Prevalence is preserved relative to the constructed working "
            "dataset, not claimed as real-world deployment prevalence."
        ),
        "seeds": SEEDS,
        "models": [
            "RandomForest",
            "LightGBM",
        ],
        "held_out_attack_families": attack_families,
        "balanced_test_per_class_cap": (
            LOAFO_TEST_PER_CLASS_CAP
        ),
        "max_test_rows": MAX_TEST_ROWS,
        "max_train_rows": MAX_TRAIN_ROWS,
        "min_train_benign_remaining_before_test_selection": (
            MIN_TRAIN_BENIGN_REMAINING
        ),
        "paired_design": (
            "For each family, seed, and model, one fitted model is evaluated "
            "on both balanced and prevalence-preserving LOAFO test sets. "
            "The same benign evaluation reserve is withheld from training "
            "for both modes."
        ),
        "leakage_control": (
            "Held-out family excluded from training; full benign evaluation "
            "reserve excluded from training; row overlap checked for both "
            "evaluation modes."
        ),
        "threshold_policy": (
            "Default classifier decision rule; no threshold calibration, "
            "to isolate evaluation-prevalence sensitivity."
        ),
        "normalized_robustness": (
            "(Macro-F1 + Balanced Accuracy + (MCC+1)/2) / 3"
        ),
    }

    with open(
        OUT_DIR
        / "prevalence_sensitivity_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        "=" * 90,
        "CICIoT2023 PREVALENCE-PRESERVING LOAFO SENSITIVITY ANALYSIS",
        "=" * 90,
        f"Dataset              : {DATA_PATH}",
        f"Attack families      : {len(attack_families)}",
        f"Seeds                : {len(SEEDS)} ({SEEDS})",
        "Models               : RandomForest, LightGBM",
        "",
        "PRIMARY DESIGN",
        "-" * 90,
        (
            "The same fitted model is evaluated on both the balanced LOAFO "
            "test set and a prevalence-preserving LOAFO test set."
        ),
        (
            "The full benign evaluation reserve is excluded from training "
            "for both evaluation modes, so test-composition sensitivity is "
            "not confounded by fitting a different model."
        ),
        "",
        "TERMINOLOGY",
        "-" * 90,
        (
            "Prevalence-preserving refers to the constructed working dataset. "
            "It is not presented as real-world deployment prevalence."
        ),
        "",
        "OUTPUTS",
        "-" * 90,
        "- prevalence_sensitivity_results_all.csv",
        "- prevalence_sensitivity_summary.csv",
        "- prevalence_sensitivity_by_family.csv",
        "- prevalence_vs_balanced_runlevel.csv",
        "- prevalence_vs_balanced_summary.csv",
        "- prevalence_test_design.csv",
        "- prevalence_sensitivity_manifest.json",
        "- prevalence_sensitivity_summary.txt",
        "=" * 90,
    ]

    summary_text = "\n".join(
        lines
    )

    with open(
        OUT_DIR
        / "prevalence_sensitivity_summary.txt",
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
