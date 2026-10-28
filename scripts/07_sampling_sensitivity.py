# 10_sampling_sensitivity.py
"""
Sampling and benign-reserve sensitivity analysis for the CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script evaluates sensitivity to:
1. the effect of controlled training-set sampling/capping, and
2. the minimum number of benign samples retained outside the balanced LOAFO test set.

It does not modify the primary benchmark results.

Design
------
To keep the sensitivity analysis focused and computationally practical, two
representative supervised models are used:
- Random Forest
- LightGBM

The analysis varies one design choice at a time around the primary benchmark:

Baseline:
    MAX_TRAIN_ROWS = 300,000
    MIN_TRAIN_BENIGN_REMAINING = 10,000

Sensitivity scenarios:
    A. lower_train_cap_150k
    B. lower_train_cap_100k
    C. lower_benign_reserve_5k
    D. higher_benign_reserve_30k

The balanced LOAFO principle is preserved in every scenario:
- the held-out attack family is completely absent from training,
- the test set contains held-out-family attacks and an equal number of benign flows,
- selected benign test rows are removed from training,
- train/test row overlap is checked explicitly.

To limit computational burden while retaining stochastic replication, the
sensitivity analysis uses three fixed seeds:
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
- Normalized robustness:
      MCC_01 = (MCC + 1) / 2
      R = (Macro-F1 + Balanced Accuracy + MCC_01) / 3

Outputs include scenario-level summaries and changes relative to baseline.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
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

OUT_DIR = Path("CICIoT2023_SAMPLING_SENSITIVITY")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

# Focused sensitivity seeds.
SEEDS = [42, 123, 2026]

# Primary benchmark test cap.
MAX_TEST_ROWS = 150_000
LOAFO_TEST_PER_CLASS_CAP = 25_000
MIN_TEST_PER_CLASS = 1_000

SAFE_N_JOBS = 1

# One-factor-at-a-time sensitivity around the original design.
SCENARIOS = {
    "baseline_300k_train_10k_benign": {
        "max_train_rows": 300_000,
        "min_train_benign_remaining": 10_000,
    },
    "train_cap_150k": {
        "max_train_rows": 150_000,
        "min_train_benign_remaining": 10_000,
    },
    "train_cap_100k": {
        "max_train_rows": 100_000,
        "min_train_benign_remaining": 10_000,
    },
    "benign_reserve_5k": {
        "max_train_rows": 300_000,
        "min_train_benign_remaining": 5_000,
    },
    "benign_reserve_30k": {
        "max_train_rows": 300_000,
        "min_train_benign_remaining": 30_000,
    },
}

BASELINE_SCENARIO = "baseline_300k_train_10k_benign"


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
    min_train_benign_remaining: int,
) -> int:
    max_benign_test = benign_count - min_train_benign_remaining

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
) -> Dict[str, float]:
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

    return {
        "accuracy": float(
            accuracy_score_binary(y_true, y_pred)
        ),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "mcc_01": float(mcc_01),
        "robustness_equal_normalized": float(robustness),
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


def accuracy_score_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(np.mean(y_true == y_pred))


# ============================================================
# LOAFO PREPARATION
# ============================================================

def prepare_scenario_data(
    df: pd.DataFrame,
    held_out_family: str,
    seed: int,
    max_train_rows: int,
    min_train_benign_remaining: int,
):
    family_df_all = df[
        df["attack_family"] == held_out_family
    ].copy()

    benign_df_all = df[
        df["binary_label"] == "BENIGN"
    ].copy()

    n_test_per_class = compute_balanced_loafo_test_size(
        family_count=len(family_df_all),
        benign_count=len(benign_df_all),
        min_train_benign_remaining=min_train_benign_remaining,
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

    if held_out_family in set(
        train_pool["attack_family"].unique()
    ):
        raise RuntimeError(
            f"Held-out family leakage: {held_out_family}"
        )

    overlap = set(
        train_pool[ROW_ID_COL]
    ).intersection(
        set(test_df[ROW_ID_COL])
    )

    if overlap:
        raise RuntimeError(
            f"Train/test row overlap: {held_out_family}"
        )

    # Verify benign reserve BEFORE the optional global train cap.
    available_benign_after_test = int(
        (train_pool["binary_label"] == "BENIGN").sum()
    )

    reserve_constraint_satisfied = (
        available_benign_after_test
        >= min_train_benign_remaining
    )

    if not reserve_constraint_satisfied:
        raise RuntimeError(
            "Benign reserve constraint failed: "
            f"available={available_benign_after_test}, "
            f"required={min_train_benign_remaining}"
        )

    train_df = safe_stratified_sample(
        train_pool,
        max_train_rows,
        seed,
        stratify_col="binary_label",
    )

    test_df = safe_stratified_sample(
        test_df,
        MAX_TEST_ROWS,
        seed,
        stratify_col="binary_label",
    )

    X_train = get_feature_matrix(
        train_df
    )

    X_test = get_feature_matrix(
        test_df
    )

    y_train = labels_to_binary_int(
        train_df["binary_label"]
    )

    y_test = labels_to_binary_int(
        test_df["binary_label"]
    )

    meta = {
        "held_out_family": held_out_family,
        "seed": seed,
        "max_train_rows": int(max_train_rows),
        "min_train_benign_remaining": int(
            min_train_benign_remaining
        ),
        "loafo_test_per_class": int(
            n_test_per_class
        ),
        "available_benign_after_test_before_train_cap": int(
            available_benign_after_test
        ),
        "reserve_constraint_satisfied": bool(
            reserve_constraint_satisfied
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
        "test_rows": int(
            len(test_df)
        ),
        "test_attack_rows": int(
            (test_df["binary_label"] == "ATTACK").sum()
        ),
        "test_benign_rows": int(
            (test_df["binary_label"] == "BENIGN").sum()
        ),
        "train_test_row_overlap": 0,
        "held_out_family_absent_from_training": True,
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        meta,
    )


# ============================================================
# SUMMARY HELPERS
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
        "training_seconds",
    ]

    scenario_summary = (
        results_df.groupby(
            ["scenario", "model"]
        )[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    scenario_summary = flatten_columns(
        scenario_summary
    )

    scenario_summary.to_csv(
        OUT_DIR / "sampling_sensitivity_summary.csv",
        index=False,
    )

    family_summary = (
        results_df.groupby(
            [
                "scenario",
                "held_out_family",
                "model",
            ]
        )[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    family_summary = flatten_columns(
        family_summary
    )

    family_summary.to_csv(
        OUT_DIR / "sampling_sensitivity_by_family.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired changes relative to baseline
    # --------------------------------------------------------

    key_cols = [
        "held_out_family",
        "seed",
        "model",
    ]

    baseline = results_df[
        results_df["scenario"] == BASELINE_SCENARIO
    ][
        key_cols + metric_cols
    ].copy()

    comparison_parts = []

    for scenario in sorted(
        results_df["scenario"].unique()
    ):
        if scenario == BASELINE_SCENARIO:
            continue

        current = results_df[
            results_df["scenario"] == scenario
        ][
            key_cols
            + metric_cols
            + [
                "loafo_test_per_class",
                "train_rows",
                "train_benign_rows",
                "available_benign_after_test_before_train_cap",
            ]
        ].copy()

        merged = baseline.merge(
            current,
            on=key_cols,
            suffixes=(
                "_baseline",
                "_scenario",
            ),
        )

        merged["scenario"] = scenario

        for metric in metric_cols:
            merged[
                f"{metric}_change"
            ] = (
                merged[
                    f"{metric}_scenario"
                ]
                - merged[
                    f"{metric}_baseline"
                ]
            )

        comparison_parts.append(
            merged
        )

    comparisons = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    comparisons.to_csv(
        OUT_DIR
        / "sampling_sensitivity_vs_baseline_runlevel.csv",
        index=False,
    )

    change_cols = [
        c for c in comparisons.columns
        if c.endswith("_change")
    ]

    comparison_summary = (
        comparisons.groupby(
            ["scenario", "model"]
        )[change_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    comparison_summary = flatten_columns(
        comparison_summary
    )

    comparison_summary.to_csv(
        OUT_DIR
        / "sampling_sensitivity_vs_baseline_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Design-property audit
    # --------------------------------------------------------

    design_cols = [
        "scenario",
        "held_out_family",
        "seed",
        "max_train_rows",
        "min_train_benign_remaining",
        "loafo_test_per_class",
        "available_benign_after_test_before_train_cap",
        "train_rows",
        "train_attack_rows",
        "train_benign_rows",
        "test_rows",
        "test_attack_rows",
        "test_benign_rows",
        "reserve_constraint_satisfied",
    ]

    design = (
        results_df[
            design_cols
        ]
        .drop_duplicates()
        .sort_values(
            [
                "scenario",
                "held_out_family",
                "seed",
            ]
        )
    )

    design.to_csv(
        OUT_DIR / "sampling_design_audit.csv",
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
        print(
            f"  {fam}: "
            f"{family_counts[fam]:,}"
        )

    print("\nSensitivity scenarios:")
    for name, cfg in SCENARIOS.items():
        print(
            f"  {name}: "
            f"max_train={cfg['max_train_rows']:,}, "
            f"min_benign_reserve="
            f"{cfg['min_train_benign_remaining']:,}"
        )

    all_rows = []

    for scenario, cfg in SCENARIOS.items():
        max_train_rows = cfg[
            "max_train_rows"
        ]

        min_benign = cfg[
            "min_train_benign_remaining"
        ]

        for seed in SEEDS:
            for held_out_family in attack_families:
                print()
                print("=" * 82)
                print(
                    f"SAMPLING SENSITIVITY | "
                    f"scenario={scenario} | "
                    f"family={held_out_family} | "
                    f"seed={seed}"
                )
                print("=" * 82)

                prepared = prepare_scenario_data(
                    df=df,
                    held_out_family=held_out_family,
                    seed=seed,
                    max_train_rows=max_train_rows,
                    min_train_benign_remaining=min_benign,
                )

                if prepared is None:
                    print(
                        "Skipped: insufficient samples."
                    )
                    continue

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    meta,
                ) = prepared

                models = make_models(
                    seed
                )

                for model_name, model in models.items():
                    print(
                        f"Training {model_name}..."
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

                        y_pred = model.predict(
                            X_test
                        ).astype(int)

                        metrics = compute_metrics(
                            y_test,
                            y_pred,
                        )

                        row = {
                            "scenario": scenario,
                            **meta,
                            "model": model_name,
                            **metrics,
                            "training_seconds": float(
                                training_seconds
                            ),
                        }

                        all_rows.append(
                            row
                        )

                        print(
                            f"  MacroF1={row['macro_f1']:.4f}, "
                            f"Recall={row['attack_recall']:.4f}, "
                            f"FPR={row['false_positive_rate']:.4f}, "
                            f"Rnorm={row['robustness_equal_normalized']:.4f}"
                        )

                    except Exception as exc:
                        print(
                            f"ERROR | {model_name} | "
                            f"{scenario} | "
                            f"{held_out_family} | "
                            f"seed={seed}: {exc}"
                        )

                    gc.collect()

                # Progressive save.
                pd.DataFrame(
                    all_rows
                ).to_csv(
                    OUT_DIR
                    / "sampling_sensitivity_results_partial.csv",
                    index=False,
                )

    results_df = pd.DataFrame(
        all_rows
    )

    if results_df.empty:
        raise RuntimeError(
            "No sensitivity-analysis results were produced."
        )

    results_df.to_csv(
        OUT_DIR / "sampling_sensitivity_results_all.csv",
        index=False,
    )

    create_summary_tables(
        results_df
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "dataset_path": str(
            DATA_PATH
        ),
        "baseline_scenario": BASELINE_SCENARIO,
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "models": [
            "RandomForest",
            "LightGBM",
        ],
        "held_out_attack_families": attack_families,
        "loafo_test_per_class_cap": LOAFO_TEST_PER_CLASS_CAP,
        "max_test_rows": MAX_TEST_ROWS,
        "sensitivity_design": (
            "One-factor-at-a-time perturbations around the original "
            "300k training cap and 10k minimum benign reserve."
        ),
        "normalized_robustness": (
            "(Macro-F1 + Balanced Accuracy + (MCC+1)/2) / 3"
        ),
        "leakage_control": (
            "Held-out family excluded from training, selected benign "
            "test rows removed from training, and row overlap checked."
        ),
    }

    with open(
        OUT_DIR / "sampling_sensitivity_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    lines = [
        "=" * 84,
        "CICIoT2023 SAMPLING AND BENIGN-RESERVE SENSITIVITY ANALYSIS",
        "=" * 84,
        f"Dataset              : {DATA_PATH}",
        f"Attack families      : {len(attack_families)}",
        f"Seeds                : {len(SEEDS)} ({SEEDS})",
        "Models               : RandomForest, LightGBM",
        "",
        "BASELINE",
        "-" * 84,
        "MAX_TRAIN_ROWS = 300,000",
        "MIN_TRAIN_BENIGN_REMAINING = 10,000",
        "",
        "SENSITIVITY SCENARIOS",
        "-" * 84,
    ]

    for name, cfg in SCENARIOS.items():
        lines.append(
            f"- {name}: "
            f"max_train={cfg['max_train_rows']:,}, "
            f"min_benign_reserve="
            f"{cfg['min_train_benign_remaining']:,}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 84,
        (
            "This is a focused sensitivity analysis rather than a replacement "
            "for the full five-model benchmark."
        ),
        (
            "The held-out family exclusion, balanced LOAFO test construction, "
            "and row-disjointness checks are retained in every scenario."
        ),
        (
            "Changes relative to the original 300k/10k design are reported "
            "at both run level and aggregate level."
        ),
        "",
        "OUTPUTS",
        "-" * 84,
        "- sampling_sensitivity_results_all.csv",
        "- sampling_sensitivity_summary.csv",
        "- sampling_sensitivity_by_family.csv",
        "- sampling_sensitivity_vs_baseline_runlevel.csv",
        "- sampling_sensitivity_vs_baseline_summary.csv",
        "- sampling_design_audit.csv",
        "- sampling_sensitivity_manifest.json",
        "- sampling_sensitivity_summary.txt",
        "=" * 84,
    ])

    summary_text = "\n".join(
        lines
    )

    with open(
        OUT_DIR / "sampling_sensitivity_summary.txt",
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
