# run_uag_ids_benchmark_balanced_final.py

import os
os.environ["MPLBACKEND"] = "Agg"

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"mergeddata\ciciot2023_working_with_source.csv"

OUT_DIR = Path(r"CICIoT2023_UAG_IDS_BALANCED_FINAL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

SEEDS = [42, 123, 2024, 2025, 2026]

TEST_SIZE = 0.20

MAX_TRAIN_ROWS = 300_000
MAX_TEST_ROWS = 150_000

# Key LOAFO settings
LOAFO_TEST_PER_CLASS_CAP = 25_000       # Maximum 25k attack + 25k benign test samples per family
MIN_TEST_PER_CLASS = 1_000              # Skip families with fewer samples
MIN_TRAIN_BENIGN_REMAINING = 10_000     # Minimum number of benign samples retained for training

SAFE_N_JOBS = 1

TASKS_FOR_BASELINE = ["binary", "coarse"]

RUN_BASELINE_SPLITS = True
RUN_LOAFO = True
SAVE_REPORTS = True


# ============================================================
# OPTIONAL MODELS
# ============================================================

HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    print("Warning: xgboost is not installed. XGBoost will be skipped.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    print("Warning: lightgbm is not installed. LightGBM will be skipped.")


# ============================================================
# LABEL FUNCTIONS
# ============================================================

def clean_label(x):
    return str(x).strip().upper()


def is_benign(label):
    label = str(label).strip().upper()
    return label in ["BENIGN", "BENIGNTRAFFIC"] or "BENIGN" in label


def map_binary(label):
    if is_benign(label):
        return "BENIGN"
    return "ATTACK"


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
        "BACKDOOR_MALWARE"
    ]:
        return "WEB_MALWARE"

    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTE_FORCE"

    if label == "VULNERABILITYSCAN":
        return "VULNERABILITY_SCAN"

    return "OTHER"


def build_task_labels(df, task):
    if task == "binary":
        return df[LABEL_COL].apply(map_binary)

    if task == "coarse":
        return df[LABEL_COL].apply(map_attack_family)

    if task == "fine":
        return df[LABEL_COL].copy()

    raise ValueError("task must be one of: binary, coarse, or fine.")


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_dataset(data_path):
    print("Loading data...")

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL} column was not found.")

    if SOURCE_COL not in df.columns:
        raise ValueError(f"{SOURCE_COL} column was not found.")

    df[LABEL_COL] = df[LABEL_COL].apply(clean_label)

    df = df[
        (df[LABEL_COL] != "NAN") &
        (df[LABEL_COL] != "") &
        (df[LABEL_COL].notna())
    ].copy()

    df = df.reset_index(drop=True)
    df[ROW_ID_COL] = np.arange(len(df))

    df["binary_label"] = df[LABEL_COL].apply(map_binary)
    df["attack_family"] = df[LABEL_COL].apply(map_attack_family)

    print("Dataset shape:", df.shape)
    print("Number of labels:", df[LABEL_COL].nunique())
    print("Number of attack families:", df["attack_family"].nunique())
    print("Number of source files:", df[SOURCE_COL].nunique())

    print("\nBinary distribution:")
    print(df["binary_label"].value_counts())

    print("\nAttack family distribution:")
    print(df["attack_family"].value_counts())

    return df


def get_feature_matrix(df):
    drop_cols = [
        LABEL_COL,
        SOURCE_COL,
        ROW_ID_COL,
        "binary_label",
        "attack_family",
        "coarse_label",
        "fine_label"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def stratified_downsample(X, y, max_rows, seed):
    if max_rows is None:
        return X, y

    if len(X) <= max_rows:
        return X, y

    idx_all = np.arange(len(X))

    try:
        _, sample_idx = train_test_split(
            idx_all,
            test_size=max_rows,
            random_state=seed,
            stratify=y
        )
    except Exception:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(idx_all, size=max_rows, replace=False)

    X_sample = X.iloc[sample_idx].copy()
    y_sample = pd.Series(y).iloc[sample_idx].copy()

    return X_sample, y_sample


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def make_models(num_classes, seed):
    models = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=SAFE_N_JOBS,
            random_state=seed
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=150,
        max_depth=24,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=SAFE_N_JOBS,
        random_state=seed
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=SAFE_N_JOBS,
        random_state=seed
    )

    if HAS_XGBOOST:
        if num_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective=objective,
            eval_metric=eval_metric,
            tree_method="hist",
            n_jobs=SAFE_N_JOBS,
            random_state=seed
        )

    if HAS_LIGHTGBM:
        if num_classes == 2:
            objective = "binary"
        else:
            objective = "multiclass"

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=64,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective=objective,
            class_weight="balanced",
            n_jobs=SAFE_N_JOBS,
            random_state=seed,
            verbose=-1
        )

    return models


# ============================================================
# METRICS
# ============================================================

def compute_standard_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }


def compute_binary_security_metrics(y_true_label, y_pred_label):
    y_true_label = pd.Series(y_true_label).astype(str).str.upper()
    y_pred_label = pd.Series(y_pred_label).astype(str).str.upper()

    tp = int(((y_true_label == "ATTACK") & (y_pred_label == "ATTACK")).sum())
    tn = int(((y_true_label == "BENIGN") & (y_pred_label == "BENIGN")).sum())
    fp = int(((y_true_label == "BENIGN") & (y_pred_label == "ATTACK")).sum())
    fn = int(((y_true_label == "ATTACK") & (y_pred_label == "BENIGN")).sum())

    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "attack_recall": attack_recall,
        "benign_recall": benign_recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    }


def compute_robustness_score(row):
    vals = [
        row.get("macro_f1", np.nan),
        row.get("balanced_accuracy", np.nan),
        row.get("mcc", np.nan)
    ]

    vals = [v for v in vals if pd.notna(v)]

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_classification_report(y_true, y_pred, labels, out_path):
    try:
        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(labels)),
            target_names=labels,
            zero_division=0
        )
        save_text(out_path, report)
    except Exception as e:
        save_text(out_path, f"Classification report could not be generated:\n{e}")


# ============================================================
# COMMON EVALUATION
# ============================================================

def evaluate_models(
    X_train,
    X_test,
    y_train_raw,
    y_test_raw,
    task,
    split_name,
    seed,
    out_dir,
    extra_info=None
):
    extra_info = extra_info or {}

    train_classes_raw = set(pd.Series(y_train_raw).astype(str).unique())
    if len(train_classes_raw) < 2:
        print("Warning: the training set contains only one class. This experiment will be skipped.")
        return []

    le = LabelEncoder()
    le.fit(pd.concat([pd.Series(y_train_raw), pd.Series(y_test_raw)], axis=0))

    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    labels = list(le.classes_)
    num_classes = len(labels)

    X_train, y_train = stratified_downsample(
        X_train,
        pd.Series(y_train),
        MAX_TRAIN_ROWS,
        seed
    )

    X_test, y_test = stratified_downsample(
        X_test,
        pd.Series(y_test),
        MAX_TEST_ROWS,
        seed
    )

    print("\n================================================")
    print(f"TASK: {task} | SPLIT: {split_name} | SEED: {seed}")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)
    print("Classes:", labels)
    print("================================================")

    models = make_models(num_classes, seed)

    results = []

    report_dir = out_dir / "classification_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        row = {
            "task": task,
            "split": split_name,
            "seed": seed,
            "model": model_name,
            "num_train": len(X_train),
            "num_test": len(X_test),
            "num_classes": num_classes,
            "status": "success"
        }

        row.update(extra_info)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            standard_metrics = compute_standard_metrics(y_test, y_pred)
            row.update(standard_metrics)

            y_test_label = le.inverse_transform(y_test)
            y_pred_label = le.inverse_transform(y_pred)

            if set(labels).issubset({"BENIGN", "ATTACK"}):
                security_metrics = compute_binary_security_metrics(
                    y_test_label,
                    y_pred_label
                )
                row.update(security_metrics)

            row["robustness_score"] = compute_robustness_score(row)

            if SAVE_REPORTS:
                report_path = report_dir / f"{task}_{split_name}_seed{seed}_{model_name}.txt"
                save_classification_report(y_test, y_pred, labels, report_path)

            print(
                f"Accuracy={row['accuracy']:.4f} | "
                f"Macro-F1={row['macro_f1']:.4f} | "
                f"MCC={row['mcc']:.4f} | "
                f"Robustness={row['robustness_score']:.4f}"
            )

        except Exception as e:
            print(f"ERROR: {model_name} failed: {e}")

            row["status"] = "failed"
            row["error"] = str(e)

            for col in [
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "macro_precision",
                "macro_recall",
                "balanced_accuracy",
                "mcc",
                "attack_recall",
                "benign_recall",
                "false_positive_rate",
                "false_negative_rate",
                "robustness_score"
            ]:
                row[col] = np.nan

        results.append(row)

        gc.collect()

    return results


# ============================================================
# RANDOM AND FILE-HELD-OUT SPLITS
# ============================================================

def run_random_split(df, task, seed, out_dir):
    y = build_task_labels(df, task)
    X = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )

    return evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task=task,
        split_name="random",
        seed=seed,
        out_dir=out_dir
    )


def run_file_held_out_split(df, task, seed, out_dir):
    y = build_task_labels(df, task)
    X = get_feature_matrix(df)
    groups = df[SOURCE_COL]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=seed
    )

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_files = set(groups.iloc[train_idx].unique())
    test_files = set(groups.iloc[test_idx].unique())
    overlap = train_files.intersection(test_files)

    if len(overlap) != 0:
        raise RuntimeError("Invalid file-held-out split: train and test sets share at least one source file.")

    split_dir = out_dir / "split_files"
    split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "train_files": pd.Series(sorted(list(train_files))),
        "test_files": pd.Series(sorted(list(test_files)))
    }).to_csv(
        split_dir / f"{task}_file_held_out_seed{seed}_files.csv",
        index=False
    )

    extra_info = {
        "train_file_count": len(train_files),
        "test_file_count": len(test_files),
        "file_overlap_count": len(overlap)
    }

    return evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task=task,
        split_name="file_held_out",
        seed=seed,
        out_dir=out_dir,
        extra_info=extra_info
    )


# ============================================================
# BALANCED LEAKAGE-CONTROLLED LOAFO
# ============================================================

def compute_balanced_loafo_test_size(family_count, benign_count):
    """
    Compute the balanced number of attack and benign test samples.

    Objectives:
    - Use the same number of attack and benign samples in the test set.
    - Retain enough benign samples for training.
    - Avoid using all samples from very large families for testing.
    """

    max_benign_test = benign_count - MIN_TRAIN_BENIGN_REMAINING

    if max_benign_test < MIN_TEST_PER_CLASS:
        max_benign_test = int(benign_count * 0.5)

    n_test = min(
        family_count,
        max_benign_test,
        LOAFO_TEST_PER_CLASS_CAP
    )

    n_test = int(n_test)

    if n_test < MIN_TEST_PER_CLASS:
        return 0

    return n_test


def run_loafo_for_family(df, held_out_family, seed, out_dir):
    """
    Balanced leakage-controlled LOAFO.

    Training:
        - The held-out family is completely removed.
        - Benign rows selected for testing are removed.
        - BENIGN and all other ATTACK families remain in training.

    Test:
        - n attack samples from the held-out family.
        - n benign samples.

    This creates a balanced 1:1 test set.
    """

    family_df_all = df[df["attack_family"] == held_out_family].copy()
    benign_df_all = df[df["binary_label"] == "BENIGN"].copy()

    family_count = len(family_df_all)
    benign_count = len(benign_df_all)

    n_test_per_class = compute_balanced_loafo_test_size(
        family_count,
        benign_count
    )

    if n_test_per_class == 0:
        print(f"{held_out_family} was skipped because there are not enough samples for a balanced test set.")
        return []

    test_attack_df = family_df_all.sample(
        n=n_test_per_class,
        random_state=seed
    ).copy()

    test_benign_df = benign_df_all.sample(
        n=n_test_per_class,
        random_state=seed
    ).copy()

    test_attack_ids = set(test_attack_df[ROW_ID_COL].tolist())
    test_benign_ids = set(test_benign_df[ROW_ID_COL].tolist())
    test_ids = test_attack_ids.union(test_benign_ids)

    test_df = pd.concat([test_attack_df, test_benign_df], axis=0)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = df[
        (df["attack_family"] != held_out_family) &
        (~df[ROW_ID_COL].isin(test_ids))
    ].copy()

    # Leakage-control checks
    if held_out_family in set(train_df["attack_family"].unique()):
        raise RuntimeError(f"LOAFO leakage detected: {held_out_family} remains in the training set.")

    train_classes = set(train_df["binary_label"].unique())
    if train_classes != {"BENIGN", "ATTACK"}:
        print(f"Warning: invalid training classes for {held_out_family}: {train_classes}")
        return []

    train_ids = set(train_df[ROW_ID_COL].tolist())
    overlap_ids = train_ids.intersection(test_ids)

    if len(overlap_ids) != 0:
        raise RuntimeError("LOAFO row-level leakage detected.")

    remaining_train_benign = int((train_df["binary_label"] == "BENIGN").sum())
    remaining_train_attack = int((train_df["binary_label"] == "ATTACK").sum())

    X_train = get_feature_matrix(train_df)
    X_test = get_feature_matrix(test_df)

    y_train = train_df["binary_label"]
    y_test = test_df["binary_label"]

    extra_info = {
        "held_out_family": held_out_family,
        "held_out_family_total_samples": family_count,
        "loafo_test_per_class": n_test_per_class,
        "test_attack_samples": n_test_per_class,
        "test_benign_samples": n_test_per_class,
        "remaining_train_benign": remaining_train_benign,
        "remaining_train_attack": remaining_train_attack,
        "loafo_train_test_row_overlap": len(overlap_ids),
        "loafo_balanced_test": True,
        "loafo_benign_leakage_fixed": True
    }

    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task="binary",
        split_name="balanced_loafo",
        seed=seed,
        out_dir=out_dir,
        extra_info=extra_info
    )

    for row in results:
        recall_value = row.get("attack_recall", np.nan)
        if pd.notna(recall_value):
            row["attack_difficulty_index"] = 1.0 - recall_value
        else:
            row["attack_difficulty_index"] = np.nan

    return results


def run_loafo(df, seed, out_dir):
    family_counts = df["attack_family"].value_counts()

    attack_families = [
        fam for fam in family_counts.index
        if fam not in ["BENIGN", "OTHER"] and family_counts[fam] >= MIN_TEST_PER_CLASS
    ]

    print("\nBalanced LOAFO attack families:")
    for fam in attack_families:
        print(f"  {fam}: {family_counts[fam]}")

    all_results = []

    for fam in attack_families:
        print("\n################################################")
        print(f"BALANCED LOAFO | Held-out family: {fam} | Seed: {seed}")
        print("################################################")

        res = run_loafo_for_family(df, fam, seed, out_dir)
        all_results.extend(res)

    return all_results


# ============================================================
# TABLES
# ============================================================

def flatten_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df


def create_generalization_gap_tables(results_df, out_dir):
    success_df = results_df[results_df["status"] == "success"].copy()

    random_binary = success_df[
        (success_df["task"] == "binary") &
        (success_df["split"] == "random")
    ].copy()

    loafo = success_df[
        (success_df["task"] == "binary") &
        (success_df["split"] == "balanced_loafo")
    ].copy()

    if random_binary.empty or loafo.empty:
        print("Random or balanced LOAFO results are missing for the generalization-gap table.")
        return

    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "false_positive_rate",
        "robustness_score"
    ]

    random_mean = random_binary.groupby("model")[metrics].mean().reset_index()
    loafo_mean = loafo.groupby("model")[metrics + ["attack_difficulty_index"]].mean().reset_index()

    merged = random_mean.merge(
        loafo_mean,
        on="model",
        suffixes=("_random", "_balanced_loafo")
    )

    for metric in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "robustness_score"
    ]:
        merged[f"{metric}_generalization_gap"] = (
            merged[f"{metric}_random"] - merged[f"{metric}_balanced_loafo"]
        )

    merged.to_csv(out_dir / "table_generalization_gap_random_vs_balanced_loafo.csv", index=False)


def create_summary_tables(results_df, out_dir):
    success_df = results_df[results_df["status"] == "success"].copy()

    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "benign_recall",
        "false_positive_rate",
        "false_negative_rate",
        "robustness_score",
        "attack_difficulty_index"
    ]

    available_metrics = [m for m in metrics if m in success_df.columns]

    summary = success_df.groupby(
        ["task", "split", "model"]
    )[available_metrics].agg(["mean", "std"]).reset_index()

    summary = flatten_columns(summary)
    summary.to_csv(out_dir / "table_summary_mean_std.csv", index=False)

    loafo = success_df[success_df["split"] == "balanced_loafo"].copy()

    if not loafo.empty:
        loafo_family = loafo.groupby(
            ["held_out_family", "model"]
        )[available_metrics].agg(["mean", "std"]).reset_index()

        loafo_family = flatten_columns(loafo_family)
        loafo_family.to_csv(out_dir / "table_balanced_loafo_by_family_mean_std.csv", index=False)

        family_model_mean = loafo.groupby(
            ["held_out_family", "model"]
        )[available_metrics].mean().reset_index()

        idx = family_model_mean.groupby("held_out_family")["robustness_score"].idxmax()
        best_by_family = family_model_mean.loc[idx].copy()
        best_by_family.to_csv(out_dir / "table_best_model_by_attack_family.csv", index=False)

        worst_case = loafo.groupby("model").agg({
            "attack_recall": ["mean", "std", "min"],
            "attack_difficulty_index": ["mean", "std", "max"],
            "false_positive_rate": ["mean", "std", "max"],
            "robustness_score": ["mean", "std", "min"],
            "macro_f1": ["mean", "std", "min"],
            "mcc": ["mean", "std", "min"]
        }).reset_index()

        worst_case = flatten_columns(worst_case)
        worst_case.to_csv(out_dir / "table_model_worst_case_robustness.csv", index=False)

    create_generalization_gap_tables(results_df, out_dir)


def save_basic_plots(results_df, out_dir):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    success_df = results_df[results_df["status"] == "success"].copy()

    loafo = success_df[success_df["split"] == "balanced_loafo"].copy()

    if not loafo.empty:
        model_order = loafo.groupby("model")["robustness_score"].mean().sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(10, 6))
        data = [
            loafo[loafo["model"] == model]["robustness_score"].dropna().values
            for model in model_order
        ]

        ax.boxplot(data, labels=model_order, showmeans=True)
        ax.set_title("Balanced LOAFO Robustness Score by Model")
        ax.set_ylabel("Robustness Score")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "balanced_loafo_robustness_score_by_model.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        difficulty = loafo.groupby("held_out_family")["attack_difficulty_index"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(difficulty.index, difficulty.values)
        ax.set_title("Attack Difficulty Index by Held-out Family")
        ax.set_ylabel("Attack Difficulty Index")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(plot_dir / "attack_difficulty_index_by_family.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    baseline = success_df[success_df["split"].isin(["random", "file_held_out"])].copy()

    if not baseline.empty:
        for metric in ["accuracy", "macro_f1", "mcc", "balanced_accuracy"]:
            pivot = baseline.groupby(["model", "split"])[metric].mean().unstack()

            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"Random vs File-held-out: {metric}")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            fig.savefig(plot_dir / f"random_vs_fileheldout_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    df = load_dataset(DATA_PATH)

    all_results = []

    if RUN_BASELINE_SPLITS:
        for seed in SEEDS:
            for task in TASKS_FOR_BASELINE:
                print("\n############################################")
                print(f"BASELINE | TASK={task} | SEED={seed}")
                print("############################################")

                res_random = run_random_split(df, task, seed, OUT_DIR)
                all_results.extend(res_random)

                res_file = run_file_held_out_split(df, task, seed, OUT_DIR)
                all_results.extend(res_file)

                pd.DataFrame(all_results).to_csv(OUT_DIR / "results_partial.csv", index=False)

    if RUN_LOAFO:
        for seed in SEEDS:
            print("\n############################################")
            print(f"BALANCED LEAKAGE-FREE LOAFO | SEED={seed}")
            print("############################################")

            res_loafo = run_loafo(df, seed, OUT_DIR)
            all_results.extend(res_loafo)

            pd.DataFrame(all_results).to_csv(OUT_DIR / "results_partial.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "results_all_uag_ids_balanced_final.csv", index=False)

    create_summary_tables(results_df, OUT_DIR)
    save_basic_plots(results_df, OUT_DIR)

    print("\nAll experiments completed.")
    print("Main result file:")
    print(OUT_DIR / "results_all_uag_ids_balanced_final.csv")


if __name__ == "__main__":
    main()