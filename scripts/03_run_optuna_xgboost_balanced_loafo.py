# run_optuna_xgboost_balanced_loafo_final.py

import os
os.environ["MPLBACKEND"] = "Agg"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"mergeddata\ciciot2023_working_with_source.csv"

OUT_DIR = Path(r"CICIoT2023_OPTUNA_XGB_BALANCED_LOAFO_FINAL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

OPTUNA_SEEDS = [42, 123, 2026]

N_TRIALS = 30

MAX_TRAIN_ROWS = 250_000
MAX_TEST_ROWS = 120_000

LOAFO_TEST_PER_CLASS_CAP = 25_000
MIN_TEST_PER_CLASS = 1_000
MIN_TRAIN_BENIGN_REMAINING = 10_000


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


def get_feature_matrix(df):
    drop_cols = [
        LABEL_COL,
        SOURCE_COL,
        ROW_ID_COL,
        "binary_label",
        "attack_family"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def safe_sample_df(df, n, seed, stratify_col=None):
    if n is None or len(df) <= n:
        return df.copy()

    if stratify_col is not None and stratify_col in df.columns:
        try:
            _, sample_idx = train_test_split(
                np.arange(len(df)),
                test_size=n,
                random_state=seed,
                stratify=df[stratify_col]
            )
            return df.iloc[sample_idx].copy()
        except Exception:
            pass

    return df.sample(n=n, random_state=seed).copy()


def compute_binary_security_metrics(y_true_raw, y_pred_raw):
    y_true_raw = pd.Series(y_true_raw).astype(str).str.upper()
    y_pred_raw = pd.Series(y_pred_raw).astype(str).str.upper()

    tp = int(((y_true_raw == "ATTACK") & (y_pred_raw == "ATTACK")).sum())
    tn = int(((y_true_raw == "BENIGN") & (y_pred_raw == "BENIGN")).sum())
    fp = int(((y_true_raw == "BENIGN") & (y_pred_raw == "ATTACK")).sum())
    fn = int(((y_true_raw == "ATTACK") & (y_pred_raw == "BENIGN")).sum())

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


def compute_metrics(y_true, y_pred, y_true_raw, y_pred_raw):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

    security_metrics = compute_binary_security_metrics(y_true_raw, y_pred_raw)
    metrics.update(security_metrics)

    metrics["attack_difficulty_index"] = (
        1.0 - metrics["attack_recall"]
        if pd.notna(metrics["attack_recall"]) else np.nan
    )

    metrics["robustness_score"] = float(np.mean([
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["mcc"]
    ]))

    return metrics


def compute_balanced_loafo_test_size(family_count, benign_count):
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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = [c.strip() for c in df.columns]
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

family_counts = df["attack_family"].value_counts()

attack_families = [
    fam for fam in family_counts.index
    if fam not in ["BENIGN", "OTHER"] and family_counts[fam] >= MIN_TEST_PER_CLASS
]

print("\nBalanced LOAFO aileleri:")
for fam in attack_families:
    print(f"{fam}: {family_counts[fam]}")


# ============================================================
# BALANCED LOAFO DATA PREPARATION
# ============================================================

def prepare_balanced_loafo_data(held_out_family, seed):
    family_df_all = df[df["attack_family"] == held_out_family].copy()
    benign_df_all = df[df["binary_label"] == "BENIGN"].copy()

    n_test_per_class = compute_balanced_loafo_test_size(
        len(family_df_all),
        len(benign_df_all)
    )

    if n_test_per_class == 0:
        return None

    test_attack_df = family_df_all.sample(
        n=n_test_per_class,
        random_state=seed
    ).copy()

    test_benign_df = benign_df_all.sample(
        n=n_test_per_class,
        random_state=seed
    ).copy()

    test_ids = set(test_attack_df[ROW_ID_COL].tolist()).union(
        set(test_benign_df[ROW_ID_COL].tolist())
    )

    test_df = pd.concat([test_attack_df, test_benign_df], axis=0)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = df[
        (df["attack_family"] != held_out_family) &
        (~df[ROW_ID_COL].isin(test_ids))
    ].copy()

    if held_out_family in set(train_df["attack_family"].unique()):
        raise RuntimeError(f"LOAFO leakage detected: {held_out_family} remains in the training set.")

    train_classes = set(train_df["binary_label"].unique())
    if train_classes != {"BENIGN", "ATTACK"}:
        print(f"Warning: invalid training classes for {held_out_family}: {train_classes}")
        return None

    train_df = safe_sample_df(
        train_df,
        MAX_TRAIN_ROWS,
        seed,
        stratify_col="binary_label"
    )

    test_df = safe_sample_df(
        test_df,
        MAX_TEST_ROWS,
        seed,
        stratify_col="binary_label"
    )

    X_train = get_feature_matrix(train_df)
    X_test = get_feature_matrix(test_df)

    y_train_raw = train_df["binary_label"]
    y_test_raw = test_df["binary_label"]

    le = LabelEncoder()
    le.fit(["ATTACK", "BENIGN"])

    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    meta = {
        "held_out_family": held_out_family,
        "test_per_class": n_test_per_class,
        "remaining_train_benign": int((train_df["binary_label"] == "BENIGN").sum()),
        "remaining_train_attack": int((train_df["binary_label"] == "ATTACK").sum())
    }

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, le, meta


def evaluate_params_on_family(params, held_out_family, seed):
    prepared = prepare_balanced_loafo_data(held_out_family, seed)

    if prepared is None:
        return None

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, le, meta = prepared

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_raw = le.inverse_transform(y_pred)

    metrics = compute_metrics(y_test, y_pred, y_test_raw, y_pred_raw)
    metrics.update(meta)
    metrics["seed"] = seed

    return metrics


# ============================================================
# OPTUNA
# ============================================================

all_seed_final_results = []
all_seed_best_params = []

# Select more challenging and representative families for optimization.
preferred_objective_families = [
    "RECON",
    "SPOOFING",
    "WEB_MALWARE",
    "VULNERABILITY_SCAN",
    "BRUTE_FORCE"
]

objective_families = [
    fam for fam in preferred_objective_families
    if fam in attack_families
]

if len(objective_families) < 3:
    objective_families = attack_families[:3]

print("\nOptuna objective families:")
print(objective_families)


for optuna_seed in OPTUNA_SEEDS:
    print("\n====================================================")
    print(f"OPTUNA-XGBOOST BALANCED LOAFO | SEED={optuna_seed}")
    print("====================================================")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        }

        scores = []

        for fam in objective_families:
            res = evaluate_params_on_family(params, fam, seed=optuna_seed)
            if res is not None:
                # Robustness is the main objective, with a penalty for high false-positive rates.
                robustness = res["robustness_score"]
                fpr_penalty = 0.25 * res["false_positive_rate"]
                score = robustness - fpr_penalty
                scores.append(score)

        if len(scores) == 0:
            return 0.0

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"Optuna_TPE_XGBoost_Balanced_LOAFO_seed_{optuna_seed}"
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params.copy()

    best_param_row = best_params.copy()
    best_param_row["seed"] = optuna_seed
    best_param_row["best_validation_score"] = study.best_value
    best_param_row["n_trials"] = N_TRIALS
    best_param_row["objective_families"] = ";".join(objective_families)

    all_seed_best_params.append(best_param_row)

    study.trials_dataframe().to_csv(
        OUT_DIR / f"optuna_xgboost_balanced_loafo_trials_seed{optuna_seed}.csv",
        index=False
    )

    print("\nBest parameters:")
    print(best_params)

    print("\nStarting final testing on all balanced LOAFO families...")

    for fam in attack_families:
        print(f"Seed={optuna_seed} | Held-out family={fam}")

        res = evaluate_params_on_family(best_params, fam, seed=optuna_seed)

        if res is None:
            continue

        res["model"] = "Optuna-TPE-XGBoost"
        res["optimizer"] = "Optuna-TPE"
        res["n_trials"] = N_TRIALS
        res["best_validation_score"] = study.best_value

        all_seed_final_results.append(res)

    pd.DataFrame(all_seed_final_results).to_csv(
        OUT_DIR / "optuna_xgboost_balanced_loafo_final_results_partial.csv",
        index=False
    )


# ============================================================
# SAVE OUTPUTS AND SUMMARIES
# ============================================================

best_params_df = pd.DataFrame(all_seed_best_params)
best_params_df.to_csv(
    OUT_DIR / "optuna_xgboost_balanced_loafo_best_params_all_seeds.csv",
    index=False
)

final_df = pd.DataFrame(all_seed_final_results)
final_df.to_csv(
    OUT_DIR / "optuna_xgboost_balanced_loafo_final_results_all_seeds.csv",
    index=False
)

summary = final_df.groupby(
    ["held_out_family", "model"]
)[
    [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "benign_recall",
        "false_positive_rate",
        "false_negative_rate",
        "attack_difficulty_index",
        "robustness_score"
    ]
].agg(["mean", "std"]).reset_index()

summary.columns = [
    "_".join([str(x) for x in col if str(x) != ""])
    if isinstance(col, tuple) else str(col)
    for col in summary.columns
]

summary.to_csv(
    OUT_DIR / "optuna_xgboost_balanced_loafo_summary_mean_std.csv",
    index=False
)

model_summary = final_df.groupby("model")[
    [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "benign_recall",
        "false_positive_rate",
        "false_negative_rate",
        "attack_difficulty_index",
        "robustness_score"
    ]
].agg(["mean", "std", "min", "max"]).reset_index()

model_summary.columns = [
    "_".join([str(x) for x in col if str(x) != ""])
    if isinstance(col, tuple) else str(col)
    for col in model_summary.columns
]

model_summary.to_csv(
    OUT_DIR / "optuna_xgboost_balanced_loafo_model_summary.csv",
    index=False
)

print("\nOptuna-XGBoost Balanced LOAFO final experiment completed.")
print("Output directory:")
print(OUT_DIR)