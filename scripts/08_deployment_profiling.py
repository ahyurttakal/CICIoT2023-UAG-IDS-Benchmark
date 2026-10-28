# 11_deployment_profiling.py
"""
Deployment-oriented profiling for the CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script provides computational and deployment-oriented profiling while
preserving the primary benchmark design.

It profiles the original five supervised classifiers under the same balanced
LOAFO setting and reports:

- training time
- test inference time
- inference latency per flow
- throughput (flows/second)
- serialized model size
- peak process-memory increase during training (when psutil is available)
- peak process-memory increase during inference (when psutil is available)
- hardware/software environment

Design
------
To reduce unnecessary repeated computation while retaining variability across
attack families and stochastic runs, profiling uses:

    8 held-out attack families x 3 seeds x 5 models

Seeds:
    42, 123, 2026

The primary benchmark settings are retained:
    MAX_TRAIN_ROWS = 300,000
    balanced LOAFO test construction
    up to 25,000 attack + 25,000 benign test samples per family
    at least 10,000 benign rows reserved before test construction

Important interpretation note
-----------------------------
Process-memory measurements are approximate deployment/profile indicators.
Peak RSS is sampled periodically at the process level and therefore includes
Python/runtime overhead in addition to model allocations. Serialized model
size is measured from a temporary joblib file.

No threshold calibration is performed here. Predictions use each classifier's
native/default decision rule, matching the primary benchmark configuration.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HAS_XGBOOST = False
HAS_LIGHTGBM = False
HAS_PSUTIL = False

try:
    import xgboost
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    xgboost = None
    XGBClassifier = None

try:
    import lightgbm
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    lightgbm = None
    LGBMClassifier = None

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("mergeddata") / "ciciot2023_working_with_source.csv"

OUT_DIR = Path("CICIoT2023_DEPLOYMENT_PROFILING")
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

# Memory monitor sampling interval.
MEMORY_POLL_SECONDS = 0.05


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
            "Update DATA_PATH if the working dataset is elsewhere."
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
    n: Optional[int],
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
        [test_attack_df, test_benign_df],
        axis=0,
    ).sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    test_ids = set(test_df[ROW_ID_COL].tolist())

    train_pool = df[
        (df["attack_family"] != held_out_family)
        & (~df[ROW_ID_COL].isin(test_ids))
    ].copy()

    if held_out_family in set(
        train_pool["attack_family"].unique()
    ):
        raise RuntimeError(
            f"Held-out family leakage detected: {held_out_family}"
        )

    overlap = set(
        train_pool[ROW_ID_COL]
    ).intersection(
        set(test_df[ROW_ID_COL])
    )

    if overlap:
        raise RuntimeError(
            f"Train/test overlap detected: {held_out_family}"
        )

    train_df = safe_stratified_sample(
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

    X_train = get_feature_matrix(train_df)
    X_test = get_feature_matrix(test_df)

    y_train = (
        train_df["binary_label"]
        .astype(str)
        .str.upper()
        .eq("ATTACK")
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_df["binary_label"]
        .astype(str)
        .str.upper()
        .eq("ATTACK")
        .astype(int)
        .to_numpy()
    )

    meta = {
        "held_out_family": held_out_family,
        "seed": seed,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_attack_rows": int(
            (test_df["binary_label"] == "ATTACK").sum()
        ),
        "test_benign_rows": int(
            (test_df["binary_label"] == "BENIGN").sum()
        ),
        "n_features": int(X_train.shape[1]),
        "held_out_family_absent_from_training": True,
        "train_test_row_overlap": 0,
    }

    return X_train, X_test, y_train, y_test, meta


# ============================================================
# MODELS
# ============================================================

def make_models(seed: int) -> Dict[str, object]:
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
# MEMORY MONITOR
# ============================================================

class PeakRSSMonitor:
    """
    Approximate peak process RSS sampled in a background thread.
    """

    def __init__(self, interval: float = MEMORY_POLL_SECONDS):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.baseline_rss = np.nan
        self.peak_rss = np.nan

    def _rss(self) -> float:
        if not HAS_PSUTIL:
            return np.nan

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss)

    def _worker(self):
        while not self._stop.is_set():
            rss = self._rss()

            if np.isfinite(rss):
                if not np.isfinite(self.peak_rss):
                    self.peak_rss = rss
                else:
                    self.peak_rss = max(
                        self.peak_rss,
                        rss,
                    )

            time.sleep(self.interval)

    def start(self):
        if not HAS_PSUTIL:
            return

        self.baseline_rss = self._rss()
        self.peak_rss = self.baseline_rss
        self._stop.clear()

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if not HAS_PSUTIL:
            return {
                "baseline_rss_mb": np.nan,
                "peak_rss_mb": np.nan,
                "peak_rss_delta_mb": np.nan,
            }

        self._stop.set()

        if self._thread is not None:
            self._thread.join(
                timeout=max(
                    1.0,
                    self.interval * 5,
                )
            )

        baseline_mb = self.baseline_rss / (1024 ** 2)
        peak_mb = self.peak_rss / (1024 ** 2)

        return {
            "baseline_rss_mb": float(baseline_mb),
            "peak_rss_mb": float(peak_mb),
            "peak_rss_delta_mb": float(
                max(
                    0.0,
                    peak_mb - baseline_mb,
                )
            ),
        }


# ============================================================
# MODEL SIZE
# ============================================================

def serialized_model_size_mb(model) -> float:
    """
    Serialize to a temporary joblib file, measure size, then delete it.
    """
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".joblib",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        joblib.dump(
            model,
            tmp_path,
            compress=3,
        )

        size_bytes = tmp_path.stat().st_size
        return float(size_bytes / (1024 ** 2))

    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ============================================================
# PROFILE ONE MODEL
# ============================================================

def profile_model(
    model_name: str,
    model,
    X_train,
    y_train,
    X_test,
    meta: Dict,
) -> Dict:
    gc.collect()

    train_monitor = PeakRSSMonitor()
    train_monitor.start()

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
    )
    training_seconds = (
        time.perf_counter()
        - t0
    )

    train_memory = train_monitor.stop()

    model_size_mb = serialized_model_size_mb(
        model
    )

    gc.collect()

    infer_monitor = PeakRSSMonitor()
    infer_monitor.start()

    t1 = time.perf_counter()
    predictions = model.predict(
        X_test
    )
    inference_seconds = (
        time.perf_counter()
        - t1
    )

    infer_memory = infer_monitor.stop()

    n_test = len(X_test)

    inference_ms_per_flow = (
        inference_seconds / n_test * 1000.0
    )

    throughput = (
        n_test / inference_seconds
        if inference_seconds > 0
        else np.nan
    )

    # Touch predictions so timing cannot be optimized away.
    predicted_attack_fraction = float(
        np.mean(
            np.asarray(predictions).astype(int)
        )
    )

    return {
        **meta,
        "model": model_name,
        "training_seconds": float(
            training_seconds
        ),
        "test_inference_seconds": float(
            inference_seconds
        ),
        "test_inference_ms_per_flow": float(
            inference_ms_per_flow
        ),
        "test_throughput_flows_per_second": float(
            throughput
        ),
        "serialized_model_size_mb": float(
            model_size_mb
        ),
        "training_baseline_rss_mb": train_memory[
            "baseline_rss_mb"
        ],
        "training_peak_rss_mb": train_memory[
            "peak_rss_mb"
        ],
        "training_peak_rss_delta_mb": train_memory[
            "peak_rss_delta_mb"
        ],
        "inference_baseline_rss_mb": infer_memory[
            "baseline_rss_mb"
        ],
        "inference_peak_rss_mb": infer_memory[
            "peak_rss_mb"
        ],
        "inference_peak_rss_delta_mb": infer_memory[
            "peak_rss_delta_mb"
        ],
        "predicted_attack_fraction": predicted_attack_fraction,
    }


# ============================================================
# ENVIRONMENT
# ============================================================

def environment_info() -> Dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_available": HAS_XGBOOST,
        "lightgbm_available": HAS_LIGHTGBM,
        "psutil_available": HAS_PSUTIL,
    }

    if HAS_XGBOOST:
        info["xgboost_version"] = xgboost.__version__

    if HAS_LIGHTGBM:
        info["lightgbm_version"] = lightgbm.__version__

    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = float(
            vm.total / (1024 ** 3)
        )

    return info


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
    metrics = [
        "training_seconds",
        "test_inference_seconds",
        "test_inference_ms_per_flow",
        "test_throughput_flows_per_second",
        "serialized_model_size_mb",
        "training_peak_rss_delta_mb",
        "inference_peak_rss_delta_mb",
    ]

    model_summary = (
        results_df.groupby(
            "model"
        )[metrics]
        .agg(
            ["mean", "std", "median", "min", "max"]
        )
        .reset_index()
    )

    model_summary = flatten_columns(
        model_summary
    )

    model_summary.to_csv(
        OUT_DIR / "deployment_profile_model_summary.csv",
        index=False,
    )

    family_summary = (
        results_df.groupby(
            ["held_out_family", "model"]
        )[metrics]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    family_summary = flatten_columns(
        family_summary
    )

    family_summary.to_csv(
        OUT_DIR / "deployment_profile_by_family.csv",
        index=False,
    )

    # Compact manuscript-oriented table with means only.
    compact = (
        results_df.groupby(
            "model"
        )[metrics]
        .mean()
        .reset_index()
    )

    compact.to_csv(
        OUT_DIR / "deployment_profile_compact_table.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    env = environment_info()

    with open(
        OUT_DIR / "deployment_environment.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            env,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Environment:")
    for key, value in env.items():
        print(
            f"  {key}: {value}"
        )

    print("\nLoading dataset...")
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
            f"  {fam}: {family_counts[fam]:,}"
        )

    available_models = list(
        make_models(
            SEEDS[0]
        ).keys()
    )

    print("\nModels:")
    for model_name in available_models:
        print(
            f"  {model_name}"
        )

    if not HAS_PSUTIL:
        print(
            "\nWARNING: psutil is not installed. "
            "Peak RSS memory columns will be NaN. "
            "Install with: pip install psutil"
        )

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

    for seed in SEEDS:
        for held_out_family in attack_families:
            print()
            print("=" * 84)
            print(
                f"DEPLOYMENT PROFILE | "
                f"family={held_out_family} | "
                f"seed={seed}"
            )
            print("=" * 84)

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
                    f"Profiling {model_name}..."
                )

                try:
                    row = profile_model(
                        model_name=model_name,
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        meta=meta,
                    )

                    all_rows.append(
                        row
                    )

                    print(
                        f"  train={row['training_seconds']:.3f}s | "
                        f"latency={row['test_inference_ms_per_flow']:.6f} ms/flow | "
                        f"throughput={row['test_throughput_flows_per_second']:.1f} flows/s | "
                        f"size={row['serialized_model_size_mb']:.3f} MB | "
                        f"train_peak_delta={row['training_peak_rss_delta_mb']:.2f} MB"
                    )

                except Exception as exc:
                    print(
                        f"ERROR | {model_name} | "
                        f"{held_out_family} | "
                        f"seed={seed}: {exc}"
                    )

                gc.collect()

            pd.DataFrame(
                all_rows
            ).to_csv(
                OUT_DIR / "deployment_profile_results_partial.csv",
                index=False,
            )

    results_df = pd.DataFrame(
        all_rows
    )

    if results_df.empty:
        raise RuntimeError(
            "No deployment profiling results were produced."
        )

    results_df.to_csv(
        OUT_DIR / "deployment_profile_results_all.csv",
        index=False,
    )

    create_summary_tables(
        results_df
    )

    manifest = {
        "dataset_path": str(
            DATA_PATH
        ),
        "held_out_attack_families": attack_families,
        "seeds": SEEDS,
        "models": sorted(
            results_df["model"]
            .dropna()
            .unique()
            .tolist()
        ),
        "max_train_rows": MAX_TRAIN_ROWS,
        "max_test_rows": MAX_TEST_ROWS,
        "loafo_test_per_class_cap": LOAFO_TEST_PER_CLASS_CAP,
        "min_train_benign_remaining_before_test_selection": (
            MIN_TRAIN_BENIGN_REMAINING
        ),
        "memory_measurement": (
            "Peak process RSS sampled every "
            f"{MEMORY_POLL_SECONDS:.2f} s using psutil; "
            "reported as peak increase over process RSS at the start "
            "of the measured operation."
            if HAS_PSUTIL
            else "psutil unavailable; memory columns are NaN."
        ),
        "model_size_measurement": (
            "Temporary joblib serialization with compress=3."
        ),
        "profiling_note": (
            "Timing and memory results are hardware/software dependent "
            "and should be reported with the accompanying environment metadata."
        ),
    }

    with open(
        OUT_DIR / "deployment_profile_manifest.json",
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
        "=" * 84,
        "CICIoT2023 DEPLOYMENT-ORIENTED PROFILING",
        "=" * 84,
        f"Dataset                    : {DATA_PATH}",
        f"Attack families            : {len(attack_families)}",
        f"Seeds                      : {len(SEEDS)} ({SEEDS})",
        (
            "Models                     : "
            + ", ".join(
                sorted(
                    results_df["model"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )
        ),
        f"Predictive features        : {int(results_df['n_features'].iloc[0])}",
        f"psutil memory profiling    : {HAS_PSUTIL}",
        "",
        "MEASURED DEPLOYMENT METRICS",
        "-" * 84,
        "- Training time",
        "- Test inference time",
        "- Inference latency per flow",
        "- Throughput (flows/s)",
        "- Serialized model size (MB)",
        "- Peak training RSS increase (MB)",
        "- Peak inference RSS increase (MB)",
        "",
        "INTERPRETATION NOTE",
        "-" * 84,
        (
            "Timing and memory values are platform-dependent. "
            "Use deployment_environment.json when reporting these results."
        ),
        (
            "Peak RSS is a process-level sampled measurement and includes "
            "runtime overhead; it should be interpreted as an approximate "
            "deployment-memory indicator rather than exact model-only memory."
        ),
        "",
        "OUTPUTS",
        "-" * 84,
        "- deployment_profile_results_all.csv",
        "- deployment_profile_model_summary.csv",
        "- deployment_profile_by_family.csv",
        "- deployment_profile_compact_table.csv",
        "- deployment_environment.json",
        "- deployment_profile_manifest.json",
        "- deployment_profile_summary.txt",
        "=" * 84,
    ]

    summary_text = "\n".join(
        lines
    )

    with open(
        OUT_DIR / "deployment_profile_summary.txt",
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
