# 07_statistical_effect_sizes.py
"""
Statistical strengthening analysis for the CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script does NOT retrain any model and does NOT modify the original
benchmark protocol. It strengthens the statistical analysis requested during
the manuscript analysis by explicitly treating attack family as the primary statistical
unit and seeds as repeated stochastic runs within each family.

Primary inferential analysis
----------------------------
1. For each model and held-out attack family, average performance across seeds.
2. Use the 8 held-out attack families as matched statistical units.
3. Apply the Friedman test across models for each metric.
4. If the Friedman test is significant, perform paired Wilcoxon signed-rank
   comparisons on family-level means.
5. Apply Holm correction for multiple pairwise comparisons.
6. Report paired rank-biserial correlation as an effect size.
7. Report family-cluster bootstrap 95% confidence intervals for:
   - model mean performance
   - pairwise model differences

Sensitivity analysis
--------------------
The original 40 family x seed matched blocks are also analyzed separately as
a sensitivity analysis. These rows are NOT treated as 40 fully independent
real-world samples; they are repeated benchmark runs.

No model fitting is performed.

Expected input
--------------
Preferred:
CICIoT2023_ROBUSTNESS_SENSITIVITY/robustness_run_level_augmented.csv

Fallback:
CICIoT2023_BENCHMARK/results_all_benchmark.csv
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# SETTINGS
# ============================================================

INPUT_CANDIDATES = [
    Path(
        "CICIoT2023_ROBUSTNESS_SENSITIVITY/"
        "robustness_run_level_augmented.csv"
    ),
    Path(
        "CICIoT2023_BENCHMARK/"
        "results_all_benchmark.csv"
    ),
    Path("robustness_run_level_augmented.csv"),
    Path("results_all_benchmark.csv"),
]

OUT_DIR = Path("CICIoT2023_STATISTICAL_ANALYSIS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TASK = "binary"
TARGET_SPLIT = "balanced_loafo"

ALPHA = 0.05

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 2026

# Metrics used in the manuscript analysis.
# Direction indicates which direction is better.
METRICS = {
    "macro_f1": "higher",
    "balanced_accuracy": "higher",
    "mcc": "higher",
    "attack_recall": "higher",
    "false_positive_rate": "lower",
    "robustness_equal_normalized": "higher",
}

MODEL_ORDER = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
]


# ============================================================
# HELPERS
# ============================================================

def resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No compatible benchmark result file was found.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Update INPUT_CANDIDATES at the top of the script if needed."
    )


def normalize_model_name(name: str) -> str:
    """
    Normalize common model-name variants while preserving readable names.
    """
    raw = str(name).strip()
    key = (
        raw.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    mapping = {
        "logisticregression": "LogisticRegression",
        "randomforest": "RandomForest",
        "extratrees": "ExtraTrees",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }

    return mapping.get(key, raw)


def ensure_normalized_robustness(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "mcc_01" not in out.columns:
        out["mcc_01"] = (
            pd.to_numeric(out["mcc"], errors="coerce") + 1.0
        ) / 2.0

    if "robustness_equal_normalized" not in out.columns:
        out["robustness_equal_normalized"] = (
            pd.to_numeric(out["macro_f1"], errors="coerce")
            + pd.to_numeric(out["balanced_accuracy"], errors="coerce")
            + pd.to_numeric(out["mcc_01"], errors="coerce")
        ) / 3.0

    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm step-down adjusted p-values.
    Returns adjusted p-values in original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        factor = m - rank
        adj = min(1.0, p[idx] * factor)
        running_max = max(running_max, adj)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted.tolist()


def paired_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired rank-biserial correlation based on signed-rank sums.

    Positive value:
        x tends to be larger than y.
    Negative value:
        y tends to be larger than x.

    Zero differences are discarded.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()

    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())

    denom = w_pos + w_neg

    if denom == 0:
        return 0.0

    return (w_pos - w_neg) / denom


def effect_magnitude(r: float) -> str:
    """
    Descriptive magnitude labels for absolute rank-biserial correlation.
    These labels are supplementary and should not replace the numeric effect.
    """
    a = abs(float(r))

    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    return "large"


def bootstrap_ci_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    iterations: int = BOOTSTRAP_ITERATIONS,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a mean.
    Resampling unit = attack family.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    n = len(values)
    stats = np.empty(iterations, dtype=float)

    for i in range(iterations):
        sample = rng.choice(values, size=n, replace=True)
        stats[i] = np.mean(sample)

    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))

    return low, high


def bootstrap_ci_paired_difference(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    iterations: int = BOOTSTRAP_ITERATIONS,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for paired mean difference x - y.
    Resampling unit = attack family.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    d = x[valid] - y[valid]

    if len(d) == 0:
        return np.nan, np.nan

    n = len(d)
    stats = np.empty(iterations, dtype=float)

    for i in range(iterations):
        sample = rng.choice(d, size=n, replace=True)
        stats[i] = np.mean(sample)

    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))

    return low, high


def ordered_models(df: pd.DataFrame) -> list[str]:
    available = list(df["model"].dropna().unique())

    ordered = [m for m in MODEL_ORDER if m in available]
    extras = sorted([m for m in available if m not in ordered])

    return ordered + extras


def favorable_transform(values: pd.Series, direction: str) -> pd.Series:
    """
    For ranking only:
    higher transformed value = better performance.
    """
    if direction == "higher":
        return values

    if direction == "lower":
        return -values

    raise ValueError(f"Unknown direction: {direction}")


# ============================================================
# DATA PREPARATION
# ============================================================

def load_analysis_data() -> tuple[pd.DataFrame, Path]:
    path = resolve_input_path()
    df = pd.read_csv(path)

    required = [
        "model",
        "seed",
        "held_out_family",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "false_positive_rate",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    if "status" in df.columns:
        df = df[
            df["status"].astype(str).str.lower() == "success"
        ].copy()

    if "task" in df.columns:
        df = df[
            df["task"].astype(str).str.lower()
            == TARGET_TASK.lower()
        ].copy()

    if "split" in df.columns:
        df = df[
            df["split"].astype(str).str.lower()
            == TARGET_SPLIT.lower()
        ].copy()

    if df.empty:
        raise ValueError(
            "No balanced LOAFO binary rows were found in the input file."
        )

    df["model"] = df["model"].map(normalize_model_name)
    df = ensure_normalized_robustness(df)

    return df, path


# ============================================================
# PRIMARY FAMILY-LEVEL ANALYSIS
# ============================================================

def create_family_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average seeds within each held-out attack family.

    Primary statistical unit = held-out attack family.
    Seeds are repeated stochastic runs within that unit.
    """
    metrics = list(METRICS.keys())

    family = (
        df.groupby(["held_out_family", "model"])[metrics]
        .mean()
        .reset_index()
    )

    return family


def model_ci_table(
    family_df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for metric, direction in METRICS.items():
        for model in models:
            values = (
                family_df.loc[
                    family_df["model"] == model,
                    metric,
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) == 0:
                continue

            ci_low, ci_high = bootstrap_ci_mean(values, rng)

            rows.append({
                "metric": metric,
                "direction": direction,
                "model": model,
                "n_attack_families": len(values),
                "family_level_mean": float(np.mean(values)),
                "family_level_std": float(
                    np.std(values, ddof=1)
                    if len(values) > 1
                    else 0.0
                ),
                "family_level_median": float(np.median(values)),
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
            })

    return pd.DataFrame(rows)


def friedman_family_analysis(
    family_df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    rows = []

    for metric, direction in METRICS.items():
        pivot = family_df.pivot(
            index="held_out_family",
            columns="model",
            values=metric,
        )

        usable_models = [
            m for m in models
            if m in pivot.columns
        ]

        pivot = pivot[usable_models].dropna()

        if len(usable_models) < 3 or len(pivot) < 2:
            continue

        transformed = {
            m: favorable_transform(pivot[m], direction).to_numpy()
            for m in usable_models
        }

        stat, p_value = friedmanchisquare(
            *[transformed[m] for m in usable_models]
        )

        # Friedman Kendall's W
        n = len(pivot)
        k = len(usable_models)
        kendall_w = float(stat / (n * (k - 1)))

        # Mean ranks: rank 1 = best.
        rank_matrix = pivot.copy()

        if direction == "higher":
            ranks = rank_matrix.rank(
                axis=1,
                ascending=False,
                method="average",
            )
        else:
            ranks = rank_matrix.rank(
                axis=1,
                ascending=True,
                method="average",
            )

        mean_ranks = ranks.mean(axis=0)

        row = {
            "metric": metric,
            "direction": direction,
            "n_attack_families": n,
            "n_models": k,
            "friedman_chi2": float(stat),
            "p_value": float(p_value),
            "kendall_w": kendall_w,
        }

        for model in usable_models:
            row[f"mean_rank_{model}"] = float(mean_ranks[model])

        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_family_analysis(
    family_df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    all_rows = []

    for metric, direction in METRICS.items():
        pivot = family_df.pivot(
            index="held_out_family",
            columns="model",
            values=metric,
        )

        metric_rows = []
        raw_p_values = []

        pair_index = 0

        for model_a, model_b in combinations(models, 2):
            if model_a not in pivot.columns or model_b not in pivot.columns:
                continue

            paired = pivot[[model_a, model_b]].dropna()

            if len(paired) < 2:
                continue

            x = paired[model_a].to_numpy(dtype=float)
            y = paired[model_b].to_numpy(dtype=float)

            difference = x - y

            # Wilcoxon can fail if all paired differences are zero.
            if np.allclose(difference, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rbc = paired_rank_biserial(x, y)

            # Direction-adjusted effect:
            # positive => model_a is better than model_b.
            favorable_rbc = (
                rbc if direction == "higher"
                else -rbc
            )

            rng = np.random.default_rng(
                BOOTSTRAP_SEED + pair_index
            )

            ci_low, ci_high = bootstrap_ci_paired_difference(
                x,
                y,
                rng,
            )

            metric_rows.append({
                "metric": metric,
                "direction": direction,
                "model_a": model_a,
                "model_b": model_b,
                "n_attack_families": len(paired),
                "mean_model_a": float(np.mean(x)),
                "mean_model_b": float(np.mean(y)),
                "mean_difference_a_minus_b": float(np.mean(difference)),
                "bootstrap_95ci_difference_low": ci_low,
                "bootstrap_95ci_difference_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial_a_minus_b": float(rbc),
                "favorable_rank_biserial_a_vs_b": float(favorable_rbc),
                "effect_magnitude": effect_magnitude(favorable_rbc),
            })

            raw_p_values.append(p_value)
            pair_index += 1

        adjusted = holm_adjust(raw_p_values)

        for row, p_holm in zip(metric_rows, adjusted):
            row["p_holm"] = float(p_holm)
            row["significant_holm_0_05"] = bool(
                p_holm < ALPHA
            )

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ============================================================
# 40-BLOCK SENSITIVITY ANALYSIS
# ============================================================

def forty_block_friedman(
    df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    """
    Reproduces the original family x seed matched-block analysis as a
    sensitivity analysis. These 40 rows are repeated benchmark blocks,
    not 40 independent attack families.
    """
    rows = []

    for metric, direction in METRICS.items():
        pivot = df.pivot_table(
            index=["held_out_family", "seed"],
            columns="model",
            values=metric,
            aggfunc="mean",
        )

        usable_models = [
            m for m in models
            if m in pivot.columns
        ]

        pivot = pivot[usable_models].dropna()

        if len(usable_models) < 3 or len(pivot) < 2:
            continue

        transformed = {
            m: favorable_transform(pivot[m], direction).to_numpy()
            for m in usable_models
        }

        stat, p_value = friedmanchisquare(
            *[transformed[m] for m in usable_models]
        )

        n = len(pivot)
        k = len(usable_models)
        kendall_w = float(stat / (n * (k - 1)))

        if direction == "higher":
            ranks = pivot.rank(
                axis=1,
                ascending=False,
                method="average",
            )
        else:
            ranks = pivot.rank(
                axis=1,
                ascending=True,
                method="average",
            )

        mean_ranks = ranks.mean(axis=0)

        row = {
            "metric": metric,
            "direction": direction,
            "n_family_seed_blocks": n,
            "n_models": k,
            "friedman_chi2": float(stat),
            "p_value": float(p_value),
            "kendall_w": kendall_w,
            "interpretation_note": (
                "Sensitivity analysis using repeated family x seed "
                "benchmark blocks; not treated as 40 independent "
                "real-world attack-family observations."
            ),
        }

        for model in usable_models:
            row[f"mean_rank_{model}"] = float(mean_ranks[model])

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df, input_path = load_analysis_data()
    models = ordered_models(df)

    family_df = create_family_level_data(df)

    n_families = int(
        family_df["held_out_family"].nunique()
    )
    n_seeds = int(df["seed"].nunique())
    n_models = int(df["model"].nunique())
    n_run_rows = int(len(df))

    print(f"Input: {input_path}")
    print(
        f"Rows={n_run_rows}, "
        f"families={n_families}, "
        f"seeds={n_seeds}, "
        f"models={n_models}"
    )

    # Save primary family-level dataset.
    family_df.to_csv(
        OUT_DIR / "family_level_seed_averaged_results.csv",
        index=False,
    )

    # Model mean + cluster bootstrap CI.
    ci_table = model_ci_table(
        family_df,
        models,
    )
    ci_table.to_csv(
        OUT_DIR / "model_family_level_bootstrap_95ci.csv",
        index=False,
    )

    # Primary Friedman.
    friedman_primary = friedman_family_analysis(
        family_df,
        models,
    )
    friedman_primary.to_csv(
        OUT_DIR / "friedman_primary_family_level.csv",
        index=False,
    )

    # Primary pairwise Wilcoxon-Holm + effect sizes.
    pairwise = pairwise_family_analysis(
        family_df,
        models,
    )
    pairwise.to_csv(
        OUT_DIR / "pairwise_wilcoxon_holm_effect_sizes.csv",
        index=False,
    )

    # Original 40-block approach as sensitivity analysis.
    friedman_40 = forty_block_friedman(
        df,
        models,
    )
    friedman_40.to_csv(
        OUT_DIR / "friedman_40block_sensitivity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compact table: pairwise comparisons involving RF/LGBM
    # --------------------------------------------------------

    focus_models = {"RandomForest", "LightGBM"}

    focus = pairwise[
        pairwise["model_a"].isin(focus_models)
        | pairwise["model_b"].isin(focus_models)
    ].copy()

    focus.to_csv(
        OUT_DIR / "pairwise_focus_randomforest_lightgbm.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "input_file": str(input_path),
        "primary_statistical_unit": "held-out attack family",
        "seed_handling": (
            "Seeds are repeated stochastic runs within each held-out "
            "attack family and are averaged before primary inferential testing."
        ),
        "primary_n_attack_families": n_families,
        "n_seeds_per_family_expected": n_seeds,
        "n_models": n_models,
        "n_run_level_rows": n_run_rows,
        "primary_tests": [
            "Friedman test on seed-averaged family-level results",
            "paired Wilcoxon signed-rank tests",
            "Holm correction",
            "paired rank-biserial correlation",
            "family-cluster bootstrap 95% confidence intervals",
        ],
        "sensitivity_analysis": (
            "Original 40 family x seed matched-block Friedman analysis "
            "retained as a sensitivity analysis."
        ),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "alpha": ALPHA,
        "metrics": METRICS,
        "models": models,
    }

    with open(
        OUT_DIR / "statistical_analysis_manifest.json",
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
        "=" * 82,
        "CICIoT2023 STATISTICAL EFFECT SIZE AND CONFIDENCE-INTERVAL ANALYSIS",
        "=" * 82,
        f"Input file                    : {input_path}",
        f"Run-level rows                : {n_run_rows}",
        f"Held-out attack families      : {n_families}",
        f"Seeds                         : {n_seeds}",
        f"Models                        : {n_models}",
        "",
        "PRIMARY STATISTICAL UNIT",
        "-" * 82,
        "Held-out attack family.",
        (
            "The five seeds are treated as repeated stochastic runs within "
            "each family and are averaged before primary inferential testing."
        ),
        "",
        "PRIMARY FRIEDMAN RESULTS",
        "-" * 82,
    ]

    for _, row in friedman_primary.iterrows():
        lines.append(
            f"- {row['metric']}: "
            f"chi2={row['friedman_chi2']:.4f}, "
            f"p={row['p_value']:.6g}, "
            f"Kendall W={row['kendall_w']:.4f}, "
            f"n={int(row['n_attack_families'])} families"
        )

    lines.extend([
        "",
        "PAIRWISE WILCOXON-HOLM RESULTS",
        "-" * 82,
        (
            "Full pairwise results, Holm-adjusted p-values, paired "
            "rank-biserial effect sizes, and bootstrap 95% confidence "
            "intervals are written to:"
        ),
        "pairwise_wilcoxon_holm_effect_sizes.csv",
        "",
        "40-BLOCK SENSITIVITY ANALYSIS",
        "-" * 82,
        (
            "The original 8 families x 5 seeds = 40 matched benchmark blocks "
            "are retained as a separate sensitivity analysis. These blocks "
            "are not described as 40 independent attack-family observations."
        ),
        "",
        "Files written:",
        "- family_level_seed_averaged_results.csv",
        "- model_family_level_bootstrap_95ci.csv",
        "- friedman_primary_family_level.csv",
        "- pairwise_wilcoxon_holm_effect_sizes.csv",
        "- pairwise_focus_randomforest_lightgbm.csv",
        "- friedman_40block_sensitivity.csv",
        "- statistical_analysis_manifest.json",
        "- statistical_analysis_summary.txt",
        "=" * 82,
    ])

    summary = "\n".join(lines)

    with open(
        OUT_DIR / "statistical_analysis_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)

    print()
    print(summary)
    print()
    print(f"Output directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
