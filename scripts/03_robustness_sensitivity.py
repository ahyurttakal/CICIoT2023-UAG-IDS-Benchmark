# 06_robustness_sensitivity.py
"""
Robustness-score normalization and sensitivity analysis for the
CICIoT2023 UAG-IDS benchmark.

Purpose
-------
This script does NOT retrain any model and does NOT modify the original
benchmark protocol. It reads the existing run-level benchmark results and:

1. Rescales MCC from [-1, 1] to [0, 1]:
       MCC_01 = (MCC + 1) / 2

2. Computes a normalized equal-weight robustness score:
       R_equal = (Macro-F1 + Balanced Accuracy + MCC_01) / 3

3. Evaluates whether model-level conclusions are sensitive to alternative
   weighting schemes.

4. Produces model-level, family-level, and ranking-stability tables for
   transparent reproducibility reporting.

No model fitting is performed in this script.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RESULTS_CANDIDATES = [
    Path("CICIoT2023_BENCHMARK") / "results_all_benchmark.csv",
    Path(r"CICIoT2023_BENCHMARK/results_all_benchmark.csv"),
    Path("results_all_benchmark.csv"),
]

OUT_DIR = Path("CICIoT2023_ROBUSTNESS_SENSITIVITY")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SPLIT = "balanced_loafo"
TARGET_TASK = "binary"

REQUIRED_COLUMNS = [
    "task",
    "split",
    "seed",
    "model",
    "held_out_family",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
]

# Alternative normalized aggregation schemes.
# All three components are on [0, 1] before weighting.
WEIGHT_SCHEMES = {
    "equal_normalized": {
        "macro_f1": 1 / 3,
        "balanced_accuracy": 1 / 3,
        "mcc_01": 1 / 3,
    },
    "f1_emphasis": {
        "macro_f1": 0.50,
        "balanced_accuracy": 0.25,
        "mcc_01": 0.25,
    },
    "balanced_accuracy_emphasis": {
        "macro_f1": 0.25,
        "balanced_accuracy": 0.50,
        "mcc_01": 0.25,
    },
    "mcc_emphasis": {
        "macro_f1": 0.25,
        "balanced_accuracy": 0.25,
        "mcc_01": 0.50,
    },
    "f1_balanced_accuracy_only": {
        "macro_f1": 0.50,
        "balanced_accuracy": 0.50,
        "mcc_01": 0.00,
    },
}


# ============================================================
# HELPERS
# ============================================================

def resolve_results_path() -> Path:
    for path in RESULTS_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in RESULTS_CANDIDATES)
    raise FileNotFoundError(
        "Benchmark result file was not found.\n"
        "Expected one of:\n"
        f"{searched}\n\n"
        "Update RESULTS_CANDIDATES at the top of the script if needed."
    )


def require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Required columns are missing from the benchmark results:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def normalize_mcc(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values + 1.0) / 2.0


def weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Weighted arithmetic mean. Weights are expected to sum to 1.
    """
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"Weights must sum to 1. Current sum = {total_weight}"
        )

    score = pd.Series(0.0, index=df.index, dtype=float)

    for col, weight in weights.items():
        score = score + pd.to_numeric(df[col], errors="coerce") * weight

    return score


def rank_descending(series: pd.Series) -> pd.Series:
    """
    Rank 1 = highest score.
    """
    return series.rank(method="average", ascending=False)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join(str(v) for v in col if str(v) != "")
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    return out


# ============================================================
# ANALYSIS
# ============================================================

def main() -> None:
    results_path = resolve_results_path()

    print(f"Reading benchmark results: {results_path}")

    df = pd.read_csv(results_path)
    require_columns(df)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()

    loafo = df[
        (df["task"].astype(str).str.lower() == TARGET_TASK.lower())
        & (df["split"].astype(str).str.lower() == TARGET_SPLIT.lower())
    ].copy()

    if loafo.empty:
        raise ValueError(
            f"No rows found for task={TARGET_TASK} and split={TARGET_SPLIT}."
        )

    # --------------------------------------------------------
    # Normalize MCC to the common [0, 1] scale
    # --------------------------------------------------------

    loafo["mcc_01"] = normalize_mcc(loafo["mcc"])

    # --------------------------------------------------------
    # Compute normalized sensitivity scores
    # --------------------------------------------------------

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        loafo[f"robustness_{scheme_name}"] = weighted_score(
            loafo,
            weights,
        )

    # Save all run-level augmented results.
    loafo.to_csv(
        OUT_DIR / "robustness_run_level_augmented.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Model-level summaries
    # --------------------------------------------------------

    score_columns = [f"robustness_{name}" for name in WEIGHT_SCHEMES]

    model_summary = (
        loafo.groupby("model")[score_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    model_summary = flatten_columns(model_summary)

    model_summary.to_csv(
        OUT_DIR / "robustness_model_summary.csv",
        index=False,
    )

    # Long-format model rankings by scheme.
    ranking_rows = []

    for score_col in score_columns:
        means = (
            loafo.groupby("model")[score_col]
            .mean()
            .sort_values(ascending=False)
        )

        ranks = rank_descending(means)

        for model, mean_score in means.items():
            ranking_rows.append({
                "score_definition": score_col,
                "model": model,
                "mean_score": float(mean_score),
                "rank": float(ranks.loc[model]),
            })

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(
        OUT_DIR / "robustness_model_rankings.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Ranking stability relative to normalized equal weights
    # --------------------------------------------------------

    rank_pivot = ranking_df.pivot(
        index="model",
        columns="score_definition",
        values="rank",
    )

    reference_col = "robustness_equal_normalized"

    if reference_col not in rank_pivot.columns:
        raise RuntimeError(
            f"Reference ranking column not found: {reference_col}"
        )

    stability_rows = []

    for col in rank_pivot.columns:
        if col == reference_col:
            reference_series = rank_pivot[reference_col].dropna()
            stability_rows.append({
                "reference": reference_col,
                "comparison": col,
                "spearman_rank_correlation": 1.0,
                "max_absolute_rank_change": 0.0,
                "same_top_model": True,
                "reference_top_model": reference_series.idxmin(),
                "comparison_top_model": reference_series.idxmin(),
            })
            continue

        paired = rank_pivot[[reference_col, col]].dropna()

        # Spearman correlation between model ranks.
        rho = paired[reference_col].corr(
            paired[col],
            method="spearman",
        )

        max_abs_rank_change = float(
            (paired[reference_col] - paired[col]).abs().max()
        )

        same_top_model = bool(
            paired[reference_col].idxmin()
            == paired[col].idxmin()
        )

        stability_rows.append({
            "reference": reference_col,
            "comparison": col,
            "spearman_rank_correlation": rho,
            "max_absolute_rank_change": max_abs_rank_change,
            "same_top_model": same_top_model,
            "reference_top_model": paired[reference_col].idxmin(),
            "comparison_top_model": paired[col].idxmin(),
        })

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(
        OUT_DIR / "robustness_ranking_stability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Family-level sensitivity
    # --------------------------------------------------------

    family_model_summary = (
        loafo.groupby(["held_out_family", "model"])[score_columns]
        .mean()
        .reset_index()
    )

    family_model_summary.to_csv(
        OUT_DIR / "robustness_by_family_and_model.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metadata / manifest
    # --------------------------------------------------------

    models = sorted(loafo["model"].dropna().astype(str).unique().tolist())
    families = sorted(
        loafo["held_out_family"].dropna().astype(str).unique().tolist()
    )
    seeds = sorted(
        pd.to_numeric(loafo["seed"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    equal_ranking = ranking_df[
        ranking_df["score_definition"] == reference_col
    ].sort_values("rank")

    manifest = {
        "input_results": str(results_path),
        "task": TARGET_TASK,
        "split": TARGET_SPLIT,
        "n_run_rows": int(len(loafo)),
        "models": models,
        "held_out_families": families,
        "seeds": seeds,
        "mcc_normalization": "(MCC + 1) / 2",
        "primary_normalized_score": (
            "(Macro-F1 + Balanced Accuracy + MCC_01) / 3"
        ),
        "weight_schemes": WEIGHT_SCHEMES,
        "normalized_equal_weight_model_order": equal_ranking["model"].tolist(),
        "note": (
            "This analysis uses existing benchmark results only and "
            "does not retrain models."
        ),
    }

    with open(
        OUT_DIR / "robustness_sensitivity_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary_lines = [
        "=" * 76,
        "CICIoT2023 ROBUSTNESS-SCORE NORMALIZATION AND SENSITIVITY ANALYSIS",
        "=" * 76,
        f"Input results                 : {results_path}",
        f"Run-level LOAFO rows          : {len(loafo)}",
        f"Models                        : {len(models)}",
        f"Held-out attack families      : {len(families)}",
        f"Seeds                         : {len(seeds)}",
        "",
        "PRIMARY NORMALIZED DEFINITION",
        "-" * 76,
        "MCC_01 = (MCC + 1) / 2",
        "R_equal = (Macro-F1 + Balanced Accuracy + MCC_01) / 3",
        "",
        "MODEL RANKING: NORMALIZED EQUAL-WEIGHT SCORE",
        "-" * 76,
    ]

    for _, row in equal_ranking.iterrows():
        summary_lines.append(
            f"{int(row['rank'])}. {row['model']}: "
            f"{row['mean_score']:.6f}"
        )

    summary_lines.extend([
        "",
        "RANKING STABILITY ACROSS ALTERNATIVE WEIGHTS",
        "-" * 76,
    ])

    for _, row in stability_df.iterrows():
        summary_lines.append(
            f"- {row['comparison']}: "
            f"Spearman rho={row['spearman_rank_correlation']:.4f}, "
            f"max rank change={row['max_absolute_rank_change']:.1f}, "
            f"same top model={row['same_top_model']}"
        )

    summary_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 76,
        "- MCC is rescaled to [0, 1] before aggregation with Macro-F1 and",
        "  Balanced Accuracy.",
        "- Alternative weights assess whether conclusions depend strongly on",
        "  the equal-weight composite definition.",
        "- No model is retrained and no benchmark split is changed.",
        "",
        "Files written:",
        "- robustness_run_level_augmented.csv",
        "- robustness_model_summary.csv",
        "- robustness_model_rankings.csv",
        "- robustness_ranking_stability.csv",
        "- robustness_by_family_and_model.csv",
        "- robustness_sensitivity_manifest.json",
        "- robustness_sensitivity_summary.txt",
        "=" * 76,
    ])

    summary_text = "\n".join(summary_lines)

    with open(
        OUT_DIR / "robustness_sensitivity_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary_text)

    print()
    print(summary_text)
    print()
    print(f"Output directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
