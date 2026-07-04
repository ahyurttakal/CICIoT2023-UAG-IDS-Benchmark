# -*- coding: utf-8 -*-
"""
PeerJ Computer Science makalesi için Section 5 istatistik tabloları ve figürleri.

Üretilen çıktılar
-----------------
1. Figure 1:
   Random split ve source-file-held-out sonuçlarını tek figürde karşılaştırır.
   Metrikler: Macro-F1, Balanced Accuracy, MCC

2. Figure 3:
   Held-out saldırı ailesi × model Macro-F1 heatmap

3. Figure 4:
   Attack Recall - False-Positive Rate trade-off grafiği

4. Table 8:
   Friedman testi, Kendall's W etki büyüklüğü ve ortalama sıralamalar

5. Supplementary tablolar:
   Holm düzeltmeli Wilcoxon signed-rank post-hoc karşılaştırmaları

Gerekli paketler
----------------
pip install pandas numpy matplotlib scipy

Örnek çalıştırma
----------------
python generate_peerj_section5_outputs.py ^
    --input "results_all_uag_ids_balanced_final.csv" ^
    --output "peerj_section5_outputs"

Windows PowerShell için:
python generate_peerj_section5_outputs.py `
    --input "results_all_uag_ids_balanced_final.csv" `
    --output "peerj_section5_outputs"
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

warnings.filterwarnings("ignore")


# ============================================================
# SABİTLER
# ============================================================

MODEL_ORDER = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
]

MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}

FAMILY_ORDER = [
    "DDOS",
    "DOS",
    "MIRAI",
    "RECON",
    "SPOOFING",
    "BRUTE_FORCE",
    "VULNERABILITY_SCAN",
    "WEB_MALWARE",
]

FAMILY_DISPLAY = {
    "DDOS": "DDoS",
    "DOS": "DoS",
    "MIRAI": "Mirai",
    "RECON": "Reconnaissance",
    "SPOOFING": "Spoofing",
    "BRUTE_FORCE": "Brute Force",
    "VULNERABILITY_SCAN": "Vulnerability Scan",
    "WEB_MALWARE": "Web Malware",
}

# Friedman testi ve post-hoc analiz için kullanılacak metrikler
METRICS = {
    "macro_f1": {
        "display": "Macro-F1",
        "direction": "higher",
    },
    "balanced_accuracy": {
        "display": "Balanced accuracy",
        "direction": "higher",
    },
    "mcc": {
        "display": "MCC",
        "direction": "higher",
    },
    "attack_recall": {
        "display": "Attack recall",
        "direction": "higher",
    },
    "false_positive_rate": {
        "display": "False-positive rate",
        "direction": "lower",
    },
    "robustness_score": {
        "display": "Robustness score",
        "direction": "higher",
    },
}

# Figure 1 içinde tek grafikte gösterilecek üç metrik
FIGURE1_METRICS = [
    ("macro_f1", "Macro-F1"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("mcc", "MCC"),
]


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Section 5 statistical tables and figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to results_all_uag_ids_balanced_final.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("peerj_section5_outputs"),
        help="Output directory. Default: peerj_section5_outputs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    """
    Hem yüksek çözünürlüklü PNG hem de vektörel PDF üretir.
    """
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_p_value(value: float) -> str:
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """
    Holm step-down correction.
    statsmodels gerektirmeden çoklu karşılaştırma düzeltmesi uygular.
    """
    p_values_array = np.asarray(p_values, dtype=float)
    m = len(p_values_array)

    order = np.argsort(p_values_array)
    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for i, idx in enumerate(order):
        adjusted = (m - i) * p_values_array[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted_values = np.empty(m, dtype=float)

    for i, idx in enumerate(order):
        adjusted_values[idx] = adjusted_sorted[i]

    return adjusted_values


def paired_rank_biserial(differences: np.ndarray) -> float:
    """
    Eşleştirilmiş karşılaştırmalar için rank-biserial effect size.

    Pozitif değer:
        model_a lehine sonuç

    Negatif değer:
        model_b lehine sonuç

    false-positive rate için yön kod içinde ters çevrilmektedir.
    Böylece pozitif değer her zaman model_a lehine yorumlanabilir.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(differences), method="average")

    w_plus = ranks[differences > 0].sum()
    w_minus = ranks[differences < 0].sum()

    denominator = w_plus + w_minus

    if denominator == 0:
        return 0.0

    return float((w_plus - w_minus) / denominator)


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "task",
        "split",
        "seed",
        "model",
        "status",
        "held_out_family",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "attack_recall",
        "false_positive_rate",
        "robustness_score",
    }

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Eksik sütunlar bulundu: "
            + ", ".join(missing_columns)
        )


def load_results(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ana sonuç dosyasını okur.
    Successful binary balanced_loafo satırlarını ayırır.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)

    success_df = df[df["status"].eq("success")].copy()

    loafo_df = success_df[
        success_df["task"].eq("binary")
        & success_df["split"].eq("balanced_loafo")
    ].copy()

    if loafo_df.empty:
        raise ValueError(
            "binary + balanced_loafo koşulunu sağlayan başarılı satır bulunamadı."
        )

    loafo_df["block"] = (
        loafo_df["held_out_family"].astype(str)
        + "__seed"
        + loafo_df["seed"].astype(str)
    )

    # Her eşleştirilmiş blokta beş modelin tamamı olmalı
    counts = loafo_df.groupby("block")["model"].nunique()
    complete_blocks = counts[counts.eq(len(MODEL_ORDER))].index

    loafo_df = loafo_df[
        loafo_df["block"].isin(complete_blocks)
    ].copy()

    if loafo_df.empty:
        raise ValueError("Tam eşleştirilmiş LOAFO bloğu bulunamadı.")

    found_models = set(loafo_df["model"].unique())
    missing_models = sorted(set(MODEL_ORDER).difference(found_models))

    if missing_models:
        raise ValueError(
            "LOAFO sonuçlarında eksik modeller var: "
            + ", ".join(missing_models)
        )

    return success_df, loafo_df


# ============================================================
# TABLE 8: FRIEDMAN + KENDALL'S W + MEAN RANKS
# ============================================================

def generate_statistical_tables(
    loafo_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Ana makale için:
        table8_friedman_mean_ranks.csv

    Supplementary material için:
        tableS4_wilcoxon_holm_posthoc.csv
        tableS5_significant_wilcoxon_holm_pairs.csv
    """
    friedman_rows: list[dict] = []
    posthoc_rows: list[dict] = []

    for metric, meta in METRICS.items():
        pivot = loafo_df.pivot(
            index="block",
            columns="model",
            values=metric,
        )

        pivot = pivot[MODEL_ORDER].dropna()

        if pivot.empty:
            continue

        # Rank 1 her zaman en iyi sonucu temsil eder
        ascending = meta["direction"] == "lower"

        ranks = pivot.rank(
            axis=1,
            ascending=ascending,
            method="average",
        )

        mean_ranks = ranks.mean(axis=0)

        # Wilcoxon ve Friedman için her metrikte "higher is better" standardı
        oriented = pivot.copy()

        if meta["direction"] == "lower":
            oriented = -oriented

        arrays = [
            oriented[model].to_numpy()
            for model in MODEL_ORDER
        ]

        chi_square, p_value = friedmanchisquare(*arrays)

        kendalls_w = chi_square / (
            len(oriented) * (len(MODEL_ORDER) - 1)
        )

        best_model = mean_ranks.idxmin()

        friedman_row = {
            "metric": meta["display"],
            "n_blocks": len(oriented),
            "friedman_chi_square": float(chi_square),
            "friedman_p_value": float(p_value),
            "kendalls_w": float(kendalls_w),
            "best_ranked_model": MODEL_DISPLAY[best_model],
            "best_mean_rank": float(mean_ranks[best_model]),
        }

        for model in MODEL_ORDER:
            friedman_row[
                f"mean_rank_{MODEL_DISPLAY[model]}"
            ] = float(mean_ranks[model])

        friedman_rows.append(friedman_row)

        # -------------------------------
        # Holm-corrected pairwise Wilcoxon
        # -------------------------------
        metric_pair_rows: list[dict] = []

        for i, model_a in enumerate(MODEL_ORDER):
            for model_b in MODEL_ORDER[i + 1:]:
                values_a = oriented[model_a].to_numpy()
                values_b = oriented[model_b].to_numpy()

                differences = values_a - values_b

                try:
                    statistic, raw_p_value = wilcoxon(
                        values_a,
                        values_b,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                except ValueError:
                    statistic, raw_p_value = 0.0, 1.0

                metric_pair_rows.append(
                    {
                        "metric": meta["display"],
                        "model_a": MODEL_DISPLAY[model_a],
                        "model_b": MODEL_DISPLAY[model_b],
                        "n_blocks": len(oriented),
                        "mean_oriented_difference_a_minus_b": float(
                            np.mean(differences)
                        ),
                        "median_oriented_difference_a_minus_b": float(
                            np.median(differences)
                        ),
                        "wilcoxon_statistic": float(statistic),
                        "raw_p_value": float(raw_p_value),
                        "rank_biserial_effect_size": paired_rank_biserial(
                            differences
                        ),
                    }
                )

        adjusted_p_values = holm_adjust(
            [
                row["raw_p_value"]
                for row in metric_pair_rows
            ]
        )

        for row, adjusted_p_value in zip(
            metric_pair_rows,
            adjusted_p_values,
        ):
            row["holm_adjusted_p_value"] = float(
                adjusted_p_value
            )
            row["significant_after_holm_0_05"] = bool(
                adjusted_p_value < 0.05
            )

            posthoc_rows.append(row)

    friedman_df = pd.DataFrame(friedman_rows)

    # Makalede kolay okunacak sütun sırası
    table8_columns = [
        "metric",
        "n_blocks",
        "friedman_chi_square",
        "friedman_p_value",
        "kendalls_w",
        "mean_rank_Logistic Regression",
        "mean_rank_Random Forest",
        "mean_rank_Extra Trees",
        "mean_rank_XGBoost",
        "mean_rank_LightGBM",
        "best_ranked_model",
        "best_mean_rank",
    ]

    friedman_df = friedman_df[table8_columns]

    friedman_df.to_csv(
        output_dir / "table8_friedman_mean_ranks.csv",
        index=False,
    )

    # Word'e veya makaleye daha rahat aktarılabilen formatlı sürüm
    formatted_df = friedman_df.copy()

    formatted_df["friedman_chi_square"] = formatted_df[
        "friedman_chi_square"
    ].map(lambda x: f"{x:.4f}")

    formatted_df["friedman_p_value"] = formatted_df[
        "friedman_p_value"
    ].map(format_p_value)

    formatted_df["kendalls_w"] = formatted_df[
        "kendalls_w"
    ].map(lambda x: f"{x:.4f}")

    mean_rank_columns = [
        column
        for column in formatted_df.columns
        if column.startswith("mean_rank_")
    ]

    for column in mean_rank_columns:
        formatted_df[column] = formatted_df[column].map(
            lambda x: f"{x:.4f}"
        )

    formatted_df["best_mean_rank"] = formatted_df[
        "best_mean_rank"
    ].map(lambda x: f"{x:.4f}")

    formatted_df.to_csv(
        output_dir / "table8_friedman_mean_ranks_formatted.csv",
        index=False,
    )

    posthoc_df = pd.DataFrame(posthoc_rows)

    posthoc_df.to_csv(
        output_dir / "tableS4_wilcoxon_holm_posthoc.csv",
        index=False,
    )

    significant_pairs_df = posthoc_df[
        posthoc_df["significant_after_holm_0_05"]
    ].copy()

    significant_pairs_df.to_csv(
        output_dir / "tableS5_significant_wilcoxon_holm_pairs.csv",
        index=False,
    )


# ============================================================
# FIGURE 1: TEK BİRLEŞİK GRAFİK
# ============================================================

def generate_figure1(
    success_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """
    Macro-F1, Balanced Accuracy ve MCC sonuçlarını
    random split ve source-file-held-out için tek grafikte verir.
    """
    baseline_df = success_df[
        success_df["task"].eq("binary")
        & success_df["split"].isin(
            ["random", "file_held_out"]
        )
    ].copy()

    if baseline_df.empty:
        raise ValueError(
            "Figure 1 için random veya file_held_out satırları bulunamadı."
        )

    summary_df = baseline_df.groupby(
        ["split", "model"]
    )[
        [metric for metric, _ in FIGURE1_METRICS]
    ].agg(["mean", "std"])

    categories = [
        (model, metric, metric_display)
        for model in MODEL_ORDER
        for metric, metric_display in FIGURE1_METRICS
    ]

    x_positions = np.arange(len(categories))
    bar_width = 0.36

    random_means: list[float] = []
    random_stds: list[float] = []
    file_means: list[float] = []
    file_stds: list[float] = []

    for model, metric, _ in categories:
        random_means.append(
            summary_df.loc[
                ("random", model),
                (metric, "mean"),
            ]
        )
        random_stds.append(
            summary_df.loc[
                ("random", model),
                (metric, "std"),
            ]
        )
        file_means.append(
            summary_df.loc[
                ("file_held_out", model),
                (metric, "mean"),
            ]
        )
        file_stds.append(
            summary_df.loc[
                ("file_held_out", model),
                (metric, "std"),
            ]
        )

    fig, ax = plt.subplots(figsize=(18, 7.5))

    ax.bar(
        x_positions - bar_width / 2,
        random_means,
        bar_width,
        yerr=random_stds,
        capsize=3,
        label="Random split",
    )

    ax.bar(
        x_positions + bar_width / 2,
        file_means,
        bar_width,
        yerr=file_stds,
        capsize=3,
        label="Source-file-held-out",
    )

    ax.set_title(
        "Comparison of Random-Split and Source-File-Held-Out Performance"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            metric_display
            for _, _, metric_display in categories
        ],
        rotation=35,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.30)
    ax.legend()

    # Her modelin altına ayrı grup etiketi
    metrics_per_model = len(FIGURE1_METRICS)

    for group_index, model in enumerate(MODEL_ORDER):
        start = group_index * metrics_per_model
        end = start + metrics_per_model - 1
        midpoint = (start + end) / 2

        ax.text(
            midpoint,
            -0.24,
            MODEL_DISPLAY[model],
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontweight="bold",
        )

        if group_index < len(MODEL_ORDER) - 1:
            ax.axvline(
                end + 0.5,
                linewidth=0.8,
                alpha=0.4,
            )

    fig.tight_layout()

    save_figure(
        fig=fig,
        output_dir=output_dir,
        name="figure1_combined_random_vs_fileheldout",
        dpi=dpi,
    )


# ============================================================
# FIGURE 3: FAMILY-LEVEL MACRO-F1 HEATMAP
# ============================================================

def generate_figure3(
    loafo_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """
    Held-out family × model Macro-F1 heatmap üretir.
    """
    heatmap_df = loafo_df.groupby(
        ["held_out_family", "model"]
    )["macro_f1"].mean().unstack("model")

    heatmap_df = heatmap_df.reindex(
        index=FAMILY_ORDER,
        columns=MODEL_ORDER,
    )

    if heatmap_df.isna().any().any():
        raise ValueError(
            "Figure 3 için family-model eşleşmelerinde eksik Macro-F1 değeri var."
        )

    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    image = ax.imshow(
        heatmap_df.to_numpy(),
        aspect="auto",
    )

    ax.set_title(
        "Family-Level Macro-F1 Scores under the Balanced LOAFO Protocol"
    )
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Held-out attack family")

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [MODEL_DISPLAY[model] for model in MODEL_ORDER],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels(
        [FAMILY_DISPLAY[family] for family in FAMILY_ORDER]
    )

    # Hücrelerin içine Macro-F1 değerlerini yaz
    for row_index in range(len(FAMILY_ORDER)):
        for column_index in range(len(MODEL_ORDER)):
            value = heatmap_df.iloc[
                row_index,
                column_index,
            ]

            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Macro-F1")

    fig.tight_layout()

    save_figure(
        fig=fig,
        output_dir=output_dir,
        name="figure3_family_level_macro_f1_heatmap",
        dpi=dpi,
    )


# ============================================================
# FIGURE 4: ATTACK RECALL - FPR TRADE-OFF
# ============================================================

def generate_figure4(
    loafo_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """
    Her model için:
    x = False-positive rate
    y = Attack recall

    Error bar:
    held-out family × seed blokları üzerindeki standart sapma
    """
    tradeoff_df = loafo_df.groupby("model")[
        [
            "attack_recall",
            "false_positive_rate",
        ]
    ].agg(["mean", "std"])

    tradeoff_df = tradeoff_df.reindex(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    for model in MODEL_ORDER:
        x_value = tradeoff_df.loc[
            model,
            ("false_positive_rate", "mean"),
        ]

        x_error = tradeoff_df.loc[
            model,
            ("false_positive_rate", "std"),
        ]

        y_value = tradeoff_df.loc[
            model,
            ("attack_recall", "mean"),
        ]

        y_error = tradeoff_df.loc[
            model,
            ("attack_recall", "std"),
        ]

        ax.errorbar(
            x_value,
            y_value,
            xerr=x_error,
            yerr=y_error,
            fmt="o",
            capsize=4,
        )

        ax.annotate(
            MODEL_DISPLAY[model],
            (x_value, y_value),
            xytext=(7, 6),
            textcoords="offset points",
        )

    ax.set_title(
        "Operational Trade-Off under the Balanced LOAFO Protocol"
    )
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Attack recall")

    ax.set_xlim(0, 0.82)
    ax.set_ylim(0.38, 1.04)

    ax.grid(alpha=0.30)

    fig.tight_layout()

    save_figure(
        fig=fig,
        output_dir=output_dir,
        name="figure4_attack_recall_vs_false_positive_rate",
        dpi=dpi,
    )


# ============================================================
# DOĞRULAMA ÖZETİ VE CAPTION DOSYASI
# ============================================================

def write_validation_summary(
    loafo_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_df = pd.DataFrame(
        {
            "item": [
                "Successful balanced LOAFO rows",
                "Complete matched blocks",
                "Models per block",
                "Held-out attack families",
                "Seeds",
            ],
            "value": [
                len(loafo_df),
                loafo_df["block"].nunique(),
                loafo_df["model"].nunique(),
                loafo_df["held_out_family"].nunique(),
                loafo_df["seed"].nunique(),
            ],
        }
    )

    summary_df.to_csv(
        output_dir / "validation_summary.csv",
        index=False,
    )


def write_figure_captions(output_dir: Path) -> None:
    captions = """Figure 1. Comparison of binary intrusion-detection performance under conventional random splitting and source-file-held-out validation. The chart reports Macro-F1, balanced accuracy, and Matthews correlation coefficient values as mean ± standard deviation across five independent random seeds.

Figure 3. Family-level Macro-F1 scores obtained by the evaluated classifiers under the balanced leakage-controlled leave-one-attack-family-out protocol. The heatmap highlights attack-family-specific variability in unseen-family generalization.

Figure 4. Operational trade-off between attack recall and false-positive rate under the balanced leakage-controlled leave-one-attack-family-out protocol. Points represent mean values and error bars indicate standard deviations across held-out attack families and independent random seeds. Models closer to the upper-left region provide a more favorable operational balance.
"""

    (output_dir / "figure_captions.txt").write_text(
        captions,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    ensure_output_dir(args.output)

    success_df, loafo_df = load_results(args.input)

    write_validation_summary(
        loafo_df=loafo_df,
        output_dir=args.output,
    )

    generate_statistical_tables(
        loafo_df=loafo_df,
        output_dir=args.output,
    )

    generate_figure1(
        success_df=success_df,
        output_dir=args.output,
        dpi=args.dpi,
    )

    generate_figure3(
        loafo_df=loafo_df,
        output_dir=args.output,
        dpi=args.dpi,
    )

    generate_figure4(
        loafo_df=loafo_df,
        output_dir=args.output,
        dpi=args.dpi,
    )

    write_figure_captions(
        output_dir=args.output,
    )

    print("\nTüm çıktılar başarıyla oluşturuldu.")
    print(f"Çıktı klasörü: {args.output.resolve()}")

    print("\nAna makale çıktıları:")
    print(" - figure1_combined_random_vs_fileheldout.png")
    print(" - figure1_combined_random_vs_fileheldout.pdf")
    print(" - figure3_family_level_macro_f1_heatmap.png")
    print(" - figure3_family_level_macro_f1_heatmap.pdf")
    print(" - figure4_attack_recall_vs_false_positive_rate.png")
    print(" - figure4_attack_recall_vs_false_positive_rate.pdf")
    print(" - table8_friedman_mean_ranks_formatted.csv")

    print("\nSupplementary çıktılar:")
    print(" - tableS4_wilcoxon_holm_posthoc.csv")
    print(" - tableS5_significant_wilcoxon_holm_pairs.csv")

    print("\nDoğrulama:")
    print(" - validation_summary.csv")
    print(" - figure_captions.txt")


if __name__ == "__main__":
    main()
