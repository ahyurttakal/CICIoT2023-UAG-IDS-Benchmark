#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct the controlled CICIoT2023 working dataset used in the study.

Expected input
--------------
Place the original CICIoT2023 CSV files in ``MERGED_CSV/``. The script first
counts fine-grained labels across all files, then retains all observations from
rare labels and probabilistically samples highly prevalent labels using an
approximate target of 50,000 observations per fine-grained class. The source
CSV filename is preserved in the ``source_file`` column so that the
source-file-held-out protocol can be reproduced.

The sampling procedure intentionally mirrors the procedure used for the
reported experiments. It should not be changed when reproducing the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "Label"
CHUNKSIZE = 200_000
RANDOM_STATE = 42
MAX_PER_CLASS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("MERGED_CSV"),
        help="Directory containing the original CICIoT2023 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mergeddata"),
        help="Directory for the constructed working dataset and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in {data_dir.resolve()}. "
            "Place the original CICIoT2023 CSV files there or use --input-dir."
        )

    print(f"CSV files found: {len(csv_files)}")

    global_label_counts = pd.Series(dtype="int64")
    file_summary = []

    for file in csv_files:
        file_rows = 0
        file_label_counts = pd.Series(dtype="int64")

        for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if LABEL_COL not in chunk.columns:
                raise ValueError(f"{file.name} does not contain column {LABEL_COL!r}.")

            chunk[LABEL_COL] = (
                chunk[LABEL_COL].astype(str).str.strip().str.upper()
            )
            vc = chunk[LABEL_COL].value_counts()
            file_label_counts = file_label_counts.add(vc, fill_value=0).astype(int)
            global_label_counts = global_label_counts.add(vc, fill_value=0).astype(int)
            file_rows += len(chunk)

        file_summary.append(
            {
                "source_file": file.name,
                "rows": file_rows,
                "num_labels": len(file_label_counts),
            }
        )

    file_summary_df = pd.DataFrame(file_summary)
    label_summary_df = global_label_counts.sort_values(ascending=False).reset_index()
    label_summary_df.columns = ["Label", "count"]
    file_summary_df.to_csv(out_dir / "file_summary.csv", index=False)
    label_summary_df.to_csv(out_dir / "label_summary.csv", index=False)

    sample_probs = {
        label: (1.0 if count <= MAX_PER_CLASS else MAX_PER_CLASS / count)
        for label, count in global_label_counts.items()
    }

    working_path = out_dir / "ciciot2023_working_with_source.csv"
    if working_path.exists():
        working_path.unlink()

    first_write = True
    for file in csv_files:
        print(f"Sampling: {file.name}")
        for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk[LABEL_COL] = (
                chunk[LABEL_COL].astype(str).str.strip().str.upper()
            )
            chunk["source_file"] = file.name
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.dropna(subset=[LABEL_COL])

            sampled_parts = []
            for label, group in chunk.groupby(LABEL_COL, sort=False):
                prob = sample_probs.get(label, 1.0)
                if prob >= 1.0:
                    sampled = group
                else:
                    # This deliberately preserves the sampling rule used in the study.
                    sampled = group.sample(frac=prob, random_state=RANDOM_STATE)
                if not sampled.empty:
                    sampled_parts.append(sampled)

            if sampled_parts:
                sampled_chunk = pd.concat(sampled_parts, axis=0)
                sampled_chunk.to_csv(
                    working_path,
                    mode="a",
                    header=first_write,
                    index=False,
                )
                first_write = False

    df_check = pd.read_csv(working_path, low_memory=False)
    print(f"Working dataset: {working_path}")
    print(f"Shape: {df_check.shape}")
    print(f"Source files represented: {df_check['source_file'].nunique()}")
    print(df_check[LABEL_COL].value_counts().head(30))


if __name__ == "__main__":
    main()
