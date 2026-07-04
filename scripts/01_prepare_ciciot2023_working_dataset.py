import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# SETTINGS
# =========================

DATA_DIR = Path(r"MERGED_CSV")  # Directory containing the source CSV files
OUT_DIR = Path(r"mergeddata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
CHUNKSIZE = 200_000
RANDOM_STATE = 42

# Maximum number of samples to retain per fine-grained class.
# The default value supports a manageable reproducible working dataset.
MAX_PER_CLASS = 50_000


# =========================
# 1. FIND CSV FILES
# =========================

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Number of CSV files found: {len(csv_files)}")

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files were found in the input directory.")

print("First 5 files:")
for f in csv_files[:5]:
    print(" -", f.name)


# =========================
# 2. COMPUTE LABEL DISTRIBUTION
# =========================

global_label_counts = pd.Series(dtype="int64")
file_summary = []

for file in csv_files:
    print(f"Counting labels in: {file.name}")
    file_rows = 0
    file_label_counts = pd.Series(dtype="int64")

    for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]

        if LABEL_COL not in chunk.columns:
            raise ValueError(f"{LABEL_COL} column was not found in {file.name}.")

        chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip().str.upper()

        vc = chunk[LABEL_COL].value_counts()
        file_label_counts = file_label_counts.add(vc, fill_value=0).astype(int)
        global_label_counts = global_label_counts.add(vc, fill_value=0).astype(int)
        file_rows += len(chunk)

    file_summary.append({
        "source_file": file.name,
        "rows": file_rows,
        "num_labels": len(file_label_counts)
    })

file_summary_df = pd.DataFrame(file_summary)
label_summary_df = global_label_counts.sort_values(ascending=False).reset_index()
label_summary_df.columns = ["Label", "count"]

file_summary_df.to_csv(OUT_DIR / "file_summary.csv", index=False)
label_summary_df.to_csv(OUT_DIR / "label_summary.csv", index=False)

print("\nOverall label distribution:")
print(label_summary_df.head(20))


# =========================
# 3. CREATE BALANCED WORKING DATASET
# =========================

# Class-level sampling is used because the full dataset is very large.
# Rare classes are retained, while highly frequent classes are sampled.

sample_probs = {}

for label, count in global_label_counts.items():
    if count <= MAX_PER_CLASS:
        sample_probs[label] = 1.0
    else:
        sample_probs[label] = MAX_PER_CLASS / count

working_path = OUT_DIR / "ciciot2023_working_with_source.csv"

if working_path.exists():
    working_path.unlink()

first_write = True

for file in csv_files:
    print(f"Sampling from: {file.name}")

    for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]

        chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip().str.upper()
        chunk["source_file"] = file.name

        # Replace infinite values with missing values.
        chunk = chunk.replace([np.inf, -np.inf], np.nan)

        # Remove rows with missing labels.
        chunk = chunk.dropna(subset=[LABEL_COL])

        sampled_parts = []

        for label, group in chunk.groupby(LABEL_COL, sort=False):
            prob = sample_probs.get(label, 1.0)

            if prob >= 1.0:
                sampled = group
            else:
                sampled = group.sample(frac=prob, random_state=RANDOM_STATE)

            if len(sampled) > 0:
                sampled_parts.append(sampled)

        if sampled_parts:
            sampled_chunk = pd.concat(sampled_parts, axis=0)

            sampled_chunk.to_csv(
                working_path,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False

print("\nWorking dataset created:")
print(working_path)


# =========================
# 4. FINAL CHECK
# =========================

df_check = pd.read_csv(working_path, low_memory=False)

print("\nWorking dataset shape:")
print(df_check.shape)

print("\nWorking dataset label distribution:")
print(df_check[LABEL_COL].value_counts().head(30))

print("\nNumber of source files:")
print(df_check["source_file"].nunique())