# CICIoT2023 UAG-IDS Benchmark

Reproducibility code for:

**Beyond Random Splits: A Leakage-Controlled Evaluation of Unseen Attack-Family Generalization in IoT Intrusion Detection Systems**

This repository provides the code used to evaluate IoT intrusion-detection models under conventional and leakage-controlled validation settings using the CICIoT2023 dataset.

## Overview

The goal of this repository is not to introduce a new classifier. Instead, it provides a reproducible benchmark for evaluating whether machine learning-based IoT intrusion-detection systems generalize to previously unseen attack families.

The benchmark includes three validation protocols:

1. Stratified random train-test split
2. Source-file-held-out validation
3. Balanced leakage-controlled leave-one-attack-family-out evaluation

The main evaluation focuses on unseen attack-family generalization. In the LOAFO protocol, one attack family is completely excluded from training and used only during testing against a balanced benign subset.

## Repository Structure

```text
CICIoT2023-UAG-IDS-Benchmark/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── scripts/
    ├── 01_prepare_ciciot2023_working_dataset.py
    ├── 02_run_uag_ids_benchmark.py
    └── 03_run_optuna_xgboost_balanced_loafo.py
   ```

## Dataset Information

This project uses the CICIoT2023 dataset as third-party data.

Original dataset source:

- Dataset name: CICIoT2023
- Provider: Canadian Institute for Cybersecurity, University of New Brunswick
- Official dataset page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Associated article: Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Sensors*, 23(13), 5941. https://doi.org/10.3390/s23135941

The original CICIoT2023 files are not redistributed in this repository. Users must download the dataset from the official source and comply with the dataset provider's terms of use.

## Requirements

Recommended environment:

- Python 3.10 or later
- Windows, Linux, or macOS
- CPU execution is supported
- GPU is not required

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
scikit-learn
matplotlib
xgboost
lightgbm
optuna
scipy
```

If XGBoost or LightGBM is not installed, the main benchmark script skips the unavailable model and continues with the remaining classifiers.

## Usage

### Step 1: Download CICIoT2023

Download the CICIoT2023 CSV files from the official dataset page:

```text
https://www.unb.ca/cic/datasets/iotdataset-2023.html
```

Place the raw CSV files in:

```text
MERGED_CSV/
```

Expected structure:

```text
project_root/
├── MERGED_CSV/
│   ├── part-00000.csv
│   ├── part-00001.csv
│   └── ...
└── scripts/
```

### Step 2: Prepare the working dataset

Run:

```bash
python scripts/01_prepare_ciciot2023_working_dataset.py
```

This creates:

```text
mergeddata/ciciot2023_working_with_source.csv
mergeddata/file_summary.csv
mergeddata/label_summary.csv
```

### Step 3: Run the main benchmark

Run:

```bash
python scripts/02_run_uag_ids_benchmark.py
```

This produces:

```text
CICIoT2023_UAG_IDS_BALANCED_FINAL/results_all_uag_ids_balanced_final.csv
CICIoT2023_UAG_IDS_BALANCED_FINAL/table_summary_mean_std.csv
CICIoT2023_UAG_IDS_BALANCED_FINAL/table_balanced_loafo_by_family_mean_std.csv
CICIoT2023_UAG_IDS_BALANCED_FINAL/table_best_model_by_attack_family.csv
CICIoT2023_UAG_IDS_BALANCED_FINAL/table_model_worst_case_robustness.csv
CICIoT2023_UAG_IDS_BALANCED_FINAL/table_generalization_gap_random_vs_balanced_loafo.csv
```

### Step 4: Run optional Optuna-XGBoost LOAFO optimization

Run:

```bash
python scripts/03_run_optuna_xgboost_balanced_loafo.py
```

This optional supplementary experiment produces Optuna-TPE-XGBoost LOAFO results.

## Methodology

The benchmark follows five main stages.

### 1. Dataset preparation

Raw CICIoT2023 CSV files are processed in chunks. Label values are standardized, source-file information is preserved, and a working dataset is generated through controlled class-level sampling.

### 2. Label mapping

Fine-grained CICIoT2023 labels are mapped into binary and coarse-grained operational labels.

Binary labels:

- `BENIGN`
- `ATTACK`

Coarse-grained labels:

- `BENIGN`
- `DDOS`
- `DOS`
- `MIRAI`
- `RECON`
- `SPOOFING`
- `WEB_MALWARE`
- `BRUTE_FORCE`
- `VULNERABILITY_SCAN`

### 3. Conventional validation

A stratified random split is used as an in-distribution baseline.

### 4. Source-file-held-out validation

A group-based split is used so that source files assigned to the test set are excluded from training.

### 5. Balanced leakage-controlled LOAFO validation

For each held-out attack family:

- All samples from the held-out family are removed from training
- The test set contains an equal number of held-out attack samples and benign samples
- Benign samples used in the test set are removed from training
- Row-level train-test overlap is explicitly checked
- The model is trained as a binary classifier on `BENIGN` versus known `ATTACK` traffic
- The model is tested on `BENIGN` versus the unseen held-out attack family

## Evaluation Metrics

The scripts report:

- Accuracy
- Macro-F1
- Weighted-F1
- Macro precision
- Macro recall
- Balanced accuracy
- Matthews correlation coefficient
- Attack recall
- Benign recall
- False-positive rate
- False-negative rate
- Robustness score

The robustness score is computed as:

```text
Robustness score = mean(Macro-F1, Balanced accuracy, MCC)
```

This score is used as a complementary summary indicator and does not replace standard evaluation metrics.

## Reproducibility

The main benchmark uses five random seeds:

```text
42, 123, 2024, 2025, 2026
```

The Optuna-XGBoost supplementary experiment uses:

```text
42, 123, 2026
```

## Third-Party Data Notice

The CICIoT2023 dataset is third-party data and is not redistributed with this repository. Users must obtain the dataset directly from the official provider and comply with its access and reuse terms.

## Citation

If you use this repository, please cite the associated manuscript and the original CICIoT2023 dataset article.

### Dataset citation

Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Sensors*, 23(13), 5941. https://doi.org/10.3390/s23135941

### Manuscript citation

Submitted

## License

The code in this repository is released under the MIT License.

The CICIoT2023 dataset is not included and remains subject to the terms of the original dataset provider.

## Contact

Corresponding author:

Ahmet Hasim Yurttakal  
Email: ahyurttakal@gmail.com
