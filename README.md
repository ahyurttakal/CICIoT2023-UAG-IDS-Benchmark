# Reproducibility Code for Unseen Attack-Family Generalization in IoT Intrusion Detection

This archive contains the reproducibility code associated with the manuscript:

**Beyond Random Splits: A Leakage-Controlled Evaluation of Unseen Attack-Family Generalization in IoT Intrusion Detection Systems**

The code implements dataset preparation, conventional and leakage-controlled evaluation protocols, balanced leave-one-attack-family-out (LOAFO) experiments, feature-space auditing, normalized robustness analysis, family-level statistical testing, decision-threshold calibration, an anomaly-oriented reference analysis, sampling and prevalence sensitivity analyses, deployment-oriented computational profiling, and manuscript table/figure generation.

## 1. Data availability

The experiments use the **CICIoT2023** dataset developed by the Canadian Institute for Cybersecurity, University of New Brunswick.

The raw dataset is third-party data and is not redistributed in this archive. Download the original CICIoT2023 CSV files from the official dataset source and place the CSV files in the following directory at the project root:

```text
MERGED_CSV/
```

The data-preparation script creates the working dataset used by the experimental pipeline:

```text
mergeddata/ciciot2023_working_with_source.csv
```

The `source_file` field is retained only to construct source-file-held-out partitions and is excluded from classifier inputs.

## 2. Package structure

```text
PeerJ_Code_Package/
├── README.md
├── requirements.txt
└── scripts/
    ├── 00_prepare_working_dataset.py
    ├── 01_run_uag_ids_benchmark.py
    ├── 02_feature_audit.py
    ├── 03_robustness_sensitivity.py
    ├── 04_statistical_effect_sizes.py
    ├── 05_threshold_operational_analysis.py
    ├── 06_anomaly_baseline.py
    ├── 07_sampling_sensitivity.py
    ├── 08_deployment_profiling.py
    ├── 09_prevalence_preserving_loafo.py
    └── 10_build_manuscript_outputs.py
```

## 3. Software environment

The main experiment environment used the following core packages:

- Python 3.11.9
- NumPy 2.4.6
- pandas 2.3.3
- scikit-learn 1.9.0
- XGBoost 3.2.0
- LightGBM 4.7.0
- SciPy
- Matplotlib
- joblib
- psutil

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The recorded deployment-profiling environment used Windows 10 (64-bit), 20 logical CPU threads, and approximately 31.63 GB of RAM. Timing and memory measurements are platform-dependent and may differ on other systems.

## 4. Reproduction workflow

Run all commands from the project root.

### Step 1 — Prepare the working dataset

```bash
python scripts/00_prepare_working_dataset.py
```

This script:

- reads the original CICIoT2023 CSV files from `MERGED_CSV/`;
- normalizes column names and class labels;
- records source-file provenance;
- applies controlled probabilistic sampling to highly prevalent fine-grained classes;
- retains rare classes without forced upsampling; and
- writes `mergeddata/ciciot2023_working_with_source.csv`.

The working-dataset sampling uses a target upper bound of approximately 50,000 observations per fine-grained label while preserving all observations for labels below that level.

### Step 2 — Run the primary benchmark

```bash
python scripts/01_run_uag_ids_benchmark.py
```

This script evaluates the five supervised classifiers under:

1. stratified random splitting;
2. source-file-held-out splitting; and
3. balanced leakage-controlled LOAFO evaluation.

Binary intrusion detection is evaluated under all three protocols. Coarse-grained attack-family classification is evaluated under the random-split and source-file-held-out protocols.

### Step 3 — Audit the predictive feature space

```bash
python scripts/02_feature_audit.py
```

The predictive matrix contains 39 flow-level features. The audit reports the feature inventory and descriptive feature groups and screens the predictors for obvious identifier-like variables, target-like variables, all-zero columns, constant columns, and extremely high-cardinality columns.

Model inputs exclude the class label, `source_file`, the internal row identifier, and derived target variables. IP addresses, source/destination port identifiers, absolute timestamps, flow identifiers, MAC addresses, and device identifiers are not part of the predictive matrix used in the experiments.

### Step 4 — Compute normalized robustness and weighting sensitivity

```bash
python scripts/03_robustness_sensitivity.py
```

MCC is mapped from `[-1, 1]` to `[0, 1]` before aggregation:

```text
MCC_01 = (MCC + 1) / 2
```

The normalized equal-weight robustness score is:

```text
R = [Macro-F1 + Balanced Accuracy + MCC_01] / 3
```

The script also evaluates alternative weighting schemes for Macro-F1, balanced accuracy, and normalized MCC.

### Step 5 — Run family-level statistical analyses

```bash
python scripts/04_statistical_effect_sizes.py
```

The **held-out attack family** is the primary inferential unit. The five random-seed repetitions are averaged within each attack family and classifier before primary statistical testing, producing eight matched family-level observations per classifier.

The statistical analysis includes:

- Friedman tests;
- Kendall's W effect sizes;
- two-sided paired Wilcoxon signed-rank tests;
- Holm correction within each metric;
- paired rank-biserial correlations; and
- 10,000-replicate attack-family bootstrap 95% confidence intervals.

A separate family × seed matched-block analysis is produced as a sensitivity analysis and is not interpreted as 40 independent attack-family observations.

### Step 6 — Run threshold calibration and operational analysis

```bash
python scripts/05_threshold_operational_analysis.py
```

For each LOAFO fold, 20% of the non-held-out development pool is used as an inner validation subset. The held-out attack family is absent from model fitting and threshold selection, and the LOAFO test set remains untouched until evaluation.

Candidate decision thresholds range from 0.05 to 0.95 in increments of 0.01. The analysis evaluates:

- the default threshold of 0.50;
- a validation threshold maximizing Macro-F1; and
- an FPR-constrained validation threshold.

For the FPR-constrained strategy, candidate thresholds with validation FPR ≤ 0.10 are considered and the threshold with the highest attack recall is selected. If no threshold satisfies the constraint, the threshold with the lowest validation FPR is selected, with higher attack recall used to break ties.

Reported operational metrics include attack precision, attack recall, attack F1, benign recall, FPR, FNR, false alarms per 1,000 benign flows, ROC-AUC, and PR-AUC.

### Step 7 — Run the anomaly-oriented reference analysis

```bash
python scripts/06_anomaly_baseline.py
```

Isolation Forest is used as a benign-only anomaly-oriented reference. For each LOAFO condition:

- the held-out attack family is excluded from model development;
- the detector is fitted only on benign development observations;
- validation data are used only for anomaly-threshold selection; and
- evaluation is performed on the untouched balanced LOAFO test set.

The analysis reports the native Isolation Forest decision boundary together with validation-based Macro-F1 and FPR-constrained threshold strategies.

### Step 8 — Run sampling sensitivity analysis

```bash
python scripts/07_sampling_sensitivity.py
```

This analysis evaluates Random Forest and LightGBM under controlled changes to:

- the training-set cap; and
- the benign reserve used during LOAFO test construction.

The evaluated settings include a 300,000-row training baseline, lower training caps, and benign-reserve values of 5,000, 10,000, and 30,000. Three fixed seeds are used for this focused sensitivity analysis: 42, 123, and 2026.

### Step 9 — Run deployment-oriented profiling

```bash
python scripts/08_deployment_profiling.py
```

This script profiles the five supervised classifiers under balanced LOAFO conditions and records:

- training time;
- inference time and latency per flow;
- throughput in flows per second;
- serialized model size; and
- approximate process-level RSS increase during training and inference.

The profiling analysis uses eight held-out attack families and seeds 42, 123, and 2026. Process-level RSS includes Python/runtime overhead and should be interpreted as an approximate comparative memory indicator.

### Step 10 — Run test-composition sensitivity analysis

```bash
python scripts/09_prevalence_preserving_loafo.py
```

This analysis compares the balanced 1:1 LOAFO test composition with an attack-versus-benign composition that preserves the prevalence observed in the constructed working dataset, subject to the test-size cap.

For each family and seed, the same fitted classifier is evaluated under both test compositions. The full benign evaluation reserve is excluded from training for both evaluations, so the comparison isolates sensitivity to test composition.

The term **prevalence-preserving** refers only to prevalence within the constructed working dataset and does not represent real-world deployment prevalence.

### Step 11 — Build manuscript tables and figures

```bash
python scripts/10_build_manuscript_outputs.py --root .
```

The script collects the generated analysis outputs and writes publication-oriented files under `MANUSCRIPT_OUTPUTS/`.

Main tables:

```text
Table_1_Dataset_Distribution.csv
Table_2_Evaluation_Protocols.csv
Table_3_Model_Configurations.csv
Table_4_Binary_Random_vs_SourceFileHeldOut.csv
Table_5_Balanced_LOAFO.csv
Table_6_Threshold_Operational.csv
Table_7_Coarse_Classification.csv
Table_8_Family_Level_Friedman.csv
Table_9_Deployment_Profile.csv
```

Supplementary tables:

```text
Table_S1_Robustness_Weighting_Sensitivity.csv
Table_S2_Pairwise_Wilcoxon_Holm_Effect_Sizes.csv
Table_S3_40Block_Sensitivity.csv
Table_S4_Isolation_Forest.csv
```

Main figures:

```text
Figure_1_Random_vs_SourceFileHeldOut
Figure_2_Normalized_Robustness_Distribution
Figure_3_Family_MacroF1_Heatmap
Figure_4_Recall_FPR_Pareto_95CI
```


## 5. Primary experimental configuration

The primary random-split, source-file-held-out, and balanced LOAFO experiments use five fixed random seeds:

```text
42, 123, 2024, 2025, 2026
```

Core sampling settings are:

```text
MAX_TRAIN_ROWS = 300,000
MAX_TEST_ROWS = 150,000
LOAFO_TEST_PER_CLASS_CAP = 25,000
MIN_TEST_PER_CLASS = 1,000
MIN_TRAIN_BENIGN_REMAINING = 10,000
```

The 10,000-observation benign reserve is applied before optional global training capping. It should not be interpreted as a guarantee that every capped training subset contains at least 10,000 benign observations.

## 6. Fixed supervised model configurations

The primary benchmark uses fixed model configurations so that the comparison focuses on the evaluation protocol.

- **Logistic Regression**: `StandardScaler`, `max_iter=1000`, `class_weight="balanced"`
- **Random Forest**: 150 trees, `max_depth=24`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`
- **Extra Trees**: 200 trees, unrestricted maximum depth, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight="balanced"`
- **XGBoost**: 250 trees, `max_depth=8`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, histogram tree method
- **LightGBM**: 300 trees, `learning_rate=0.06`, `num_leaves=64`, `subsample=0.85`, `colsample_bytree=0.85`, `class_weight="balanced"`

`StandardScaler` is fitted only within the Logistic Regression training pipeline. Tree-based classifiers are trained without feature standardization.

## 7. Evaluation protocols

### Stratified random split

An 80%/20% observation-level stratified train/test split is used as an in-distribution reference.

### Source-file-held-out split

Source CSV files are assigned to training and testing partitions with approximately 80%/20% separation. A source file cannot appear in both partitions.

### Balanced leakage-controlled LOAFO

For each attack family:

1. the complete target attack family is excluded from training;
2. the test set contains held-out-family attack observations and an equal number of benign observations;
3. benign observations selected for testing are removed from the training pool;
4. row-level train/test disjointness is checked explicitly; and
5. the held-out attack family remains absent from threshold calibration when calibration is performed.

The LOAFO protocol measures **unseen attack-family generalization**. It does not establish detection of genuine operational zero-day exploits.

## 8. Script-to-output mapping

| Script | Main purpose | Principal manuscript output |
|---|---|---|
| `00_prepare_working_dataset.py` | Construct the working dataset and preserve source-file provenance | Table 1 inputs; data preparation |
| `01_run_uag_ids_benchmark.py` | Random split, source-file-held-out evaluation, and balanced LOAFO benchmark | Tables 4, 5, 7; Figure 1 inputs |
| `02_feature_audit.py` | Characterize the 39-feature predictive matrix | Feature-space specification |
| `03_robustness_sensitivity.py` | Normalize MCC and evaluate robustness-score weighting sensitivity | Table 5; Figure 2; Table S1 |
| `04_statistical_effect_sizes.py` | Family-level inference, effect sizes, bootstrap intervals, matched-block sensitivity | Table 8; Figure 4 inputs; Tables S2–S3 |
| `05_threshold_operational_analysis.py` | Threshold calibration and operational metrics | Table 6 |
| `06_anomaly_baseline.py` | Benign-only Isolation Forest reference | Table S4 |
| `07_sampling_sensitivity.py` | Training-cap and benign-reserve sensitivity | Sampling-sensitivity results |
| `08_deployment_profiling.py` | Computational and deployment-oriented profiling | Table 9 |
| `09_prevalence_preserving_loafo.py` | Test-composition sensitivity | Test-composition sensitivity results |
| `10_build_manuscript_outputs.py` | Assemble publication-oriented tables and figures | Tables 1–9, Tables S1–S4, Figures 1–4 |

## 9. Reproducibility notes

- The raw CICIoT2023 data are not included in this archive.
- Reproducing all experiments from the raw CSV files is computationally intensive.
- XGBoost and LightGBM are required to reproduce all five supervised-classifier results.
- `psutil` is required for process-level memory profiling.
- Random seeds are specified explicitly in the scripts.
- Output directories are created automatically by the corresponding scripts.
- Timing, throughput, and memory values depend on the execution platform and should not be expected to match exactly across different hardware/software environments.

## 10. Manuscript correspondence

The code package is organized to reproduce the computational procedures and analyses reported in:

**Beyond Random Splits: A Leakage-Controlled Evaluation of Unseen Attack-Family Generalization in IoT Intrusion Detection Systems**

The dataset itself remains subject to the terms of its original provider and must be obtained from the official CICIoT2023 source.
