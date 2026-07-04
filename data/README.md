# Data Directory

This repository does not redistribute the CICIoT2023 dataset.

Download the dataset from the official source:

https://www.unb.ca/cic/datasets/iotdataset-2023.html

Place the raw CICIoT2023 CSV files in a folder named `MERGED_CSV` at the project root:

```text
project_root/
├── MERGED_CSV/
│   ├── part-00000.csv
│   ├── part-00001.csv
│   └── ...
└── scripts/
```

Then run:

```bash
python scripts/01_prepare_ciciot2023_working_dataset.py
```

The processed working dataset will be generated under:

```text
mergeddata/ciciot2023_working_with_source.csv
```
