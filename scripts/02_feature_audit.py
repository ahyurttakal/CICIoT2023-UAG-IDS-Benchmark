# 05_feature_audit.py
"""
Feature-space audit for the CICIoT2023 UAG-IDS benchmark.

This script does not modify the benchmark, dataset, train/test splits,
or model results. It only audits the feature space used by the existing
benchmark pipeline.

Outputs
-------
- feature_audit_summary.txt
- feature_audit_manifest.json
- feature_inventory.csv
- feature_groups.csv
- potential_identifier_features.csv
- potential_target_leakage_features.csv
- high_cardinality_predictors.csv
- all_zero_after_numeric_coercion.csv
- constant_predictors.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("mergeddata") / "ciciot2023_working_with_source.csv"

OUT_DIR = Path("CICIoT2023_FEATURE_AUDIT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_ROWS = 200_000

LABEL_COL = "Label"
SOURCE_COL = "source_file"
ROW_ID_COL = "_row_id"

# Mirrors the exclusions used in the benchmark scripts.
EXPLICIT_EXCLUDE = [
    LABEL_COL,
    SOURCE_COL,
    ROW_ID_COL,
    "binary_label",
    "attack_family",
    "coarse_label",
    "fine_label",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-./\\]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def identifier_reasons(column: str) -> List[str]:
    """
    Conservative name-based screen for variables that may encode endpoint
    identity, connection identity, or acquisition time.

    A match is a review flag, not an automatic exclusion decision.
    """
    n = normalize_name(column)
    c = compact_name(column)
    reasons: List[str] = []

    # IP address fields
    if n in {
        "ip", "src_ip", "source_ip", "dst_ip", "dest_ip", "destination_ip",
        "src_ip_address", "source_ip_address", "dst_ip_address",
        "destination_ip_address", "srcip", "dstip", "sourceip", "destinationip",
    } or c in {
        "srcip", "dstip", "sourceip", "destinationip",
        "srcipaddress", "dstipaddress", "sourceipaddress", "destinationipaddress",
    }:
        reasons.append("IP address / endpoint identifier")

    # Port fields
    if n in {
        "port", "src_port", "source_port", "dst_port", "dest_port",
        "destination_port", "sport", "dport", "srcport", "dstport",
        "sourceport", "destinationport",
    } or c in {
        "srcport", "dstport", "sourceport", "destinationport", "sport", "dport",
    }:
        reasons.append("Port / endpoint-specific variable")

    # Absolute time fields; duration is deliberately not flagged.
    if n in {
        "timestamp", "time_stamp", "datetime", "date_time",
        "start_time", "end_time", "flow_start_time", "flow_end_time",
        "capture_time", "capture_timestamp", "frame_time_epoch",
        "time_first", "time_last", "first_seen", "last_seen",
    } or c in {
        "timestamp", "datetime", "starttime", "endtime",
        "flowstarttime", "flowendtime", "capturetime",
        "capturetimestamp", "frametimeepoch", "firstseen", "lastseen",
    }:
        reasons.append("Absolute timestamp / acquisition-time variable")

    # Flow/session/connection IDs
    if n in {
        "flow_id", "flowid", "session_id", "sessionid",
        "connection_id", "connectionid", "conn_id", "connid",
        "socket_id", "socketid", "uid", "uuid",
    } or c in {
        "flowid", "sessionid", "connectionid", "connid",
        "socketid", "uid", "uuid",
    }:
        reasons.append("Flow/session/connection identifier")

    # MAC/device/host identity
    if n in {
        "src_mac", "source_mac", "dst_mac", "destination_mac",
        "mac", "mac_address", "device_id", "deviceid",
        "host_id", "hostid", "client_id", "clientid",
        "server_id", "serverid",
    } or c in {
        "srcmac", "sourcemac", "dstmac", "destinationmac",
        "macaddress", "deviceid", "hostid", "clientid", "serverid",
    }:
        reasons.append("MAC/device/host identifier")

    # Provenance fields
    if n in {"source_file", "sourcefile", "file_id", "filename", "file_name"}:
        reasons.append("Dataset/source provenance identifier")

    return sorted(set(reasons))


def leakage_name_reasons(column: str) -> List[str]:
    """
    Name-based screen for columns that may encode targets or derived labels.
    This does not prove leakage; it identifies columns requiring verification.
    """
    n = normalize_name(column)
    c = compact_name(column)
    reasons: List[str] = []

    if n in {
        "label", "class", "target", "y",
        "binary_label", "attack_family", "coarse_label", "fine_label",
        "attack_label", "attack_type", "attack_category",
    } or c in {
        "binarylabel", "attackfamily", "coarselabel", "finelabel",
        "attacklabel", "attacktype", "attackcategory",
    }:
        reasons.append("Target/label-like field")

    return reasons


def infer_feature_group(column: str) -> str:
    """
    Name-based descriptive grouping for reporting only.
    It is not used for preprocessing or model fitting.
    """
    n = normalize_name(column)

    if identifier_reasons(column):
        return "identifier_or_context"

    if any(k in n for k in [
        "flag", "fin_", "syn_", "rst_", "psh_", "ack_", "ece_", "cwr_"
    ]):
        return "tcp_flag_features"

    if any(k in n for k in [
        "rate", "srate", "drate", "packets_per", "bytes_per"
    ]):
        return "rate_features"

    if any(k in n for k in [
        "duration", "iat", "interarrival", "inter_arrival", "idle", "active"
    ]):
        return "temporal_flow_features"

    if any(k in n for k in [
        "packet", "pkt", "length", "len", "bytes", "header",
        "payload", "size", "magnitude", "weight"
    ]):
        return "packet_size_and_volume_features"

    if any(k in n for k in [
        "mean", "std", "variance", "var", "min", "max", "sum",
        "covariance", "radius"
    ]):
        return "statistical_features"

    if any(k in n for k in [
        "protocol", "proto", "tcp", "udp", "icmp", "arp", "dns", "http", "https"
    ]):
        return "protocol_related_features"

    return "other_flow_features"


# ============================================================
# DATA LOADING
# ============================================================

def read_header(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n  {path}\n\n"
            "Update DATA_PATH at the top of this script if needed."
        )

    header = pd.read_csv(path, nrows=0)
    return [str(c).strip() for c in header.columns]


def read_profile_sample(path: Path, nrows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


# ============================================================
# AUDIT
# ============================================================

def audit_feature_space() -> None:
    columns = read_header(DATA_PATH)

    excluded_present = [c for c in EXPLICIT_EXCLUDE if c in columns]
    predictive_features = [c for c in columns if c not in EXPLICIT_EXCLUDE]

    sample = read_profile_sample(DATA_PATH, PROFILE_ROWS)
    sample.columns = [str(c).strip() for c in sample.columns]

    inventory_rows: List[Dict] = []

    for col in columns:
        is_excluded = col in EXPLICIT_EXCLUDE
        is_predictor = not is_excluded

        s = sample[col]
        dtype_original = str(s.dtype)
        missing_rate = float(s.isna().mean())
        unique_count = int(s.nunique(dropna=True))
        nonmissing = int(s.notna().sum())

        unique_ratio = (
            float(unique_count / nonmissing)
            if nonmissing > 0
            else np.nan
        )

        if is_predictor:
            numeric = pd.to_numeric(s, errors="coerce")
            finite = numeric.replace([np.inf, -np.inf], np.nan)
            numeric_nonmissing_rate = float(finite.notna().mean())

            after_fill = finite.fillna(0)
            effective_unique = int(after_fill.nunique(dropna=False))
            all_zero = bool((after_fill == 0).all())
            constant = bool(effective_unique <= 1)
        else:
            numeric_nonmissing_rate = np.nan
            effective_unique = np.nan
            all_zero = np.nan
            constant = np.nan

        id_reasons = identifier_reasons(col)
        leak_reasons = leakage_name_reasons(col)

        inventory_rows.append({
            "column": col,
            "used_as_predictor": is_predictor,
            "explicitly_excluded": is_excluded,
            "feature_group": (
                infer_feature_group(col)
                if is_predictor
                else "excluded_metadata_or_target"
            ),
            "identifier_flag": bool(id_reasons),
            "identifier_reason": "; ".join(id_reasons),
            "target_or_leakage_name_flag": bool(leak_reasons),
            "target_or_leakage_name_reason": "; ".join(leak_reasons),
            "original_dtype_profile_sample": dtype_original,
            "missing_rate_profile_sample": missing_rate,
            "unique_count_profile_sample": unique_count,
            "unique_ratio_profile_sample": unique_ratio,
            "numeric_nonmissing_rate_after_coercion": numeric_nonmissing_rate,
            "unique_count_after_numeric_coercion_and_zero_fill": effective_unique,
            "all_zero_after_numeric_coercion": all_zero,
            "constant_after_numeric_coercion": constant,
        })

    inventory = pd.DataFrame(inventory_rows)
    predictor_inventory = inventory[inventory["used_as_predictor"]].copy()

    id_flags = predictor_inventory[
        predictor_inventory["identifier_flag"]
    ].copy()

    leakage_flags = predictor_inventory[
        predictor_inventory["target_or_leakage_name_flag"]
    ].copy()

    high_cardinality = predictor_inventory[
        predictor_inventory["unique_ratio_profile_sample"].fillna(0) >= 0.90
    ].copy()

    all_zero_cols = predictor_inventory[
        predictor_inventory["all_zero_after_numeric_coercion"] == True
    ].copy()

    constant_cols = predictor_inventory[
        predictor_inventory["constant_after_numeric_coercion"] == True
    ].copy()

    feature_groups = (
        predictor_inventory.groupby("feature_group", dropna=False)
        .agg(
            feature_count=("column", "count"),
            features=("column", lambda x: "; ".join(map(str, x))),
        )
        .reset_index()
        .sort_values(
            ["feature_count", "feature_group"],
            ascending=[False, True],
        )
    )

    # Save outputs
    inventory.to_csv(OUT_DIR / "feature_inventory.csv", index=False)
    id_flags.to_csv(OUT_DIR / "potential_identifier_features.csv", index=False)
    leakage_flags.to_csv(
        OUT_DIR / "potential_target_leakage_features.csv",
        index=False,
    )
    high_cardinality.to_csv(
        OUT_DIR / "high_cardinality_predictors.csv",
        index=False,
    )
    all_zero_cols.to_csv(
        OUT_DIR / "all_zero_after_numeric_coercion.csv",
        index=False,
    )
    constant_cols.to_csv(
        OUT_DIR / "constant_predictors.csv",
        index=False,
    )
    feature_groups.to_csv(
        OUT_DIR / "feature_groups.csv",
        index=False,
    )

    status = (
        "PASS"
        if len(id_flags) == 0 and len(leakage_flags) == 0
        else "REVIEW_REQUIRED"
    )

    manifest = {
        "dataset_path": str(DATA_PATH),
        "profile_rows_requested": PROFILE_ROWS,
        "profile_rows_read": int(len(sample)),
        "total_columns_exact": int(len(columns)),
        "predictive_feature_count_exact": int(len(predictive_features)),
        "explicitly_excluded_columns_present": excluded_present,
        "predictive_features": predictive_features,
        "potential_identifier_predictors": id_flags["column"].tolist(),
        "potential_target_leakage_predictors": leakage_flags["column"].tolist(),
        "high_cardinality_predictors_profile_sample": (
            high_cardinality["column"].tolist()
        ),
        "all_zero_predictors_after_numeric_coercion_profile_sample": (
            all_zero_cols["column"].tolist()
        ),
        "constant_predictors_profile_sample": (
            constant_cols["column"].tolist()
        ),
        "audit_status": status,
        "note": (
            "Identifier and leakage flags are conservative name-based "
            "review flags and do not by themselves prove leakage."
        ),
    }

    with open(
        OUT_DIR / "feature_audit_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "=" * 72,
        "CICIoT2023 FEATURE-SPACE AUDIT",
        "=" * 72,
        f"Dataset                     : {DATA_PATH}",
        f"Profile rows read           : {len(sample):,}",
        f"Total columns (exact)       : {len(columns)}",
        f"Predictive features (exact) : {len(predictive_features)}",
        (
            "Explicit excluded columns   : "
            + (
                ", ".join(excluded_present)
                if excluded_present
                else "None found"
            )
        ),
        "",
        "POTENTIAL IDENTIFIER / SHORTCUT FEATURES",
        "-" * 72,
    ]

    if id_flags.empty:
        summary_lines.append(
            "None detected among predictive features by the conservative "
            "name-based audit."
        )
    else:
        for _, row in id_flags.iterrows():
            summary_lines.append(
                f"- {row['column']}: {row['identifier_reason']}"
            )

    summary_lines.extend([
        "",
        "POTENTIAL TARGET / LABEL-LIKE FEATURES",
        "-" * 72,
    ])

    if leakage_flags.empty:
        summary_lines.append(
            "None detected among predictive features by the name-based audit."
        )
    else:
        for _, row in leakage_flags.iterrows():
            summary_lines.append(
                f"- {row['column']}: "
                f"{row['target_or_leakage_name_reason']}"
            )

    summary_lines.extend([
        "",
        "NUMERIC-COERCION CHECK",
        "-" * 72,
        f"All-zero predictors after coercion : {len(all_zero_cols)}",
        f"Constant predictors after coercion : {len(constant_cols)}",
        (
            "High-cardinality predictors "
            f"(>=90% unique in profile sample): {len(high_cardinality)}"
        ),
        "",
        "FEATURE GROUPS",
        "-" * 72,
    ])

    for _, row in feature_groups.iterrows():
        summary_lines.append(
            f"- {row['feature_group']}: {int(row['feature_count'])}"
        )

    summary_lines.extend([
        "",
        "AUDIT STATUS",
        "-" * 72,
        status,
        "",
        "Interpretation:",
        "- PASS means no obvious identifier- or target-like predictor was found",
        "  by the conservative name-based screen.",
        "- REVIEW_REQUIRED means one or more columns should be inspected before",
        "  making a manuscript claim about identifier exclusion.",
        "- High cardinality alone does not imply leakage.",
        "- This audit does not alter the benchmark or any reported result.",
        "",
        "Files written:",
        "- feature_audit_summary.txt",
        "- feature_audit_manifest.json",
        "- feature_inventory.csv",
        "- feature_groups.csv",
        "- potential_identifier_features.csv",
        "- potential_target_leakage_features.csv",
        "- high_cardinality_predictors.csv",
        "- all_zero_after_numeric_coercion.csv",
        "- constant_predictors.csv",
        "=" * 72,
    ])

    summary_text = "\n".join(summary_lines)

    with open(
        OUT_DIR / "feature_audit_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nOutput directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    audit_feature_space()
