#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the numerical manuscript tables and final Figures 1–4.

This script consumes the outputs of scripts 01–09. It does not retrain models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = [
    "LogisticRegression", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM"
]
MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}
FAMILY_ORDER = [
    "DDOS", "DOS", "MIRAI", "RECON", "SPOOFING",
    "BRUTE_FORCE", "VULNERABILITY_SCAN", "WEB_MALWARE",
]
FAMILY_DISPLAY = {
    "DDOS": "DDoS", "DOS": "DoS", "MIRAI": "Mirai",
    "RECON": "Reconnaissance", "SPOOFING": "Spoofing",
    "BRUTE_FORCE": "Brute Force", "VULNERABILITY_SCAN": "Vulnerability Scan",
    "WEB_MALWARE": "Web Malware",
}
REPRESENTATIVE_LABELS = {
    "BENIGN": "Benign traffic",
    "DDOS": "ACK fragmentation, UDP flood, ICMP flood, SYN flood, HTTP flood, and related variants",
    "DOS": "TCP flood, HTTP flood, SYN flood, and UDP flood",
    "MIRAI": "GRE-IP flood, GRE Ethernet flood, and UDP-plain traffic",
    "RECON": "Ping sweep, operating-system scan, port scan, and host discovery",
    "SPOOFING": "Domain name system spoofing and address resolution protocol spoofing",
    "VULNERABILITY_SCAN": "Vulnerability-scanning traffic",
    "WEB_MALWARE": "Browser hijacking, command injection, structured query language injection, cross-site scripting, uploading attacks, and backdoor malware",
    "BRUTE_FORCE": "Dictionary-based brute-force attacks",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("MANUSCRIPT_OUTPUTS"))
    return p.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def mean_sd(mean, sd, decimals=4):
    return f"{float(mean):.{decimals}f} ± {float(sd):.{decimals}f}"


def round_half_up(value, decimals=3):
    quantum = Decimal("1").scaleb(-decimals)
    return format(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP), f".{decimals}f")


def save_fig(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def map_family(label: str) -> str:
    label = str(label).strip().upper()
    if "BENIGN" in label:
        return "BENIGN"
    if label.startswith("DDOS"):
        return "DDOS"
    if label.startswith("DOS"):
        return "DOS"
    if label.startswith("MIRAI"):
        return "MIRAI"
    if label.startswith("RECON"):
        return "RECON"
    if label in {"DNS_SPOOFING", "MITM-ARPSPOOFING"}:
        return "SPOOFING"
    if label in {"BROWSERHIJACKING", "COMMANDINJECTION", "SQLINJECTION", "XSS", "UPLOADING_ATTACK", "BACKDOOR_MALWARE"}:
        return "WEB_MALWARE"
    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTE_FORCE"
    if label == "VULNERABILITYSCAN":
        return "VULNERABILITY_SCAN"
    return "OTHER"


def table1_dataset_distribution(working: pd.DataFrame) -> pd.DataFrame:
    fam = working["Label"].map(map_family)
    counts = fam.value_counts()
    if int(counts.get("OTHER", 0)) != 0:
        raise ValueError(
            "The working dataset contains labels outside the manuscript taxonomy; "
            "verify map_family() before generating Table 1."
        )
    order = ["BENIGN", "DDOS", "DOS", "MIRAI", "RECON", "SPOOFING", "VULNERABILITY_SCAN", "WEB_MALWARE", "BRUTE_FORCE"]
    total = int(len(working))
    rows = []
    for f in order:
        n = int(counts.get(f, 0))
        rows.append({
            "Traffic family": "Benign" if f == "BENIGN" else FAMILY_DISPLAY.get(f, f),
            "Representative fine-grained labels": REPRESENTATIVE_LABELS[f],
            "Number of observations": n,
            "Proportion of working dataset": f"{100*n/total:.2f}%",
        })
    rows.append({
        "Traffic family": "Total", "Representative fine-grained labels": "—",
        "Number of observations": total, "Proportion of working dataset": "100.00%",
    })
    return pd.DataFrame(rows)


def table2_protocols() -> pd.DataFrame:
    return pd.DataFrame([
        ["Stratified random split", "Traffic observations are randomly divided while preserving class proportions", "Binary and coarse-grained classification", "Measure conventional in-distribution performance"],
        ["Source-file-held-out split", "Source CSV files assigned to the test set are completely excluded from training", "Binary and coarse-grained classification", "Examine robustness to source-file separation"],
        ["Balanced leakage-controlled LOAFO", "One complete attack family is removed from training and evaluated against an equally sized benign subset", "Binary classification", "Assess unseen attack-family generalization"],
    ], columns=["Evaluation protocol", "Training–testing separation principle", "Classification task", "Primary objective"])


def table3_models() -> pd.DataFrame:
    return pd.DataFrame([
        ["Logistic Regression", "StandardScaler; maximum iterations = 1,000; class weight = balanced"],
        ["Random Forest", "Number of trees = 150; maximum depth = 24; minimum samples for split = 4; minimum samples per leaf = 2; class weight = balanced subsample"],
        ["Extra Trees", "Number of trees = 200; unrestricted maximum depth; minimum samples for split = 2; minimum samples per leaf = 1; class weight = balanced"],
        ["XGBoost", "Number of trees = 250; maximum depth = 8; learning rate = 0.08; subsample ratio = 0.85; column-sampling ratio = 0.85; tree method = histogram"],
        ["LightGBM", "Number of trees = 300; learning rate = 0.06; number of leaves = 64; subsample ratio = 0.85; column-sampling ratio = 0.85; class weight = balanced"],
    ], columns=["Model", "Main hyperparameters"])


def aggregate_baseline(raw: pd.DataFrame, task: str) -> pd.DataFrame:
    d = raw[(raw["status"] == "success") & (raw["task"] == task) & raw["split"].isin(["random", "file_held_out"])].copy()
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]
    a = d.groupby(["split", "model"])[metrics].agg(["mean", "std"])
    rows=[]
    for split in ["random", "file_held_out"]:
        for model in MODEL_ORDER:
            if (split, model) not in a.index:
                continue
            rows.append({
                "Validation protocol": "Random split" if split == "random" else "Source-file-held-out",
                "Model": MODEL_DISPLAY[model],
                "Accuracy": mean_sd(a.loc[(split,model),("accuracy","mean")], a.loc[(split,model),("accuracy","std")]),
                "Macro-F1": mean_sd(a.loc[(split,model),("macro_f1","mean")], a.loc[(split,model),("macro_f1","std")]),
                "Balanced accuracy": mean_sd(a.loc[(split,model),("balanced_accuracy","mean")], a.loc[(split,model),("balanced_accuracy","std")]),
                "MCC": mean_sd(a.loc[(split,model),("mcc","mean")], a.loc[(split,model),("mcc","std")]),
            })
    return pd.DataFrame(rows)


def table5_loafo(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy","macro_f1","mcc","attack_recall","benign_recall","false_positive_rate","robustness_equal_normalized"]
    a=run_df.groupby("model")[metrics].agg(["mean","std"])
    rows=[]
    for m in ["LightGBM","RandomForest","LogisticRegression","XGBoost","ExtraTrees"]:
        if m not in a.index: continue
        rows.append({
            "Model": MODEL_DISPLAY[m],
            "Accuracy": mean_sd(a.loc[m,("accuracy","mean")],a.loc[m,("accuracy","std")]),
            "Macro-F1": mean_sd(a.loc[m,("macro_f1","mean")],a.loc[m,("macro_f1","std")]),
            "MCC": mean_sd(a.loc[m,("mcc","mean")],a.loc[m,("mcc","std")]),
            "Attack recall": mean_sd(a.loc[m,("attack_recall","mean")],a.loc[m,("attack_recall","std")]),
            "Benign recall": mean_sd(a.loc[m,("benign_recall","mean")],a.loc[m,("benign_recall","std")]),
            "False-positive rate": mean_sd(a.loc[m,("false_positive_rate","mean")],a.loc[m,("false_positive_rate","std")]),
            "Normalized robustness score": mean_sd(a.loc[m,("robustness_equal_normalized","mean")],a.loc[m,("robustness_equal_normalized","std")]),
        })
    return pd.DataFrame(rows)


def table6_threshold(op: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for m in MODEL_ORDER:
        for strategy, label in [("default_0.5","Default (0.50)"),("validation_fpr_constrained","FPR-constrained")]:
            q=op[(op["model"]==m)&(op["threshold_strategy"]==strategy)]
            if q.empty: continue
            r=q.iloc[0]
            rows.append({
                "Model": MODEL_DISPLAY[m], "Threshold setting": label,
                "Attack precision": f"{r['attack_precision_mean']:.4f}",
                "Attack recall": f"{r['attack_recall_mean']:.4f}",
                "Attack F1": f"{r['attack_f1_mean']:.4f}",
                "FPR": f"{r['false_positive_rate_mean']:.4f}",
                "False alarms / 1,000 benign": f"{r['false_alarms_per_1000_benign_mean']:.1f}",
                "ROC-AUC": f"{r['roc_auc_mean']:.4f}",
                "PR-AUC": f"{r['pr_auc_mean']:.4f}",
            })
    return pd.DataFrame(rows)


def table8_friedman(stats: pd.DataFrame) -> pd.DataFrame:
    label={"macro_f1":"Macro-F1","balanced_accuracy":"Balanced accuracy","mcc":"MCC","attack_recall":"Attack recall","false_positive_rate":"False-positive rate","robustness_equal_normalized":"Normalized robustness"}
    rows=[]
    for _,r in stats.iterrows():
        rows.append({
            "Metric":label.get(r["metric"],r["metric"]),"n families":int(r["n_attack_families"]),
            "Friedman χ²":f"{r['friedman_chi2']:.3f}","p-value":"<0.0001" if r["p_value"]<0.0001 else f"{r['p_value']:.4f}","Kendall’s W":f"{r['kendall_w']:.3f}",
            "Logistic Regression rank":round_half_up(r["mean_rank_LogisticRegression"],3),"Random Forest rank":round_half_up(r["mean_rank_RandomForest"],3),"Extra Trees rank":round_half_up(r["mean_rank_ExtraTrees"],3),"XGBoost rank":round_half_up(r["mean_rank_XGBoost"],3),"LightGBM rank":round_half_up(r["mean_rank_LightGBM"],3),
        })
    return pd.DataFrame(rows)


def table9_deployment(dep: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for m in ["LogisticRegression","LightGBM","XGBoost","RandomForest","ExtraTrees"]:
        q=dep[dep["model"]==m]
        if q.empty: continue
        r=q.iloc[0]
        rows.append({"Model":MODEL_DISPLAY[m],"Training time (s)":f"{r['training_seconds']:.3f}","Inference latency (ms/flow)":f"{r['test_inference_ms_per_flow']:.6f}","Throughput (flows/s)":f"{r['test_throughput_flows_per_second']:,.0f}","Serialized model size (MB)":f"{r['serialized_model_size_mb']:.3f}","Peak training RSS Δ (MB)":f"{r['training_peak_rss_delta_mb']:.1f}"})
    return pd.DataFrame(rows)


def figure1(raw: pd.DataFrame, out: Path):
    d=raw[(raw["status"]=="success")&(raw["task"]=="binary")&raw["split"].isin(["random","file_held_out"])].copy()
    metrics=[("macro_f1","Macro-F1"),("balanced_accuracy","Balanced accuracy"),("mcc","MCC")]
    a=d.groupby(["split","model"])[[x for x,_ in metrics]].agg(["mean","std"])
    cats=[(m,k,label) for m in MODEL_ORDER for k,label in metrics]
    x=np.arange(len(cats)); w=.36
    rm=[];rs=[];fm=[];fs=[]
    for m,k,_ in cats:
        rm.append(a.loc[("random",m),(k,"mean")]); rs.append(a.loc[("random",m),(k,"std")])
        fm.append(a.loc[("file_held_out",m),(k,"mean")]); fs.append(a.loc[("file_held_out",m),(k,"std")])
    fig,ax=plt.subplots(figsize=(18,7.5))
    ax.bar(x-w/2,rm,w,yerr=rs,capsize=3,label="Random split")
    ax.bar(x+w/2,fm,w,yerr=fs,capsize=3,label="Source-file-held-out")
    ax.set_title("Comparison of Random-Split and Source-File-Held-Out Performance"); ax.set_ylabel("Score"); ax.set_ylim(0,1.05)
    ax.set_xticks(x); ax.set_xticklabels([lab for _,_,lab in cats],rotation=35,ha="right"); ax.grid(axis="y",alpha=.30); ax.legend()
    n=len(metrics)
    for i,m in enumerate(MODEL_ORDER):
        start=i*n; end=start+n-1; mid=(start+end)/2
        ax.text(mid,-.24,MODEL_DISPLAY[m],ha="center",va="top",transform=ax.get_xaxis_transform(),fontweight="bold")
        if i<len(MODEL_ORDER)-1: ax.axvline(end+.5,linewidth=.8,alpha=.4)
    fig.tight_layout(); save_fig(fig,out/"Figure_1_Random_vs_SourceFileHeldOut")


def figure2(run: pd.DataFrame, out: Path):
    data=[];labels=[]
    for m in MODEL_ORDER:
        v=run.loc[run["model"]==m,"robustness_equal_normalized"].dropna().to_numpy()
        if len(v): data.append(v); labels.append(MODEL_DISPLAY[m])
    fig,ax=plt.subplots(figsize=(6.8,3.8)); ax.boxplot(data,vert=False,tick_labels=labels,showmeans=True,meanprops={"marker":"^","markersize":5},medianprops={"linewidth":1.3}); ax.set_xlabel("Normalized robustness score"); ax.set_xlim(0,1); ax.grid(axis="x",alpha=.25); save_fig(fig,out/"Figure_2_Normalized_Robustness_Distribution")


def figure3(family: pd.DataFrame, out: Path):
    pivot=family.pivot(index="held_out_family",columns="model",values="macro_f1")
    rows=[f for f in FAMILY_ORDER if f in pivot.index]; cols=[m for m in MODEL_ORDER if m in pivot.columns]
    mat=pivot.loc[rows,cols].to_numpy(float); fig,ax=plt.subplots(figsize=(7.4,4.7)); im=ax.imshow(mat,aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels([MODEL_DISPLAY[m] for m in cols],rotation=30,ha="right")
    ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([FAMILY_DISPLAY[f] for f in rows])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val=mat[i,j]; rgba=im.cmap(im.norm(val)); lum=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]; color="black" if lum>.55 else "white"
            ax.text(j,i,f"{val:.3f}",ha="center",va="center",fontsize=7.5,color=color)
    c=fig.colorbar(im,ax=ax,fraction=.035,pad=.03); c.set_label("Macro-F1"); fig.tight_layout(); save_fig(fig,out/"Figure_3_Family_MacroF1_Heatmap")


def figure4(ci: pd.DataFrame, out: Path):
    rec=ci[ci["metric"]=="attack_recall"].set_index("model"); fpr=ci[ci["metric"]=="false_positive_rate"].set_index("model")
    fig,ax=plt.subplots(figsize=(6.2,4.6)); offsets={"LogisticRegression":(6,4,"left"),"RandomForest":(6,5,"left"),"LightGBM":(6,5,"left"),"XGBoost":(-8,6,"right"),"ExtraTrees":(8,6,"left")}
    for m in MODEL_ORDER:
        if m not in rec.index or m not in fpr.index: continue
        x=float(fpr.loc[m,"family_level_mean"]); y=float(rec.loc[m,"family_level_mean"])
        xerr=np.array([[x-float(fpr.loc[m,"bootstrap_95ci_low"])],[float(fpr.loc[m,"bootstrap_95ci_high"])-x]])
        yerr=np.array([[y-float(rec.loc[m,"bootstrap_95ci_low"])],[float(rec.loc[m,"bootstrap_95ci_high"])-y]])
        ax.errorbar(x,y,xerr=xerr,yerr=yerr,marker="o",markersize=6,capsize=3,linestyle="none")
        dx,dy,ha=offsets.get(m,(6,4,"left")); ax.annotate(MODEL_DISPLAY[m],(x,y),xytext=(dx,dy),textcoords="offset points",fontsize=7.5,ha=ha)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("Attack recall"); ax.set_xlim(left=0); ax.set_ylim(0,1.03); ax.grid(alpha=.25); fig.tight_layout(); save_fig(fig,out/"Figure_4_Recall_FPR_Pareto_95CI")


def supplementary_s1(rankings: pd.DataFrame) -> pd.DataFrame:
    mapping={"robustness_equal_normalized":"Equal","robustness_f1_emphasis":"F1 emphasis","robustness_balanced_accuracy_emphasis":"BA emphasis","robustness_mcc_emphasis":"MCC emphasis","robustness_f1_balanced_accuracy_only":"F1 + BA only"}
    p=rankings[rankings["score_definition"].isin(mapping)].pivot(index="model",columns="score_definition",values="mean_score")
    rows=[]
    for m in ["LightGBM","RandomForest","LogisticRegression","XGBoost","ExtraTrees"]:
        if m not in p.index: continue
        row={"Model":MODEL_DISPLAY[m]}
        for src,dst in mapping.items(): row[dst]=f"{p.loc[m,src]:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def supplementary_s4(anom: pd.DataFrame) -> pd.DataFrame:
    labels={"default_iforest":"Default Isolation Forest","validation_fpr_constrained":"Validation FPR-constrained","validation_macro_f1":"Validation Macro-F1"}
    rows=[]
    for s in ["default_iforest","validation_fpr_constrained","validation_macro_f1"]:
        q=anom[anom["threshold_strategy"]==s]
        if q.empty: continue
        r=q.iloc[0]
        row={"Threshold strategy":labels[s]}
        for src,dst in [("balanced_accuracy","Balanced accuracy"),("macro_f1","Macro-F1"),("mcc","MCC"),("attack_precision","Attack precision"),("attack_recall","Attack recall"),("attack_f1","Attack F1"),("false_positive_rate","FPR"),("false_alarms_per_1000_benign","FA / 1,000 benign"),("roc_auc","ROC-AUC"),("pr_auc","PR-AUC")]:
            dec=1 if src=="false_alarms_per_1000_benign" else 4
            row[dst]=mean_sd(r[f"{src}_mean"],r[f"{src}_std"],dec)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args=parse_args(); root=args.root.resolve(); out=(root/args.out).resolve(); main_tables=out/"main_tables"; main_figs=out/"main_figures"; supp=out/"supplementary_tables"
    for p in [main_tables,main_figs,supp]: p.mkdir(parents=True,exist_ok=True)
    paths={
        "working": require(root/"mergeddata"/"ciciot2023_working_with_source.csv"),
        "raw": require(root/"CICIoT2023_BENCHMARK"/"results_all_benchmark.csv"),
        "run": require(root/"CICIoT2023_ROBUSTNESS_SENSITIVITY"/"robustness_run_level_augmented.csv"),
        "rankings": require(root/"CICIoT2023_ROBUSTNESS_SENSITIVITY"/"robustness_model_rankings.csv"),
        "family": require(root/"CICIoT2023_STATISTICAL_ANALYSIS"/"family_level_seed_averaged_results.csv"),
        "stats": require(root/"CICIoT2023_STATISTICAL_ANALYSIS"/"friedman_primary_family_level.csv"),
        "s3": require(root/"CICIoT2023_STATISTICAL_ANALYSIS"/"friedman_40block_sensitivity.csv"),
        "pairwise": require(root/"CICIoT2023_STATISTICAL_ANALYSIS"/"pairwise_wilcoxon_holm_effect_sizes.csv"),
        "ci": require(root/"CICIoT2023_STATISTICAL_ANALYSIS"/"model_family_level_bootstrap_95ci.csv"),
        "op": require(root/"CICIoT2023_THRESHOLD_OPERATIONAL"/"operational_metrics_model_summary.csv"),
        "anom": require(root/"CICIoT2023_ANOMALY_BASELINE"/"anomaly_baseline_summary.csv"),
        "dep": require(root/"CICIoT2023_DEPLOYMENT_PROFILING"/"deployment_profile_compact_table.csv"),
    }
    working=pd.read_csv(paths["working"],low_memory=False); raw=pd.read_csv(paths["raw"]); run=pd.read_csv(paths["run"]); rankings=pd.read_csv(paths["rankings"]); family=pd.read_csv(paths["family"]); stats=pd.read_csv(paths["stats"]); s3=pd.read_csv(paths["s3"]); pairwise=pd.read_csv(paths["pairwise"]); ci=pd.read_csv(paths["ci"]); op=pd.read_csv(paths["op"]); anom=pd.read_csv(paths["anom"]); dep=pd.read_csv(paths["dep"])
    table1_dataset_distribution(working).to_csv(main_tables/"Table_1_Dataset_Distribution.csv",index=False)
    table2_protocols().to_csv(main_tables/"Table_2_Evaluation_Protocols.csv",index=False)
    table3_models().to_csv(main_tables/"Table_3_Model_Configurations.csv",index=False)
    aggregate_baseline(raw,"binary").to_csv(main_tables/"Table_4_Binary_Random_vs_SourceFileHeldOut.csv",index=False)
    table5_loafo(run).to_csv(main_tables/"Table_5_Balanced_LOAFO.csv",index=False)
    table6_threshold(op).to_csv(main_tables/"Table_6_Threshold_Operational.csv",index=False)
    aggregate_baseline(raw,"coarse").to_csv(main_tables/"Table_7_Coarse_Classification.csv",index=False)
    table8_friedman(stats).to_csv(main_tables/"Table_8_Family_Level_Friedman.csv",index=False)
    table9_deployment(dep).to_csv(main_tables/"Table_9_Deployment_Profile.csv",index=False)
    supplementary_s1(rankings).to_csv(supp/"Table_S1_Robustness_Weighting_Sensitivity.csv",index=False)
    pairwise.to_csv(supp/"Table_S2_Pairwise_Wilcoxon_Holm_Effect_Sizes.csv",index=False)
    s3.to_csv(supp/"Table_S3_40Block_Sensitivity.csv",index=False)
    supplementary_s4(anom).to_csv(supp/"Table_S4_Isolation_Forest.csv",index=False)
    figure1(raw,main_figs); figure2(run,main_figs); figure3(family,main_figs); figure4(ci,main_figs)
    print(f"Manuscript outputs written to: {out}")


if __name__ == "__main__":
    main()
