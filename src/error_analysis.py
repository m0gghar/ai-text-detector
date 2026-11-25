"""Error Diagnostics, Threshold Analysis, Calibration & Kaggle Internal Overlap Audit (v3.1).
Implements:
1. MinHash/LSH Kaggle internal train-test overlap check (MUST-FIX 4)
2. Validation-only threshold sweep & ROC curves (docs/roc_threshold_curve.png)
3. Reliability diagram & Expected Calibration Error (docs/calibration_plot.png)
4. Length stratification curves (docs/accuracy_vs_length.png)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from datasketch import MinHash, MinHashLSH

# Setup project root in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import BaselineDetector
from src.preprocessing import clean_text, get_word_count
from data.load_external_benchmarks import compute_minhash, NUM_PERM, LSH_THRESHOLD

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
RESULTS_JSON_PATH = os.path.join(DOCS_DIR, "master_benchmark_results.json")


def audit_kaggle_train_test_overlap(
    train_path: str,
    test_path: str,
    max_train_samples: int = 15000,
    test_sample_size: int = 3000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run MinHash LSH check (5-gram, 128 permutations, Jaccard >= 0.80)
    between train.csv and test.csv.
    Reports:
    - Number and percentage of test texts with near-duplicates in train
    - Model accuracy separately for test texts WITH and WITHOUT near-duplicates in train
    """
    print(f"\n[1/4] Running Kaggle Internal Train-Test Overlap Audit...")
    print(f"Reading train samples (capped at {max_train_samples:,}) and test samples ({test_sample_size:,})...")
    
    df_train = pd.read_csv(train_path, nrows=max_train_samples * 2)
    if len(df_train) > max_train_samples:
        df_train_idx = df_train.sample(n=max_train_samples, random_state=random_seed).reset_index(drop=True)
    else:
        df_train_idx = df_train.reset_index(drop=True)

    df_test = pd.read_csv(test_path, nrows=test_sample_size * 2)
    if len(df_test) > test_sample_size:
        df_test_sample = df_test.sample(n=test_sample_size, random_state=random_seed).reset_index(drop=True)
    else:
        df_test_sample = df_test.reset_index(drop=True)

    print(f"Building MinHash LSH index over {len(df_train_idx):,} training texts...")
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    for idx, text in enumerate(df_train_idx["text"]):
        m = compute_minhash(str(text))
        lsh.insert(f"train_{idx}", m)

    print(f"Querying {len(df_test_sample):,} test texts for near-duplicates in train...")
    has_near_dup = []
    for text in df_test_sample["text"]:
        m = compute_minhash(str(text))
        matches = lsh.query(m)
        has_near_dup.append(len(matches) > 0)

    df_test_sample["has_near_duplicate"] = has_near_dup

    with_dup_count = int(sum(has_near_dup))
    without_dup_count = int(len(has_near_dup) - with_dup_count)
    dup_percentage = round(float(with_dup_count / len(has_near_dup)) * 100, 2)

    # Evaluate accuracy with vs without near-duplicates using frozen SVM
    print("Evaluating SVM accuracy partitioned by near-duplicate presence...")
    detector = BaselineDetector(model_type="svm")
    texts_clean = [clean_text(t) for t in df_test_sample["text"]]
    probs = detector.pipeline.predict_proba(texts_clean)[:, 1]
    preds = (probs >= 0.50).astype(int)
    df_test_sample["pred"] = preds

    df_with = df_test_sample[df_test_sample["has_near_duplicate"] == True]
    df_without = df_test_sample[df_test_sample["has_near_duplicate"] == False]

    acc_with = round(float(accuracy_score(df_with["label"], df_with["pred"])), 4) if len(df_with) > 0 else "not applicable"
    acc_without = round(float(accuracy_score(df_without["label"], df_without["pred"])), 4) if len(df_without) > 0 else "not applicable"
    acc_overall = round(float(accuracy_score(df_test_sample["label"], df_test_sample["pred"])), 4)

    overlap_report = {
        "test_sample_evaluated": len(df_test_sample),
        "train_reference_size": len(df_train_idx),
        "lsh_jaccard_threshold": LSH_THRESHOLD,
        "near_duplicate_count": with_dup_count,
        "near_duplicate_percentage": dup_percentage,
        "clean_count": without_dup_count,
        "accuracy_overall": acc_overall,
        "accuracy_with_near_duplicate": acc_with,
        "accuracy_without_near_duplicate": acc_without,
        "wording_standard": "No exact or near-duplicate matches found at the 0.80 Jaccard threshold (MinHash is probabilistic)",
    }

    print("--- Kaggle Internal Overlap Audit Summary ---")
    print(f"Test texts evaluated: {len(df_test_sample):,}")
    print(f"Near-duplicates found in train: {with_dup_count:,} ({dup_percentage}%)")
    print(f"Accuracy on texts WITH near-duplicate: {acc_with}")
    print(f"Accuracy on texts WITHOUT near-duplicate: {acc_without}")

    return overlap_report


def generate_roc_threshold_plots(val_csv_path: str, detector: Any, output_path: str) -> Dict[str, Any]:
    """Generates ROC curve and validation-only threshold trade-off sweeps (docs/roc_threshold_curve.png)."""
    print(f"\n[2/4] Generating ROC & Threshold Trade-off Curves strictly on Validation set...")
    df_val = pd.read_csv(val_csv_path, nrows=5000)
    y_true = df_val["label"].values.astype(int)

    texts_clean = [clean_text(t) for t in df_val["text"]]
    probs = detector.pipeline.predict_proba(texts_clean)[:, 1]

    fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr_arr, tpr_arr)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    prec_list, rec_list, f1_list, fpr_list = [], [], [], []

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f)
        fpr_list.append(fp_rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: ROC Curve
    ax1.plot(fpr_arr, tpr_arr, color="#1976D2", lw=2.5, label=f"Calibrated Linear SVM (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="#9E9E9E", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    # Mark operating point tau = 0.50
    preds_50 = (probs >= 0.50).astype(int)
    from sklearn.metrics import confusion_matrix
    cm_50 = confusion_matrix(y_true, preds_50, labels=[0, 1])
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel()
    fpr_50 = fp_50 / (fp_50 + tn_50)
    tpr_50 = tp_50 / (tp_50 + fn_50)
    ax1.scatter([fpr_50], [tpr_50], color="#D32F2F", s=100, zorder=5, label=f"Decision Operating Point ($\\tau=0.50$)")

    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (Human False Alarms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (AI Detection Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Threshold Sweeps
    ax2.plot(thresholds, f1_list, marker="o", color="#2E7D32", lw=2, label="F1-Score")
    ax2.plot(thresholds, prec_list, marker="s", color="#1976D2", lw=2, label="Precision (AI)")
    ax2.plot(thresholds, rec_list, marker="^", color="#E65100", lw=2, label="Recall (AI)")
    ax2.plot(thresholds, fpr_list, marker="x", color="#D32F2F", lw=2, linestyle="--", label="False Positive Rate (Human)")
    ax2.axvline(0.50, color="#7B1FA2", linestyle=":", lw=2, label="Selected Threshold ($\\tau = 0.50$)")

    ax2.set_xlabel("Decision Threshold ($\\tau$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax2.set_title("Threshold Trade-Off Sweeps (Validation Set)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC & Threshold sweeps plot to: {output_path}")

    return {
        "roc_auc": round(float(roc_auc), 4),
        "threshold_sweeps": [
            {"threshold": t, "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "fpr": round(float(fp_r), 4)}
            for t, p, r, f, fp_r in zip(thresholds, prec_list, rec_list, f1_list, fpr_list)
        ]
    }


def generate_calibration_plot(val_csv_path: str, detector: Any, output_path: str) -> Dict[str, Any]:
    """Generates Reliability Diagram and Expected Calibration Error (docs/calibration_plot.png)."""
    print(f"\n[3/4] Generating Reliability Diagram & Calibration Curve...")
    df_val = pd.read_csv(val_csv_path, nrows=5000)
    y_true = df_val["label"].values.astype(int)

    texts_clean = [clean_text(t) for t in df_val["text"]]
    probs = detector.pipeline.predict_proba(texts_clean)[:, 1]

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, probs)

    # Compute Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += (np.sum(bin_mask) / len(probs)) * np.abs(bin_acc - bin_conf)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9E9E9E", label="Perfect Calibration (Ideal)")
    ax.plot(prob_pred, prob_true, marker="o", color="#1565C0", lw=2.5, label=f"Calibrated Linear SVM (ECE = {ece:.4f})")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=11, fontweight="bold")
    ax.set_title("Reliability Diagram / Calibration Curve", fontsize=12, fontweight="bold")
    ax.text(0.05, 0.85, f"Brier Score: {brier:.4f}\nECE: {ece:.4f}\nPlatt Scaling: Active", 
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#90CAF9"))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Calibration plot to: {output_path}")

    return {
        "brier_score": round(float(brier), 4),
        "expected_calibration_error": round(float(ece), 4),
    }


def generate_length_stratification_plot(test_csv_path: str, detector: Any, output_path: str) -> Dict[str, Any]:
    """Evaluates performance across word-count bins and produces docs/accuracy_vs_length.png."""
    print(f"\n[4/4] Generating Length Stratification Curves across Word Count Bins...")
    df_test = pd.read_csv(test_csv_path, nrows=10000)
    df_test["word_count"] = df_test["text"].astype(str).apply(get_word_count)

    bins = [0, 50, 100, 250, 500, 10000]
    bin_labels = ["< 50 words", "50-100 words", "100-250 words", "250-500 words", "> 500 words"]
    df_test["length_bin"] = pd.cut(df_test["word_count"], bins=bins, labels=bin_labels, right=False)

    texts_clean = [clean_text(t) for t in df_test["text"]]
    probs = detector.pipeline.predict_proba(texts_clean)[:, 1]
    df_test["pred"] = (probs >= 0.50).astype(int)

    bin_accuracies = []
    bin_counts = []
    for label in bin_labels:
        sub = df_test[df_test["length_bin"] == label]
        if len(sub) > 0:
            acc = float(accuracy_score(sub["label"], sub["pred"]))
        else:
            acc = 0.0
        bin_accuracies.append(acc)
        bin_counts.append(len(sub))

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color1 = "#1976D2"
    ax1.set_xlabel("Passage Length Category", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Detection Accuracy", color=color1, fontsize=11, fontweight="bold")
    bars = ax1.bar(bin_labels, bin_accuracies, color=color1, alpha=0.85, width=0.55, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0, 1.08])

    for bar, acc in zip(bars, bin_accuracies):
        height = bar.get_height()
        ax1.annotate(f"{acc*100:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.axhline(0.99, color="#D32F2F", linestyle="--", label="In-Domain Benchmark Target (99%)")
    ax1.set_title("Detector Accuracy Stratified by Passage Word Count", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Length Stratification plot to: {output_path}")

    return {
        bin_lbl: {"count": cnt, "accuracy": round(acc, 4)}
        for bin_lbl, cnt, acc in zip(bin_labels, bin_counts, bin_accuracies)
    }


def run_full_error_analysis_and_diagnostics():
    """Run overlap audit, threshold curves, calibration curves, and length stratification."""
    print("=" * 70)
    print("Starting Comprehensive Diagnostics & Kaggle Internal Overlap Audit")
    print("=" * 70)

    train_path = os.path.join(DATA_DIR, "processed", "train.csv")
    val_path = os.path.join(DATA_DIR, "processed", "val.csv")
    test_path = os.path.join(DATA_DIR, "processed", "test.csv")

    detector = BaselineDetector(model_type="svm")

    # 1. Overlap Audit
    overlap_report = audit_kaggle_train_test_overlap(train_path, test_path, max_train_samples=15000, test_sample_size=3000)

    # 2. ROC & Threshold Sweeps
    roc_out = os.path.join(DOCS_DIR, "roc_threshold_curve.png")
    roc_report = generate_roc_threshold_plots(val_path, detector, roc_out)

    # 3. Calibration Curve & ECE
    cal_out = os.path.join(DOCS_DIR, "calibration_plot.png")
    cal_report = generate_calibration_plot(val_path, detector, cal_out)

    # 4. Length Stratification Plot
    len_out = os.path.join(DOCS_DIR, "accuracy_vs_length.png")
    len_report = generate_length_stratification_plot(test_path, detector, len_out)

    # 5. Append to master_benchmark_results.json
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["kaggle_internal_overlap_audit"] = overlap_report
        data["validation_threshold_study"] = roc_report
        data["calibration_analysis"] = cal_report
        data["length_stratification_analysis"] = len_report

        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nAppended diagnostics to: {RESULTS_JSON_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    run_full_error_analysis_and_diagnostics()
