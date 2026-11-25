"""Multi-Benchmark Evaluation Engine (v3.1).
Implements uniform length filtering (>=50 words), single-class metric handling ('not applicable'),
segregated short-text analysis, fine-grained RAID stratification slices,
and 95% bootstrap confidence intervals (B=1000, seed=42).
Produces docs/master_benchmark_results.json as the single source of truth.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# Setup project root in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import BaselineDetector
from src.preprocessing import clean_text, get_word_count

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
BENCHMARKS_DIR = os.path.join(DATA_DIR, "external_benchmarks")
OUTPUT_JSON_PATH = os.path.join(DOCS_DIR, "master_benchmark_results.json")
OUTPUT_TABLE_PATH = os.path.join(DOCS_DIR, "master_benchmark_table.md")

MIN_WORD_THRESHOLD = 50
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash for reproducibility provenance."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "git_commit_not_available_or_clean_repo"


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Any,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval (2.5th and 97.5th percentiles)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 5:
        return 0.0, 1.0

    scores = []
    for _ in range(rounds):
        idx = rng.randint(0, n, size=n)
        s_true = y_true[idx]
        s_pred = y_pred[idx]
        try:
            score = metric_func(s_true, s_pred)
            scores.append(score)
        except Exception:
            pass

    if not scores:
        return 0.0, 1.0

    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return round(lower, 4), round(upper, 4)


def evaluate_dataset_robust(
    df_raw: pd.DataFrame,
    dataset_name: str,
    detector: Any,
    apply_50_word_filter: bool = True,
    batch_size: int = 500,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate dataset with:
    - Length accounting: rows_before_filter, rows_removed, rows_evaluated
    - Single-class detection: handling 'not applicable' for metrics that cannot be defined
    - Bootstrap 95% confidence intervals
    - Batch inference for high throughput
    """
    rows_before = len(df_raw)
    df = df_raw.copy()
    df["word_count"] = df["text"].astype(str).apply(get_word_count)

    if apply_50_word_filter:
        df_short = df[df["word_count"] < MIN_WORD_THRESHOLD].copy().reset_index(drop=True)
        df_eval = df[df["word_count"] >= MIN_WORD_THRESHOLD].copy().reset_index(drop=True)
        rows_removed = len(df_short)
    else:
        df_short = pd.DataFrame()
        df_eval = df.reset_index(drop=True)
        rows_removed = 0

    rows_evaluated = len(df_eval)

    if rows_evaluated == 0:
        return {
            "dataset_name": dataset_name,
            "rows_before_filter": rows_before,
            "rows_removed": rows_removed,
            "rows_evaluated": 0,
            "accuracy": "not applicable",
            "accuracy_ci95": "not applicable",
            "false_positive_rate": "not applicable",
            "specificity": "not applicable",
            "recall": "not applicable",
            "precision": "not applicable",
            "f1_score": "not applicable",
            "f1_score_ci95": "not applicable",
            "balanced_accuracy": "not applicable",
            "macro_f1": "not applicable",
            "roc_auc": "not applicable",
            "status": "empty_after_filter",
        }, df_short

    y_true = df_eval["label"].values.astype(int)

    # Fast batch inference in chunks
    texts = df_eval["text"].astype(str).tolist()
    y_prob = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        chunk_clean = [clean_text(t) for t in chunk]
        probs = detector.pipeline.predict_proba(chunk_clean)[:, 1]
        y_prob.extend(probs)

    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.50).astype(int)

    unique_classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Base Metrics
    acc = accuracy_score(y_true, y_pred)
    acc_ci = compute_bootstrap_ci(y_true, y_pred, accuracy_score)

    result: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "rows_before_filter": int(rows_before),
        "rows_removed": int(rows_removed),
        "rows_evaluated": int(rows_evaluated),
        "class_distribution": {
            "human_count": int((y_true == 0).sum()),
            "ai_count": int((y_true == 1).sum()),
        },
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "accuracy": round(float(acc), 4),
        "accuracy_ci95": acc_ci,
    }

    # Case A: Dataset has ONLY Human text (e.g. formal_human)
    if len(unique_classes) == 1 and unique_classes[0] == 0:
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 1.0
        result.update({
            "false_positive_rate": round(fpr, 4),
            "specificity": round(specificity, 4),
            "recall": "not applicable",
            "precision": "not applicable",
            "f1_score": "not applicable",
            "f1_score_ci95": "not applicable",
            "balanced_accuracy": "not applicable",
            "macro_f1": "not applicable",
            "roc_auc": "not applicable",
            "note": "Single-class dataset (Human-only). Recall, Precision, F1 and ROC-AUC are mathematically not applicable.",
        })

    # Case B: Dataset has ONLY AI text (e.g. paraphrased_test)
    elif len(unique_classes) == 1 and unique_classes[0] == 1:
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        result.update({
            "recall": round(recall, 4),
            "false_positive_rate": "not applicable",
            "specificity": "not applicable",
            "precision": "not applicable",
            "f1_score": "not applicable",
            "f1_score_ci95": "not applicable",
            "balanced_accuracy": "not applicable",
            "macro_f1": "not applicable",
            "roc_auc": "not applicable",
            "note": "Single-class dataset (AI-only). FPR, Specificity, Precision, F1 and ROC-AUC are mathematically not applicable.",
        })

    # Case C: Standard Binary Dataset (both classes present)
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 1.0

        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.50

        f1_ci = compute_bootstrap_ci(
            y_true, y_pred,
            lambda t, p: precision_recall_fscore_support(t, p, average="binary", zero_division=0)[2]
        )
        bal_acc_ci = compute_bootstrap_ci(y_true, y_pred, balanced_accuracy_score)

        result.update({
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "f1_score_ci95": f1_ci,
            "balanced_accuracy": round(float(bal_acc), 4),
            "balanced_accuracy_ci95": bal_acc_ci,
            "macro_f1": round(float(macro_f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "specificity": round(float(specificity), 4),
            "roc_auc": round(float(auc), 4),
        })

    return result, df_short


def evaluate_short_text_slice(df_short: pd.DataFrame, source_name: str, detector: Any) -> Dict[str, Any]:
    """Evaluates short-text (<50 words) partition separately."""
    if len(df_short) == 0:
        return {
            "source": source_name,
            "count": 0,
            "accuracy": "not applicable",
            "human_count": 0,
            "ai_count": 0,
        }

    texts = [clean_text(t) for t in df_short["text"]]
    probs = detector.pipeline.predict_proba(texts)[:, 1]
    preds = (probs >= 0.50).astype(int)
    y = df_short["label"].values.astype(int)
    acc = round(float(accuracy_score(y, preds)), 4)

    return {
        "source": source_name,
        "count": len(df_short),
        "accuracy": acc,
        "human_count": int((y == 0).sum()),
        "ai_count": int((y == 1).sum()),
    }


def generate_master_benchmark_markdown_table(data: Dict[str, Any], output_path: str):
    """Render master_benchmark_table.md strictly from data dictionary."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Master Benchmark Results Table\n\n")
        f.write(f"**Generated**: {data['metadata']['execution_timestamp']}  \n")
        f.write(f"**Git Commit**: `{data['metadata']['git_commit']}`  \n")
        f.write(f"**Random Seed**: {data['metadata']['random_seed']} | **Bootstrap Rounds**: {BOOTSTRAP_ROUNDS}  \n")
        f.write(f"**Decision Threshold**: {data['metadata']['decision_threshold']} | **Model**: `{data['metadata']['model_name']}`\n\n")

        f.write("### 1. Main Benchmark Performance (Passages $\\ge 50$ Words)\n\n")
        f.write("| Dataset | Rows (Before / Removed / Eval) | Accuracy (95% CI) | Balanced Acc | Macro-F1 | Binary F1 | Recall (AI) | FPR (Human False Alarm) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for key, res in data["benchmarks"].items():
            name = res["dataset_name"]
            counts = f"{res['rows_before_filter']} / {res['rows_removed']} / **{res['rows_evaluated']}**"
            acc_str = f"{res['accuracy']*100:.2f}% [{res['accuracy_ci95'][0]*100:.1f}-{res['accuracy_ci95'][1]*100:.1f}]" if isinstance(res['accuracy'], (int, float)) else str(res['accuracy'])
            bal_str = f"{res['balanced_accuracy']*100:.2f}%" if isinstance(res['balanced_accuracy'], (int, float)) else str(res['balanced_accuracy'])
            mac_str = f"{res['macro_f1']*100:.2f}%" if isinstance(res['macro_f1'], (int, float)) else str(res['macro_f1'])
            f1_str = f"{res['f1_score']*100:.2f}%" if isinstance(res['f1_score'], (int, float)) else str(res['f1_score'])
            rec_str = f"{res['recall']*100:.2f}%" if isinstance(res['recall'], (int, float)) else str(res['recall'])
            fpr_str = f"{res['false_positive_rate']*100:.2f}%" if isinstance(res['false_positive_rate'], (int, float)) else str(res['false_positive_rate'])

            f.write(f"| **{name}** | {counts} | {acc_str} | {bal_str} | {mac_str} | **{f1_str}** | {rec_str} | {fpr_str} |\n")

        f.write("\n---\n\n")
        f.write("### 2. RAID Benchmark Fine-Grained Stratification (Passages $\\ge 50$ Words)\n\n")
        f.write("#### Performance by Generator\n")
        f.write("| Generator | Count | Accuracy | AI Detection Rate (Recall) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for gen, g_res in data["raid_fine_grained_slices"]["by_generator"].items():
            rec_disp = f"{g_res['recall']*100:.2f}%" if isinstance(g_res['recall'], (int, float)) else g_res['recall']
            f.write(f"| `{gen}` | {g_res['count']} | {g_res['accuracy']*100:.2f}% | {rec_disp} |\n")

        f.write("\n#### Performance by Domain\n")
        f.write("| Domain | Count | Accuracy |\n")
        f.write("| :--- | :---: | :---: |\n")
        for dom, d_res in data["raid_fine_grained_slices"]["by_domain"].items():
            f.write(f"| `{dom}` | {d_res['count']} | {d_res['accuracy']*100:.2f}% |\n")

        f.write("\n#### Robustness Against Adversarial Attacks\n")
        f.write("| Attack Type | Count | Accuracy | AI Detection Rate |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for att, a_res in data["raid_fine_grained_slices"]["by_adversarial_attack"].items():
            det_disp = f"{a_res['ai_detection_rate']*100:.2f}%" if isinstance(a_res['ai_detection_rate'], (int, float)) else a_res['ai_detection_rate']
            f.write(f"| `{att}` | {a_res['count']} | {a_res['accuracy']*100:.2f}% | {det_disp} |\n")

        f.write("\n---\n\n")
        f.write("### 3. Segregated Short-Text Analysis (Texts $< 50$ Words)\n\n")
        f.write("| Dataset Source | Short Texts Count | Accuracy | Human Count | AI Count |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for key, s_res in data["short_text_analysis"].items():
            acc_val = f"{s_res['accuracy']*100:.2f}%" if isinstance(s_res['accuracy'], (int, float)) else s_res['accuracy']
            f.write(f"| **{s_res['source']}** | {s_res['count']} | {acc_val} | {s_res['human_count']} | {s_res['ai_count']} |\n")

        f.write("\n> *Note: Primary benchmarks strictly enforce the $W \\ge 50$ words threshold. Short texts are analyzed separately to avoid sample-size distortion.*\n")


def run_full_multi_benchmark_evaluation():
    """Execute complete multi-benchmark evaluation pipeline using frozen baseline Linear SVM."""
    print("=" * 70)
    print("Starting Multi-Benchmark Robustness & Generalization Evaluation (v3.1)")
    print("=" * 70)

    # 1. Load frozen baseline detector
    detector = BaselineDetector(model_type="svm")
    print(f"Loaded frozen detector: {detector.model_type.upper()} from {detector.model_path}")

    # Verify model hash
    import hashlib
    with open(detector.model_path, "rb") as f:
        svm_hash = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"Verified Frozen SVM SHA-256: {svm_hash}")

    datasets_to_eval = [
        # In-domain Kaggle
        ("kaggle_test_5k", "Kaggle In-Domain (5k Slice)", os.path.join(DATA_DIR, "processed", "test.csv"), 5000),
        ("kaggle_test_full", "Kaggle In-Domain (Full 73k)", os.path.join(DATA_DIR, "processed", "test.csv"), None),
        # External Benchmarks
        ("formal_human", "Pre-2022 Formal Human / ESL (arXiv)", os.path.join(BENCHMARKS_DIR, "formal_human", "test.csv"), None),
        ("hc3", "HC3 Benchmark (Human vs ChatGPT QA)", os.path.join(BENCHMARKS_DIR, "hc3", "test.csv"), None),
        ("raid", "RAID Benchmark (Multi-Generator)", os.path.join(BENCHMARKS_DIR, "raid", "test.csv"), None),
        ("m4", "M4 Benchmark (Wang et al., 2023)", os.path.join(BENCHMARKS_DIR, "m4", "test.csv"), None),
        # Stress Sets
        ("ood_creative", "Out-of-Domain Creative Fiction", os.path.join(DATA_DIR, "ood_test", "test.csv"), None),
        ("paraphrased_adversarial", "Paraphrased AI Evasion Test", os.path.join(DATA_DIR, "paraphrased_test", "test.csv"), None),
    ]

    benchmarks_results = {}
    short_text_results = {}

    for key, display_name, file_path, sample_n in datasets_to_eval:
        print(f"\nEvaluating: {display_name}...")
        if not os.path.exists(file_path):
            print(f" [WARNING] File not found: {file_path}, skipping.")
            continue

        df_raw = pd.read_csv(file_path)
        if sample_n is not None and len(df_raw) > sample_n:
            df_raw = df_raw.sample(n=sample_n, random_state=42).reset_index(drop=True)

        res, df_short = evaluate_dataset_robust(
            df_raw=df_raw,
            dataset_name=display_name,
            detector=detector,
            apply_50_word_filter=True,
            batch_size=500,
        )

        benchmarks_results[key] = res
        print(f" -> Kept: {res['rows_evaluated']}, Removed: {res['rows_removed']}, Accuracy: {res['accuracy']}")

        # Short text slice
        short_res = evaluate_short_text_slice(df_short, display_name, detector)
        short_text_results[f"{key}_short"] = short_res

    # 2. Detailed RAID Stratification Slices
    raid_path = os.path.join(BENCHMARKS_DIR, "raid", "test.csv")
    raid_slices = {"by_generator": {}, "by_domain": {}, "by_adversarial_attack": {}}
    if os.path.exists(raid_path):
        print("\nComputing fine-grained RAID stratification slices...")
        df_raid = pd.read_csv(raid_path)
        df_raid["wc"] = df_raid["text"].astype(str).apply(get_word_count)
        df_raid_clean = df_raid[df_raid["wc"] >= MIN_WORD_THRESHOLD].reset_index(drop=True)

        raid_texts = [clean_text(t) for t in df_raid_clean["text"]]
        raid_probs = detector.pipeline.predict_proba(raid_texts)[:, 1]
        df_raid_clean["pred"] = (raid_probs >= 0.50).astype(int)

        # By Generator
        for gen, grp in df_raid_clean.groupby("generator"):
            acc = float(accuracy_score(grp["label"], grp["pred"]))
            ai_sub = grp[grp["label"] == 1]
            rec = float((ai_sub["pred"] == 1).mean()) if len(ai_sub) > 0 else "not applicable"
            raid_slices["by_generator"][gen] = {
                "count": len(grp),
                "accuracy": round(acc, 4),
                "recall": round(rec, 4) if isinstance(rec, float) else rec,
            }

        # By Domain
        for dom, grp in df_raid_clean.groupby("domain"):
            acc = float(accuracy_score(grp["label"], grp["pred"]))
            raid_slices["by_domain"][dom] = {
                "count": len(grp),
                "accuracy": round(acc, 4),
            }

        # By Adversarial Attack
        for att, grp in df_raid_clean.groupby("adversarial_attack"):
            acc = float(accuracy_score(grp["label"], grp["pred"]))
            ai_sub = grp[grp["label"] == 1]
            rec = float((ai_sub["pred"] == 1).mean()) if len(ai_sub) > 0 else "not applicable"
            raid_slices["by_adversarial_attack"][att] = {
                "count": len(grp),
                "accuracy": round(acc, 4),
                "ai_detection_rate": round(rec, 4) if isinstance(rec, float) else rec,
            }

    # 3. Assemble Master Data Payload
    master_data = {
        "metadata": {
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit_hash(),
            "random_seed": 42,
            "model_type": "svm",
            "model_file": "models/baseline_svm.joblib",
            "model_sha256": svm_hash,
            "decision_threshold": 0.50,
            "min_word_threshold": MIN_WORD_THRESHOLD,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "model_name": "Calibrated Linear SVM (Sublinear TF-IDF + 5 Stylometrics)",
        },
        "benchmarks": benchmarks_results,
        "raid_fine_grained_slices": raid_slices,
        "short_text_analysis": short_text_results,
    }

    # 4. Save JSON and Markdown
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2)
    print(f"\nSaved master benchmark results JSON to: {OUTPUT_JSON_PATH}")

    generate_master_benchmark_markdown_table(master_data, OUTPUT_TABLE_PATH)
    print(f"Saved master benchmark Markdown table to: {OUTPUT_TABLE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_multi_benchmark_evaluation()
