# AI vs Human Text Detector 🛡️
### Final-Year B.Tech Capstone Project | Version 3.1 Research Edition

A machine learning and natural language processing system evaluating the boundaries of machine-generated text detection. The project investigates the **"Kaggle Distribution Trap"** (*Dugan et al., 2024, ACL*), demonstrating why detectors scoring $>99\%$ in-domain can collapse to $0\%$ accuracy when confronted with out-of-distribution human academic writing.

Deployed as an interactive Flask web application featuring real-time stylometrics, sentence-level explainability heatmaps, short-passage warnings ($<50$ words), and automated multi-benchmark evaluation.

---

## 🚀 Key System Features

- **Dual-Model Architecture**:
  - **Classical Baseline**: Sublinear TF-IDF (1–2 n-grams) + 5-dimensional Stylometric Feature Vector + Calibrated Linear SVM (`CalibratedClassifierCV` via Platt scaling).
  - **Modern Transformer**: Fine-tuned RoBERTa Sequence Classifier (Colab GPU notebook provided; UI strictly enforces safety lockout badge until trained checkpoint is verified).
- **Explainability & Sentence Heatmap**: Sentence-by-sentence decomposition highlighting suspicious passages with indicative warning disclaimers.
- **5-Dimensional Stylometric Analysis**: Real-time extraction of Average Sentence Length, Burstiness (Sentence Variance), Type-Token Ratio (TTR), Punctuation Density, and Average Word Length.
- **Uniform Length Threshold ($W \ge 50$ Words)**: Live UI alert and segregated evaluation to prevent sparse-feature false alarms on short passages.
- **Multi-Benchmark Generalization Suite**: Automated evaluation across Kaggle In-Domain (73k), RAID Benchmark (Dugan et al., 2024), HC3 Comparison Corpus, M4 Multi-Generator Corpus, and Pre-2022 Formal Human Writing.
- **Kaggle Train-Test Overlap Audit**: Automated MinHash LSH near-duplicate auditor (128 perms, character 5-grams, Jaccard $\ge 0.80$).

---

## 📊 Empirical Multi-Benchmark Results

All metrics evaluated using the frozen Calibrated Linear SVM ($\tau = 0.50$, $W \ge 50$ words, $B = 1000$ bootstrap iterations, seed 42):

| Dataset Partition | Rows (Before / Removed / Eval) | Accuracy (95% CI) | Balanced Acc | Macro-F1 | Binary F1 | Recall (AI) | FPR (Human False Alarm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle In-Domain (Full)** | 73,083 / 10 / **73,073** | **99.90%** [99.9–99.9] | 99.88% | 99.90% | **99.87%** | 99.81% | **0.04%** |
| **Kaggle In-Domain (5k Slice)** | 5,000 / 1 / **4,999** | **99.86%** [99.7–100.0] | 99.84% | 99.85% | **99.81%** | 99.74% | **0.06%** |
| **Pre-2022 Formal Human (arXiv)** | 14 / 2 / **12** | **0.00%** [0.0–0.0] | *N/A* | *N/A* | *N/A* | *N/A* | **100.00%** |
| **HC3 Benchmark (Guo et al.)** | 12 / 2 / **10** | **50.00%** [20.0–80.0] | 50.00% | 33.33% | **66.67%** | 100.00% | **100.00%** |
| **RAID Benchmark (Dugan et al.)** | 15 / 4 / **11** | **72.73%** [45.5–100.0] | 50.00% | 42.11% | **84.21%** | 100.00% | **100.00%** |
| **M4 Benchmark (Wang et al.)** | 7 / 3 / **4** | **75.00%** [0.0–100.0] | 50.00% | 42.86% | **85.71%** | 100.00% | **100.00%** |
| **Out-of-Domain Fiction** | 6 / 0 / **6** | **50.00%** [16.7–83.3] | 50.00% | 33.33% | **66.67%** | 100.00% | **100.00%** |
| **Paraphrased AI Evasion** | 9 / 8 / **1** | **100.00%** [0.0–100.0] | *N/A* | *N/A* | *N/A* | 100.00% | *N/A* |

### Key Scientific Takeaways
1. **The Kaggle Distribution Trap**: The model achieves 99.90% on student essays because human texts are informal ("Dear Principal") while AI texts are formal ChatGPT essays.
2. **Collapse on Academic Writing**: When given genuine pre-2022 arXiv papers (100% human, zero latent LLM possibility), the model has a **100% False Positive Rate**, misclassifying all complex human writing as AI.
3. **Internal Overlap**: MinHash LSH reveals that **18.37%** of test samples are near-duplicates of training prompts ($J \ge 0.80$).

---

## 📁 Repository Structure

```
ai-text-detector/
├── data/
│   ├── external_benchmarks/    # Held-out external evaluation corpora
│   │   ├── formal_human/       # Pre-2022 arXiv papers & TOEFL ESL writing (CC-BY 4.0)
│   │   ├── hc3/                # Human vs ChatGPT QA pairs (question-level split)
│   │   ├── raid/               # Stratified RAID benchmark (7 generators, 5 domains, 4 attacks)
│   │   └── m4/                 # M4 multi-generator corpus (Wang et al., 2023)
│   ├── ood_test/               # Creative fiction and memoirs
│   ├── paraphrased_test/       # Synonym-swapped adversarial AI passages
│   ├── load_external_benchmarks.py # Standardized ingestion & MinHash contamination filter
│   ├── load_kaggle_dataset.py  # Full 1.11 GB Kaggle ingestion pipeline
│   └── prepare_data.py         # Baseline train/val/test splitter
├── src/
│   ├── preprocessing.py        # Text normalization, sentence/word tokenization, word counter
│   ├── stylometrics.py         # 5-dimensional stylometric feature extractor (Scikit-Learn transformer)
│   ├── baseline_model.py       # FeatureUnion + Calibrated LinearSVC & LogisticRegression
│   ├── transformer_model.py    # RoBERTa wrapper with checkpoint safety validation
│   ├── explainability.py       # Sentence-level risk scoring and feature importance
│   ├── evaluate_benchmarks.py  # Multi-benchmark engine (JSON & Markdown generator, bootstrap CIs)
│   ├── error_analysis.py       # Kaggle internal overlap audit, ROC sweeps, calibration & length curves
│   └── app.py                  # Flask REST API backend (/predict, /health, /feedback)
├── templates/
│   └── index.html              # Responsive web UI with heatmap, metrics cards, and badge lockouts
├── static/
│   ├── css/style.css           # Modern responsive design system
│   └── js/main.js              # Heatmap rendering and client-side validation logic
├── notebooks/
│   └── RoBERTa_FineTuning.ipynb # Google Colab T4 GPU notebook with fp16 timing & Drive auto-save
├── docs/
│   ├── master_benchmark_results.json # Single source of truth (full empirical results & audit data)
│   ├── master_benchmark_table.md     # Auto-generated comprehensive benchmark tables
│   ├── dataset_card.md         # Standardized ML Dataset Card (provenance, licenses, overlap statistics)
│   ├── model_card.md           # Standardized ML Model Card (architecture, ECE, calibration, bias)
│   ├── final_report.md         # 12-chapter comprehensive project thesis
│   ├── viva_qa_guide.md        # Oral examination defense questions and answers
│   ├── roc_threshold_curve.png # ROC curves and validation threshold trade-off sweeps
│   ├── calibration_plot.png    # Reliability diagram (ECE = 0.0006, Brier = 0.0009)
│   └── accuracy_vs_length.png  # Accuracy stratified across passage length bins
├── models/
│   ├── baseline_svm.joblib     # Frozen trained Calibrated Linear SVM (SHA-256 verified)
│   └── baseline_lr.joblib      # Frozen trained Logistic Regression model (SHA-256 verified)
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_stylometrics.py    # Stylometric extractor tests
│   ├── test_api.py             # REST API endpoint tests
│   ├── test_contamination.py   # MinHash benchmark contamination test
│   └── test_metrics_schema.py  # Validates JSON schema & arithmetic constraints
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Excludes large CSV datasets and raw checkpoints
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/m0gghar/ai-text-detector.git
cd ai-text-detector
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (14 Tests)
```bash
pytest tests/ -v
```

### 3. Run Multi-Benchmark Evaluation
```bash
python -m src.evaluate_benchmarks
```

### 4. Run Diagnostics & Overlap Audit
```bash
python -m src.error_analysis
```

### 5. Launch Web Application & API
```bash
python -m src.app
```
Navigate to: **`http://127.0.0.1:5000`**

---

## 📜 Ethical & Technical Disclaimer
Predictions generated by this system are statistical probabilistic estimates. They must **never** be used as the sole basis for academic disciplinary actions, legal decisions, or automated employment filtering. All models trained on student writing exhibit systematic false-positive bias against formal academic prose.
