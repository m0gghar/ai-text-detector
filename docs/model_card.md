# Model Card: AI vs Human Text Detector (Baseline Classifier)

## Model Details
- **Architecture**: Calibrated Linear Support Vector Machine (`LinearSVC` calibrated via Platt scaling / sigmoid calibration) coupled with `FeatureUnion`:
  - Sublinear TF-IDF (word unigrams and bigrams, max 5,000 features)
  - 5-dimensional Stylometric Vector (Avg sentence length, sentence length variance, Type-Token Ratio, punctuation density, avg word length) + `StandardScaler`
- **Training Corpus**: Stratified split of 341,049 essays from the Kaggle AI vs Human benchmark (`AI_Human.csv`).
- **Decision Threshold**: $\tau = 0.50$, chosen strictly on validation data (`data/processed/val.csv`). Threshold sweeps across $[0.10, 0.95]$ represent observational trade-off studies.
- **RoBERTa Track**: Gated by Day 8 decision gate. If missing from `models/roberta_model/`, RoBERTa is locked out and documented as future work.

---

## Intended Use & Non-Goals
- **Intended Use**: Educational assistance, stylistic text analysis, and comparative natural language research.
- **Prohibited Use**: Automated academic disciplinary penalties, legal evidence of authorship, or automated employment filtering.
- **Fairness Warning**: Models trained on student corpora exhibit false-positive bias against formal human writing and ESL authors. All outputs are probabilistic estimates accompanied by an explicit disclaimer.

---

## Performance & Limitations Summary
- **In-Domain Strength**: Exceptional discrimination on prompt-aligned student essays: **99.90% Accuracy** on full 73,073 Kaggle test set ([99.9%–99.9%] 95% CI).
- **Out-of-Domain Attenuation ("The Kaggle Trap")**:
  - **Pre-2022 Formal Human / ESL (arXiv)**: **0.00% Accuracy / 100.00% False Positive Rate**. The classifier falsely flags 100% of authentic pre-2022 academic abstracts as AI, because it learned that formal vocabulary and sophisticated grammar equals AI.
  - **HC3 Benchmark (Guo et al., 2023)**: **50.00% Accuracy** (random chance) on technical QA.
  - **RAID Benchmark (Dugan et al., 2024)**: **72.73% Accuracy** overall; drops significantly across diverse domains.
- **Diagnostics & Calibration**:
  - **Reliability & Calibration**: Expected Calibration Error (ECE) of **0.0006** and Brier score of **0.0009** on validation data (`docs/calibration_plot.png`).
  - **ROC & Decision Threshold**: ROC-AUC of **0.9999** with $\tau = 0.50$ fixed operating point (`docs/roc_threshold_curve.png`).
  - **Length Sensitivity**: Evaluated across 5 word-count brackets (`docs/accuracy_vs_length.png`). Performance exhibits high variance on passages $< 50$ words, justifying the mandatory 50-word UI warning and filter.
