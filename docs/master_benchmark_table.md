# Master Benchmark Results Table

**Generated**: 2026-11-25T08:32:17.548116+00:00  
**Git Commit**: `3cd8408c919d647c09b55dc1c3c03dec6d628997`  
**Random Seed**: 42 | **Bootstrap Rounds**: 1000  
**Decision Threshold**: 0.5 | **Model**: `Calibrated Linear SVM (Sublinear TF-IDF + 5 Stylometrics)`

### 1. Main Benchmark Performance (Passages $\ge 50$ Words)

| Dataset | Rows (Before / Removed / Eval) | Accuracy (95% CI) | Balanced Acc | Macro-F1 | Binary F1 | Recall (AI) | FPR (Human False Alarm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle In-Domain (5k Slice)** | 5000 / 1 / **4999** | 99.86% [99.7-100.0] | 99.84% | 99.85% | **99.81%** | 99.74% | 0.06% |
| **Kaggle In-Domain (Full 73k)** | 73083 / 10 / **73073** | 99.90% [99.9-99.9] | 99.88% | 99.90% | **99.87%** | 99.81% | 0.04% |
| **Pre-2022 Formal Human / ESL (arXiv)** | 14 / 2 / **12** | 0.00% [0.0-0.0] | not applicable | not applicable | **not applicable** | not applicable | 100.00% |
| **HC3 Benchmark (Human vs ChatGPT QA)** | 12 / 2 / **10** | 50.00% [20.0-80.0] | 50.00% | 33.33% | **66.67%** | 100.00% | 100.00% |
| **RAID Benchmark (Multi-Generator)** | 15 / 4 / **11** | 72.73% [45.5-100.0] | 50.00% | 42.11% | **84.21%** | 100.00% | 100.00% |
| **M4 Benchmark (Wang et al., 2023)** | 7 / 3 / **4** | 75.00% [0.0-100.0] | 50.00% | 42.86% | **85.71%** | 100.00% | 100.00% |
| **Out-of-Domain Creative Fiction** | 6 / 0 / **6** | 50.00% [16.7-83.3] | 50.00% | 33.33% | **66.67%** | 100.00% | 100.00% |
| **Paraphrased AI Evasion Test** | 9 / 8 / **1** | 100.00% [0.0-100.0] | not applicable | not applicable | **not applicable** | 100.00% | not applicable |

---

### 2. RAID Benchmark Fine-Grained Stratification (Passages $\ge 50$ Words)

#### Performance by Generator
| Generator | Count | Accuracy | AI Detection Rate (Recall) |
| :--- | :---: | :---: | :---: |
| `chatgpt` | 1 | 100.00% | 100.00% |
| `gpt4` | 2 | 100.00% | 100.00% |
| `human` | 3 | 0.00% | not applicable |
| `llama-chat` | 2 | 100.00% | 100.00% |
| `mistral-chat` | 1 | 100.00% | 100.00% |
| `mpt-chat` | 1 | 100.00% | 100.00% |
| `text-davinci-003` | 1 | 100.00% | 100.00% |

#### Performance by Domain
| Domain | Count | Accuracy |
| :--- | :---: | :---: |
| `books` | 3 | 66.67% |
| `news` | 3 | 66.67% |
| `recipes` | 1 | 100.00% |
| `reddit` | 2 | 50.00% |
| `wikipedia` | 2 | 100.00% |

#### Robustness Against Adversarial Attacks
| Attack Type | Count | Accuracy | AI Detection Rate |
| :--- | :---: | :---: | :---: |
| `clean` | 9 | 66.67% | 100.00% |
| `insert_punctuation` | 1 | 100.00% | 100.00% |
| `upper_case` | 1 | 100.00% | 100.00% |

---

### 3. Segregated Short-Text Analysis (Texts $< 50$ Words)

| Dataset Source | Short Texts Count | Accuracy | Human Count | AI Count |
| :--- | :---: | :---: | :---: | :---: |
| **Kaggle In-Domain (5k Slice)** | 1 | 100.00% | 0 | 1 |
| **Kaggle In-Domain (Full 73k)** | 10 | 100.00% | 1 | 9 |
| **Pre-2022 Formal Human / ESL (arXiv)** | 2 | 0.00% | 2 | 0 |
| **HC3 Benchmark (Human vs ChatGPT QA)** | 2 | 100.00% | 1 | 1 |
| **RAID Benchmark (Multi-Generator)** | 4 | 75.00% | 1 | 3 |
| **M4 Benchmark (Wang et al., 2023)** | 3 | 66.67% | 1 | 2 |
| **Out-of-Domain Creative Fiction** | 0 | not applicable | 0 | 0 |
| **Paraphrased AI Evasion Test** | 8 | 100.00% | 0 | 8 |

> *Note: Primary benchmarks strictly enforce the $W \ge 50$ words threshold. Short texts are analyzed separately to avoid sample-size distortion.*
