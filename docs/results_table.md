# Model Performance & Comparison Table

### 1. In-Domain Test Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (TF-IDF + Stylometry)** | 94.1% | 88.9% | 100.0% | **94.1%** | 0.972 |
| **Logistic Regression** | 88.2% | 87.5% | 87.5% | **87.5%** | 0.958 |
| **Fine-Tuned RoBERTa** | 52.9% | 0.0% | 0.0% | **0.0%** | 0.306 |

### 2. Generalization & Robustness (OOD vs Paraphrased)

| Test Set | Linear SVM F1 | RoBERTa F1 | SVM AI Detection Rate | RoBERTa AI Detection Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Out-of-Domain (OOD Fiction/Memoir)** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Paraphrased AI Evasion Test** | 40.0% | 0.0% | 25.0% | 0.0% |

### 3. Length Sensitivity Analysis (< 50 words vs >= 50 words)

| Length Bracket | Sample Count | Linear SVM Accuracy | RoBERTa Accuracy |
| :--- | :---: | :---: | :---: |
| **short_texts (<50 words)** | 4 | 75.0% | 75.0% |
| **long_texts (>=50 words)** | 13 | 100.0% | 46.2% |
