# Viva & Technical Interview Preparation Guide
## AI vs Human Text Detector (Final-Year B.Tech Capstone Project)

This guide provides precise, mathematically grounded answers to the core viva and technical interview questions specified in **PRD Section 14**.

---

### Q1: Why start with a classical baseline instead of jumping directly to transformers?
**Answer:**
1. **Benchmark Establishment**: A baseline establishes an empirical performance lower-bound. Without a baseline, one cannot determine whether the multi-million parameter complexity and latency of a deep transformer are justified.
2. **Computational Cost & Resource Profiling**: The classical baseline (TF-IDF + Stylometry + Linear SVM) trains in under 2 seconds and evaluates inference in under 5 milliseconds on a standard CPU. RoBERTa requires GPU resources for training and carries higher CPU inference latency (~1–3 seconds).
3. **Interpretability & Error Diagnostics**: Classical linear models allow direct feature weight inspection ($w_i \cdot x_i$), revealing whether predictions are driven by authentic stylistic signals or dataset artifacts (e.g., specific punctuation or topic keywords).

---

### Q2: Why choose RoBERTa over BERT, GPT, or other architectures?
**Answer:**
1. **Robust Pre-training Recipe**: RoBERTa (*Robustly Optimized BERT Approach*, Liu et al., 2019) improves upon original BERT by:
   - Removing the Next Sentence Prediction (NSP) objective, which was shown to harm single-sequence classification.
   - Employing dynamic token masking across epochs rather than static masking.
   - Pre-training on 160 GB of diverse uncompressed text (BookCorpus, CC-News, OpenWebText, Stories) with much larger mini-batch sizes (8K).
2. **Bidirectional Contextual Encoder**: Unlike autoregressive decoder models (e.g., GPT-2/3/4) that only condition on preceding tokens $P(w_t \mid w_{<t})$, RoBERTa uses bidirectional self-attention $P(w_t \mid w_{\setminus t})$, allowing representation of global discourse structure, syntactic symmetry, and semantic continuity across the entire document.

---

### Q3: What does TF-IDF capture, and what does it miss?
**Answer:**
- **What it captures**:
  - Unigram and bigram term frequency-inverse document frequency statistics.
  - Surface lexical indicators: AI texts frequently employ transition words ("moreover", "in conclusion", "crucial", "testament to", "delve"), while human texts display distinct colloquial or specialized terminology.
- **What it misses**:
  - **Word Order & Syntax**: It operates as an unordered "bag-of-words" or bag-of-n-grams. It cannot understand syntax trees, clausal subordination, or long-range coreference.
  - **Contextual Polysemy**: A word with multiple senses (e.g., "bank") has a static identifier, whereas transformer embeddings vary dynamically based on surrounding tokens.
  - **Paraphrase Resilience**: If an AI text is reworded using synonyms, TF-IDF representations shift completely, causing evasion.

---

### Q4: Why might detection accuracy look unrealistically high (e.g., 99%+), and how do you investigate it?
**Answer:**
A near-perfect accuracy score is almost always a warning sign of **data leakage** or **spurious correlations**:
1. **Topic / Domain Leakage**: If human texts are drawn from Wikipedia history articles and AI texts from ChatGPT physics answers, the classifier learns to classify *physics vs history*, not *AI vs human*.
2. **Formatting Artifacts**: Inconsistent Markdown styling, HTML tags, carriage returns (`\r\n` vs `\n`), or unique quotation marks (`"` vs `“`) can become trivial classification shortcuts.
3. **Length Disparities**: If AI responses are consistently 300 words while human posts are 60 words, length alone becomes a proxy.
4. **Mitigation**: We mitigate this through domain-paired corpora (HC3 prompts where both human and AI answered the exact same prompt), uniform text cleaning, and evaluating on a strictly held-out Out-Of-Domain (OOD) test set.

---

### Q5: Why evaluate with F1-Score and ROC-AUC rather than raw Accuracy?
**Answer:**
1. **Class Prevalence Skew**: If a test dataset consists of 90% human texts and 10% AI texts, a trivial classifier that always predicts "Human" achieves 90% accuracy while having an AI recall of 0.0%.
2. **Asymmetric Cost of Errors**: In academic and publishing contexts, a **False Positive** (falsely accusing an innocent student of AI generation) has catastrophic consequences, whereas a **False Negative** (missing an AI text) is merely an omission. F1 balances precision and recall:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Threshold Invariance**: ROC-AUC measures discriminative capacity across all possible classification decision thresholds, independent of the default 0.50 cutoff.

---

### Q6: What are the primary failure modes and limitations of current AI text detectors?
**Answer:**
1. **Short Text Inputs (< 50 words)**: Statistical stylometry collapses because sample size is insufficient to estimate lexical diversity (TTR) or sentence length variance.
2. **Paraphrasing & Rewriting**: Passing AI text through paraphrasing tools (e.g., QuillBot, synonym swappers) removes characteristic n-grams while preserving semantic meaning, causing detection rates to drop significantly.
3. **Non-Native English Speakers (ESL)**: Studies show that ESL writers often write in a constrained, formal, and grammatically uniform vocabulary with low perplexity, frequently triggering false-positive accusations.
4. **Emergence of Newer LLM Generations**: Detectors trained on GPT-3.5/ChatGPT-3.5 exhibit degraded performance when confronted with newer models (Claude 3.5, GPT-4o, Gemini 1.5) that produce more varied human-like prose.

---

### Q7: How does sentence-level highlighting work in this project (FR8)?
**Answer:**
The system decomposes the submitted text into constituent sentences using regex-based boundary tokenization that respects abbreviations. Each sentence is independently evaluated through the inference engine to compute a localized probability score $P(\text{AI})$. Sentences with $P(\text{AI}) \ge 0.70$ are tagged as High Risk (red), $0.45 \le P < 0.70$ as Mixed/Uncertain (yellow), and $P < 0.45$ as Low Risk (green). This provides transparent visual explainability rather than an opaque single scalar score.

---

### Q8: What would be the next architectural steps (Future Scope)?
**Answer:**
1. **Ensemble Modeling**: Stacking RoBERTa with DeBERTa-v3 (which incorporates disentangled attention and enhanced mask decoder pre-training).
2. **Perplexity and Burstiness Features**: Integrating statistical language model perplexity ($PPL = \exp(-\frac{1}{N}\sum \log P(w_i))$) and cross-sentence burstiness variance.
3. **Adversarial Training**: Augmented fine-tuning on back-translated, paraphrased, and prompt-injected texts to increase robustness against evasion techniques.
