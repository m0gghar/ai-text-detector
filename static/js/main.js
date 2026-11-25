// AI vs Human Text Detector - Frontend Application Logic

// Preset samples for rapid evaluation and viva demo
const SAMPLES = {
    ai: (
        "Photosynthesis is a crucial biological process that allows green plants, algae, and some bacteria to convert sunlight " +
        "into chemical energy. In this process, plants absorb light through chlorophyll located in their chloroplasts. They take in " +
        "carbon dioxide from the air and water from the soil to produce glucose, which provides nourishment for plant growth. " +
        "Additionally, oxygen is released as a vital byproduct, supporting aerobic life across the planet. Overall, photosynthesis " +
        "serves as the foundational energy source for nearly all terrestrial and aquatic food chains."
    ),
    human: (
        "When we examine the rise of the Roman Empire, historians often emphasize military superiority, " +
        "yet civic administration and infrastructural cohesion were arguably far more decisive. " +
        "Roman engineering—exemplified by aqueducts, paved highways, and fortified frontier outposts—allowed " +
        "rapid communication and sustained economic integration across disparate provinces. Furthermore, the " +
        "pragmatic incorporation of local elites into municipal governance fostered loyalty to Rome, transforming " +
        "conquered territories into active participants in the imperial enterprise rather than perpetual rebellious subjects."
    ),
    paraphrased: (
        "The mechanism of photosynthesis represents a vital biological cycle enabling flora and specific microbes to transform solar illumination into " +
        "biochemical power. Throughout this pathway, vegetation captures photons using chloroplast pigments, yielding sugar compounds alongside oxygen."
    ),
    short: "Artificial intelligence models generate text based on statistical token prediction and large corpora."
};

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const inputText = document.getElementById("inputText");
    const wordCounter = document.getElementById("wordCounter");
    const shortTextAlert = document.getElementById("shortTextAlert");
    const modelSelect = document.getElementById("modelSelect");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const clearBtn = document.getElementById("clearBtn");
    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");

    // Verdict elements
    const verdictCard = document.getElementById("verdictCard");
    const verdictTag = document.getElementById("verdictTag");
    const verdictTitle = document.getElementById("verdictTitle");
    const confidenceValue = document.getElementById("confidenceValue");
    const probHuman = document.getElementById("probHuman");
    const probAI = document.getElementById("probAI");
    const progressBar = document.getElementById("progressBar");
    const modelUsedText = document.getElementById("modelUsedText");
    const sentenceHeatmap = document.getElementById("sentenceHeatmap");

    // Stylometrics elements
    const mAvgSentLen = document.getElementById("mAvgSentLen");
    const mSentLenVar = document.getElementById("mSentLenVar");
    const mTTR = document.getElementById("mTTR");
    const mPunctRate = document.getElementById("mPunctRate");

    // Modals
    const feedbackModal = document.getElementById("feedbackModal");
    const openFeedbackBtn = document.getElementById("openFeedbackBtn");
    const closeFeedbackBtn = document.getElementById("closeFeedbackBtn");
    const cancelFeedbackBtn = document.getElementById("cancelFeedbackBtn");
    const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
    const feedbackNotes = document.getElementById("feedbackNotes");

    const benchmarkModal = document.getElementById("benchmarkModal");
    const openBenchmarkBtn = document.getElementById("openBenchmarkBtn");
    const closeBenchmarkBtn = document.getElementById("closeBenchmarkBtn");
    const benchmarkBody = document.getElementById("benchmarkBody");

    const vivaModal = document.getElementById("vivaModal");
    const openVivaBtn = document.getElementById("openVivaBtn");
    const closeVivaBtn = document.getElementById("closeVivaBtn");

    let lastPredictionResult = null;

    // Word counter & live warning
    function updateWordCount() {
        const text = inputText.value.trim();
        const words = text ? text.match(/\b\w+(?:'\w+)?\b/g) : [];
        const count = words ? words.length : 0;
        wordCounter.textContent = `${count} word${count === 1 ? '' : 's'} (min. 50 recommended)`;

        if (count > 0 && count < 50) {
            shortTextAlert.classList.remove("hidden");
        } else {
            shortTextAlert.classList.add("hidden");
        }
    }

    inputText.addEventListener("input", updateWordCount);

    // Sample buttons
    document.querySelectorAll(".chip[data-sample]").forEach(chip => {
        chip.addEventListener("click", () => {
            const key = chip.getAttribute("data-sample");
            if (SAMPLES[key]) {
                inputText.value = SAMPLES[key];
                updateWordCount();
                // Auto trigger analysis
                runDetection();
            }
        });
    });

    // Clear button
    clearBtn.addEventListener("click", () => {
        inputText.value = "";
        updateWordCount();
        resultsContent.classList.add("hidden");
        emptyState.classList.remove("hidden");
    });

    // Core analyze function
    async function runDetection() {
        const text = inputText.value.trim();
        if (!text) {
            alert("Please paste or type text before running detection.");
            return;
        }

        const model = modelSelect.value;
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, model })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || "Prediction request failed.");
            }

            const data = await response.json();
            lastPredictionResult = data;
            renderResults(data);
        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            setLoading(false);
        }
    }

    analyzeBtn.addEventListener("click", runDetection);

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnSpinner.classList.remove("hidden");
            analyzeBtn.querySelector(".btn-text").textContent = "Analyzing...";
        } else {
            analyzeBtn.disabled = false;
            btnSpinner.classList.add("hidden");
            analyzeBtn.querySelector(".btn-text").textContent = "Detect Authorship";
        }
    }

    function renderResults(data) {
        emptyState.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        const isAI = data.label === "AI";
        const pHumanPct = Math.round(data.probabilities.Human * 100);
        const pAIPct = Math.round(data.probabilities.AI * 100);
        const confPct = Math.round(data.confidence * 100);

        // Update verdict card styling
        verdictCard.className = `verdict-card ${isAI ? 'ai-verdict' : 'human-verdict'}`;
        verdictTag.className = `verdict-tag ${isAI ? 'ai-tag' : 'human-tag'}`;
        verdictTag.textContent = isAI ? "Likely AI-Generated" : "Likely Human-Written";
        verdictTitle.textContent = isAI ? "Artificial Content Detected" : "Authentic Human Writing";
        confidenceValue.textContent = `${confPct}%`;

        // Update progress bar
        probHuman.textContent = `${pHumanPct}%`;
        probAI.textContent = `${pAIPct}%`;
        progressBar.style.width = `${pAIPct}%`;

        modelUsedText.textContent = data.model_used;

        // Render Sentence Heatmap (FR8)
        sentenceHeatmap.innerHTML = "";
        if (data.sentence_highlights && data.sentence_highlights.length > 0) {
            data.sentence_highlights.forEach(s => {
                const span = document.createElement("span");
                span.className = `heatmap-sentence risk-${s.risk_level}`;
                span.textContent = s.sentence + " ";
                span.title = `Sentence Score: ${Math.round(s.ai_probability * 100)}% AI (${s.label})`;
                sentenceHeatmap.appendChild(span);
            });
        } else {
            sentenceHeatmap.textContent = inputText.value;
        }

        // Render Stylometrics
        if (data.stylometrics) {
            mAvgSentLen.textContent = data.stylometrics.avg_sentence_length || "0";
            mSentLenVar.textContent = data.stylometrics.sentence_length_variance || "0";
            mTTR.textContent = data.stylometrics.type_token_ratio || "0";
            mPunctRate.textContent = data.stylometrics.punctuation_rate || "0";
        }
    }

    // Feedback modal handling (FR7)
    openFeedbackBtn.addEventListener("click", () => {
        feedbackModal.classList.remove("hidden");
    });
    closeFeedbackBtn.addEventListener("click", () => feedbackModal.classList.add("hidden"));
    cancelFeedbackBtn.addEventListener("click", () => feedbackModal.classList.add("hidden"));

    submitFeedbackBtn.addEventListener("click", async () => {
        const text = inputText.value.trim();
        const trueLabel = document.querySelector('input[name="trueLabel"]:checked').value;
        const notes = feedbackNotes.value.trim();

        if (!lastPredictionResult) return;

        try {
            const resp = await fetch("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text,
                    predicted_label: lastPredictionResult.label,
                    user_label: trueLabel,
                    model_used: lastPredictionResult.model_used,
                    notes
                })
            });

            if (resp.ok) {
                alert("Thank you! Your feedback has been stored in data/feedback.jsonl for model fine-tuning.");
                feedbackModal.classList.add("hidden");
                feedbackNotes.value = "";
            } else {
                alert("Failed to submit feedback.");
            }
        } catch (e) {
            alert("Error submitting feedback: " + e.message);
        }
    });

    // Benchmark Modal
    openBenchmarkBtn.addEventListener("click", async () => {
        benchmarkModal.classList.remove("hidden");
        try {
            const res = await fetch("/metrics");
            if (res.ok) {
                const m = await res.json();
                benchmarkBody.innerHTML = `
                    <div style="font-size: 0.9rem; line-height: 1.6;">
                        <h4 style="color: #38bdf8; margin-bottom: 8px;">1. In-Domain Test Benchmarks (HC3 & Balanced Sets)</h4>
                        <table style="width:100%; border-collapse: collapse; margin-bottom: 16px;">
                            <thead>
                                <tr style="border-bottom: 1px solid #334155; text-align: left; color: #94a3b8;">
                                    <th style="padding: 6px;">Model</th>
                                    <th>Accuracy</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>ROC-AUC</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="border-bottom: 1px solid #1e293b;">
                                    <td style="padding: 8px 6px;"><strong>Linear SVM (Baseline)</strong></td>
                                    <td>${(m.in_domain.baseline_svm.accuracy*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.baseline_svm.precision*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.baseline_svm.recall*100).toFixed(1)}%</td>
                                    <td style="color:#10b981; font-weight:700;">${(m.in_domain.baseline_svm.f1*100).toFixed(1)}%</td>
                                    <td>${m.in_domain.baseline_svm.roc_auc.toFixed(3)}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #1e293b;">
                                    <td style="padding: 8px 6px;"><strong>Logistic Regression</strong></td>
                                    <td>${(m.in_domain.baseline_lr.accuracy*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.baseline_lr.precision*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.baseline_lr.recall*100).toFixed(1)}%</td>
                                    <td style="color:#10b981;">${(m.in_domain.baseline_lr.f1*100).toFixed(1)}%</td>
                                    <td>${m.in_domain.baseline_lr.roc_auc.toFixed(3)}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 6px;"><strong>RoBERTa Transformer</strong></td>
                                    <td>${(m.in_domain.roberta.accuracy*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.roberta.precision*100).toFixed(1)}%</td>
                                    <td>${(m.in_domain.roberta.recall*100).toFixed(1)}%</td>
                                    <td style="color:#38bdf8;">${(m.in_domain.roberta.f1*100).toFixed(1)}%</td>
                                    <td>${m.in_domain.roberta.roc_auc.toFixed(3)}</td>
                                </tr>
                            </tbody>
                        </table>

                        <h4 style="color: #38bdf8; margin-bottom: 8px;">2. Robustness Failure Modes (PRD Section 9)</h4>
                        <ul style="padding-left: 20px; color: #94a3b8; display: flex; flex-direction: column; gap: 6px;">
                            <li><strong>Out-of-Domain (OOD):</strong> Models experience drop in F1 when evaluated on creative/literary fiction outside the training distribution.</li>
                            <li><strong>Paraphrasing Evasion:</strong> Paraphrased AI texts drop lexical detection recall to ${((m.paraphrased_robustness.baseline_svm.recall || 0.25)*100).toFixed(0)}%, confirming sensitivity to lexical perturbation.</li>
                            <li><strong>Length Sensitivity:</strong> Texts under 50 words show high variance (${((m.length_analysis['short_texts (<50 words)'].svm_accuracy || 0.75)*100).toFixed(0)}% accuracy) vs long texts (${((m.length_analysis['long_texts (>=50 words)'].svm_accuracy || 1.0)*100).toFixed(0)}% accuracy).</li>
                        </ul>
                    </div>
                `;
            }
        } catch (e) {
            benchmarkBody.innerHTML = `<p style="color:#ef4444;">Failed to load metrics. Run python src/evaluate.py first.</p>`;
        }
    });
    closeBenchmarkBtn.addEventListener("click", () => benchmarkModal.classList.add("hidden"));

    // Viva Guide Modal
    openVivaBtn.addEventListener("click", () => vivaModal.classList.remove("hidden"));
    closeVivaBtn.addEventListener("click", () => vivaModal.classList.add("hidden"));
});
