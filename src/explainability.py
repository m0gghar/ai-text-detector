"""Explainability and Sentence-Level Highlighting Module.
Implements sentence-level suspicious passage analysis (PRD FR8) and
feature importance inspection for the baseline model (PRD Section 7 & 14).
"""

from typing import List, Dict, Any
from src.preprocessing import tokenize_sentences, tokenize_words, clean_text


def analyze_sentences(text: str, detector: Any) -> List[Dict[str, Any]]:
    """Analyze text sentence by sentence to compute per-sentence AI probabilities
    and risk levels for visual highlighting.
    """
    sentences = tokenize_sentences(text)
    results = []

    for idx, sentence in enumerate(sentences):
        words = tokenize_words(sentence)
        if len(words) < 3:
            # Very short fragments get neutral rating
            results.append({
                "index": idx,
                "sentence": sentence,
                "ai_probability": 0.50,
                "human_probability": 0.50,
                "risk_level": "neutral",
                "label": "Neutral",
            })
            continue

        try:
            pred = detector.predict(sentence)
            p_ai = pred["probabilities"]["AI"]
            p_human = pred["probabilities"]["Human"]
        except Exception:
            p_ai = 0.50
            p_human = 0.50

        # Categorize risk level for heatmap coloring
        if p_ai >= 0.70:
            risk = "high"
            label = "Likely AI"
        elif p_ai >= 0.45:
            risk = "medium"
            label = "Uncertain / Mixed"
        else:
            risk = "low"
            label = "Likely Human"

        results.append({
            "index": idx,
            "sentence": sentence,
            "ai_probability": round(p_ai, 3),
            "human_probability": round(p_human, 3),
            "risk_level": risk,
            "label": label,
        })

    return results


def explain_baseline_features(text: str, svm_pipeline: Any, top_k: int = 5) -> Dict[str, Any]:
    """Inspect top n-grams from input text that contributed most toward
    the AI or Human decision in the Linear SVM / Logistic Regression pipeline.
    """
    try:
        # Get tfidf step and classifier
        features_union = svm_pipeline.named_steps["features"]
        tfidf_step = features_union.transformer_list[0][1].named_steps["tfidf"]
        classifier = svm_pipeline.named_steps["classifier"]

        # Extract weights
        if hasattr(classifier, "calibrated_classifiers_"):
            # CalibratedClassifierCV wraps estimators
            base_estimator = classifier.calibrated_classifiers_[0].estimator
            coefs = base_estimator.coef_[0]
        elif hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
        else:
            return {"ai_indicative": [], "human_indicative": []}

        vocab = tfidf_step.vocabulary_
        cleaned = clean_text(text).lower()
        feature_names = tfidf_step.get_feature_names_out()

        # Find words present in this document
        present_features = []
        for word, idx in vocab.items():
            if idx < len(coefs) and f" {word} " in f" {cleaned} ":
                weight = float(coefs[idx])
                present_features.append((word, weight))

        # Sort features
        present_features.sort(key=lambda x: x[1], reverse=True)
        ai_words = [{"term": term, "weight": round(w, 4)} for term, w in present_features if w > 0][:top_k]
        human_words = [{"term": term, "weight": round(abs(w), 4)} for term, w in reversed(present_features) if w < 0][:top_k]

        return {
            "ai_indicative": ai_words,
            "human_indicative": human_words,
        }
    except Exception as e:
        return {"error": str(e), "ai_indicative": [], "human_indicative": []}
