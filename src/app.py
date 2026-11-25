"""Flask REST API Backend for AI vs Human Text Detector.
Exposes /predict (POST), /health (GET), /feedback (POST), and /metrics (GET)
as specified in PRD Section 6 (FR1 - FR8).
"""

import os
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.preprocessing import clean_text, get_word_count, check_length_warning
from src.baseline_model import BaselineDetector
from src.transformer_model import RoBERTaDetector
from src.explainability import analyze_sentences, explain_baseline_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FEEDBACK_FILE = os.path.join(BASE_DIR, "data", "feedback.jsonl")
RESULTS_JSON_FILE = os.path.join(BASE_DIR, "docs", "evaluation_results.json")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
)
CORS(app)

# Initialize detector singletons
print("Initializing detectors for Flask app...")
baseline_svm_detector = BaselineDetector(model_type="svm")
transformer_detector = RoBERTaDetector()


@app.route("/", methods=["GET"])
def index():
    """Serve the web application dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (PRD FR5)."""
    return jsonify({
        "status": "healthy",
        "service": "AI vs Human Text Detector",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "baseline_svm": baseline_svm_detector.pipeline is not None,
            "transformer_trained": RoBERTaDetector.has_trained_checkpoint(),
        },
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether input text is AI-generated or Human-written (PRD FR1, FR2, FR3, FR4, FR8)."""
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON request payload."}), 400

    raw_text = str(data.get("text", "")).strip()
    if not raw_text:
        return jsonify({"error": "Provided text is empty."}), 400

    # Select model (PRD FR4) with RoBERTa lockout check
    model_choice = data.get("model", "baseline").lower()
    model_override_note = None
    if model_choice in ["baseline", "svm", "classical"]:
        detector = baseline_svm_detector
    else:
        if not RoBERTaDetector.has_trained_checkpoint():
            detector = baseline_svm_detector
            model_override_note = "RoBERTa transformer is not trained yet (models/roberta_model/ absent). Evaluated using Calibrated Linear SVM."
        else:
            detector = transformer_detector

    # Validate length warning (PRD FR3)
    word_count = get_word_count(raw_text)
    is_warning, warning_msg = check_length_warning(raw_text, min_words=50)

    # Core prediction
    try:
        prediction_result = detector.predict(raw_text)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Sentence-level highlighting analysis (PRD FR8)
    sentence_highlights = analyze_sentences(raw_text, detector)

    # Explainability features if baseline model
    feature_importance = None
    if detector == baseline_svm_detector and baseline_svm_detector.pipeline is not None:
        feature_importance = explain_baseline_features(raw_text, baseline_svm_detector.pipeline, top_k=5)

    response = {
        "label": prediction_result["label"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "model_used": prediction_result["model_name"],
        "word_count": word_count,
        "warning": (f"{model_override_note + ' ' if model_override_note else ''}{warning_msg if is_warning else ''}".strip()) if (model_override_note or is_warning) else None,
        "stylometrics": prediction_result["stylometrics"],
        "sentence_highlights": sentence_highlights,
        "feature_importance": feature_importance,
        "disclaimer": (
            "Disclaimer: Predictions are probabilistic statistical estimates based on lexical and stylistic patterns. "
            "They should never be used as sole or definitive proof of authorship."
        ),
    }
    return jsonify(response), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect user feedback/label corrections (PRD FR7)."""
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data or "user_label" not in data:
        return jsonify({"error": "Missing 'text' or 'user_label' in feedback payload."}), 400

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": data.get("text"),
        "predicted_label": data.get("predicted_label"),
        "user_label": data.get("user_label"),
        "model_used": data.get("model_used"),
        "notes": data.get("notes", ""),
    }

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonify({
        "success": True,
        "message": "Feedback submitted successfully. Thank you for contributing to dataset refinement!",
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Retrieve benchmark evaluation metrics."""
    if os.path.exists(RESULTS_JSON_FILE):
        with open(RESULTS_JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200
    return jsonify({"message": "Metrics not yet generated. Run evaluate.py first."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI vs Human Text Detector API on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
