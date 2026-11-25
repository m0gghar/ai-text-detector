"""Classical Machine Learning Baseline Model.
Implements TF-IDF + Stylometric Features + Calibrated Linear SVM and Logistic Regression
as specified in PRD Section 8.3.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.preprocessing import clean_text
from src.stylometrics import StylometricExtractor, extract_stylometric_dict

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_feature_pipeline() -> FeatureUnion:
    """Build the FeatureUnion combining TF-IDF word n-grams and scaled stylometric features."""
    tfidf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            preprocessor=clean_text,
        ))
    ])

    stylometric_pipe = Pipeline([
        ("extractor", StylometricExtractor()),
        ("scaler", StandardScaler()),
    ])

    union = FeatureUnion([
        ("tfidf_features", tfidf_pipe),
        ("stylometric_features", stylometric_pipe),
    ])
    return union


def train_baseline_models(train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Train both Calibrated Linear SVM and Logistic Regression models.
    Returns metrics and trained pipeline instances.
    """
    X_train = train_df["text"].values
    y_train = train_df["label"].values

    # 1. Train Calibrated Linear SVM
    svm_pipeline = Pipeline([
        ("features", build_feature_pipeline()),
        ("classifier", CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000),
            cv=3,
        )),
    ])
    print("Fitting Calibrated Linear SVM pipeline...")
    svm_pipeline.fit(X_train, y_train)

    # 2. Train Logistic Regression
    lr_pipeline = Pipeline([
        ("features", build_feature_pipeline()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    print("Fitting Logistic Regression pipeline...")
    lr_pipeline.fit(X_train, y_train)

    results = {
        "svm_pipeline": svm_pipeline,
        "lr_pipeline": lr_pipeline,
    }

    if val_df is not None:
        X_val = val_df["text"].values
        y_val = val_df["label"].values

        for name, pipe in [("SVM", svm_pipeline), ("LogisticRegression", lr_pipeline)]:
            preds = pipe.predict(X_val)
            probs = pipe.predict_proba(X_val)[:, 1]
            acc = accuracy_score(y_val, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            try:
                auc = roc_auc_score(y_val, probs)
            except Exception:
                auc = 0.5
            print(f"Validation [{name}] -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
            results[f"{name}_val_metrics"] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    svm_path = os.path.join(MODELS_DIR, "baseline_svm.joblib")
    lr_path = os.path.join(MODELS_DIR, "baseline_lr.joblib")
    joblib.dump(svm_pipeline, svm_path)
    joblib.dump(lr_pipeline, lr_path)
    print(f"Saved baseline models to {svm_path} and {lr_path}")

    return results


class BaselineDetector:
    """Inference wrapper for the trained baseline model."""

    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type
        model_filename = "baseline_svm.joblib" if model_type == "svm" else "baseline_lr.joblib"
        self.model_path = os.path.join(MODELS_DIR, model_filename)
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            print(f"Model file not found at {self.model_path}. Please run train_baseline_models first.")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict whether input text is AI-generated or Human-written.
        Returns label, confidence score, class probabilities, and stylometric features.
        """
        if self.pipeline is None:
            self._load_model()
            if self.pipeline is None:
                raise RuntimeError("Baseline model is not loaded or trained yet.")

        cleaned = clean_text(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        # Label 0 = Human, Label 1 = AI
        p_human = float(probs[0])
        p_ai = float(probs[1])

        label = "AI" if p_ai >= 0.5 else "Human"
        confidence = p_ai if label == "AI" else p_human

        stylometrics = extract_stylometric_dict(text)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Human": round(p_human, 4),
                "AI": round(p_ai, 4),
            },
            "stylometrics": stylometrics,
            "model_name": f"Baseline ({self.model_type.upper()}: TF-IDF + Stylometry)",
        }


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print("Training baseline models...")
        train_baseline_models(train_df, val_df)
    else:
        print("Dataset not found. Run data/prepare_data.py first.")
