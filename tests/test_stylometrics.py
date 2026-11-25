"""Unit tests for stylometric feature extraction."""

import numpy as np
from src.stylometrics import StylometricExtractor, extract_stylometric_dict


def test_extract_stylometric_dict():
    text = (
        "Artificial intelligence transforms modern technology. "
        "It provides deep predictive modeling capabilities across industries."
    )
    d = extract_stylometric_dict(text)
    assert d["total_sentences"] == 2
    assert d["total_words"] > 10
    assert d["avg_sentence_length"] > 0
    assert 0.0 <= d["type_token_ratio"] <= 1.0
    assert d["punctuation_rate"] >= 0.0
    assert d["avg_word_length"] > 0


def test_stylometric_extractor_sklearn():
    texts = [
        "Short sentence one. Another sentence two.",
        "A much longer and more elaborate sentence detailing multiple complex propositions.",
    ]
    extractor = StylometricExtractor()
    features = extractor.fit_transform(texts)
    assert isinstance(features, np.ndarray)
    assert features.shape == (2, 5)
