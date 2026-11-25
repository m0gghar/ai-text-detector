"""Stylometric feature extraction for classical ML baseline.
Implements the 5 features specified in PRD Section 8.3:
1. Average sentence length (words per sentence)
2. Sentence length variance
3. Type-Token Ratio (TTR: vocabulary richness)
4. Punctuation rate (punctuation frequency per token)
5. Average word length (characters per word)
"""

import math
import string
from typing import Dict, List, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing import tokenize_sentences, tokenize_words


def extract_stylometric_dict(text: str) -> Dict[str, float]:
    """Compute stylometric statistics dictionary for a given text.
    Useful for API responses and explainability cards.
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0 or total_sentences == 0:
        return {
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "type_token_ratio": 0.0,
            "punctuation_rate": 0.0,
            "avg_word_length": 0.0,
            "total_words": 0,
            "total_sentences": 0,
        }

    # 1. Average sentence length
    sentence_lengths = [len(tokenize_words(s)) for s in sentences]
    avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0

    # 2. Sentence length variance
    sentence_len_var = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0

    # 3. Type-Token Ratio (lexical diversity)
    unique_words = len(set(w.lower() for w in words))
    ttr = float(unique_words / total_words) if total_words > 0 else 0.0

    # 4. Punctuation rate (punctuation occurrences per word token)
    punct_chars = set(string.punctuation + "“”‘’—–…")
    punct_count = sum(1 for ch in text if ch in punct_chars)
    punct_rate = float(punct_count / total_words) if total_words > 0 else 0.0

    # 5. Average word length (characters per word)
    total_word_chars = sum(len(w) for w in words)
    avg_word_len = float(total_word_chars / total_words) if total_words > 0 else 0.0

    return {
        "avg_sentence_length": round(avg_sentence_len, 2),
        "sentence_length_variance": round(sentence_len_var, 2),
        "type_token_ratio": round(ttr, 4),
        "punctuation_rate": round(punct_rate, 4),
        "avg_word_length": round(avg_word_len, 2),
        "total_words": total_words,
        "total_sentences": total_sentences,
    }


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer that extracts a 5-dimensional
    stylometric feature vector from text inputs.
    """

    FEATURE_NAMES = [
        "avg_sentence_length",
        "sentence_length_variance",
        "type_token_ratio",
        "punctuation_rate",
        "avg_word_length",
    ]

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """Transform a collection of text documents into a 2D numpy array of stylometric features."""
        features_list = []
        for text in X:
            d = extract_stylometric_dict(str(text))
            row = [
                d["avg_sentence_length"],
                d["sentence_length_variance"],
                d["type_token_ratio"],
                d["punctuation_rate"],
                d["avg_word_length"],
            ]
            features_list.append(row)
        return np.array(features_list, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)
