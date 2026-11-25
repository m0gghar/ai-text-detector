"""Unit tests for text preprocessing."""

import pytest
from src.preprocessing import (
    clean_text,
    tokenize_sentences,
    tokenize_words,
    get_word_count,
    check_length_warning,
)


def test_clean_text():
    raw = "  Hello   world!  \n\n\nThis is a test. <p>Html tag</p> “Smart quote” "
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert ' "Smart quote" ' in cleaned or '"Smart quote"' in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello world!")


def test_tokenize_sentences():
    text = "Photosynthesis is vital. Dr. Smith proved it in 1999! How does it work? It creates glucose."
    sentences = tokenize_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Photosynthesis is vital."
    assert "Dr. Smith" in sentences[1]


def test_tokenize_words():
    text = "Don't count the days, make the days count!"
    words = tokenize_words(text)
    assert "Don't" in words or "dont" in words or "Don" in words
    assert len(words) == 8


def test_word_count():
    text = "One two three four five."
    assert get_word_count(text) == 5


def test_length_warning():
    short_text = "This is a brief note."
    is_warn, msg = check_length_warning(short_text, min_words=50)
    assert is_warn is True
    assert "Warning" in msg

    long_text = " ".join(["word"] * 60) + "."
    is_warn, msg = check_length_warning(long_text, min_words=50)
    assert is_warn is False
    assert msg == ""
