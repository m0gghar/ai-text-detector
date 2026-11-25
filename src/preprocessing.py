"""Text preprocessing utilities for AI vs Human Text Detector.
Handles cleaning, sentence tokenization, word tokenization, and length validation.
"""

import re
import unicodedata
from typing import List, Tuple


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters like curly quotes, dashes, and ellipsis."""
    if not text:
        return ""
    # Normalize unicode forms
    text = unicodedata.normalize("NFKC", text)
    # Replace special quotes and hyphens
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " - ",
        "\u2013": " - ",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace, normalizing quotes, and trimming."""
    if not text:
        return ""
    text = normalize_unicode(text)
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize multiple whitespace characters but preserve sentence flow
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection.
    Handles standard sentence terminators (. ! ?) while avoiding false breaks
    on common abbreviations (e.g., Mr., Dr., etc., vs.).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Common abbreviations to prevent false splits
    abbrevs = r"\b(?:e\.g|i\.e|mr|mrs|ms|dr|prof|vs|approx|etc|inc|ltd)\."
    
    # Temporarily mask period in known abbreviations
    def mask_abbrev(match):
        return match.group(0).replace(".", "@@DOT@@")

    masked = re.sub(abbrevs, mask_abbrev, cleaned, flags=re.IGNORECASE)

    # Split on sentence ending punctuation followed by space or newline
    sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])|\n+")
    raw_sentences = sentence_end.split(masked)

    sentences = []
    for s in raw_sentences:
        unmasked = s.replace("@@DOT@@", ".").strip()
        if unmasked:
            sentences.append(unmasked)

    # Fallback if no split occurred but text has multiple lines
    if len(sentences) <= 1:
        fallback = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(fallback) > len(sentences):
            return fallback

    return sentences if sentences else [cleaned]


def tokenize_words(text: str) -> List[str]:
    """Extract word tokens from text, preserving apostrophes in contractions."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Match words including contractions (e.g., "don't", "it's")
    return re.findall(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b", cleaned)


def get_word_count(text: str) -> int:
    """Return the total word count for a piece of text."""
    return len(tokenize_words(text))


def check_length_warning(text: str, min_words: int = 50) -> Tuple[bool, str]:
    """Check if the text meets the minimum recommended word count.
    Implements PRD FR3: System warns when the text is too short for a reliable result.
    """
    count = get_word_count(text)
    if count == 0:
        return True, "Input text is empty. Please paste text to analyze."
    if count < min_words:
        return True, (
            f"Warning: Text contains only {count} words (recommended minimum is {min_words} words). "
            "Predictions on short texts have significantly higher uncertainty and risk of false classification."
        )
    return False, ""
