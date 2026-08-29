"""Small text-normalization helpers shared by retrieval and ranking."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "have", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with", "would",
    "you", "looking", "what", "matters", "need", "prefer", "preference",
}


def flatten_text(value: object) -> str:
    """Convert nested catalog fields into one searchable string."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    """Return normalized, non-trivial search terms while preserving order."""

    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def normalize_text(text: str) -> str:
    """Normalize punctuation and whitespace for phrase comparisons."""

    return " ".join(token.lower() for token in TOKEN_RE.findall(text))


def token_coverage(required: str, candidate: str) -> float:
    """Return the fraction of unique requirement terms present in a candidate."""

    required_terms = set(terms(required))
    if not required_terms:
        return 0.0
    candidate_terms = set(terms(candidate))
    return len(required_terms & candidate_terms) / len(required_terms)
