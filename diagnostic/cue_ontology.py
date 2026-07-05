"""Text normalization helpers used by manual cue-case filtering."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not needle:
        return True
    return f" {needle} " in f" {haystack} "
