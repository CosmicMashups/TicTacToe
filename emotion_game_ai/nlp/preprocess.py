"""NLP preprocessing utilities for sentiment analysis."""

from __future__ import annotations

import re
from typing import Iterable

import nltk


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def ensure_nltk() -> None:
    """Ensure required NLTK resources are present."""
    for pkg in ("punkt", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_and_remove_stopwords(text: str, stopwords: set[str]) -> str:
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(tokens)


def preprocess_texts(texts: Iterable[str]) -> list[str]:
    ensure_nltk()
    from nltk.corpus import stopwords as nltk_stopwords

    sw = set(nltk_stopwords.words("english"))
    out: list[str] = []
    for t in texts:
        t2 = normalize_text("" if t is None else str(t))
        out.append(tokenize_and_remove_stopwords(t2, sw))
    return out

