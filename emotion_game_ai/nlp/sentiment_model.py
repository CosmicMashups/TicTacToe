"""Emotion + sentiment model training/inference using TF-IDF + Logistic Regression."""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from emotion_game_ai.nlp.preprocess import preprocess_texts


@dataclass
class SentimentArtifacts:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    labels: list[str]


class SentimentModel:
    """
    Train and run emotion classification (13 labels) and derive sentiment groupings.

    - Emotion labels: dataset categories (e.g. love, anger, worry, ...).
    - Derived runtime sentiment labels: positive, neutral, negative.
    """

    def __init__(self, model_dir: str = os.path.join("data", "models")) -> None:
        self.model_dir = model_dir
        self.artifacts: Optional[SentimentArtifacts] = None

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, "sentiment_model.pkl")

    def try_load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self.artifacts = SentimentArtifacts(
            vectorizer=payload["vectorizer"],
            model=payload["model"],
            labels=payload["labels"],
        )
        return True

    def save(self) -> None:
        if self.artifacts is None:
            raise RuntimeError("No artifacts to save. Train first.")
        os.makedirs(self.model_dir, exist_ok=True)
        payload = {
            "vectorizer": self.artifacts.vectorizer,
            "model": self.artifacts.model,
            "labels": self.artifacts.labels,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def train(
        self,
        dataset_path: str = os.path.join("assets", "sentiment_data.csv"),
        sample_rows: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        df = pd.read_csv(dataset_path)
        required = {"ID", "text", "emotion"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=random_state)

        # Normalize and optionally filter emotions before training.
        df = df.copy()
        df["emotion_norm"] = df["emotion"].astype(str).str.strip().str.lower()
        # Drop dominant neutral rows so the model focuses on the other emotions.
        df = df[df["emotion_norm"] != "neutral"]

        texts = preprocess_texts(df["text"].astype(str).tolist())
        y = df["emotion_norm"].astype(str).tolist()
        labels = sorted({v for v in y if v})
        if not labels:
            raise ValueError("No emotion labels found in dataset.")

        x_train, x_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=random_state, stratify=y
        )

        vectorizer = TfidfVectorizer(max_features=10000)
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        # Dataset is heavily imbalanced toward "neutral"; balancing improves robustness.
        model = LogisticRegression(max_iter=800, solver="lbfgs", class_weight="balanced")
        model.fit(x_train_vec, y_train)

        preds = model.predict(x_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, preds, labels=labels, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, preds, labels=labels))

        self.artifacts = SentimentArtifacts(vectorizer=vectorizer, model=model, labels=labels)

    def predict_emotion(self, text: str) -> str:
        if self.artifacts is None:
            return "neutral"
        cleaned = preprocess_texts([text])[0]
        x = self.artifacts.vectorizer.transform([cleaned])
        pred = self.artifacts.model.predict(x)[0]
        return str(pred).strip().lower() or "neutral"

    def predict(self, text: str) -> str:
        """Backward-compatible: returns derived runtime sentiment (positive/neutral/negative)."""
        emotion = self.predict_emotion(text)
        return sentiment_from_emotion(emotion)

    def predict_group(self, text: str) -> str:
        """Returns a 3-way group: positive | negative | low_engagement."""
        emotion = self.predict_emotion(text)
        return emotion_group(emotion)


def _normalize_emotions(raws: list[str]) -> list[str]:
    out: list[str] = []
    for raw in raws:
        r = (raw or "").strip().lower()
        out.append(r if r else "neutral")
    return out


def emotion_group(emotion: str) -> str:
    """
    Map a predicted emotion to a higher-level group:
    - positive: love, happiness, fun, enthusiasm, relief
    - negative: sadness, hate, anger, worry
    - low_engagement: neutral, empty, boredom, surprise
    """
    e = (emotion or "").strip().lower()
    if e in {"love", "happiness", "fun", "enthusiasm", "relief"}:
        return "positive"
    if e in {"sadness", "hate", "anger", "worry"}:
        return "negative"
    return "low_engagement"


def sentiment_from_emotion(emotion: str) -> str:
    """
    Backwards-compatible mapping into runtime sentiment:
    positive -> positive, negative -> negative, low_engagement -> neutral.
    """
    g = emotion_group(emotion)
    if g == "positive":
        return "positive"
    if g == "negative":
        return "negative"
    return "neutral"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Train and save the sentiment model")
    p.add_argument("--dataset", default=os.path.join("assets", "sentiment_data.csv"))
    p.add_argument("--model-dir", default=os.path.join("data", "models"))
    p.add_argument("--sample-rows", type=int, default=0, help="Optional row sample for faster training")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sm = SentimentModel(model_dir=args.model_dir)
    if args.train:
        sample = args.sample_rows if args.sample_rows and args.sample_rows > 0 else None
        sm.train(dataset_path=args.dataset, sample_rows=sample)
        sm.save()
        print(f"Saved: {sm.model_path}")
        return 0
    loaded = sm.try_load()
    print("Loaded model." if loaded else "No saved model found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

