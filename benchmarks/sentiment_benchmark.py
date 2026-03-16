"""Simple benchmark for sentiment classification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import click
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def load_data(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


@click.command()
@click.option("--data", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to CSV dataset.")
def main(data: Path) -> None:
    texts, labels = load_data(data)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    click.echo(f"Accuracy on hold‑out set: {acc:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
