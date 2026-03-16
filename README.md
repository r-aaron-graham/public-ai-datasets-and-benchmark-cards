# Public AI Datasets and Benchmark Cards

This repository hosts example datasets along with accompanying
documentation (data cards) and simple benchmark scripts to evaluate
models on the data.  Publishing datasets with clear metadata helps
others understand their contents, quality, and appropriate uses.

## Contents

- `data/` – Sample datasets in CSV format.
- `data_cards/` – Markdown files describing each dataset (columns,
  source, intended use, ethical considerations).
- `benchmarks/` – Scripts to run simple tasks (e.g., classification,
  summarization) on the datasets and compute metrics.

## Example Dataset: `sentiment_dataset.csv`

This synthetic dataset contains short text reviews labeled as
positive or negative.  It can be used to train or evaluate sentiment
analysis models.  See `data_cards/sentiment_dataset.md` for details.

## Running Benchmarks

Install dependencies and run the benchmark:

```bash
pip install -r requirements.txt
python benchmarks/sentiment_benchmark.py --data data/sentiment_dataset.csv
```

The script trains a simple logistic regression classifier and reports
accuracy on a hold‑out set.

## Extending

Add your own datasets under `data/` and create corresponding data cards
in `data_cards/`.  Write benchmark scripts that demonstrate common
tasks and report relevant metrics.