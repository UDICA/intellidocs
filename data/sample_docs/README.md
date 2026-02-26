# Sample Documents

This directory contains sample documents for demonstrating IntelliDocs' document ingestion and retrieval capabilities.

## Files

| File | Format | Description |
|------|--------|-------------|
| `python_overview.txt` | Plain text | Overview of the Python programming language — history, features, and ecosystem |
| `machine_learning_basics.md` | Markdown | Fundamentals of machine learning — types, algorithms, evaluation metrics, and workflow |
| `world_cities.csv` | CSV | Data on 15 major world cities with population figures and descriptions |

## Usage

Ingest these documents through the web UI upload feature or via the CLI:

```bash
python -m backend.src.cli data/sample_docs/
```

Once ingested, try queries like:

- "What are the main use cases for Python?"
- "Explain the difference between supervised and unsupervised learning"
- "Which are the most populous cities in Asia?"
- "What is overfitting and how can it be prevented?"

## Notes

All content is factual, publicly available information suitable for demonstration purposes. Population figures are approximate and based on recent metropolitan area estimates.
