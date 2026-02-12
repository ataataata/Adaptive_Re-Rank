# Adaptive Re-Ranking

A framework that dynamically routes queries to the most cost-effective re-ranking strategy based on query complexity, instead of applying a heavy re-ranker to every query.

Queries are routed to one of three pipelines:
- **Class 0 (BM25):** No re-ranker, fastest execution
- **Class 1 (MiniLM-L6):** Light re-ranker, balanced speed and quality
- **Class 2 (BGE-v2-m3):** Heavy re-ranker, highest quality at higher latency

However, users can change the re-ranker acording to their needs.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

The project has two stages: **labeling** and **training**.

### 1. Label queries

`labelling.py` runs each query through all three retrieval strategies (BM25, MiniLM-L6, BGE), measures effectiveness (nDCG@10 + MRR@10) and latency, then assigns each query to the best strategy using a utility function:

```
util(q) = (1 - lambda) * eff(q) + lambda * (1 - lat(q) / max_lat(q))
```

The `lambda` parameter (default `0.05`) controls latency sensitivity. Increase it to prioritize speed over quality.

To run labeling, uncomment the datasets you want in `labelling.py`:

```python
datasets_to_run = [
    ("beir/fiqa/train", "fiqa-train"),
    ("beir/nfcorpus/train", "nfcorpus-train"),
    # add more BEIR datasets here
]
```

Then run:

```bash
python labelling.py
```

This outputs:
- CSV files in `labeled_queries/` (one per dataset)
- A summary in `stats.txt`

### 2. Train the router

`trainning_classifier.py` reads all labeled CSVs, downsamples to balance classes, and fine-tunes `bert-base-uncased` as a 3-class query router.

```bash
python trainning_classifier.py
```

This will:
- Load all CSVs from `labeled_queries/`
- Downsample to the minority class size (user can opt not to do that)
- Fine-tune BERT for 20 epochs (resumes from checkpoint if available)
- Save the model to `./results/models/`
- Plot training loss and validation accuracy curves

## Project Structure

```
adaptive/
├── labelling.py              # Labels queries using BM25, L6, and BGE
├── trainning_classifier.py   # Fine-tunes BERT router on labeled data
├── labeled_queries/          # Output directory for labeled CSVs
├── results/                  # Training checkpoints and saved model
│   └── models/               # Final saved model and tokenizer
├── stats.txt                 # Labeling run statistics
└── requirements.txt          # Python dependencies
```

## Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `lambda_cost` | `labelling.py` | `0.05` | Latency penalty weight. Higher = faster routing preferences |
| `num_train_epochs` | `trainning_classifier.py` | `20` | Number of fine-tuning epochs |
| `learning_rate` | `trainning_classifier.py` | `3e-5` | AdamW learning rate |
| `k` (top-k) | `labelling.py` | `50` | Number of BM25 candidates to re-rank |
| `DOWNSAMPLE` | `trainning_classifier.py` | `True` | Balance classes by downsampling to minority class size. Set `False` to use all data |
