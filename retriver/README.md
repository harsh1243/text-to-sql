# Schema Retriever — Multi-Stage Pipeline for Text-to-SQL

A production-ready schema retrieval module that selects the **relevant tables, columns, and foreign keys** from a database schema for a given natural language question. Designed as the retrieval component of a Text-to-SQL system.

## What It Does

Given:
- A **natural language question** (e.g., *"List all singer names in concerts in year 2014"*)
- A **database schema** (`.sql` file with `CREATE TABLE` statements)

It returns the **exact model input** matching the training data format:

```
question: List all singer names in concerts in year 2014. | schema: singer ( Singer_ID [PK], Name ) | concert ( concert_ID [PK], Stadium_ID, Year ) | singer_in_concert ( concert_ID [PK], Singer_ID ) | foreign keys: singer_in_concert.Singer_ID -> singer.Singer_ID ; singer_in_concert.concert_ID -> concert.concert_ID
```

Only the tables and columns relevant to answering the question are included — not the entire database.

---

## Pipeline Architecture

The retriever uses a **5-stage pipeline** where each stage progressively narrows down the schema:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Full Database Schema (.sql)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Multi-Signal Fusion Scoring                         │
│  2 signals: Bi-encoder + BM25  │
│  Query-adaptive weights (lexical vs semantic queries)           │
│  Output: Top 6 candidate tables with fusion scores             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Cross-Encoder Reranking                             │
│  Joint question+table scoring with ms-marco cross-encoder      │
│  Combined with fusion scores (0.6 CE + 0.4 fusion)             │
│  Output: Reranked candidates with combined scores              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Adaptive Threshold Selection                        │
│  Ratio-based cutoff (DROP_RATIO=0.65) with gap guard           │
│  Output: 1–5 selected tables                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3.5 — FK Neighbor Expansion                             │
│  Add FK neighbors only if table/column names match the         │
│  question (prevents over-inclusion)                            │
│  Output: Expanded table set                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — Bridge BFS                                          │
│  Find intermediate junction/bridge tables between selected     │
│  tables via BFS on FK graph (iterative until stable)           │
│  Output: Final table set with bridge tables                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4.5 — Column Pruning                                    │
│  Keep: structural (PK/FK) + text-matched + value-matched       │
│        + superlative-matched + cross-encoder scored             │
│  Output: Pruned columns per table                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — Format Output                                       │
│  Format as training data string:                               │
│  question: ... | schema: ... | foreign keys: ...               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
retriever/
├── __init__.py      # Package entry — exports parse_schema, build_fk_graph, retrieve
├── config.py        # All hyperparameters & constants (single place to tune)
├── models.py        # Lazy-loaded bi-encoder & cross-encoder singletons
├── parser.py        # Stage 0: Parse .sql → schema dict + FK graph
├── scoring.py       # Stage 1 + 2: Fusion scoring + cross-encoder reranking
├── selection.py     # Stage 3 + 3.5 + 4 + 4.5: Table selection + column pruning
├── formatter.py     # Stage 5: Output formatting (training data format)
├── pipeline.py      # Main retrieve() orchestrator
└── README.md        # This file
```

| File | Stage | Responsibility |
|------|-------|----------------|
| `config.py` | — | All tunable hyperparameters (weights, thresholds, stopwords) |
| `models.py` | — | Singleton model loading (bi-encoder, cross-encoder) |
| `parser.py` | 0 | Parse `CREATE TABLE` SQL → structured schema dict + FK graph |
| `scoring.py` | 1, 2 | 4-signal fusion scoring + cross-encoder reranking |
| `selection.py` | 3, 3.5, 4, 4.5 | Adaptive threshold, FK expansion, bridge BFS, column pruning |
| `formatter.py` | 5 | Format output to match training data format exactly |
| `pipeline.py` | All | `retrieve()` — chains all stages together |

---

## Quick Start

### Installation

```bash
pip install sentence-transformers
```

### Usage

```python
from retriever import parse_schema, build_fk_graph, retrieve

# 1. Parse schema ONCE per database
with open("schema.sql") as f:
    sql_text = f.read()

schema   = parse_schema(sql_text)
fk_graph = build_fk_graph(schema)

# 2. Retrieve for each question
result = retrieve(
    question="List all singer names in concerts in year 2014.",
    schema=schema,
    fk_graph=fk_graph,
)

# 3. Use model_input directly as input to your trained model
print(result["model_input"])
# → question: List all singer names in concerts in year 2014. | schema: singer ( Singer_ID [PK], Name ) | concert ( concert_ID [PK], Stadium_ID, Year ) | singer_in_concert ( concert_ID [PK], Singer_ID ) | foreign keys: singer_in_concert.Singer_ID -> singer.Singer_ID ; singer_in_concert.concert_ID -> concert.concert_ID
```

### With Verbose Debug Output

```python
result = retrieve(
    question="Show the stadium names without any concerts.",
    schema=schema,
    fk_graph=fk_graph,
    verbose=True,  # prints stage-by-stage debug info
)
```

### With Domain-Specific Synonyms (Optional)

```python
result = retrieve(
    question="Which venue hosted the most events?",
    schema=schema,
    fk_graph=fk_graph,
    synonyms={
        "venue":  ["stadium", "arena", "location"],
        "event":  ["concert", "show", "performance"],
    },
)
```

---

## Output Format

The `retrieve()` function returns a dictionary:

```python
{
    "model_input":     str,   # ← Use this — matches training data format exactly
    "selected_tables": list,  # Final selected table names
    "schema_str":      str,   # Multi-line schema (for inspection)
    "fk_str":          str,   # Multi-line foreign keys (for inspection)
    "debug": {                # Per-stage debug info
        "stage1":         [...],  # Fusion candidates with scores
        "ranked":         [...],  # After cross-encoder reranking
        "stage3":         [...],  # After adaptive threshold
        "pruned_columns": {...},  # Columns kept per table
    }
}
```

**`model_input`** is the primary output — feed it directly to your Text-to-SQL model.

---

## Models Used

| Model | Purpose | Stage |
|-------|---------|-------|
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Bi-encoder for semantic table similarity | Stage 1 |
| [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Cross-encoder for joint question-table scoring | Stage 2, 4.5 |

Both models are loaded lazily (on first use) and cached as singletons.

---

## Hyperparameters

All tunable constants live in [`config.py`](config.py). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W_BIENCODER_BASE` | 0.60 | Base weight for bi-encoder signal |
| `W_BM25_BASE` | 0.4 | Base weight for BM25 signal |
| `DROP_RATIO` | 0.65 | Adaptive threshold: keep tables above `top_score × ratio` |
| `GAP_RATIO` | 0.25 | Gap guard: stop if score gap > `top_score × ratio` |
| `MAX_TABLES` | 5 | Maximum tables to select |
| `CE_WEIGHT` | 0.60 | Cross-encoder weight in combined score |
| `FUSION_WEIGHT` | 0.40 | Fusion weight in combined score |
| `JUNCTION_PENALTY` | -0.15 | Score penalty for junction tables not mentioned in question |

---


