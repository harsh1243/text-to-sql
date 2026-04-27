# Schema-Aware Text-to-SQL

> **A Multi-Stage Retrieval & Dual Transformer Framework for Natural Language Database Queries**  
> Making databases accessible to everyone — no SQL knowledge required.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Flan--T5--XL%20%2B%20LoRA-orange)](https://huggingface.co/google/flan-t5-xl)
[![Dataset](https://img.shields.io/badge/Dataset-Spider-purple)](https://yale-lily.github.io/spider)
[![NLP Course Project](https://img.shields.io/badge/Course-NLP%20Project-red)]()

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [File Manifest](#file-manifest)
- [Installation](#installation)
- [Configuration](#configuration)
- [Operating Instructions](#operating-instructions)
- [Datasets](#datasets)
- [Results](#results)
- [Known Bugs and Issues](#known-bugs-and-issues)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project addresses the problem of **natural language to SQL (Text-to-SQL) translation** for non-technical users. Instead of requiring full database schema as input (which is impractical for large databases), our system:

1. **Retrieves** only the relevant tables and columns using a 5-stage multi-signal schema retriever
2. **Plans** an intermediate execution plan (SCAN → FILTER → AGGREGATE → PROJECT)
3. **Generates** the final SQL query using a fine-tuned Flan-T5-XL model with LoRA

We implement and compare **two architectures**:
- **Single Transformer**: Question + Schema → Plan + SQL (one model)
- **Dual Transformer**: Question + Schema → Plan (T1) → SQL (T2)

### Key Results

| Model | Dataset | Plan Token F1 | SQL Token F1 |
|---|---|---|---|
| Single Transformer | Spider (Gold Schema) | 93.93 | 94.02 |
| Dual Transformer | Spider (Gold Schema) | **95.18** | **94.80** |
| Single Transformer | Spider (Retriever) | 84.95 | 84.68 |
| Dual Transformer | Spider (Retriever) | 83.99 | 83.96 |

**Retriever Performance:** Table F1: 75.58 | Column F1: 68.52 | FK F1: 70.66

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Natural Language Question                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Stage Schema Retriever                        │
│  Stage 1: Two-Signal Fusion (BM25 + Bi-encoder)                 │
│  Stage 2: Cross-Encoder Reranking (MS-MARCO)                     │
│  Stage 3: Adaptive Threshold Selection (DROP_RATIO=0.65)         │
│  Stage 3.5: FK Neighbor Expansion                                │
│  Stage 4: Bridge BFS (FK Graph)                                  │
│  Stage 4.5: Column Pruning                                       │
│  Stage 5: Format → question | schema | foreign keys              │
└─────────────────────┬───────────────────────────────────────────┘
                      │  Pruned Schema
                      ▼
        ┌─────────────────────────────┐
        │   Single Transformer        │         ┌──────────────────┐
        │   Flan-T5-XL + LoRA         │   OR    │ Dual Transformer │
        │   Q+Schema → Plan+SQL       │         │ T1: Q+S → Plan   │
        └─────────────┬───────────────┘         │ T2: Q+S+P → SQL  │
                      │                         └────────┬─────────┘
                      ▼                                  ▼
              [ SQL Query Output ]              [ SQL Query Output ]
```

---

## File Manifest

```
text-to-sql/
│
├── README.md                                        ← This file
│
├── dataset/                                         ← Dataset files
│   ├── dev_final.json                               ← Processed dev/validation split
│   ├── train_final.json                             ← Processed training split
│   └── readme.md                                    ← Dataset-specific notes
│
├── experiments/                                     ← All training & evaluation notebooks
│   ├── plan2sql(spider).ipynb                       ← Dual T2: Plan→SQL on Spider dataset
│   ├── plan2sql(synthetic).ipynb                    ← Dual T2: Plan→SQL on Synthetic dataset
│   ├── text2plan(spider).ipynb                      ← Dual T1: Text→Plan on Spider dataset
│   ├── text2plan(synthetic).ipynb                   ← Dual T1: Text→Plan on Synthetic dataset
│   ├── single_transformer(spider).ipynb             ← Single Transformer on Spider dataset
│   ├── single_transformer(synthetic).ipynb          ← Single Transformer on Synthetic dataset
│   ├── retriver_based_single_transformer_synthetic_.ipynb  ← Full pipeline: Retriever + Single Transformer
│   ├── retriver_text_to_plan(synthetic).ipynb       ← Full pipeline: Retriever + Dual T1 (Plan)
│   └── readme.txt                                   ← Notes on running experiments
│
├── lora-training/                                   ← LoRA fine-tuning notebook
│   ├── text_to_plan.ipynb                           ← LoRA training script for Flan-T5-XL
│   └── readme.txt                                   ← Training setup instructions
│
├── lora_weights/                                    ← Saved LoRA adapter checkpoints
│   └── weights.txt                                  ← Instructions / links to download full weights
│
├── query_to_plan/                                   ← Dual Transformer SQL generation module
│   ├── query_to_plan.py                             ← Core script: Question + Plan → SQL
│   └── README.md                                    ← Module-level documentation
│
└── retriver/                                        ← Multi-stage schema retriever module
    ├── __init__.py                                  ← Package initializer
    ├── config.py                                    ← All tunable hyperparameters (W_BIENCODER_BASE, DROP_RATIO, etc.)
    ├── models.py                                    ← Bi-encoder and cross-encoder model loading
    ├── scoring.py                                   ← BM25 + Bi-encoder fusion scoring (Stage 1)
    ├── selection.py                                 ← Adaptive threshold + FK expansion (Stages 3 & 3.5)
    ├── pipeline.py                                  ← Main 5-stage retriever pipeline orchestrator
    ├── parser.py                                    ← SQL schema (.sql) parser and FK graph builder
    ├── formatter.py                                 ← Output formatter → "question | schema | foreign keys"
    └── README.md                                    ← Retriever-specific documentation
```

> **Note:** Large model files (base Flan-T5-XL weights) are not included in the repository.  
> Download them from HuggingFace: `google/flan-t5-xl`

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (minimum 16GB VRAM recommended for Flan-T5-XL)
- Google Colab (Pro recommended) or equivalent cloud GPU environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/harsh1243/text-to-sql.git
cd text-to-sql
```

### Step 2: Install Dependencies

```bash
pip install -r lora-training/requirements.txt
```

The key dependencies are:

```
torch>=2.0.0
transformers==4.40.0
peft==0.10.0
accelerate==0.29.3
sentence-transformers>=2.6.0
rank-bm25>=0.2.2
datasets>=2.18.0
evaluate>=0.4.0
numpy>=1.24.0
tqdm>=4.65.0
```

### Step 3: Download Base Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
```

### Step 4: Download Spider Dataset

Download from the official Spider repository and place in `dataset/spider/`:

```bash
# Manual download from:
# https://yale-lily.github.io/spider
# Place train_spider.json, dev.json, and database/ folder inside dataset/spider/
```

---

## Configuration

All retriever hyperparameters are stored in `retriver/config.py`:

```python
# ── Fusion Scoring Weights ──────────────────────────────────────
W_BIENCODER_BASE   = 0.60   # Weight for bi-encoder (dense) signal
W_BM25_BASE        = 0.40   # Weight for BM25 (lexical) signal

# ── Threshold Selection ─────────────────────────────────────────
DROP_RATIO         = 0.65   # Keep tables above top_score × ratio
GAP_RATIO          = 0.25   # Gap guard: stop if gap > top_score × ratio
MAX_TABLES         = 6      # Maximum tables to select

# ── Cross-Encoder Reranking ─────────────────────────────────────
CE_WEIGHT          = 0.60   # Cross-encoder weight in combined score
FUSION_WEIGHT      = 0.40   # Fusion score weight in combined score

# ── Junction Table Handling ─────────────────────────────────────
JUNCTION_PENALTY   = -0.15  # Penalty for junction tables not in question
```

> ⚠️ **Important:** These hyperparameter values were **not tuned empirically**. They were selected using LLM-suggested defaults. Hyperparameter tuning is planned as future work and may significantly improve retriever performance.

### Generation Model Configuration

```python
# Inference settings (used in all evaluation notebooks)
MAX_INPUT_LENGTH  = 512
MAX_NEW_TOKENS    = 512
NUM_BEAMS         = 4
EARLY_STOPPING    = True
LENGTH_PENALTY    = 1.0
```

---

## Operating Instructions

> **Recommended:** All experiments can be run directly as **Jupyter/Colab notebooks** from the `experiments/` folder. You only need the trained LoRA adapter weights — you do not need to re-train from scratch.

---

### Option A: Run Directly Using Experiment Notebooks (Recommended)

This is the easiest way to reproduce results or run inference. Each notebook in `experiments/` is self-contained and only requires:
1. The base model (`google/flan-t5-xl`) — downloaded automatically from HuggingFace
2. The trained LoRA adapter `.zip` file — from `lora_weights/`

**Step 1: Upload your LoRA adapter zip to Colab**

```python
from google.colab import files
uploaded = files.upload()   # upload your lora_adapter_*.zip file
```

**Step 2: Install dependencies (first cell of every notebook)**

```python
!pip install -q transformers==4.40.0 peft==0.10.0 accelerate==0.29.3
```

**Step 3: Load base model + LoRA adapter**

```python
import zipfile, os, torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

# Unzip LoRA adapter
ADAPTER_ZIP = "./lora_adapter_plan2sql.zip"
ADAPTER_DIR = "./lora_adapter_plan2sql"
if not os.path.exists(ADAPTER_DIR):
    with zipfile.ZipFile(ADAPTER_ZIP, 'r') as z:
        z.extractall(ADAPTER_DIR)

# Auto-select precision based on GPU capability
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load Flan-T5-XL base model
base_model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl",
    torch_dtype=dtype,
    device_map="auto",
    offload_folder="./offload",
)

# Apply LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
model.eval()
model.config.use_cache = True
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
```

**Step 4: Run inference**

```python
def generate_output(input_text, max_input=384, max_target=384):
    inp = tokenizer(input_text, return_tensors="pt",
                    max_length=max_input, truncation=True)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_target,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.0,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Example
result = generate_output("question: How many singers are there? | schema: singer ( singer_id [PK] ) | foreign keys: none")
print(result)
# step1: SCAN | table: singer || step2: AGGREGATE | Scalar aggregate -> compute COUNT(*) || step3: PROJECT | columns: COUNT(*)
```

---

### Experiment Notebooks — What Each Does

| Notebook | Task | Input | Output |
|---|---|---|---|
| `text2plan(spider).ipynb` | Text → Plan (Dual T1) | Question + Schema | Execution Plan |
| `text2plan(synthetic).ipynb` | Text → Plan on synthetic data | Question + Schema | Execution Plan |
| `plan2sql(spider).ipynb` | Plan → SQL (Dual T2) | Question + Plan | SQL Query |
| `plan2sql(synthetic).ipynb` | Plan → SQL on synthetic data | Question + Plan | SQL Query |
| `single_transformer(spider).ipynb` | Q+Schema → Plan+SQL (Single) | Question + Schema | Plan + SQL |
| `single_transformer(synthetic).ipynb` | Single Transformer on synthetic | Question + Schema | Plan + SQL |
| `retriver_based_single_transformer_synthetic_.ipynb` | **Full pipeline** (Retriever + Single) | Question + DB Schema file | SQL Query |
| `retriver_text_to_plan(synthetic).ipynb` | **Full pipeline** (Retriever + Dual T1) | Question + DB Schema file | Plan |

---

### Option B: LoRA Training from Scratch

If you want to train your own LoRA adapter, open `lora-training/text_to_plan.ipynb` in Google Colab.

**What the training notebook does:**

```
1. Loads google/flan-t5-xl as the base model
2. Applies LoRA config (low-rank adapter layers, base weights frozen)
3. Trains on (input_text → target_text) pairs from train_final.json
4. Saves the LoRA adapter as a zip file (lora_adapter_*.zip)
   → This zip is what all experiment notebooks load for inference
```

**Minimum requirements for training:**
- Google Colab Pro with A100 GPU (40GB VRAM)
- `torch_dtype = bfloat16` (A100) or `float16` (V100/T4)
- `device_map = "auto"` with offload folder for large model shards

**Training data format** (`train_final.json`):

```json
[
  {
    "input":  "question: How many singers are there? | schema: singer ( singer_id [PK] ) | foreign keys: none",
    "output": "step1: SCAN | table: singer || step2: AGGREGATE | Scalar aggregate -> compute COUNT(*) || step3: PROJECT | columns: COUNT(*)"
  },
  ...
]
```

**After training**, download your adapter zip and use it in any experiment notebook as shown in Option A.

---

### Running Retriever Only

```python
import sys
sys.path.append("retriver/")
from pipeline import MultiStageRetrieverPipeline

pipeline = MultiStageRetrieverPipeline(db_schema_path="path/to/schema.sql")
result = pipeline.retrieve("How many heads of departments are older than 56?")
print(result)
# question: How many heads of departments are older than 56?
# | schema: head ( head_ID [PK], age )
# | foreign keys: none
```

---

### Token F1 Evaluation

The experiment notebooks compute Token F1 automatically. Here is the **actual output** from a real run of `plan2sql(spider).ipynb`:

```
══════════════════════════════════════
    SQL EVALUATION (n=150)
══════════════════════════════════════
  Token F1 : 94.80%
══════════════════════════════════════

── Sample predictions ──

[0] SQL F1=1.000
  PRED PLAN : step1: SCAN | table: singer || step2: AGGREGATE | Scalar aggregate (no GROUP BY) -> compute COUNT(*) || step3: PROJECT | columns: COUNT(*)
  GOLD SQL  : SELECT count(*) FROM singer
  PRED SQL  : SELECT count(*) FROM singer

[10] SQL F1=1.000
  PRED PLAN : step1: SCAN | table: singer || step2: AGGREGATE | group_by: Country | compute: COUNT(*) || step3: PROJECT | columns: Country, COUNT(*)
  GOLD SQL  : SELECT country , count(*) FROM singer GROUP BY country
  PRED SQL  : SELECT Country, COUNT(*) FROM singer GROUP BY Country

[50] SQL F1=1.000
  PRED PLAN : step1: SCAN | table: Pets || step2: AGGREGATE | group_by: PetType | compute: MAX(weight) || step3: PROJECT | columns: MAX(weight), PetType
  GOLD SQL  : SELECT max(weight) , petType FROM pets GROUP BY petType
  PRED SQL  : SELECT max(weight), PetType FROM Pets GROUP BY PetType
```

The Token F1 formula used internally:

```python
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def token_f1(pred, gold):
    pred_tok = tokenize(pred)
    gold_tok = tokenize(gold)
    if not pred_tok and not gold_tok: return 1.0
    if not pred_tok or not gold_tok:  return 0.0
    overlap   = sum((Counter(pred_tok) & Counter(gold_tok)).values())
    precision = overlap / len(pred_tok)
    recall    = overlap / len(gold_tok)
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)
```

---

## Datasets

### Spider Benchmark (Primary)
- **Paper:** Yu et al., EMNLP 2018
- **Download:** https://yale-lily.github.io/spider
- **Description:** 10,181 questions, 200 databases, 138 domains
- **Usage:** Primary benchmark for training and evaluation

### Synthetic Dataset (New)
- **Location:** `dataset/synthetic/`
- **Description:** Newly generated dataset with unseen schemas for zero-shot generalization testing
- **Usage:** Tests true generalization to databases not seen during training

### Related Repositories and Resources

| Resource | Link |
|---|---|
| Spider Dataset | https://yale-lily.github.io/spider |
| Flan-T5-XL (HuggingFace) | https://huggingface.co/google/flan-t5-xl |
| MS-MARCO Cross-Encoder | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Sentence-Transformers (Bi-encoder) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| PEFT Library | https://github.com/huggingface/peft |
| rank_bm25 | https://github.com/dorianbrown/rank_bm25 |
| DAIL-SQL (baseline) | https://github.com/BeachWang/DAIL-SQL |
| PICARD (baseline) | https://github.com/ServiceNow/picard |

---

## Results

### Transformer Model Results (Gold Schema Input)

| Model | Dataset | Plan Token F1 (%) | SQL Token F1 (%) |
|---|---|---|---|
| Single Transformer | Synthetic (New) | 90.01 | 90.99 |
| Single Transformer | Spider | 93.93 | 94.02 |
| Dual Transformer | Synthetic (New) | 90.74 | 92.32 |
| **Dual Transformer** | **Spider** | **95.18** | **94.80** |

### Full System Results (Retriever Schema Input)

| Model | Plan Token F1 (%) | SQL Token F1 (%) |
|---|---|---|
| Single Transformer (Spider) | 84.95 | 84.68 |
| Dual Transformer — T1 Plan (Spider) | 83.99 | — |
| Dual Transformer — T2 SQL (Spider) | — | 83.96 |

### Retriever Results

| Component | Metric | Score (%) |
|---|---|---|
| Table Selection | Table F1 | 75.58 |
| Column Selection | Column F1 | 68.52 |
| FK Connections | FK F1 | 70.66 |

---

## Known Bugs and Issues

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | **Retriever hyperparameters not tuned** — Values (W_BIENCODER_BASE, DROP_RATIO, etc.) were LLM-suggested defaults, not empirically validated. May be suboptimal. | Medium | Open |
| 2 | **Column F1 is low (68.52)** — Column retrieval is the weakest component. Tables with many similarly-named columns cause confusion. | Medium | Open |
| 3 | **Error propagation in Dual Transformer** — If T1 (Planner) generates a wrong plan due to retrieval error, T2 (SQL Generator) will also produce wrong SQL. | Medium | Open |
| 4 | **Context length limit (512 tokens)** — Very complex schemas with many tables and columns may get truncated, losing information. | Medium | Open |

| 5 | **No support for multi-turn dialogue** — Each question is treated independently; context from previous queries in a session is not preserved. | Low | Open |
| 6 | **Large base model size** — Flan-T5-XL (3B parameters) requires significant GPU memory. Running on CPU is very slow. | Low | By Design |
| 7 | **Column Mention and Value Pattern signals removed** — Earlier design had 4 retriever signals; current version uses only 2 (BM25 + Bi-encoder). Removing these may have hurt column-level retrieval. | Low | Open |

---

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Fix:** Reduce batch size or use gradient checkpointing. For inference, use `float16` precision:
```python
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", torch_dtype=torch.float16)
```
Alternatively, use Google Colab Pro with A100 GPU.

---

### PEFT / LoRA Adapter Not Loading

**Error:** `ValueError: adapter_model.bin not found`

**Fix:** Ensure the LoRA adapter folder contains both `adapter_model.bin` and `adapter_config.json`. If training from scratch, make sure the training script completed successfully before loading:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "lora_weights/lora_adapter_combined/")
```

---

### Spider Dataset Path Error

**Error:** `FileNotFoundError: dataset/spider/database not found`

**Fix:** Download the Spider dataset manually from https://yale-lily.github.io/spider and place the `database/` folder, `train_spider.json`, and `dev.json` inside `dataset/spider/`:
```
dataset/
└── spider/
    ├── train_spider.json
    ├── dev.json
    └── database/
        ├── department_management/
        │   └── department_management.sqlite
        └── ...
```

---

### Retriever Returns Empty Schema

**Error:** Retriever returns empty schema string or `| schema: | foreign keys: none`

**Fix:** This happens when no table scores above the DROP_RATIO threshold. Lower the threshold in `retriver/config.py`:
```python
DROP_RATIO = 0.50   # Try a lower value (default is 0.65)
```

---

### Sentence Transformers Model Download Fails

**Error:** `OSError: Can't load tokenizer for 'all-MiniLM-L6-v2'`

**Fix:**
```bash
pip install --upgrade sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

### BM25 ImportError

**Error:** `ModuleNotFoundError: No module named 'rank_bm25'`

**Fix:**
```bash
pip install rank-bm25
```

---

### Transformers Version Conflict

**Error:** Model generates garbage output or truncated sequences

**Fix:** Ensure exact versions are installed:
```bash
pip install transformers==4.40.0 peft==0.10.0 accelerate==0.29.3
```
Version mismatches between PEFT and Transformers can cause silent failures in LoRA adapter loading.

---



### Academic References

This project builds on the following foundational work:

- **Spider Dataset** — Yu et al., EMNLP 2018. The primary benchmark used for training and evaluation.
- **DIN-SQL** — Pourreza & Rafiei, NeurIPS 2023. Inspiration for decomposed query generation.
- **DAIL-SQL** — Gao et al., 2023. State-of-the-art Text-to-SQL baseline we compared against.
- **CHESS** — Gao et al., 2024. Retrieval-based Text-to-SQL approach we studied.
- **MAC-SQL** — Wang et al., 2023. Multi-agent SQL generation framework.
- **PICARD** — Scholak et al., EMNLP 2021. Constrained decoding approach for SQL generation.
- **RATSQL** — Wang et al., ACL 2020. Graph-based schema-linking approach.
- **Flan-T5** — Chung et al., 2022. Base generation model used in this project.
- **LoRA** — Hu et al., ICLR 2022. Parameter-efficient fine-tuning method used for adapting Flan-T5-XL.

### Tools and Libraries

- [Hugging Face Transformers](https://github.com/huggingface/transformers) — Model loading and inference
- [PEFT](https://github.com/huggingface/peft) — LoRA fine-tuning
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) — Bi-encoder and cross-encoder
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — BM25 scoring
- [Google Colab](https://colab.research.google.com/) — Training environment



## License

```
None
```

---

*NLP Course Project · 2026 · Schema-Aware Text-to-SQL*
