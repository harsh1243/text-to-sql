"""
Scoring Stages
===============
Stage 1 — Dual-signal fusion scoring (bi-encoder + BM25)
Stage 2 — Cross-encoder reranking

Functions
---------
- stage1_fusion(question, schema, synonyms, top_k) → ranked candidate list
- stage2_crossencoder(question, candidates, schema)  → reranked list

Shared helpers (also imported by selection.py)
----------------------------------------------
- expand_query(question, synonyms)
- _soft_match(token_a, token_b)
- _get_distinctive_parts(col_low)
"""

import re
import math
from collections import defaultdict
from sentence_transformers import util

from .config import (
    W_BIENCODER_BASE, W_BM25_BASE,
    LEXICAL_SHIFT, SEMANTIC_SHIFT, JUNCTION_PENALTY,
    CE_WEIGHT, FUSION_WEIGHT,
)
from .models import get_biencoder, get_crossencoder
from .parser import classify_table_type


# ─── Text helpers (also used by selection.py) ─────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def _soft_match(token_a: str, token_b: str) -> bool:
    """Fuzzy token match — handles simple plurals / suffixes (name↔names)."""
    if token_a == token_b:
        return True
    shorter, longer = sorted([token_a, token_b], key=len)
    return longer.startswith(shorter) and (len(longer) - len(shorter)) <= 3


def _get_distinctive_parts(col_low: str) -> set:
    """
    Return non-trivial parts of a column name.
    e.g. song_name  → {"song"}   ("name" is too generic)
         song_release_year → {"song", "release", "year"}
    """
    from .config import TRIVIAL_COL_PARTS
    TRIVIAL = {'id', 'num', 'no', 'is', 'has', 'the', 'a', 'an'} | TRIVIAL_COL_PARTS
    parts = set(re.split(r'[_\s]+', col_low)) - TRIVIAL
    return {p for p in parts if len(p) > 2}


def expand_query(question: str, synonyms: dict) -> set:
    """
    Expand question tokens using the provided synonym map.
    Pass an empty dict for no expansion.
    """
    words = set(re.findall(r'\b\w+\b', question.lower()))
    expanded = set(words)
    for w in words:
        if w in synonyms:
            expanded.update(synonyms[w])
    return expanded


def make_table_description(table_name: str, info: dict) -> str:
    col_descs = ", ".join(f"{n.lower()} {t}" for n, t in info["columns"])
    return f"table {table_name} columns: {col_descs}"


# ─── BM25 ─────────────────────────────────────────────────────────────────────

def compute_doc_freqs(all_doc_tokens: list) -> dict:
    """Compute document frequency for each token across all table descriptions."""
    df = defaultdict(int)
    for doc in all_doc_tokens:
        for token in set(doc):
            df[token] += 1
    return dict(df)


def bm25_score(q_tokens: list, doc_tokens: list, avgdl: float,
               doc_freqs: dict, n_docs: int,
               k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 with proper corpus-level IDF per token."""
    if not doc_tokens:
        return 0.0
    dl   = len(doc_tokens)
    freq = defaultdict(int)
    for t in doc_tokens:
        freq[t] += 1
    score = 0.0
    for qt in set(q_tokens):
        f = freq.get(qt, 0)
        if f == 0:
            continue
        df  = doc_freqs.get(qt, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf  = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1)))
        score += idf * tf
    return score


# ─── Query-type detection ──────────────────────────────────────────────────────

def _detect_query_type(question: str, schema: dict) -> str:
    """
    'lexical'  → has 2+ direct token overlaps with schema → boost BM25
    'semantic' → paraphrase-style question → boost bi-encoder
    """
    q_tokens = set(_tokenize(question))
    schema_tokens = set()
    for table, info in schema.items():
        schema_tokens.update(re.split(r'[_\s]+', table))
        for (col, _) in info["columns"]:
            schema_tokens.update(re.split(r'[_\s]+', col.lower()))
    schema_tokens -= {'id', 'num', 'no', 'is', 'has', 'the', 'a', 'an'}
    return "lexical" if len(q_tokens & schema_tokens) >= 2 else "semantic"


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FUSION SCORING (bi-encoder + BM25 only)
# ═════════════════════════════════════════════════════════════════════════════

def stage1_fusion(question: str, schema: dict,
                  synonyms: dict,
                  top_k_candidates: int = 6) -> list:
    """
    Score every table with 2 signals and return top_k_candidates.

    Signals: bi-encoder cosine similarity, BM25.
    Also applies keyword boost and junction table penalty.

    Returns
    -------
    list of (name, fusion, bi, bm) sorted by fusion desc.
    """
    model        = get_biencoder()
    table_names  = list(schema.keys())
    descriptions = [make_table_description(t, schema[t]) for t in table_names]

    # Bi-encoder scores
    q_emb     = model.encode(question,     convert_to_tensor=True)
    t_embs    = model.encode(descriptions, convert_to_tensor=True)
    bi_scores = util.cos_sim(q_emb, t_embs)[0].cpu().tolist()

    # BM25 with corpus-level IDF
    q_tokens  = _tokenize(question)
    doc_toks  = [_tokenize(d) for d in descriptions]
    avgdl     = sum(len(d) for d in doc_toks) / max(len(doc_toks), 1)
    doc_freqs = compute_doc_freqs(doc_toks)
    n_docs    = len(doc_toks)

    bm25_raw = [bm25_score(q_tokens, doc_toks[i], avgdl, doc_freqs, n_docs)
                for i in range(len(table_names))]

    # Min-max normalize BM25 to [0, 1]
    bm_min, bm_max = min(bm25_raw), max(bm25_raw)
    bm_range  = bm_max - bm_min
    bm25_norm = [(s - bm_min) / bm_range if bm_range > 0 else 0.0
                 for s in bm25_raw]

    q_lower = question.lower()

    # Query-adaptive weights — always sum to 1.0
    qtype = _detect_query_type(question, schema)
    if qtype == "lexical":
        w_bi = W_BIENCODER_BASE - LEXICAL_SHIFT   # 0.50
        w_bm = W_BM25_BASE      + LEXICAL_SHIFT   # 0.50
    else:  # semantic
        w_bi = W_BIENCODER_BASE + SEMANTIC_SHIFT  # 0.70
        w_bm = W_BM25_BASE      - SEMANTIC_SHIFT  # 0.30

    results = []
    for i, name in enumerate(table_names):
        bi = bi_scores[i]
        bm = bm25_norm[i]

        # Hard keyword boost — table name literally in question
        kb = 0.0
        if name in q_lower or name.replace("_", " ") in q_lower:
            kb = 0.30

        # Junction table penalty — suppress unless explicitly mentioned
        jp = 0.0
        if classify_table_type(name, schema[name]) == "junction" and kb == 0.0:
            jp = JUNCTION_PENALTY

        fusion = w_bi * bi + w_bm * bm + kb + jp
        results.append((name, fusion, bi, bm))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k_candidates]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CROSS-ENCODER RERANKING
# ═════════════════════════════════════════════════════════════════════════════

def stage2_crossencoder(question: str, candidates: list, schema: dict) -> list:
    """
    Cross-encoder reads question + table description jointly.
    Combines normalized CE scores with Stage 1 fusion scores (preserves
    keyword boosts, junction penalties, etc.).

    Returns
    -------
    list of (table_name, combined_score) sorted desc.
    """
    ce_model = get_crossencoder()
    pairs, names, fusion_scores = [], [], []
    for (name, fusion, *_) in candidates:
        pairs.append([question, make_table_description(name, schema[name])])
        names.append(name)
        fusion_scores.append(fusion)

    ce_raw = ce_model.predict(pairs).tolist()

    # Min-max normalize CE scores to [0, 1]
    ce_min, ce_max = min(ce_raw), max(ce_raw)
    ce_range = ce_max - ce_min
    ce_norm  = [(s - ce_min) / ce_range if ce_range > 0 else 0.5
                for s in ce_raw]

    # Combine: CE score + fusion (preserves keyword boosts / junction penalties)
    combined = [CE_WEIGHT * cn + FUSION_WEIGHT * fs
                for cn, fs in zip(ce_norm, fusion_scores)]

    return sorted(zip(names, combined), key=lambda x: x[1], reverse=True)
