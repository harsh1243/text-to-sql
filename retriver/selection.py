"""
Table & Column Selection
=========================
Stage 3   — Adaptive threshold selection
Stage 3.5 — FK neighbor expansion
Stage 4   — Bridge BFS (iterative until stable)
Stage 4.5 — Column pruning (structural + text + semantic)

Functions
---------
- adaptive_select(ranked)                   → selected table list
- fk_neighbor_expansion(selected, ...)      → expanded table list
- find_bridge_tables(selected, fk_graph)    → final table list
- stage_column_pruning(tables, question, …) → {table: [(col, type), ...]}
"""

import re
from collections import deque

from .config import (
    DROP_RATIO, GAP_RATIO, MAX_TABLES,
    COL_CE_FLOOR, COL_CE_GAP,
    COLUMN_STOPWORDS, TRIVIAL_COL_PARTS,
)
from .models import get_crossencoder
from .scoring import _tokenize, _soft_match, _get_distinctive_parts, expand_query


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ADAPTIVE THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_select(ranked: list) -> list:
    """
    Ratio-based table selection.  Combined scores (CE_norm + fusion) are
    always in [0, ~1] range, so ratio thresholds work correctly.
    Keeps tables whose score >= top * DROP_RATIO with gap guard.
    Guarantees at least 1 table.
    """
    if not ranked:
        return []
    top_score = ranked[0][1]
    threshold = max(top_score * DROP_RATIO, 0.05)   # floor prevents 0*ratio
    selected  = []
    for i, (name, score) in enumerate(ranked[:MAX_TABLES]):
        if score < threshold:
            break
        if i > 0 and (ranked[i-1][1] - score) > max(top_score * GAP_RATIO, 0.10):
            break
        selected.append(name)
    return selected if selected else [ranked[0][0]]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3.5 — FK NEIGHBOR EXPANSION
# ═════════════════════════════════════════════════════════════════════════════

def _collect_fk_candidates(selected: list, schema: dict,
                            fk_graph: dict, expanded: set) -> set:
    """Collect direct FK neighbors not already in the expanded set."""
    candidates = set()
    for table in selected:
        for (_, neighbor, _) in fk_graph.get(table, []):
            if neighbor in schema and neighbor not in expanded:
                candidates.add(neighbor)
    return candidates


def _filter_by_table_name(nb: str, q_lower: str) -> bool:
    """Return True if the neighbor table name appears in the question."""
    return nb in q_lower or nb.replace('_', ' ') in q_lower


def _filter_by_column_text(nb: str, schema: dict,
                            q_lower: str, q_tokens: set,
                            trivial: set) -> bool:
    """
    Return True if any non-structural column of *nb* matches a question token.
    """
    info = schema[nb]
    pks = set(info.get('pks', []))
    fk_cols = {fc for (fc, _, _) in info.get('fks', [])}
    structural = pks | fk_cols

    for (col, _) in info['columns']:
        col_low = col.lower()
        if col_low in structural:
            continue
        parts = set(re.split(r'[_\s]+', col_low)) - trivial
        if not parts:
            continue
        if len(parts) == 1:
            p = next(iter(parts))
            if p not in (COLUMN_STOPWORDS | TRIVIAL_COL_PARTS) and any(
                    _soft_match(p, q) for q in q_tokens):
                return True
        else:
            col_phrase = col_low.replace('_', ' ')
            if col_phrase in q_lower:
                return True
    return False


def fk_neighbor_expansion(selected: list, question: str,
                           schema: dict, fk_graph: dict) -> list:
    """
    Text-based FK neighbor expansion — add a direct FK neighbor only if:
      (a) its table name appears in the question, OR
      (b) a non-structural column (not PK/FK) matches a question token.

    Junction tables are NOT added here — they are handled by Bridge BFS
    (Stage 4)
    """
    if not selected:
        return selected

    q_lower  = question.lower()
    q_tokens = _tokenize(q_lower)
    trivial  = COLUMN_STOPWORDS | TRIVIAL_COL_PARTS
    expanded = set(selected)

    candidates = _collect_fk_candidates(selected, schema, fk_graph, expanded)

    for nb in candidates:
        if (_filter_by_table_name(nb, q_lower) or
                _filter_by_column_text(nb, schema, q_lower, q_tokens, trivial)):
            expanded.add(nb)

    # preserve original order, then append new tables
    result = list(selected)
    for nb in candidates:
        if nb in expanded and nb not in result:
            result.append(nb)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — BRIDGE BFS
# ═════════════════════════════════════════════════════════════════════════════

def find_bridge_tables(selected: list, schema: dict, fk_graph: dict) -> list:
    """
    BFS to find junction/bridge tables that connect two or more selected
    tables via FK edges.  Runs iteratively until the selected set is stable.
    """
    selected_set = set(selected)
    changed = True

    while changed:
        changed = False
        candidates = set()
        for table in list(selected_set):
            for (_, neighbor, _) in fk_graph.get(table, []):
                if neighbor not in selected_set and neighbor in schema:
                    candidates.add(neighbor)

        for candidate in candidates:
            neighbors_in_selected = sum(
                1 for (_, nb, _) in fk_graph.get(candidate, [])
                if nb in selected_set
            )
            if neighbors_in_selected >= 2:
                selected_set.add(candidate)
                changed = True

    # preserve original order, then append bridge tables
    result = list(selected)
    for t in selected_set:
        if t not in result:
            result.append(t)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4.5 — COLUMN PRUNING
# ═════════════════════════════════════════════════════════════════════════════

def _build_structural_columns(info: dict) -> set:
    """Return the set of structural column names (PKs + FK source columns)."""
    pks = set(info.get('pks', []))
    fk_cols = {fc for (fc, _, _) in info.get('fks', [])}
    return pks | fk_cols


def _is_superlative_matched(question: str) -> bool:
    """
    Return True if the question contains a superlative or extremal keyword
    that implies an ORDER BY / LIMIT pattern.
    """
    superlatives = {
        'most', 'least', 'highest', 'lowest', 'largest', 'smallest',
        'greatest', 'fewest', 'maximum', 'minimum', 'max', 'min',
        'top', 'bottom', 'best', 'worst', 'first', 'last',
    }
    tokens = set(_tokenize(question.lower()))
    return bool(tokens & superlatives)


def _is_text_matched(col: str, col_type: str, q_lower: str,
                     q_tokens: set, trivial: set,
                     expanded_q: str) -> bool:
    """
    Return True if *col* has a meaningful text overlap with the question.
    Handles single-part and multi-part column names separately.
    """
    col_low = col.lower()
    col_phrase = col_low.replace('_', ' ')

    # Direct phrase match
    if col_phrase in q_lower or col_phrase in expanded_q:
        return True

    parts = set(re.split(r'[_\s]+', col_low)) - trivial
    if not parts:
        return False

    if len(parts) == 1:
        p = next(iter(parts))
        if p in COLUMN_STOPWORDS:
            return False
        return any(_soft_match(p, q) for q in q_tokens)

    # Multi-part: require phrase-level match (already checked above)
    return False


def _score_columns_with_ce(table: str, columns: list, question: str,
                            ce_model) -> dict:
    """
    Run cross-encoder scoring for all columns in *table* against *question*.
    Returns a dict mapping col_name → ce_score.
    """
    pairs = [(question, f"{table} {col}") for (col, _) in columns]
    scores = ce_model.predict(pairs)
    return {col: float(score) for (col, _), score in zip(columns, scores)}


def _apply_ce_pruning(columns: list, ce_scores: dict,
                      structural: set) -> list:
    """
    Apply cross-encoder gap pruning to *columns*, always keeping structural
    columns.  Returns the pruned column list.
    """
    if not ce_scores:
        return columns

    scored = [(col, typ, ce_scores.get(col, 0.0)) for (col, typ) in columns]
    scored.sort(key=lambda x: x[2], reverse=True)

    top_score = scored[0][2] if scored else 0.0
    threshold = max(top_score - COL_CE_GAP, COL_CE_FLOOR)

    kept = []
    for col, typ, score in scored:
        col_low = col.lower()
        if col_low in structural or score >= threshold:
            kept.append((col, typ))

    return kept


def _prune_by_text_match(columns: list, structural: set,
                          q_lower: str, q_tokens: set,
                          trivial: set, expanded_q: str) -> list:
    """
    Keep a column if it is structural OR has a text match with the question.
    Returns the filtered column list.
    """
    kept = []
    for (col, col_type) in columns:
        col_low = col.lower()
        if col_low in structural:
            kept.append((col, col_type))
            continue
        if _is_text_matched(col, col_type, q_lower, q_tokens,
                             trivial, expanded_q):
            kept.append((col, col_type))
    return kept


def _prune_by_superlative(columns: list, structural: set,
                           col_scores: dict) -> list:
    """
    When a superlative is detected, also keep the single highest-scored
    non-structural numeric/integer column (if not already kept).
    Returns the (possibly augmented) column list.
    """
    kept_names = {col for (col, _) in columns}
    best_col = None
    best_score = -1.0

    for col, typ in col_scores.items():
        col_low = col.lower() if isinstance(col, str) else col
        if col_low in structural or col_low in kept_names:
            continue
        score = col_scores.get(col, 0.0) if isinstance(col_scores, dict) else 0.0
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is not None:
        columns = list(columns) + [(best_col, 'number')]
    return columns


def _prune_single_table(table: str, info: dict, question: str,
                         q_lower: str, q_tokens: set,
                         trivial: set, expanded_q: str,
                         ce_model, use_superlative: bool) -> list:
    """
    Apply all pruning stages to a single table's columns and return the
    surviving (col, type) pairs.
    """
    columns = info['columns']
    structural = _build_structural_columns(info)

    # Stage A: text-match pruning
    text_kept = _prune_by_text_match(
        columns, structural, q_lower, q_tokens, trivial, expanded_q
    )

    # Stage B: cross-encoder scoring & gap pruning
    ce_scores = {}
    if ce_model is not None and columns:
        ce_scores = _score_columns_with_ce(table, columns, question, ce_model)
        ce_kept = _apply_ce_pruning(columns, ce_scores, structural)
    else:
        ce_kept = columns

    # Merge: keep union of text-kept and CE-kept, preserving original order
    kept_names = {col for (col, _) in text_kept} | {col for (col, _) in ce_kept}
    merged = [(col, typ) for (col, typ) in columns if col in kept_names]

    # Stage C: superlative augmentation
    if use_superlative and ce_scores:
        # Build a score lookup keyed by col name for _prune_by_superlative
        merged = _prune_by_superlative(merged, structural, ce_scores)

    # Always guarantee at least structural columns
    if not merged:
        merged = [(col, typ) for (col, typ) in columns
                  if col.lower() in structural]
    if not merged and columns:
        merged = [columns[0]]

    return merged


def stage_column_pruning(tables: list, question: str,
                          schema: dict, fk_graph: dict,
                          use_ce: bool = True) -> dict:
    """
    Stage 4.5 — Column pruning orchestrator.

    For each selected table, applies:
      1. Text-match pruning  (_prune_by_text_match)
      2. Cross-encoder gap pruning (_apply_ce_pruning)
      3. Superlative augmentation (_prune_by_superlative)

    Returns
    -------
    dict mapping table_name → [(col_name, col_type), ...]
    """
    q_lower    = question.lower()
    q_tokens   = _tokenize(q_lower)
    trivial    = COLUMN_STOPWORDS | TRIVIAL_COL_PARTS
    expanded_q = expand_query(question).lower()
    use_superlative = _is_superlative_matched(question)
    ce_model   = get_crossencoder() if use_ce else None

    result = {}
    for table in tables:
        if table not in schema:
            continue
        info = schema[table]
        result[table] = _prune_single_table(
            table, info, question,
            q_lower, q_tokens, trivial, expanded_q,
            ce_model, use_superlative,
        )

    return result