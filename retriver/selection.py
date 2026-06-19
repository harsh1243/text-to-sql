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
    (Stage 4) which discovers them as intermediate tables.
    """
    expanded = set(selected)
    candidates = _collect_fk_candidates(selected, schema, fk_graph, expanded)

    if not candidates:
        return selected

    q_lower  = question.lower()
    q_tokens = set(_tokenize(question))
    TRIVIAL  = {'id', 'num', 'no', 'is', 'has', 'the', 'a', 'an'}

    for nb in candidates:
        if _filter_by_table_name(nb, q_lower):
            expanded.add(nb)
        elif _filter_by_column_text(nb, schema, q_lower, q_tokens, TRIVIAL):
            expanded.add(nb)

    return [t for t in selected] + [t for t in expanded if t not in set(selected)]