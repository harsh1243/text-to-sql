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
    candidates = set()

    for table in selected:
        for (_, neighbor, _) in fk_graph.get(table, []):
            if neighbor in schema and neighbor not in expanded:
                candidates.add(neighbor)

    if not candidates:
        return selected

    q_lower  = question.lower()
    q_tokens = set(_tokenize(question))
    TRIVIAL  = {'id', 'num', 'no', 'is', 'has', 'the', 'a', 'an'}

    for nb in candidates:
        # (a) Table name appears in question as substring
        if nb in q_lower or nb.replace('_', ' ') in q_lower:
            expanded.add(nb)
            continue

        # (b) Non-structural columns match question tokens
        info = schema[nb]
        pks      = set(info.get('pks', []))
        fk_cols  = {fc for (fc, _, _) in info.get('fks', [])}
        structural = pks | fk_cols

        matched = False
        for (col, _) in info['columns']:
            col_low = col.lower()
            if col_low in structural:
                continue
            parts = set(re.split(r'[_\s]+', col_low)) - TRIVIAL
            if not parts:
                continue
            if len(parts) == 1:
                p = next(iter(parts))
                if p not in (COLUMN_STOPWORDS | TRIVIAL_COL_PARTS) and any(
                        _soft_match(p, q) for q in q_tokens):
                    matched = True
                    break
            else:
                col_phrase = col_low.replace('_', ' ')
                if col_phrase in q_lower:
                    matched = True
                    break
        if matched:
            expanded.add(nb)

    return list(expanded)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — BRIDGE BFS (iterative until stable)
# ═════════════════════════════════════════════════════════════════════════════

def find_bridge_tables(selected: list, fk_graph: dict, max_hops: int = 2) -> list:
    """
    For every pair of selected tables, BFS to find intermediate bridge tables.
    Runs in a while loop until no new tables are added — handles chains like
    A → B → C where B is discovered on one pass and needs C on the next.
    """
    selected_set = set(selected)

    def bfs_path(src, dst):
        if src == dst:
            return []
        visited = {src}
        queue   = deque([(src, [])])
        while queue:
            node, path = queue.popleft()
            for (_, nb, _) in fk_graph.get(node, []):
                if nb not in visited:
                    np2 = path + [nb]
                    if nb == dst:
                        return np2
                    if len(np2) < max_hops:
                        visited.add(nb)
                        queue.append((nb, np2))
        return None

    changed = True
    while changed:
        changed = False
        tables  = list(selected_set)
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                path = bfs_path(tables[i], tables[j])
                if path:
                    for b in path:
                        if b not in selected_set:
                            selected_set.add(b)
                            changed = True

    return list(selected_set)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4.5 — COLUMN PRUNING
# ═════════════════════════════════════════════════════════════════════════════

def stage_column_pruning(selected_tables: list, question: str,
                         schema: dict, synonyms: dict,
                         primary_tables: set = None) -> dict:
    """
    Column-level pruning (pure rule-based, no fine-tuning needed).

    For each selected table, keep only columns that are:
      (a) structural  — PK or FK endpoint (always kept)
      (b) COUNT-only  — skip non-structural entirely for COUNT(*) queries
      (c) text-match  — column's DISTINCTIVE parts fuzzy-match question tokens
      (d) value-match — proper nouns/years in question → categorical/int cols
      (e) superlative — "youngest/oldest/highest" → numeric cols for ORDER BY
      (f) cross-encoder scored above adaptive threshold (PRIMARY tables only)

    Supporting tables (bridge/FK-expansion) only get (a)+(c)+(d).
    """
    ce_model = get_crossencoder()
    expanded = expand_query(question, synonyms)
    q_lower  = question.lower()
    q_tokens = set(re.findall(r'\b\w+\b', q_lower))

    # ── Detect SQL operation type ─────────────────────────────────────────────
    AGGR_WORDS_SET = {'average', 'avg', 'maximum', 'minimum', 'max', 'min',
                      'sum', 'total'}
    ORDER_SUPERLATIVES = {
        'oldest', 'youngest', 'tallest', 'shortest', 'heaviest', 'lightest',
        'earliest', 'latest', 'newest', 'oldest', 'highest', 'lowest',
        'largest', 'smallest', 'biggest', 'most', 'least', 'best', 'worst',
        'first', 'last', 'top', 'bottom', 'greatest', 'fewest'
    }

    is_count_only = any(re.search(rf'\b{p}\b', q_lower)
                        for p in ['how many', 'count', 'number of', 'total number'])
    has_aggr      = bool(q_tokens & AGGR_WORDS_SET)
    has_order_sup = bool(q_tokens & ORDER_SUPERLATIVES)

    # Proper nouns in question (capitalized words after prepositions)
    proper_nouns = set(re.findall(
        r'\b(?:from|by|named?|called|in|at|of)\s+([A-Z][a-zA-Z]+)',
        question))   # use ORIGINAL question, not lowercased

    # ── Helpers ───────────────────────────────────────────────────────────────
    TRIVIAL = {'id', 'num', 'no', 'is', 'has', 'the', 'a', 'an'}

    def is_text_matched(col_orig: str) -> bool:
        col_low  = col_orig.lower()
        distinct = _get_distinctive_parts(col_low)
        all_parts = set(re.split(r'[_\s]+', col_low)) - TRIVIAL

        if not all_parts:
            return False

        if len(all_parts) == 1:
            p = next(iter(all_parts))
            if p not in COLUMN_STOPWORDS and len(p) > 2:
                return any(_soft_match(p, q) for q in expanded)
        else:
            if distinct:
                return any(
                    any(_soft_match(d, q) for q in expanded)
                    for d in distinct
                )
            return col_orig.lower().replace('_', ' ') in q_lower

        return False

    def is_value_matched(col_orig: str, col_type: str) -> bool:
        col_low   = col_orig.lower()
        col_parts = set(re.split(r'[_\s]+', col_low))

        if proper_nouns and col_type in ('text', 'varchar', 'char', 'string'):
            geo_hints = {'country', 'nation', 'location', 'city', 'state',
                         'region', 'place', 'area', 'continent', 'nationality',
                         'origin', 'home', 'birth', 'source'}
            if col_parts & geo_hints:
                return True

        if re.search(r'\b(19|20)\d{2}\b', question):
            time_hints = {'year', 'date', 'time', 'period', 'when', 'era'}
            primary_time = _get_distinctive_parts(col_low) & time_hints
            if primary_time and len(_get_distinctive_parts(col_low)) == 1:
                return True

        return False

    def is_superlative_matched(col_orig: str, col_type: str) -> bool:
        if not has_order_sup:
            return False
        if col_type not in ('int', 'integer', 'number', 'numeric', 'real', 'float'):
            return False
        col_low  = col_orig.lower()
        distinct = _get_distinctive_parts(col_low)
        if not distinct:
            return False
        if is_text_matched(col_orig):
            return False
        return True

    # ── Main pruning loop ─────────────────────────────────────────────────────
    is_primary = primary_tables if primary_tables is not None else set(selected_tables)
    pruned = {}

    for table in selected_tables:
        info = schema.get(table, {})
        pks  = set(info.get("pks", []))
        fk_source = {fc for (fc, _, _) in info.get("fks", [])}
        fk_target = set()
        for other in selected_tables:
            if other == table:
                continue
            for (_, rt, rc) in schema.get(other, {}).get("fks", []):
                if rt == table:
                    fk_target.add(rc)
        structural = pks | fk_source | fk_target

        kept     = []
        to_score = []

        for (col_orig, col_type) in info.get("columns", []):
            col_low = col_orig.lower()

            # (a) Always keep structural
            if col_low in structural:
                kept.append((col_orig, col_type))
                continue

            # (b) COUNT-only → skip all non-structural
            if is_count_only and not has_aggr:
                continue

            # (c) Text match on distinctive parts
            if is_text_matched(col_orig):
                kept.append((col_orig, col_type))
                continue

            # (d) Value match (proper nouns, years)
            if is_value_matched(col_orig, col_type):
                kept.append((col_orig, col_type))
                continue

            # (e) Superlative → implicit ORDER BY numeric col
            if table in is_primary and is_superlative_matched(col_orig, col_type):
                kept.append((col_orig, col_type))
                continue

            # (f) Cross-encoder scoring — primary tables only
            if table in is_primary:
                to_score.append((col_orig, col_type))

        # Cross-encoder for remaining candidates (primary tables)
        if to_score and table in is_primary:
            pairs  = [[question, f"{table} column {c.replace('_', ' ')}"]
                      for (c, _) in to_score]
            scores = ce_model.predict(pairs).tolist()

            sorted_desc = sorted(scores, reverse=True)
            threshold = COL_CE_FLOOR
            for k in range(1, len(sorted_desc)):
                if sorted_desc[k - 1] - sorted_desc[k] > COL_CE_GAP:
                    threshold = max(
                        (sorted_desc[k - 1] + sorted_desc[k]) / 2,
                        COL_CE_FLOOR)
                    break

            for idx, (col_orig, col_type) in enumerate(to_score):
                if scores[idx] >= threshold:
                    kept.append((col_orig, col_type))

        pruned[table] = kept

    return pruned
