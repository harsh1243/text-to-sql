"""
Pipeline Orchestrator
======================
Main `retrieve()` function that chains all 5 stages together.

Usage::

    from retriever import retrieve
    from retriever.parser import parse_schema, build_fk_graph

    schema   = parse_schema(sql_text)   # once per DB
    fk_graph = build_fk_graph(schema)   # once per DB

    result = retrieve(question, schema, fk_graph)
    model_input = result["model_input"]
"""

from .scoring import (
    stage1_fusion, stage2_crossencoder,
    expand_query, _detect_query_type,   # expand_query restored in scoring.py
)
from .selection import (
    adaptive_select, fk_neighbor_expansion,
    find_bridge_tables, stage_column_pruning,
)
from .formatter import format_schema_output, build_input_string


def retrieve(
    question:          str,
    schema:            dict,   # from parse_schema()
    fk_graph:          dict,   # from build_fk_graph()
    synonyms:          dict  = None,   # optional domain-specific synonym map
    use_cross_encoder: bool  = True,
    verbose:           bool  = False,
) -> dict:
    """
    Full 5-stage retrieval pipeline.

    Call parse_schema() + build_fk_graph() ONCE per database,
    then pass them here for every question.

    Args
    ----
    question          : Natural language question.
    schema            : Parsed schema dict from parse_schema().
    fk_graph          : FK adjacency list from build_fk_graph().
    synonyms          : Optional dict mapping NL words to schema-relevant
                        aliases, e.g. {"venue": ["stadium", "arena"]}.
                        Defaults to no expansion — works on any schema
                        without hardcoded assumptions.
    use_cross_encoder : True = more accurate, slightly slower.
                        False = fusion score only, faster.
    verbose           : Print stage-by-stage debug output.

    Returns
    -------
    dict with keys:
        model_input     — formatted string matching training data format
        selected_tables — list of final table names
        schema_str      — multi-line schema (for inspection)
        fk_str          — multi-line foreign keys (for inspection)
        debug           — per-stage debug info (candidates, scores, etc.)
    """
    if synonyms is None:
        synonyms = {}

    # Stage 1: fusion scoring → top 6 candidates
    candidates = stage1_fusion(question, schema, synonyms, top_k_candidates=6)

    # Stage 2: cross-encoder reranking
    if use_cross_encoder and len(candidates) > 1:
        ranked = stage2_crossencoder(question, candidates, schema)
    else:
        ranked = [(name, score) for (name, score, *_) in candidates]

    # Stage 3: adaptive threshold
    selected = adaptive_select(ranked)
    primary_tables = set(selected)   # tables chosen by scoring, not expansion

    # Stage 3.5: FK neighbor expansion
    selected_expanded = fk_neighbor_expansion(selected, question, schema, fk_graph)

    # Stage 4: bridge BFS (iterative until stable)
    final_tables = find_bridge_tables(selected_expanded, fk_graph)

    # Stage 4.5: column pruning — primary tables get CE columns,
    #            supporting tables only get structural + text-matched
    pruned_columns = stage_column_pruning(
        final_tables, question, schema, synonyms, primary_tables)

    # Stage 5: format
    schema_str, fk_str = format_schema_output(final_tables, schema, pruned_columns)
    model_input = build_input_string(question, schema_str, fk_str)

    if verbose:
        print("\n" + "="*65)
        print(f"QUESTION : {question}")
        # FIX Bug 4: _detect_query_type now takes (question, schema) only
        qtype = _detect_query_type(question, schema)
        print(f"Query type: {qtype}")
        print(f"\nStage 1 — Fusion (top candidates):")
        # FIX Bug 3: tuple is now (name, fusion, bi, bm) — 4 elements not 6
        for name, fs, bi, bm in candidates:
            print(f"  {name:<25} fusion={fs:.3f}  bi={bi:.3f}  bm25={bm:.3f}")
        if use_cross_encoder:
            print(f"\nStage 2 — Cross-encoder reranked:")
            for name, cs in ranked:
                print(f"  {name:<25} cross={cs:.3f}")
        print(f"\nStage 3  selected (scoring)  : {selected}")
        print(f"Stage 3.5 after FK expansion : {selected_expanded}")
        print(f"Stage 4  after BFS           : {final_tables}")
        print(f"\nStage 4.5 columns kept:")
        for t, cols in pruned_columns.items():
            print(f"  {t}: {[c for c, _ in cols]}")
        print(f"\nSCHEMA:\n{schema_str}")
        print(f"\nFOREIGN KEYS:\n{fk_str}")
        print(f"\nMODEL INPUT:\n{model_input}")
        print("="*65)

    return {
        "model_input":     model_input,
        "selected_tables": final_tables,
        "schema_str":      schema_str,
        "fk_str":          fk_str,
        "debug": {
            "stage1":          candidates,
            "ranked":          ranked,
            "stage3":          selected,
            "pruned_columns":  pruned_columns,
        }
    }
