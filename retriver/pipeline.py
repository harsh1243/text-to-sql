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


def _validate_and_init(synonyms):
    """Validate and initialize default arguments."""
    if synonyms is None:
        return {}
    return synonyms


def _fetch_candidates(question, schema, synonyms, use_cross_encoder):
    """Stage 1 & 2: Fetch and rank candidates via fusion and optional cross-encoder."""
    candidates = stage1_fusion(question, schema, synonyms, top_k_candidates=6)

    if use_cross_encoder and len(candidates) > 1:
        ranked = stage2_crossencoder(question, candidates, schema)
    else:
        ranked = [(name, score) for (name, score, *_) in candidates]

    return candidates, ranked


def _expand_and_bridge(ranked, question, schema, fk_graph):
    """Stage 3 & 4: Select, expand via FK neighbors, and find bridge tables."""
    selected = adaptive_select(ranked)
    primary_tables = set(selected)

    selected_expanded = fk_neighbor_expansion(selected, question, schema, fk_graph)
    final_tables = find_bridge_tables(selected_expanded, fk_graph)

    return selected, primary_tables, final_tables


def _prune_and_format(final_tables, question, schema, synonyms, primary_tables):
    """Stage 4.5 & 5: Prune columns and format the schema output."""
    pruned_columns = stage_column_pruning(
        final_tables, question, schema, synonyms, primary_tables)

    schema_str, fk_str = format_schema_output(final_tables, schema, pruned_columns)
    model_input = build_input_string(question, schema_str, fk_str)

    return pruned_columns, schema_str, fk_str, model_input


def _print_verbose(question, schema, candidates, ranked, selected,
                   primary_tables, final_tables, pruned_columns,
                   schema_str, fk_str, use_cross_encoder):
    """Print stage-by-stage debug output when verbose=True."""
    print("\n" + "="*65)
    print(f"QUESTION : {question}")
    qtype = _detect_query_type(question, schema)
    print(f"Query type: {qtype}")
    print(f"\nStage 1 — Fusion (top candidates):")
    for name, fs, bi, bm in candidates:
        print(f"  {name:<25} fusion={fs:.3f}  bi={bi:.3f}  bm25={bm:.3f}")
    if use_cross_encoder:
        print(f"\nStage 2 — Cross-encoder reranked:")
        for name, score in ranked:
            print(f"  {name:<25} ce_score={score:.3f}")
    print(f"\nStage 3 — Adaptive select: {selected}")
    print(f"Stage 3.5 — FK expanded:   {list(primary_tables)}")
    print(f"Stage 4 — Bridge tables:   {final_tables}")
    print(f"\nStage 4.5 — Pruned columns:")
    for tbl, cols in pruned_columns.items():
        print(f"  {tbl}: {cols}")
    print(f"\nStage 5 — Schema:\n{schema_str}")
    print(f"FKs:\n{fk_str}")
    print("="*65 + "\n")


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
    synonyms = _validate_and_init(synonyms)

    candidates, ranked = _fetch_candidates(question, schema, synonyms, use_cross_encoder)

    selected, primary_tables, final_tables = _expand_and_bridge(
        ranked, question, schema, fk_graph)

    pruned_columns, schema_str, fk_str, model_input = _prune_and_format(
        final_tables, question, schema, synonyms, primary_tables)

    if verbose:
        _print_verbose(
            question, schema, candidates, ranked, selected,
            primary_tables, final_tables, pruned_columns,
            schema_str, fk_str, use_cross_encoder,
        )

    return {
        "model_input":     model_input,
        "selected_tables": final_tables,
        "schema_str":      schema_str,
        "fk_str":          fk_str,
        "debug": {
            "candidates":     candidates,
            "ranked":         ranked,
            "selected":       selected,
            "primary_tables": primary_tables,
            "final_tables":   final_tables,
            "pruned_columns": pruned_columns,
        },
    }