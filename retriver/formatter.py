"""
Output Formatter
=================
Stage 5 — Format selected tables + pruned columns into the model input string
that matches the training data format:

    question: <NL question> | schema: <table1> | <table2> | foreign keys: <fk1> ; <fk2>

Functions
---------
- format_schema_output(tables, schema, pruned) → (schema_str, fk_str)
- build_input_string(question, schema_str, fk_str) → model_input
"""


def _orig_col(schema: dict, table: str, col_low: str) -> str:
    """Look up original-cased column name from lowercase key."""
    for (cn, _) in schema.get(table, {}).get("columns", []):
        if cn.lower() == col_low:
            return cn
    return col_low


def format_schema_output(selected_tables: list, schema: dict,
                         pruned_columns: dict = None) -> tuple:
    """
    Returns (schema_str, fk_str).

    schema_str — one table per line, original casing, [PK] tagged.
    fk_str     — only FKs where:
                   (1) both endpoint tables are in selected_tables, AND
                   (2) both endpoint columns are present in pruned_columns
                       (so FK line is only emitted if cols actually appear in schema_str)
    or "none"
    """
    selected_set = set(selected_tables)

    # Build set of (table, col_lower) pairs that are actually output
    # so FK emission is consistent with what's shown in schema_str
    output_cols = set()
    for table in selected_tables:
        info    = schema.get(table, {})
        columns = (pruned_columns[table]
                   if pruned_columns and table in pruned_columns
                   else info.get("columns", []))
        for (col_orig, _) in columns:
            output_cols.add((table, col_orig.lower()))

    schema_lines = []
    for table in selected_tables:
        info    = schema.get(table, {})
        columns = (pruned_columns[table]
                   if pruned_columns and table in pruned_columns
                   else info.get("columns", []))
        col_parts = []
        for (col_orig, _) in columns:
            tag = " [PK]" if col_orig.lower() in info.get("pks", []) else ""
            col_parts.append(f"{col_orig}{tag}")
        schema_lines.append(f"{table} ( {', '.join(col_parts)} )")

    fk_lines, seen = [], set()
    for table in selected_tables:
        for (fc, rt, rc) in schema.get(table, {}).get("fks", []):
            if rt not in selected_set:
                continue
            # Only emit if both endpoint cols are actually in output
            if (table, fc) not in output_cols:
                continue
            if (rt, rc) not in output_cols:
                continue
            fc_orig = _orig_col(schema, table, fc)
            rc_orig = _orig_col(schema, rt,    rc)
            entry   = f"{table}.{fc_orig} -> {rt}.{rc_orig}"
            if entry not in seen:
                fk_lines.append(entry)
                seen.add(entry)

    schema_str = "\n".join(schema_lines)
    fk_str     = "\n".join(fk_lines) if fk_lines else "none"
    return schema_str, fk_str


def build_input_string(question: str, schema_str: str, fk_str: str) -> str:
    """
    Format the final model input string matching the training data format:

        question: <NL question> | schema: <table1> | <table2> | foreign keys: <fk1> ; <fk2>
    """
    schema_flat = " | ".join(schema_str.split("\n"))
    fk_flat     = " ; ".join(fk_str.split("\n")) if fk_str != "none" else "none"
    return f"question: {question} | schema: {schema_flat} | foreign keys: {fk_flat}"
