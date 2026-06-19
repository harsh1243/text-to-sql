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


def _get_table_columns(table: str, schema: dict, pruned_columns: dict) -> list:
    """Return the effective column list for a table, respecting pruned_columns if provided."""
    info = schema.get(table, {})
    if pruned_columns and table in pruned_columns:
        return pruned_columns[table]
    return info.get("columns", [])


def _build_output_cols(selected_tables: list, schema: dict,
                       pruned_columns: dict) -> set:
    """Build set of (table, col_lower) pairs that are actually output in schema_str."""
    output_cols = set()
    for table in selected_tables:
        columns = _get_table_columns(table, schema, pruned_columns)
        for (col_orig, _) in columns:
            output_cols.add((table, col_orig.lower()))
    return output_cols


def _format_table_columns(table: str, schema: dict, pruned_columns: dict) -> str:
    """Format a single table's columns into a schema line string."""
    info = schema.get(table, {})
    columns = _get_table_columns(table, schema, pruned_columns)
    col_parts = []
    for (col_orig, _) in columns:
        tag = " [PK]" if col_orig.lower() in info.get("pks", []) else ""
        col_parts.append(f"{col_orig}{tag}")
    return f"{table} ( {', '.join(col_parts)} )"


def _build_schema_lines(selected_tables: list, schema: dict,
                        pruned_columns: dict) -> list:
    """Build list of formatted schema lines, one per selected table."""
    return [
        _format_table_columns(table, schema, pruned_columns)
        for table in selected_tables
    ]


def _build_fk_lines(selected_tables: list, schema: dict,
                    selected_set: set, output_cols: set) -> list:
    """Build list of FK relationship strings filtered to selected tables and output columns."""
    fk_lines, seen = [], set()
    for table in selected_tables:
        for (fc, rt, rc) in schema.get(table, {}).get("fks", []):
            if rt not in selected_set:
                continue
            if (table, fc) not in output_cols:
                continue
            if (rt, rc) not in output_cols:
                continue
            fc_orig = _orig_col(schema, table, fc)
            rc_orig = _orig_col(schema, rt, rc)
            entry = f"{table}.{fc_orig} -> {rt}.{rc_orig}"
            if entry not in seen:
                fk_lines.append(entry)
                seen.add(entry)
    return fk_lines


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
    output_cols = _build_output_cols(selected_tables, schema, pruned_columns)
    schema_lines = _build_schema_lines(selected_tables, schema, pruned_columns)
    fk_lines = _build_fk_lines(selected_tables, schema, selected_set, output_cols)

    schema_str = "\n".join(schema_lines)
    fk_str = "\n".join(fk_lines) if fk_lines else "none"
    return schema_str, fk_str


def build_input_string(question: str, schema_str: str, fk_str: str) -> str:
    """
    Format the final model input string matching the training data format:

        question: <NL question> | schema: <table1> | <table2> | foreign keys: <fk1> ; <fk2>
    """
    schema_flat = " | ".join(schema_str.split("\n"))
    fk_flat = " ; ".join(fk_str.split("\n")) if fk_str != "none" else "none"
    return f"question: {question} | schema: {schema_flat} | foreign keys: {fk_flat}"