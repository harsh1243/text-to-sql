"""
Schema Parser
==============
Stage 0 — Parse CREATE TABLE statements from a .sql file into a structured
dictionary, build an undirected FK graph, and classify table types.

Functions
---------
- parse_schema(sql_text)    → schema dict keyed by lowercase table name
- build_fk_graph(schema)    → undirected FK adjacency list
- classify_table_type(...)  → "entity" | "junction"
"""

import re
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers for parse_schema
# ═════════════════════════════════════════════════════════════════════════════

_KEYWORDS = {
    'primary', 'key', 'foreign', 'references', 'unique', 'check',
    'constraint', 'index', 'not', 'null', 'default',
    'insert', 'into', 'values', 'on', 'pragma'
}

_TABLE_BLOCK_RE = re.compile(
    r'CREATE\s+TABLE\s+[`"\']?(\w+)[`"\']?\s*\((.*?)\);',
    re.IGNORECASE | re.DOTALL
)
_PK_RE = re.compile(r'primary\s+key\s*\(([^)]+)\)', re.IGNORECASE)
_FK_RE = re.compile(
    r'foreign\s+key\s*\(([^)]+)\)\s+references\s+[`"\']?(\w+)[`"\']?\s*\(([^)]+)\)',
    re.IGNORECASE
)
_COL_RE = re.compile(r'^[`"\']?(\w+)[`"\']?\s+(\w+)', re.IGNORECASE)


def _extract_table_blocks(sql_text: str):
    """Return list of (table_name, body) tuples from CREATE TABLE statements."""
    return _TABLE_BLOCK_RE.findall(sql_text)


def _parse_fk(line: str):
    """
    Parse a FOREIGN KEY line.

    Returns
    -------
    tuple or None
        (from_col_lower, ref_table_lower, ref_col_lower) or None
    """
    m = _FK_RE.match(line)
    if not m:
        return None
    return (
        m.group(1).strip().strip('"\'`').lower(),
        m.group(2).strip().lower(),
        m.group(3).strip().strip('"\'`').lower(),
    )


def _parse_pk(line: str):
    """
    Parse a PRIMARY KEY constraint line.

    Returns
    -------
    list or None
        List of pk column names (lower) or None if not a PK line.
    """
    m = _PK_RE.match(line)
    if not m:
        return None
    return [c.strip().strip('"\'`').lower() for c in m.group(1).split(',')]


def _parse_column(line: str):
    """
    Parse a column definition line.

    Returns
    -------
    tuple or None
        (orig_name, type_str) or None if line is a keyword/constraint.
    """
    m = _COL_RE.match(line)
    if not m:
        return None
    cn, ct = m.group(1), m.group(2).lower()
    if cn.lower() in _KEYWORDS or ct in _KEYWORDS:
        return None
    return (cn, ct)


def _parse_table_body(body: str):
    """
    Parse the body of a CREATE TABLE statement.

    Returns
    -------
    tuple
        (columns, pks, fks) where:
          columns = [(orig_name, type_str), ...]
          pks     = [col_lower, ...]
          fks     = [(from_col_lower, ref_table_lower, ref_col_lower), ...]
    """
    columns, pks, fks = [], [], []

    for raw_line in body.split('\n'):
        line = raw_line.strip().rstrip(',').strip()
        if not line:
            continue

        fk = _parse_fk(line)
        if fk is not None:
            fks.append(fk)
            continue

        pk_cols = _parse_pk(line)
        if pk_cols is not None:
            pks.extend(pk_cols)
            continue

        col = _parse_column(line)
        if col is not None:
            columns.append(col)

    return columns, pks, fks


# ═════════════════════════════════════════════════════════════════════════════
# parse_schema
# ═════════════════════════════════════════════════════════════════════════════

def parse_schema(sql_text: str) -> dict:
    """
    Parse CREATE TABLE statements.

    Returns
    -------
    dict
        {
          table_lower: {
            "table_orig": str,
            "columns":    [(orig_name, type_str), ...],
            "pks":        [col_lower, ...],
            "fks":        [(from_col_lower, ref_table_lower, ref_col_lower), ...]
          }
        }
    """
    schema = {}
    for table_name, body in _extract_table_blocks(sql_text):
        table_low = table_name.lower()
        columns, pks, fks = _parse_table_body(body)
        schema[table_low] = {
            "table_orig": table_name,
            "columns":    columns,
            "pks":        pks,
            "fks":        fks,
        }
    return schema


# ═════════════════════════════════════════════════════════════════════════════
# build_fk_graph
# ═════════════════════════════════════════════════════════════════════════════

def build_fk_graph(schema: dict) -> dict:
    """
    Undirected FK graph — both FK direction and reverse stored.

    Returns
    -------
    dict
        graph[table] = [(from_col, neighbor_table, neighbor_col), ...]
    """
    graph = defaultdict(list)
    for table, info in schema.items():
        for (fc, rt, rc) in info["fks"]:
            graph[table].append((fc, rt, rc))
            graph[rt].append((rc, table, fc))   # reverse
    return dict(graph)


# ═════════════════════════════════════════════════════════════════════════════
# classify_table_type
# ═════════════════════════════════════════════════════════════════════════════

def classify_table_type(table: str, info: dict) -> str:
    """
    Classify table as 'entity' or 'junction'.

    Junction tables have ≤ 4 columns where most non-PK columns are FK sources
    (e.g. singer_in_concert with only concert_ID and Singer_ID).
    """
    columns = info.get("columns", [])
    pks = set(info.get("pks", []))
    fk_sources = {fk[0] for fk in info.get("fks", [])}

    if len(columns) > 4:
        return "entity"

    non_pk_cols = [c for c, _ in columns if c.lower() not in pks]
    if not non_pk_cols:
        return "entity"

    fk_ratio = sum(1 for c in non_pk_cols if c.lower() in fk_sources) / len(non_pk_cols)
    return "junction" if fk_ratio >= 0.5 else "entity"