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
    blocks = re.findall(
        r'CREATE\s+TABLE\s+[`"\']?(\w+)[`"\']?\s*\((.*?)\);',
        sql_text, re.IGNORECASE | re.DOTALL
    )
    KEYWORDS = {
        'primary', 'key', 'foreign', 'references', 'unique', 'check',
        'constraint', 'index', 'not', 'null', 'default',
        'insert', 'into', 'values', 'on', 'pragma'
    }
    for table_name, body in blocks:
        table_low = table_name.lower()
        columns, pks, fks = [], [], []

        for line in body.split('\n'):
            line = line.strip().rstrip(',').strip()
            if not line:
                continue

            pk_match = re.match(
                r'primary\s+key\s*\(([^)]+)\)', line, re.IGNORECASE)
            fk_match = re.match(
                r'foreign\s+key\s*\(([^)]+)\)\s+references\s+[`"\']?(\w+)[`"\']?\s*\(([^)]+)\)',
                line, re.IGNORECASE)
            col_match = re.match(
                r'^[`"\']?(\w+)[`"\']?\s+(\w+)', line, re.IGNORECASE)

            if fk_match:
                fks.append((
                    fk_match.group(1).strip().strip('"\'`').lower(),
                    fk_match.group(2).strip().lower(),
                    fk_match.group(3).strip().strip('"\'`').lower(),
                ))
            elif pk_match:
                pks.extend([c.strip().strip('"\'`').lower()
                             for c in pk_match.group(1).split(',')])
            elif col_match:
                cn, ct = col_match.group(1), col_match.group(2).lower()
                if cn.lower() not in KEYWORDS and ct not in KEYWORDS:
                    columns.append((cn, ct))

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
    fk_sources = {fc for (fc, _, _) in info.get("fks", [])}
    if len(columns) <= 1:
        return "entity"
    non_pk_cols = [c for (c, _) in columns if c.lower() not in pks]
    if not non_pk_cols:
        return "junction" if len(columns) <= 4 else "entity"
    fk_ratio = sum(1 for c in non_pk_cols if c.lower() in fk_sources) / len(non_pk_cols)
    return "junction" if len(columns) <= 4 and fk_ratio >= 0.5 else "entity"
