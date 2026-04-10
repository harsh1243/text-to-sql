"""
Schema Retriever — Multi-Stage Pipeline for Text-to-SQL
========================================================
Given a natural language question and a database schema (.sql file),
retrieve the relevant tables, columns, and foreign keys formatted
exactly as the training data expects:

    question: <NL> | schema: <table1> | <table2> | foreign keys: <fk1> ; <fk2>

Quick start::

    from retriever import retrieve
    from retriever.parser import parse_schema, build_fk_graph

    schema   = parse_schema(open("schema.sql").read())
    fk_graph = build_fk_graph(schema)

    result   = retrieve("How many singers do we have?", schema, fk_graph)
    print(result["model_input"])
"""

from .parser   import parse_schema, build_fk_graph
from .pipeline import retrieve

__all__ = ["parse_schema", "build_fk_graph", "retrieve"]
