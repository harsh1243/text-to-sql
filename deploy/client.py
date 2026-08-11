"""
Client for the Deployed Modal Endpoints
========================================
Calls the deployed models over Modal's own RPC rather than HTTP, which avoids
the 150-second web-endpoint limit that a cold start can exceed.

Auth comes from ``~/.modal.toml`` (written by ``modal setup``) or from the
``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` environment variables.  The
proxy-auth tokens are only needed for raw HTTP calls, not for this path.

USAGE
-----
    # full dual pipeline from a raw question + a schema .sql file
    python deploy/client.py --question "How many singers do we have?" \
                            --schema-file path/to/schema.sql

    # single-transformer model instead of the dual pipeline
    python deploy/client.py --question "..." --schema-file ... --model single

    # skip the retriever, pass a pre-built input string
    python deploy/client.py --input "question: ... | schema: ... | foreign keys: none"

    # plan -> SQL only
    python deploy/client.py --model plan2sql --input "step1: SCAN | table: singer || ..."

The retriever runs locally and costs nothing; only generation touches the GPU.
"""

import argparse
import json
import sys

import modal

APP_NAME = "text-to-sql"


def _cls(name: str):
    """Look up a deployed class.  Fails clearly if not deployed yet."""
    try:
        return modal.Cls.from_name(APP_NAME, name)
    except Exception as exc:
        sys.exit(f"could not find {APP_NAME}/{name} — deploy first with:\n"
                 f"    modal deploy deploy/modal_app.py\n({exc})")


def build_input(question: str, schema_file: str) -> str:
    """Run the local retriever to produce the model's ``input`` string."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retriver import retrieve
    from retriver.parser import parse_schema, build_fk_graph

    with open(schema_file, encoding="utf-8") as f:
        sql_text = f.read()

    schema = parse_schema(sql_text)
    if not schema:
        sys.exit(f"parsed 0 tables from {schema_file} — is it CREATE TABLE DDL?")
    fk_graph = build_fk_graph(schema)

    result = retrieve(question, schema, fk_graph)
    print(f"retriever selected: {result['selected_tables']}\n")
    return result["model_input"]


def main():
    ap = argparse.ArgumentParser(description="Call the deployed Modal models.")
    ap.add_argument("--model", default="pipeline",
                    choices=["pipeline", "planner", "plan2sql", "single"],
                    help="pipeline = T1 then T2 (default)")
    ap.add_argument("--question", help="Natural language question")
    ap.add_argument("--schema-file", help="A .sql file of CREATE TABLE statements")
    ap.add_argument("--input", dest="raw_input",
                    help="Pre-built input string (skips the retriever)")
    args = ap.parse_args()

    if args.raw_input:
        text = args.raw_input
    elif args.question and args.schema_file:
        text = build_input(args.question, args.schema_file)
    elif args.question:
        ap.error("--question also needs --schema-file (or use --input)")
    else:
        ap.error("pass --input, or --question with --schema-file")

    print(f"input: {text}\n")
    print("calling Modal (a cold start loads 11 GB — first call is slow) ...\n")

    if args.model == "planner":
        out = {"plan": _cls("Planner")().plan.remote(text)}

    elif args.model == "plan2sql":
        out = {"sql": _cls("Plan2Sql")().to_sql.remote(text)}

    elif args.model == "single":
        out = _cls("Single")().run.remote(text)

    else:
        plan = _cls("Planner")().plan.remote(text)
        print(f"plan: {plan}\n")
        out = {"plan": plan, "sql": _cls("Plan2Sql")().to_sql.remote(plan)}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
