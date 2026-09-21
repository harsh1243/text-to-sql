"""
Streamlit UI for the Schema-Aware Text-to-SQL project.

Calls the deployed Modal endpoints for inference. Runs the schema retriever
locally to build the ``input`` string the endpoints consume.

Run locally:
    pip install streamlit requests sentence-transformers rank-bm25
    streamlit run streamlit_app.py

Deploy on Modal (see deploy/modal_streamlit.py):
    modal deploy deploy/modal_streamlit.py
"""

import os
import sys
import time
from pathlib import Path

# Make the `retriver` package importable when this file is launched from
# anywhere in the repo (local dev or Modal image).
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import requests
import streamlit as st

try:
    from retriver import parse_schema, build_fk_graph, retrieve
except ImportError:
    parse_schema = build_fk_graph = retrieve = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Dual pipeline endpoint — T1 plans, T2 writes SQL.
ENDPOINT_URL = os.getenv(
    "ENDPOINT_URL",
    "https://harsh1243--text-to-sql-pipeline.modal.run",
)

# Proxy-auth headers. Defaults below are committed in-repo so a Streamlit
# Cloud deploy works without configuring its own secrets UI. Override via
# env vars if needed.
MODAL_KEY = os.getenv("MODAL_KEY", "wk-7OTNaNa9rCSRY1o1iUYjDp")
MODAL_SECRET = os.getenv("MODAL_SECRET", "ws-wIg3aqnjVBTeQw0Tzwp9mN")


def call_modal(payload: dict, key: str, secret: str, timeout: int = 600) -> dict:
    """POST to the Modal endpoint with proxy auth. Follows redirects because
    Modal returns 303 if a cold start outstays 150 s."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Modal-Key"] = key
    if secret:
        headers["Modal-Secret"] = secret
    r = requests.post(ENDPOINT_URL, json=payload, headers=headers,
                      allow_redirects=True, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Text-to-SQL", page_icon="🗃️", layout="wide")
st.title("Text-to-SQL")

# Modal credentials — sidebar
with st.sidebar:
    key = st.text_input("Modal-Key (wk-…)",
                        value=MODAL_KEY or st.session_state.get("modal_key", ""),
                        type="password")
    secret = st.text_input("Modal-Secret (ws-…)",
                          value=MODAL_SECRET or st.session_state.get("modal_secret", ""),
                          type="password")
    if key:
        st.session_state["modal_key"] = key
    if secret:
        st.session_state["modal_secret"] = secret


# 1. Schema upload
schema_file = st.file_uploader("Upload schema.sql", type=["sql"])
schema_ready = False
if schema_file is not None:
    text = schema_file.read().decode("utf-8", errors="replace")
    if retrieve is None:
        st.error("Retriever not importable. Run from the repo root and install "
                 "`sentence-transformers rank-bm25`.")
    else:
        try:
            with st.spinner("Parsing schema…"):
                schema = parse_schema(text)
                fk_graph = build_fk_graph(schema)
            st.session_state["schema"] = schema
            st.session_state["fk_graph"] = fk_graph
            n = len(schema)
            preview = ", ".join(list(schema.keys())[:5])
            extra = f" (+{n - 5} more)" if n > 5 else ""
            st.success(f"Parsed {n} tables: {preview}{extra}")
            schema_ready = True
        except Exception as e:
            st.error(f"Failed to parse schema: {e}")


# 2. Question + Generate SQL
question = st.text_input("Question",
                          placeholder="e.g. What are the names of authors who have written papers in the 'Database' domain?",
                          disabled=not schema_ready)

creds_ready = bool(key and secret)
generate = st.button("Generate SQL",
                     type="primary",
                     disabled=not schema_ready or not creds_ready)

if generate:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Running retriever + model — first request may take 30–90 s for cold start."):
            try:
                retriever_out = retrieve(
                    question,
                    st.session_state["schema"],
                    st.session_state["fk_graph"],
                )
                model_input = retriever_out["model_input"]
                data = call_modal({"input": model_input}, key, secret)
            except requests.HTTPError as e:
                st.error(f"Modal returned {e.response.status_code}. "
                         f"Body: {e.response.text[:200]}")
                data = None
            except Exception as e:
                st.error(f"Failed: {e}")
                data = None

        if data:
            plan = (data.get("plan") or "").strip()
            sql = (data.get("sql") or "").strip()
            col_plan, col_sql = st.columns(2)
            with col_plan:
                st.subheader("Execution plan")
                st.code(plan or "(no plan returned)", language="text")
            with col_sql:
                st.subheader("SQL")
                st.code(sql or "(no SQL returned)", language="sql")
