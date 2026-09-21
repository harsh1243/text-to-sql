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

PIPELINE_URL = os.getenv(
    "PIPELINE_URL",
    "https://harsh1243--text-to-sql-pipeline.modal.run",
)
SINGLE_URL = os.getenv(
    "SINGLE_URL",
    "https://harsh1243--text-to-sql-single-web.modal.run",
)

# Proxy-auth headers. Defaults below are committed in-repo so a Streamlit
# Cloud deploy works without configuring its own secrets UI.
MODAL_KEY = os.getenv("MODAL_KEY", "wk-7OTNaNa9rCSRY1o1iUYjDp")
MODAL_SECRET = os.getenv("MODAL_SECRET", "ws-wIg3aqnjVBTeQw0Tzwp9mN")

MODELS = {
    "Dual transformer (T1→T2)": PIPELINE_URL,
    "Single transformer": SINGLE_URL,
}

# Generic question used to wake the containers. Picked so it works for any
# schema — the retriever still does the full table/column selection.
_WARM_QUESTION = "How many records are there in total?"


def call_modal(url: str, payload: dict, key: str, secret: str,
               timeout: int = 600) -> dict:
    """POST to a Modal endpoint with proxy auth. Follows redirects because
    Modal returns 303 if a cold start outstays 150 s."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Modal-Key"] = key
    if secret:
        headers["Modal-Secret"] = secret
    r = requests.post(url, json=payload, headers=headers,
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
            # Reset warm state — schema changed, model needs to re-warm.
            st.session_state["warmed_model"] = None
            n = len(schema)
            preview = ", ".join(list(schema.keys())[:5])
            extra = f" (+{n - 5} more)" if n > 5 else ""
            st.success(f"Parsed {n} tables: {preview}{extra}")
            schema_ready = True
        except Exception as e:
            st.error(f"Failed to parse schema: {e}")


# 2. Pick a model
creds_ready = bool(key and secret)
model_label = st.radio("Model", list(MODELS.keys()), horizontal=True,
                       disabled=not schema_ready)
model_url = MODELS[model_label]

warm_status = st.session_state.get("warmed_model") == model_label


# 3. Warm up
warm_clicked = st.button(
    "Warm up GPU",
    disabled=not schema_ready or not creds_ready or warm_status,
)

if warm_clicked:
    with st.spinner(f"Warming up {model_label} — cold start can take 30–90 s…"):
        try:
            retriever_out = retrieve(
                _WARM_QUESTION,
                st.session_state["schema"],
                st.session_state["fk_graph"],
            )
            call_modal(
                model_url,
                {"input": retriever_out["model_input"]},
                key, secret,
            )
            st.session_state["warmed_model"] = model_label
            st.session_state["warmed_url"] = model_url
        except Exception as e:
            st.error(f"Warm-up failed: {e}")

# Live status line — only one of these shows at a time.
if st.session_state.get("warmed_model") == model_label:
    st.write(f"Status: **{model_label} is warm**.")
elif st.session_state.get("warmed_model"):
    st.write(f"Status: currently warm = **{st.session_state['warmed_model']}**. "
             f"Click **Warm up GPU** again to switch.")


# 4. Question + Generate SQL
question = st.text_input(
    "Question",
    placeholder="Type a question, then click Generate SQL.",
    disabled=not warm_status,
)

generate = st.button(
    "Generate SQL",
    type="primary",
    disabled=not warm_status or not creds_ready,
)

if generate:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Running retriever + model…"):
            try:
                retriever_out = retrieve(
                    question,
                    st.session_state["schema"],
                    st.session_state["fk_graph"],
                )
                model_input = retriever_out["model_input"]
                data = call_modal(
                    st.session_state["warmed_url"],
                    {"input": model_input},
                    key, secret,
                )
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
