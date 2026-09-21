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
# Config — override via env vars (Modal secrets) or the sidebar
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_URL = os.getenv(
    "PIPELINE_URL",
    "https://harsh1243--text-to-sql-pipeline.modal.run",
)
SINGLE_URL = os.getenv(
    "SINGLE_URL",
    "https://harsh1243--text-to-sql-single-web.modal.run",
)

# Proxy-auth headers. When deployed to Modal, set these as a Modal Secret
# named ``modal-proxy-auth`` with keys MODAL_KEY / MODAL_SECRET.
# Defaults below are committed in-repo so a Streamlit Cloud deploy works
# without configuring its own secrets UI. Override via env vars if needed.
MODAL_KEY = os.getenv("MODAL_KEY", "wk-7OTNaNa9rCSRY1o1iUYjDp")
MODAL_SECRET = os.getenv("MODAL_SECRET", "ws-wIg3aqnjVBTeQw0Tzwp9mN")

# (Warm-up now builds its question from the user's schema inside warm_up().)

MODELS = {
    "Dual transformer (T1→T2)": {
        "url": PIPELINE_URL,
        "kind": "pipeline",
        "blurb": "T1 plans, T2 writes SQL — best quality, ~30-50% slower.",
    },
    "Single transformer": {
        "url": SINGLE_URL,
        "kind": "single",
        "blurb": "One model emits plan and SQL — faster, slightly lower SQL F1.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def warm_up(model_label: str, schema: dict, fk_graph, key: str,
            secret: str) -> tuple[float, dict, str]:
    """Run a sample question through the user's *uploaded* schema end-to-end
    (retriever → model) so the warm-up output is meaningful to them, not
    a hardcoded singer-table sanity check.

    For the dual pipeline this single call wakes both Planner and Plan2Sql
    pools because ``pipeline`` calls each via ``.remote()`` internally.

    Returns (elapsed_seconds, response_json, question_used).
    """
    cfg = MODELS[model_label]
    # Generic question that works for any schema: pick the first table
    # so the model has something concrete to emit.
    first_table = next(iter(schema.keys())) if schema else "records"
    sample_question = f"How many rows are in {first_table}?"

    retriever_out = retrieve(sample_question, schema, fk_graph)
    model_input = retriever_out["model_input"]

    t0 = time.time()
    data = call_modal(cfg["url"], {"input": model_input}, key, secret)
    return time.time() - t0, data, sample_question


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Text-to-SQL",
    page_icon="🗃️",
    layout="wide",
)
st.title("Schema-Aware Text-to-SQL")
st.caption(
    "Upload a schema → warm the GPUs → ask questions in natural language. "
    "Each question is independent — nothing is stored between turns."
)

# Session-state defaults — survive reruns, cleared on schema reload.
st.session_state.setdefault("schema", None)
st.session_state.setdefault("fk_graph", None)
st.session_state.setdefault("schema_tables", [])
st.session_state.setdefault("warmed_model", None)
st.session_state.setdefault("warm_seconds", None)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Modal proxy auth + endpoint URLs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Modal credentials")
    st.caption(
        "Create at modal.com → Settings → Proxy Auth Tokens. "
        "On a deployed Modal UI these come from the `modal-proxy-auth` secret."
    )
    key = st.text_input(
        "Modal-Key (wk-…)",
        value=MODAL_KEY or st.session_state.get("modal_key", ""),
        type="password",
    )
    secret = st.text_input(
        "Modal-Secret (ws-…)",
        value=MODAL_SECRET or st.session_state.get("modal_secret", ""),
        type="password",
    )
    if key:
        st.session_state["modal_key"] = key
    if secret:
        st.session_state["modal_secret"] = secret

    with st.expander("Endpoint URLs"):
        st.code(PIPELINE_URL, language="text")
        st.code(SINGLE_URL, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema upload
# ─────────────────────────────────────────────────────────────────────────────

st.header("1. Upload schema")
schema_file = st.file_uploader(
    "schema.sql",
    type=["sql"],
    key="schema_file",
    help="The CREATE TABLE statements for the database you want to query.",
)

if schema_file is not None:
    text = schema_file.read().decode("utf-8", errors="replace")
    if retrieve is None:
        st.error(
            "Retriever not importable. Run from the repo root and install "
            "`sentence-transformers rank-bm25`."
        )
    else:
        try:
            with st.spinner(
                "Parsing schema and loading retriever models (first run "
                "downloads ~80 MB) …"
            ):
                schema = parse_schema(text)
                fk_graph = build_fk_graph(schema)
            st.session_state["schema"] = schema
            st.session_state["fk_graph"] = fk_graph
            st.session_state["schema_tables"] = list(schema.keys())
            # Reset warm state — different schema may need different tables
            # selected, but more importantly we want a clean "ready?" state.
            st.session_state["warmed_model"] = None
            st.session_state["warm_seconds"] = None

            n = len(schema)
            preview = ", ".join(list(schema.keys())[:5])
            extra = f" (+{n - 5} more)" if n > 5 else ""
            st.success(f"Parsed {n} tables: {preview}{extra}")
        except Exception as e:
            st.error(f"Failed to parse schema: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pick a model and warm up the GPUs
# ─────────────────────────────────────────────────────────────────────────────

st.header("2. Choose a model and warm up")

schema_ready = st.session_state["schema"] is not None
creds_ready = bool(key and secret)

model = st.radio(
    "Model",
    list(MODELS.keys()),
    index=0,
    horizontal=True,
    disabled=not schema_ready,
    help=MODELS[list(MODELS.keys())[0]]["blurb"]
    if schema_ready else "Upload a schema first.",
)

st.caption(MODELS[model]["blurb"])

col_warm, col_status = st.columns([1, 3])
with col_warm:
    warm_clicked = st.button(
        "Warm up GPUs",
        disabled=not schema_ready or not creds_ready,
        help="Sends a sample question so the first real query is fast. "
             "Cold start can take 30–90 s.",
    )

if warm_clicked:
    with st.spinner(f"Warming up {model} — cold start can take 30–90 s …"):
        try:
            secs, warm_data, warm_q = warm_up(
                model,
                st.session_state["schema"],
                st.session_state["fk_graph"],
                key, secret,
            )
            st.session_state["warmed_model"] = model
            st.session_state["warm_seconds"] = secs
            st.session_state["last_warm_data"] = warm_data
            st.session_state["last_warm_question"] = warm_q
            # Toast = top-right corner, very visible, auto-dismisses.
            st.toast(
                f"✅ {model} is warm and ready "
                f"(cold start took {secs:.1f}s — next call will be faster)",
                icon="🟢",
            )
        except Exception as e:
            st.toast(f"❌ Warm-up failed: {e}", icon="🔴")
            st.error(f"Warm-up failed: {e}")

if st.session_state["warmed_model"]:
    with col_status:
        warm_model = st.session_state["warmed_model"]
        secs = st.session_state["warm_seconds"]
        if warm_model == model:
            st.success(
                f"**Status: warm.** {warm_model} took {secs:.1f}s on the "
                f"warm-up call — your next question will be faster."
            )
        else:
            st.warning(
                f"⚠ Currently warm: **{warm_model}** ({secs:.1f}s ago). "
                f"You selected **{model}** — click **Warm up GPUs** again "
                f"to switch."
            )

    # Show what the model produced during warm-up as proof it's live.
    if "last_warm_data" in st.session_state:
        d = st.session_state["last_warm_data"]
        st.caption(
            f"Warm-up ran your uploaded schema through the retriever + "
            f"model with the question: *“{st.session_state.get('last_warm_question', '')}”*"
        )
        cp, cs = st.columns(2)
        with cp:
            st.caption("Warm-up output — Plan")
            st.code(d.get("plan", ""), language="text")
        with cs:
            st.caption("Warm-up output — SQL")
            st.code(d.get("sql", ""), language="sql")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ask a question — independent each time, no history stored
# ─────────────────────────────────────────────────────────────────────────────

st.header("3. Ask a question")

warm_ready = st.session_state["warmed_model"] == model
can_ask = schema_ready and warm_ready and creds_ready

question = st.text_input(
    "Question",
    placeholder="e.g. How many singers do we have?",
    disabled=not schema_ready,
)

generate = st.button(
    "Generate SQL",
    type="primary",
    disabled=not can_ask or not question,
    help="Disabled until a schema is loaded and the selected model is warm."
         if not can_ask else "Run the retriever and call the model.",
)

if generate:
    with st.spinner("Retrieving schema → calling model …"):
        try:
            t0 = time.time()
            retriever_out = retrieve(
                question,
                st.session_state["schema"],
                st.session_state["fk_graph"],
            )
            model_input = retriever_out["model_input"]
            retrieval_ms = (time.time() - t0) * 1000

            cfg = MODELS[model]
            data = call_modal(cfg["url"], {"input": model_input}, key, secret)
            total_ms = (time.time() - t0) * 1000
        except requests.HTTPError as e:
            st.error(
                f"Modal returned {e.response.status_code}. "
                f"Check that Modal-Key/Secret are correct and the endpoint "
                f"exists. Body: {e.response.text[:200]}"
            )
            data = None
        except Exception as e:
            st.error(f"Failed: {e}")
            data = None

    if data:
        plan = (data.get("plan") or "").strip()
        sql = (data.get("sql") or "").strip()
        model_secs = (total_ms - retrieval_ms) / 1000
        st.success(
            f"Done in {total_ms / 1000:.1f}s "
            f"(retriever {retrieval_ms / 1000:.1f}s, model {model_secs:.1f}s)"
        )

        col_plan, col_sql = st.columns(2)
        with col_plan:
            st.subheader("Execution plan")
            st.code(plan or "(no plan returned)", language="text")
        with col_sql:
            st.subheader("SQL")
            st.code(sql or "(no SQL returned)", language="sql")

        with st.expander("What the retriever sent to the model"):
            st.code(model_input, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# Footer hint
# ─────────────────────────────────────────────────────────────────────────────

if not schema_ready:
    st.info("Upload a `schema.sql` to start.")
elif not creds_ready:
    st.info("Add Modal-Key and Modal-Secret in the sidebar.")
elif not warm_ready:
    st.info(f"Click **Warm up GPUs** to load **{model}** before asking.")
