"""
Modal Serverless Deployment — Text-to-SQL LoRA Models
======================================================
Deploys all three fine-tuned Flan-T5-XL + LoRA adapters as on-demand
serverless GPU endpoints on Modal (https://modal.com).

Containers scale to zero when idle.  You are billed only while a request is
being served, plus a short idle window (``SCALEDOWN_WINDOW``) that keeps the
model warm for follow-up calls.

Three models, three independent container pools
-----------------------------------------------
    Planner    question + schema  ->  execution plan     (T1 of dual pipeline)
    Plan2Sql   execution plan     ->  SQL                (T2 of dual pipeline)
    Single     question + schema  ->  plan [SQL] sql     (single transformer)

Each mirrors its notebook in ``experiments/`` exactly: same base model, same
dtype rule, same beam-search settings, same tokenizer source.  The schema
retriever is NOT deployed here — it runs locally for free and produces the
``input`` string these endpoints consume.

ONE-TIME SETUP
--------------
    pip install modal
    modal setup

    # 1. Pull flan-t5-xl (~11 GB) from HuggingFace into the Volume.
    #    Runs cloud-to-cloud; nothing crosses your own connection.
    modal run deploy/modal_app.py::download_base

    # 2. Push the three unzipped LoRA adapters from your machine.
    python deploy/upload_adapters.py --local-dir ./lora_zips

    # 3. Create a proxy-auth token pair in the Modal dashboard
    #    (Settings -> Proxy Auth Tokens).  Unauthenticated requests are
    #    rejected at Modal's edge BEFORE a GPU starts, so a stray request
    #    or bot scan cannot burn your credits.

DEPLOY
------
    modal deploy deploy/modal_app.py

CALL
----
    curl -L -X POST \
      -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
      -H 'Content-Type: application/json' \
      -d '{"input": "question: How many singers do we have? | schema: singer ( Singer_ID [PK] ) | foreign keys: none"}' \
      https://<workspace>--text-to-sql-planner-web.modal.run

The ``-L`` is not optional: Modal returns a 303 redirect if a request
outstays 150 s, which a cold start can.  ``deploy/client.py`` calls over
Modal's own RPC instead and has no such limit.
"""

import modal

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

BASE_REPO    = "google/flan-t5-xl"
MODEL_DIR    = "/models"
BASE_DIR     = f"{MODEL_DIR}/flan-t5-xl"
ADAPTER_ROOT = f"{MODEL_DIR}/adapters"

GPU  = "L40S"                 # $0.000542/sec ~= $1.95/hr
SCALEDOWN_WINDOW = 900        # idle seconds before scale-to-zero (15 min)
FN_TIMEOUT       = 900        # generous: covers an 11 GB cold load + beams

# Adapter subdirectory names inside ADAPTER_ROOT.  upload_adapters.py writes
# these; keep the two files in agreement.
ADAPTER_PLANNER  = "planner"
ADAPTER_PLAN2SQL = "plan2sql"
ADAPTER_SINGLE   = "single"

volume = modal.Volume.from_name("t2s-model-weights", create_if_missing=True)

# Versions pinned to the notebooks so generation behaviour is identical.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.40.0",
        "peft==0.10.0",
        "accelerate==0.29.3",
        "sentencepiece==0.2.0",
        "huggingface_hub==0.23.0",
        "fastapi[standard]==0.115.4",
    )
)

app = modal.App("text-to-sql", image=image)


# ═════════════════════════════════════════════════════════════════════════════
# One-time: populate the Volume with the base model
# ═════════════════════════════════════════════════════════════════════════════

@app.function(volumes={MODEL_DIR: volume}, timeout=60 * 60)
def download_base():
    """Download flan-t5-xl into the Volume.  Idempotent — safe to re-run."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=BASE_DIR,
        # Keep only the safetensors shards.  The repo also ships TensorFlow,
        # Flax and legacy pytorch_model-*.bin copies of the same weights —
        # downloading those would roughly double an already 11 GB pull for
        # files transformers will never open.
        ignore_patterns=["*.h5", "*.msgpack", "*.ot", "pytorch_model*.bin"],
    )
    volume.commit()
    print(f"base model ready at {BASE_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
# Shared loading + generation
#   Kept as module-level helpers rather than a Modal base class: each @app.cls
#   below is an independent container pool, and plain functions are the least
#   surprising way to share code across them.
# ═════════════════════════════════════════════════════════════════════════════

def _load_adapter(adapter_name: str):
    """
    Load flan-t5-xl from the Volume and apply one LoRA adapter.

    Returns (model, tokenizer).  Mirrors the notebooks: bfloat16 on Ampere and
    newer (L40S is compute capability 8.9, so bfloat16), float16 otherwise.
    """
    import os
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel

    adapter_dir = f"{ADAPTER_ROOT}/{adapter_name}"
    if not os.path.isdir(adapter_dir):
        raise RuntimeError(
            f"adapter '{adapter_name}' not found at {adapter_dir}. "
            f"Run: python deploy/upload_adapters.py --local-dir <dir>")

    dtype = (torch.bfloat16
             if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    print(f"loading base model ({dtype}) ...")
    base = T5ForConditionalGeneration.from_pretrained(
        BASE_DIR, torch_dtype=dtype, device_map="auto")

    print(f"applying LoRA adapter '{adapter_name}' ...")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    # The notebooks read the tokenizer out of the adapter zip.  Fall back to
    # the base model for adapters that were saved without tokenizer files.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    except Exception:
        print("  no tokenizer in adapter dir — using base tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)

    return model, tokenizer


def _generate(model, tokenizer, text: str,
              max_input: int, max_target: int) -> str:
    """Beam search with the notebooks' exact decoding settings."""
    import torch

    inp = tokenizer(text, return_tensors="pt",
                    max_length=max_input, truncation=True)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_target,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.0,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


_CLS_KWARGS = dict(
    gpu=GPU,
    volumes={MODEL_DIR: volume},
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=FN_TIMEOUT,
)


# ═════════════════════════════════════════════════════════════════════════════
# T1 — Planner:  question + schema  ->  execution plan
# ═════════════════════════════════════════════════════════════════════════════

@app.cls(**_CLS_KWARGS)
class Planner:
    MAX_INPUT  = 512
    MAX_TARGET = 384

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = _load_adapter(ADAPTER_PLANNER)

    @modal.method()
    def plan(self, text: str) -> str:
        return _generate(self.model, self.tokenizer, text,
                         self.MAX_INPUT, self.MAX_TARGET)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def web(self, item: dict):
        text = item.get("input") or item.get("text")
        if not text:
            return {"error": "body must contain 'input'"}
        return {"plan": self.plan.local(text)}

# ═════════════════════════════════════════════════════════════════════════════
# T2 — Plan2Sql:  execution plan  ->  SQL
# ═════════════════════════════════════════════════════════════════════════════

@app.cls(**_CLS_KWARGS)
class Plan2Sql:
    MAX_INPUT  = 384        # notebook uses 384 here, not 512
    MAX_TARGET = 384

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = _load_adapter(ADAPTER_PLAN2SQL)

    @modal.method()
    def to_sql(self, plan_text: str) -> str:
        return _generate(self.model, self.tokenizer, plan_text,
                         self.MAX_INPUT, self.MAX_TARGET)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def web(self, item: dict):
        text = item.get("plan") or item.get("input")
        if not text:
            return {"error": "body must contain 'plan'"}
        return {"sql": self.to_sql.local(text)}


# ═════════════════════════════════════════════════════════════════════════════
# Single transformer:  question + schema  ->  plan [SQL] sql
# ═════════════════════════════════════════════════════════════════════════════

@app.cls(**_CLS_KWARGS)
class Single:
    MAX_INPUT  = 512
    MAX_TARGET = 512
    SQL_SEP    = "[SQL]"

    @modal.enter()
    def load(self):
        self.model, self.tokenizer = _load_adapter(ADAPTER_SINGLE)

    @modal.method()
    def run(self, text: str) -> dict:
        raw = _generate(self.model, self.tokenizer, text,
                        self.MAX_INPUT, self.MAX_TARGET)
        # Trained to emit "<plan> [SQL] <sql>" as one sequence.
        if self.SQL_SEP in raw:
            plan, sql = raw.split(self.SQL_SEP, 1)
            return {"plan": plan.strip(), "sql": sql.strip(), "raw": raw}
        return {"plan": raw.strip(), "sql": "", "raw": raw}

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def web(self, item: dict):
        text = item.get("input") or item.get("text")
        if not text:
            return {"error": "body must contain 'input'"}
        return self.run.local(text)


# ═════════════════════════════════════════════════════════════════════════════
# Dual pipeline — T1 then T2 in one call
#
# This runs on CPU and costs nothing extra: it just orchestrates two remote
# GPU calls.  Each leg wakes its own container, so a fully cold dual call
# pays two 11 GB base-model loads.  That is the cost of keeping the two
# adapters in separate pools.
# ═════════════════════════════════════════════════════════════════════════════

@app.function(timeout=FN_TIMEOUT)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def pipeline(item: dict):
    text = item.get("input") or item.get("text")
    if not text:
        return {"error": "body must contain 'input'"}

    plan = Planner().plan.remote(text)
    sql  = Plan2Sql().to_sql.remote(plan)
    return {"plan": plan, "sql": sql}


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test:  modal run deploy/modal_app.py
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """Exercise all three models with one known-good example."""
    sample = ("question: How many singers do we have? | "
              "schema: singer ( Singer_ID [PK] ) | foreign keys: none")

    print("\n--- dual pipeline (T1 -> T2) ---")
    plan = Planner().plan.remote(sample)
    print("plan:", plan)
    print("sql :", Plan2Sql().to_sql.remote(plan))

    print("\n--- single transformer ---")
    print(Single().run.remote(sample))


