# Serverless Deployment on Modal

Deploys the three fine-tuned Flan-T5-XL + LoRA models as on-demand GPU
endpoints. Containers scale to zero when idle — you pay only while a request
runs, plus a short warm window.

| Model | Class | Input | Output |
|---|---|---|---|
| T1 planner | `Planner` | `question: ... \| schema: ... \| foreign keys: ...` | execution plan |
| T2 plan→SQL | `Plan2Sql` | execution plan | SQL |
| Single transformer | `Single` | `question: ... \| schema: ...` | plan + SQL |
| Dual pipeline | `pipeline` | `question: ... \| schema: ...` | plan + SQL (T1→T2) |

The schema retriever is **not** deployed. It runs locally for free and builds
the `input` string these endpoints consume — GPU time is spent only on
generation.

## Live endpoints

Deployed to workspace `harsh1243`, app `text-to-sql`:

| Endpoint | URL |
|---|---|
| Dual pipeline (T1→T2) | `https://harsh1243--text-to-sql-pipeline.modal.run` |
| T1 planner | `https://harsh1243--text-to-sql-planner-web.modal.run` |
| T2 plan→SQL | `https://harsh1243--text-to-sql-plan2sql-web.modal.run` |
| Single transformer | `https://harsh1243--text-to-sql-single-web.modal.run` |

Dashboard: https://modal.com/apps/harsh1243/main/deployed/text-to-sql

All four return **HTTP 401** without credentials — verified by probing each one.
Note that `modal deploy` only prints the 🔑 badge next to `pipeline`; the three
class endpoints are equally protected despite the missing icon.

## Web UI (Streamlit)

`streamlit_app.py` at the repo root is a thin Streamlit front-end that calls
the endpoints above. Flow: **upload schema → warm GPUs → ask questions**. Each
question is independent — nothing is stored between turns.

```bash
pip install streamlit requests sentence-transformers rank-bm25
streamlit run streamlit_app.py
```

Or deploy it on Modal (so the UI is reachable without running anything
locally):

```bash
# One-time: create the secret in the Modal dashboard.
#   Secrets → New → name "modal-proxy-auth"
#   keys MODAL_KEY (wk-…) and MODAL_SECRET (ws-…)
modal deploy deploy/modal_streamlit.py
```

Modal prints a URL like `https://harsh1243--text-to-sql-ui.modal.run`. The
deployed UI reads its proxy-auth token from the Modal Secret, so users of the
deployed UI don't need to paste credentials themselves.

## Cost

L40S is **$0.000542/sec (~$1.95/hr)**. Your $30 of credits is roughly
**15 GPU-hours**.

Billing is per-container, and each of the three classes loads its own copy of
the 11 GB base model. A cold `pipeline` call therefore pays two base-model
loads — the tradeoff for keeping the adapters in separate pools.

`SCALEDOWN_WINDOW = 900` in `modal_app.py` means a container lingers 15 minutes
after your last request. That costs ~$0.49 per burst but makes follow-up calls
skip the cold start entirely. Lower it to `120` to minimise idle spend if you
only ever send one request at a time.

Nothing runs, and nothing is billed, until a request arrives.

## Setup

```bash
pip install modal
modal setup                     # opens a browser to authenticate
```

### 1. Load the base model into a Volume

```bash
modal run deploy/modal_app.py::download_base
```

Pulls `google/flan-t5-xl` (~11 GB) from HuggingFace directly into the
`t2s-model-weights` Volume. Runs cloud-to-cloud, so it does not touch your own
bandwidth. Idempotent — safe to re-run.

### 2. Upload the LoRA adapters

Download all three zips from the Drive folder in `lora_weights/weights.txt`,
put them in one directory, then:

```bash
python deploy/upload_adapters.py --local-dir ./lora_zips
```

The script unzips them, picks out the files PEFT needs, and uploads. It
tolerates browser duplicate-download names like `lora_adapter (1) (2).zip`.

Verify:

```bash
modal volume ls t2s-model-weights adapters
```

### 3. Create a proxy-auth token

In the Modal dashboard: **Settings → Proxy Auth Tokens → New**. Save the
Token ID (`wk-...`) and Token Secret (`ws-...`).

Endpoints use `requires_proxy_auth=True`, so Modal rejects unauthorized
requests **at its edge, before any GPU starts** — a bot scan cannot burn your
credits. Never commit these tokens.

### 4. Deploy

```bash
modal deploy deploy/modal_app.py
```

Prints the live URL for each endpoint. Read them from that output rather than
guessing — the pattern is
`https://<workspace>--text-to-sql-<class>-web.modal.run`.

## Calling the API

### Python (recommended)

Uses Modal's RPC, which has no request-duration limit:

```bash
# full dual pipeline, retriever runs locally
python deploy/client.py --question "How many singers do we have?" \
                        --schema-file path/to/schema.sql

# single-transformer model
python deploy/client.py --question "..." --schema-file ... --model single

# skip the retriever
python deploy/client.py --input "question: How many singers do we have? | schema: singer ( Singer_ID [PK] ) | foreign keys: none"
```

### modal curl — does NOT work with these endpoints

`modal curl` is documented as sending authenticated requests without proxy
headers, but it was tested against this deployment and returns:

```
modal-http: missing credentials for proxy authorization
```

`requires_proxy_auth=True` is enforced at Modal's edge and is not satisfied by
your workspace API credentials. Use `client.py` (RPC) or a proxy-auth token.

### plain curl — needs a proxy-auth token

Create one at **modal.com → Settings → Proxy Auth Tokens**. The CLI cannot mint
these; the dashboard is the only route. You get a Token ID (`wk-...`) and a
Token Secret (`ws-...`).

```bash
export MODAL_KEY=wk-xxxxxxxx
export MODAL_SECRET=ws-xxxxxxxx

curl -L -X POST \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -H 'Content-Type: application/json' \
  -d '{"input": "question: How many singers do we have? | schema: singer ( Singer_ID [PK] ) | foreign keys: none"}' \
  https://harsh1243--text-to-sql-pipeline.modal.run
```

**`-L` is required.** Modal returns a 303 redirect if a request outstays 150
seconds, which a cold start can. Without `-L`, curl reports the redirect and
you lose a result that actually succeeded. Many HTTP clients also refuse to
follow redirects on POST by default — check yours, or use `client.py`.

Request bodies: `{"input": "..."}` for `Planner`, `Single`, and `pipeline`;
`{"plan": "..."}` for `Plan2Sql`.

## Smoke test

```bash
modal run deploy/modal_app.py
```

Runs one known-good example through the dual pipeline and the single model.

## Notes

Local development with hot reload — but note that URLs get a `-dev` suffix and
the containers stay alive while it runs, so it bills continuously:

```bash
modal serve deploy/modal_app.py
```

Latency: a cold start loads 11 GB from the Volume before generating, so expect
tens of seconds. Beam search (`num_beams=4`, matching the notebooks) is not
fast either. Warm requests skip the load entirely.

To retrain and swap an adapter, re-upload with `--force` and redeploy:

```bash
python deploy/upload_adapters.py --local-dir ./lora_zips --force
modal deploy deploy/modal_app.py
```

The `single` adapter mapping (`lora_adapter_single.zip` →
`lora_adapter_combined`) is inferred from the notebooks' directory names, not
from archive contents. If the single-transformer endpoint returns odd output,
check that mapping first in `upload_adapters.py`.
