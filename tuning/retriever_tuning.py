"""
Retriever Hyperparameter Tuning
===============================
Grid search over the retriever's hyperparameters, scored against the GOLD
pruned schema that is already embedded in the processed training file
(`train_final.json` — its `input` field IS the schema the retriever should
have produced).

Score (as specified):

    table_f1   — F1 over the set of selected TABLES
    column_f1  — F1 over the set of selected COLUMNS   (weight 40%)
    combined   = 0.60 * table_f1  +  0.40 * column_f1

Grid search uses plain nested for-loops. No new dependencies — standard
library only, plus whatever the retriever itself already needs.


TWO THINGS THAT MAKE THIS WORK
------------------------------
1. Parameters are injected by patching the *consuming* modules, not
   `config.py`.  `scoring.py` / `selection.py` do `from .config import X`,
   which copies the value at import time — writing to `config.X` afterwards
   has no effect.  See `apply_params()`.

2. Neural scores are cached.  The bi-encoder embeddings and cross-encoder
   logits do NOT depend on any tuned parameter, so they are computed once and
   replayed for every config.  Without this a 700-config grid would re-encode
   the same sentences ~700 times and take days.  With it, the first config
   warms the cache and the rest are nearly pure arithmetic.


USAGE
-----
    # quick sanity run (small grid, 150 questions)
    python tuning/retriever_tuning.py \
        --data "C:/Users/harsh/Downloads/train_final (1).json" \
        --schemas "C:/Users/harsh/Downloads/spider.zip" \
        --limit 150 --quick

    # full grid
    python tuning/retriever_tuning.py \
        --data "C:/Users/harsh/Downloads/train_final (1).json" \
        --schemas "C:/Users/harsh/Downloads/spider.zip" \
        --limit 400

`--schemas` accepts either spider.zip directly or an extracted tables.json.

Results are written to tuning/tuning_results.json (all configs, ranked) and
the winner to tuning/config_best.py, which you can copy into retriver/config.py.

REQUIREMENTS: `pip install sentence-transformers rank-bm25`.  Runs on CPU;
the two models are ~120 MB total.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Schema loading — Spider tables.json → retriever schema dict
# ═════════════════════════════════════════════════════════════════════════════

def schema_from_tables_entry(entry: dict) -> dict:
    """
    Convert one tables.json entry into the dict shape parse_schema() returns.

    tables.json is used instead of the per-db schema.sql files because it
    covers all 146 dbs in the training data, whereas spider.zip ships only
    148 schema.sql files that miss 13 of them (college_1, chinook_1, ...).
    """
    tnames = [t.lower() for t in entry["table_names_original"]]
    cols_by_table = [[] for _ in tnames]
    cid_to_tid, cid_to_name = {}, {}

    for cid, (tid, cname) in enumerate(entry["column_names_original"]):
        if tid < 0:                      # the synthetic "*" column
            continue
        ctype = entry["column_types"][cid]
        cols_by_table[tid].append((cname, ctype))
        cid_to_tid[cid] = tid
        cid_to_name[cid] = cname.lower()

    pks_by_table = {}
    for cid in entry.get("primary_keys", []):
        if isinstance(cid, list):        # composite PK
            for c in cid:
                if c in cid_to_tid:
                    pks_by_table.setdefault(cid_to_tid[c], []).append(cid_to_name[c])
        elif cid in cid_to_tid:
            pks_by_table.setdefault(cid_to_tid[cid], []).append(cid_to_name[cid])

    # FKs are per-table: (from_col, ref_table, ref_col), all lowercase
    fks_by_table = {}
    for fcid, rcid in entry.get("foreign_keys", []):
        if fcid not in cid_to_tid or rcid not in cid_to_tid:
            continue
        f_tid = cid_to_tid[fcid]
        fks_by_table.setdefault(f_tid, []).append(
            (cid_to_name[fcid], tnames[cid_to_tid[rcid]], cid_to_name[rcid]))

    return {
        tname: {
            "table_orig": entry["table_names_original"][ti],
            "columns":    cols_by_table[ti],
            "pks":        pks_by_table.get(ti, []),
            "fks":        fks_by_table.get(ti, []),
        }
        for ti, tname in enumerate(tnames)
    }


def load_schemas(path: str) -> dict:
    """Load tables.json from either spider.zip or a plain tables.json path."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("tables.json"))
            entries = json.loads(z.read(name).decode("utf-8"))
    else:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    return {e["db_id"]: schema_from_tables_entry(e) for e in entries}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Gold parsing + F1
# ═════════════════════════════════════════════════════════════════════════════

_TABLE_RE = re.compile(r'^\s*([^()|]+?)\s*\(\s*(.*?)\s*\)\s*$', re.DOTALL)


def parse_gold(input_field: str):
    """
    'question: Q | schema: t1 ( c1 [PK], c2 ) | t2 ( c3 ) | foreign keys: ...'
        → (question, {t1, t2}, {'t1.c1', 't1.c2', 't2.c3'})
    """
    question = input_field.split(" | schema:")[0].replace("question:", "", 1).strip()
    if " | schema:" not in input_field:
        return question, set(), set()
    body = input_field.split(" | schema:", 1)[1]
    body = body.split("| foreign keys:", 1)[0]

    tables, cols = set(), set()
    for chunk in body.split(" | "):
        m = _TABLE_RE.match(chunk.strip())
        if not m:
            continue
        tname = m.group(1).strip().lower()
        if not tname:
            continue
        tables.add(tname)
        for col in m.group(2).split(","):
            col = col.replace("[PK]", "").strip().lower()
            if col:
                cols.add(f"{tname}.{col}")
    return question, tables, cols


def f1(pred: set, gold: set) -> float:
    """Set-level F1. Both empty → 1.0 (nothing to retrieve, nothing retrieved)."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Caching model proxies
#    Neural scores don't depend on tuned params, so compute once and replay.
# ═════════════════════════════════════════════════════════════════════════════

class _ScoreList(list):
    """Mimics the numpy array returned by CrossEncoder.predict()."""
    def tolist(self):
        return list(self)


class CachedBiEncoder:
    def __init__(self, real):
        self._real = real
        self._cache = {}

    def encode(self, x, convert_to_tensor=False, **kw):
        key = x if isinstance(x, str) else tuple(x)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._real.encode(x, convert_to_tensor=convert_to_tensor, **kw)
            self._cache[key] = hit
        return hit


class CachedCrossEncoder:
    def __init__(self, real):
        self._real = real
        self._cache = {}

    def predict(self, pairs, **kw):
        pairs = [tuple(p) for p in pairs]
        missing = [p for p in pairs if p not in self._cache]
        if missing:
            fresh = self._real.predict([list(p) for p in missing], **kw)
            for p, s in zip(missing, list(fresh)):
                self._cache[p] = float(s)
        return _ScoreList(self._cache[p] for p in pairs)


def install_caches():
    """Patch the module-level names the retriever actually calls."""
    import retriver.models as models
    import retriver.scoring as scoring
    import retriver.selection as selection

    bi = CachedBiEncoder(models.get_biencoder())
    ce = CachedCrossEncoder(models.get_crossencoder())

    scoring.get_biencoder = lambda: bi
    scoring.get_crossencoder = lambda: ce
    selection.get_crossencoder = lambda: ce
    return bi, ce


# ═════════════════════════════════════════════════════════════════════════════
# 4. Parameter injection
#    scoring/selection did `from .config import X`, binding X into their own
#    namespace. Patching config.X would be a silent no-op — patch them.
# ═════════════════════════════════════════════════════════════════════════════

# param name -> modules that read it
PARAM_TARGETS = {
    "W_BIENCODER_BASE": ["retriver.scoring"],
    "W_BM25_BASE":      ["retriver.scoring"],
    "LEXICAL_SHIFT":    ["retriver.scoring"],
    "SEMANTIC_SHIFT":   ["retriver.scoring"],
    "JUNCTION_PENALTY": ["retriver.scoring"],
    "CE_WEIGHT":        ["retriver.scoring"],
    "FUSION_WEIGHT":    ["retriver.scoring"],
    "DROP_RATIO":       ["retriver.selection"],
    "GAP_RATIO":        ["retriver.selection"],
    "MAX_TABLES":       ["retriver.selection"],
    "COL_CE_FLOOR":     ["retriver.selection"],
    "COL_CE_GAP":       ["retriver.selection"],
}


def apply_params(params: dict):
    for name, value in params.items():
        for modname in PARAM_TARGETS.get(name, []):
            setattr(sys.modules[modname], name, value)


def verify_injection():
    """Fail loudly if patching ever stops working (e.g. after a refactor)."""
    import retriver.selection as selection
    original = selection.DROP_RATIO
    apply_params({"DROP_RATIO": 0.123})
    ok = selection.DROP_RATIO == 0.123
    apply_params({"DROP_RATIO": original})
    if not ok:
        raise RuntimeError(
            "Parameter injection failed — retriver.selection.DROP_RATIO did not "
            "change. The grid would score every config identically.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def build_examples(data_path, schemas, limit):
    """Pre-parse gold + attach schema/fk_graph once (not per config)."""
    from retriver import build_fk_graph

    with open(data_path, encoding="utf-8") as f:
        rows = json.load(f)

    graph_cache = {}
    examples, skipped = [], 0
    for row in rows:
        db = row.get("db_id")
        schema = schemas.get(db)
        if schema is None:
            skipped += 1
            continue
        question, g_tables, g_cols = parse_gold(row["input"])
        if not question or not g_tables:
            skipped += 1
            continue
        if db not in graph_cache:
            graph_cache[db] = build_fk_graph(schema)
        examples.append((question, schema, graph_cache[db], g_tables, g_cols))
        if limit and len(examples) >= limit:
            break
    return examples, skipped


def eval_config(examples, use_cross_encoder=True):
    """Return (table_f1, column_f1, combined, n_errors)."""
    from retriver import retrieve

    t_scores, c_scores, errors = [], [], 0
    for question, schema, fk_graph, g_tables, g_cols in examples:
        try:
            res = retrieve(question, schema, fk_graph,
                           use_cross_encoder=use_cross_encoder)
        except Exception:
            errors += 1
            t_scores.append(0.0)
            c_scores.append(0.0)
            continue

        p_tables = {t.lower() for t in res["selected_tables"]}
        p_cols = {
            f"{t.lower()}.{c.lower()}"
            for t, cols in res["debug"]["pruned_columns"].items()
            for c, _ in cols
        }
        t_scores.append(f1(p_tables, g_tables))
        c_scores.append(f1(p_cols, g_cols))

    n = max(len(t_scores), 1)
    table_f1 = sum(t_scores) / n
    column_f1 = sum(c_scores) / n
    combined = 0.60 * table_f1 + 0.40 * column_f1
    return table_f1, column_f1, combined, errors


# ═════════════════════════════════════════════════════════════════════════════
# 6. Grid search — explicit nested for-loops
# ═════════════════════════════════════════════════════════════════════════════

FULL_GRID = {
    "W_BIENCODER_BASE": [0.40, 0.50, 0.60, 0.70],
    "DROP_RATIO":       [0.50, 0.60, 0.65, 0.75],
    "GAP_RATIO":        [0.15, 0.25, 0.35],
    "JUNCTION_PENALTY": [0.00, -0.15, -0.30],
    "MAX_TABLES":       [4, 5, 6],
    "COL_CE_GAP":       [1.0, 2.0, 3.0],
}

QUICK_GRID = {
    "W_BIENCODER_BASE": [0.50, 0.60],
    "DROP_RATIO":       [0.60, 0.65],
    "GAP_RATIO":        [0.25],
    "JUNCTION_PENALTY": [0.00, -0.15],
    "MAX_TABLES":       [5],
    "COL_CE_GAP":       [2.0],
}


def run_grid(examples, grid, use_cross_encoder=True):
    results = []
    total = 1
    for v in grid.values():
        total *= len(v)

    print(f"\ngrid: {total} configs x {len(examples)} questions")
    print("(first config is slow — it fills the neural cache)\n")

    i = 0
    t0 = time.time()

    # Explicit nested loops over every parameter.
    for w_bi in grid["W_BIENCODER_BASE"]:
        for drop in grid["DROP_RATIO"]:
            for gap in grid["GAP_RATIO"]:
                for pen in grid["JUNCTION_PENALTY"]:
                    for maxt in grid["MAX_TABLES"]:
                        for col_gap in grid["COL_CE_GAP"]:
                            params = {
                                "W_BIENCODER_BASE": w_bi,
                                "W_BM25_BASE":      round(1.0 - w_bi, 4),
                                "DROP_RATIO":       drop,
                                "GAP_RATIO":        gap,
                                "JUNCTION_PENALTY": pen,
                                "MAX_TABLES":       maxt,
                                "COL_CE_GAP":       col_gap,
                            }
                            apply_params(params)
                            t_f1, c_f1, comb, errs = eval_config(
                                examples, use_cross_encoder)
                            results.append({
                                "combined":  round(comb, 4),
                                "table_f1":  round(t_f1, 4),
                                "column_f1": round(c_f1, 4),
                                "errors":    errs,
                                "params":    params,
                            })
                            i += 1
                            el = time.time() - t0
                            eta = (el / i) * (total - i)
                            print(f"[{i:>4}/{total}] combined={comb:.4f} "
                                  f"table={t_f1:.4f} col={c_f1:.4f} "
                                  f"| {el:.0f}s elapsed, ETA {eta:.0f}s")

    results.sort(key=lambda r: r["combined"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 7. Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Grid-search retriever hyperparameters.")
    ap.add_argument("--data", required=True,
                    help="Processed training file (train_final.json)")
    ap.add_argument("--schemas", required=True,
                    help="spider.zip, or an extracted tables.json")
    ap.add_argument("--limit", type=int, default=200,
                    help="Questions to evaluate per config (0 = all). Default 200.")
    ap.add_argument("--quick", action="store_true",
                    help="Small 8-config grid for a sanity check")
    ap.add_argument("--no-cross-encoder", action="store_true",
                    help="Skip Stage 2 reranking (faster, lower accuracy)")
    ap.add_argument("--top", type=int, default=15, help="How many rows to print")
    args = ap.parse_args()

    print("loading schemas ...")
    schemas = load_schemas(args.schemas)
    print(f"  {len(schemas)} databases")

    print("loading + pre-parsing examples ...")
    examples, skipped = build_examples(args.data, schemas, args.limit)
    print(f"  {len(examples)} usable, {skipped} skipped (db not in tables.json)")
    if not examples:
        sys.exit("No usable examples — check --data / --schemas paths.")

    print("loading models (first run downloads ~120 MB) ...")
    install_caches()
    verify_injection()
    print("  parameter injection verified")

    grid = QUICK_GRID if args.quick else FULL_GRID
    results = run_grid(examples, grid, use_cross_encoder=not args.no_cross_encoder)

    print(f"\n{'='*72}")
    print(f"TOP {args.top}  —  combined = 0.60*table_F1 + 0.40*column_F1")
    print(f"{'='*72}")
    for i, r in enumerate(results[:args.top], 1):
        p = r["params"]
        print(f"#{i:<3} combined={r['combined']:.4f}  "
              f"table={r['table_f1']:.4f}  column={r['column_f1']:.4f}")
        print(f"     w_bi={p['W_BIENCODER_BASE']} drop={p['DROP_RATIO']} "
              f"gap={p['GAP_RATIO']} junction={p['JUNCTION_PENALTY']} "
              f"max_tables={p['MAX_TABLES']} col_ce_gap={p['COL_CE_GAP']}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(out_dir, "tuning_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    cfg_path = os.path.join(out_dir, "config_best.py")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write('"""Best config from tuning/retriever_tuning.py.\n\n')
        f.write(f'combined={best["combined"]}  table_f1={best["table_f1"]}  ')
        f.write(f'column_f1={best["column_f1"]}\n')
        f.write(f'Tuned on {len(examples)} questions.\n')
        f.write('Copy these values into retriver/config.py to adopt them.\n"""\n\n')
        for k, v in best["params"].items():
            f.write(f"{k} = {v!r}\n")

    print(f"\nall results -> {res_path}")
    print(f"best config -> {cfg_path}")

    # Baseline comparison against the shipped defaults
    import retriver.config as cfg
    apply_params({
        "W_BIENCODER_BASE": cfg.W_BIENCODER_BASE,
        "W_BM25_BASE":      cfg.W_BM25_BASE,
        "DROP_RATIO":       cfg.DROP_RATIO,
        "GAP_RATIO":        cfg.GAP_RATIO,
        "JUNCTION_PENALTY": cfg.JUNCTION_PENALTY,
        "MAX_TABLES":       cfg.MAX_TABLES,
        "COL_CE_GAP":       cfg.COL_CE_GAP,
    })
    b_t, b_c, b_comb, _ = eval_config(examples, not args.no_cross_encoder)
    print(f"\nshipped defaults : combined={b_comb:.4f} table={b_t:.4f} column={b_c:.4f}")
    print(f"best from grid   : combined={best['combined']:.4f} "
          f"table={best['table_f1']:.4f} column={best['column_f1']:.4f}")
    delta = best["combined"] - b_comb
    print(f"improvement      : {delta:+.4f} ({100*delta/max(b_comb,1e-9):+.1f}%)")


if __name__ == "__main__":
    main()
