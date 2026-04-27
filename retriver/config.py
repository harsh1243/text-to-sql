"""
Hyperparameters & Constants
===========================
All tunable knobs for the multi-stage schema retriever live here.
Import this module from any stage file to keep magic numbers out of logic.
"""

# ─── Model identifiers ────────────────────────────────────────────────────────
BIENCODER_MODEL    = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Generic trigger words ────────────────────────────────────────────────────
# Domain-agnostic NL patterns that signal negation or aggregation.
# They are NOT table/column names — safe to keep as constants.

NEGATION_WORDS = {"without", "never", "no", "not", "except",
                  "missing", "none", "lack", "excluding"}
AGGR_WORDS     = {"number", "count", "how many", "total", "average",
                  "maximum", "minimum", "most", "least", "each", "per"}

# ─── Fusion signal weights — base values, adjusted per query type ─────────────
W_BIENCODER_BASE = 0.60
W_BM25_BASE      = 0.45


# Query-adaptive weight shifts
LEXICAL_SHIFT  = 0.10
SEMANTIC_SHIFT = 0.10

# ─── Adaptive threshold (Stage 3) ────────────────────────────────────────────
DROP_RATIO  = 0.65
GAP_RATIO   = 0.25
MAX_TABLES  = 5

# ─── Junction table penalty ──────────────────────────────────────────────────
JUNCTION_PENALTY = -0.15

# ─── Cross-encoder + fusion combination weight (Stage 2) ─────────────────────
CE_WEIGHT     = 0.60
FUSION_WEIGHT = 0.40

# ─── Column pruning — cross-encoder adaptive threshold (Stage 4.5) ────────────
COL_CE_FLOOR = 0.0
COL_CE_GAP   = 2.0

# ─── Superlative / function words — should NOT trigger column text matching ───
# NOTE: 'average' and 'total' are NOT included — they are real column names
COLUMN_STOPWORDS = {'highest', 'lowest', 'most', 'least', 'largest', 'smallest',
                    'biggest', 'best', 'worst', 'first', 'last', 'top', 'bottom',
                    'maximum', 'minimum', 'greatest', 'fewest'}

# ─── Trivial column-name parts — too generic to meaningfully match ────────────
# e.g. "song_name" → distinctive="song", trivial="name"
# Without this, "name" in question matches song_name, concert_name, etc.
TRIVIAL_COL_PARTS = {'name', 'type', 'status', 'description', 'title',
                     'value', 'number', 'code', 'info', 'data', 'flag',
                     'date', 'time', 'desc', 'label', 'text', 'val'}
