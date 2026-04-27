"""
Hyperparameters & Constants
===========================
All tunable knobs for the multi-stage schema retriever live here.
Import this module from any stage file to keep magic numbers out of logic.
"""

# ─── Model identifiers ────────────────────────────────────────────────────────
BIENCODER_MODEL    = "all-MiniLM-L6-v2"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"



NEGATION_WORDS = {"without", "never", "no", "not", "except",
                  "missing", "none", "lack", "excluding"}
AGGR_WORDS     = {"number", "count", "how many", "total", "average",
                  "maximum", "minimum", "most", "least", "each", "per"}


W_BIENCODER_BASE = 0.60
W_BM25_BASE      = 0.40

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


COLUMN_STOPWORDS = {'highest', 'lowest', 'most', 'least', 'largest', 'smallest',
                    'biggest', 'best', 'worst', 'first', 'last', 'top', 'bottom',
                    'maximum', 'minimum', 'greatest', 'fewest'}


TRIVIAL_COL_PARTS = {'name', 'type', 'status', 'description', 'title',
                     'value', 'number', 'code', 'info', 'data', 'flag',
                     'date', 'time', 'desc', 'label', 'text', 'val'}
