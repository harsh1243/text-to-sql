# Query-to-Plan Generator

Converts a SQL query into a **human-readable, step-by-step execution plan** using [`sqlglot`](https://github.com/tobymao/sqlglot) for AST parsing. Designed as the plan generation component of a Text-to-SQL training data pipeline — given a gold SQL query, it produces the plan that the model learns to generate.

## What It Does

Given a SQL query:
```sql
SELECT T2.name
FROM singer_in_concert AS T1
JOIN singer AS T2 ON T1.singer_id = T2.singer_id
JOIN concert AS T3 ON T1.concert_id = T3.concert_id
WHERE T3.year = 2014
```

It produces a structured execution plan:
```
Step 01  [SCAN]
         Sequential scan on table 'singer_in_concert' AS 'T1'

Step 02  [SCAN]
         Sequential scan on table 'singer' AS 'T2'

Step 03  [HASH-JOIN]
         INNER JOIN T2 ON T1.singer_id = T2.singer_id

Step 04  [SCAN]
         Sequential scan on table 'concert' AS 'T3'

Step 05  [HASH-JOIN]
         INNER JOIN T3 ON T1.concert_id = T3.concert_id

Step 06  [FILTER]
         Apply WHERE: T3.year = 2014

Step 07  [PROJECT]
         SELECT T2.name
```

---

## Installation

```bash
pip install sqlglot
```

No other dependencies required. Pure Python, no database connection needed.

---

## Usage

```python
from query_to_plan import generate_plan

sql = "SELECT name, age FROM singer WHERE country = 'France' ORDER BY age DESC"
plan = generate_plan(sql)
print(plan)
```

Output:
```
Step 01  [SCAN]
         Sequential scan on table 'singer'

Step 02  [FILTER]
         Apply WHERE: country = 'France'

Step 03  [PROJECT]
         SELECT name, age

Step 04  [SORT]
         Sort by: age DESC
```

---

## How It Works

The generator parses SQL into an AST using `sqlglot` (SQLite dialect), then walks the tree to emit plan steps in logical execution order:

```
SQL String
    │
    ▼
┌──────────────────────────┐
│  sqlglot.parse_one()     │  Parse SQL → AST
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  _walk(ast)              │  Recursive AST walker
│                          │  Dispatches to handlers:
│  ├─ SELECT  → _plan_select()
│  ├─ UNION   → left/right recursion
│  ├─ INTERSECT / EXCEPT
│  └─ WITH/CTE
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  _render(steps)          │  Format steps as numbered text
└──────────────────────────┘
```

### SELECT Planning Order (`_plan_select`)

For each `SELECT` statement, steps are emitted in logical SQL execution order:

| Order | Clause | Plan Step Type |
|-------|--------|----------------|
| 1 | `WITH` | `CTE` — Materialise each CTE |
| 2 | `FROM` | `SCAN` — Sequential scan on table or derived table |
| 3 | `JOIN` | `SCAN` + `HASH-JOIN` / `NESTED-LOOP-JOIN` |
| 4 | `WHERE` | `FILTER` — Apply predicates |
| 5 | `GROUP BY` | `AGGREGATE` — Group + compute aggregates |
| 6 | `HAVING` | `HAVING-FILTER` — Filter groups |
| 7 | Window functions | `WINDOW` — Compute RANK, ROW_NUMBER, etc. |
| 8 | `SELECT` columns | `PROJECT` — Column projection |
| 9 | `DISTINCT` | `DISTINCT` — Remove duplicates |
| 10 | `ORDER BY` | `SORT` — Sort results |
| 11 | `OFFSET` / `LIMIT` | `OFFSET` / `LIMIT` — Pagination |

---

## Supported SQL Features

### Core
| Feature | Plan Step | Example |
|---------|-----------|---------|
| Table scan | `SCAN` | `FROM students` |
| Derived table (subquery in FROM) | `SUBQUERY-SCAN` | `FROM (SELECT ...) AS sub` |
| WHERE filter | `FILTER` | `WHERE age > 20` |
| GROUP BY + aggregates | `AGGREGATE` | `GROUP BY dept → compute COUNT(*)` |
| Scalar aggregate | `AGGREGATE` | `SELECT AVG(salary) FROM ...` |
| HAVING | `HAVING-FILTER` | `HAVING COUNT(*) > 5` |
| SELECT projection | `PROJECT` | `SELECT name, age` |
| DISTINCT | `DISTINCT` | `SELECT DISTINCT name` |
| ORDER BY | `SORT` | `ORDER BY gpa DESC` |
| LIMIT / OFFSET | `LIMIT` / `OFFSET` | `LIMIT 10 OFFSET 5` |

### Joins
| Feature | Plan Step | Strategy |
|---------|-----------|----------|
| INNER JOIN (equi) | `HASH-JOIN` | Equality condition detected |
| LEFT/RIGHT/FULL JOIN | `HASH-JOIN` | With join type label |
| CROSS JOIN | `NESTED-LOOP-JOIN` | No join condition |
| Comma-join (`FROM a, b`) | `NESTED-LOOP-JOIN` | Implicit cross product |
| NATURAL JOIN | `HASH-JOIN` | Implicit column matching |
| JOIN USING | `HASH-JOIN` | Explicit shared columns |

### Subqueries
| Feature | Plan Step | Notes |
|---------|-----------|-------|
| `IN (SELECT ...)` | `SUBQUERY` | Labelled as IN |
| `NOT IN (SELECT ...)` | `SUBQUERY` | Labelled as NOT IN |
| `EXISTS (SELECT ...)` | `SUBQUERY` | Labelled as EXISTS |
| `NOT EXISTS (SELECT ...)` | `SUBQUERY` | Labelled as NOT EXISTS |
| Scalar subquery in SELECT | `SUBQUERY` | e.g., `(SELECT COUNT(*) FROM ...)` |
| Subquery in JOIN ON | `SUBQUERY` | Expanded before join step |
| Subquery in ORDER BY | `SUBQUERY` | Correlated sort expression |
| `ANY` / `ALL` subquery | `SUBQUERY` | Quantified comparison |
| `CASE WHEN ... IN (subq)` | `SUBQUERY` | Recursive CASE expansion |

### Set Operations
| Feature | Plan Step |
|---------|-----------|
| `UNION` | `UNION` (remove duplicates) |
| `UNION ALL` | `UNION ALL` (keep duplicates) |
| `INTERSECT` | `INTERSECT` |
| `INTERSECT ALL` | `INTERSECT ALL` |
| `EXCEPT` | `EXCEPT` |
| `EXCEPT ALL` | `EXCEPT ALL` |

### Advanced
| Feature | Plan Step |
|---------|-----------|
| `WITH` / CTE | `CTE` — Materialise before main query |
| Window functions (RANK, ROW_NUMBER, etc.) | `WINDOW` |
| LATERAL subquery | `LATERAL-SCAN` |
| No FROM clause (dual) | `DUAL` |

---

## Join Strategy Selection

The generator chooses between two join strategies:

| Strategy | Condition |
|----------|-----------|
| **`HASH-JOIN`** | Equality condition (`=`) in ON clause, or `USING`, or `NATURAL` |
| **`NESTED-LOOP-JOIN`** | No condition (CROSS JOIN), or non-equality condition only |

---

## Architecture

```
query_to_plan.py
│
├── generate_plan(sql)           # Public API — entry point
│
├── _walk(node, steps, depth)    # Main recursive AST walker
│   ├── handles: SELECT, UNION, INTERSECT, EXCEPT, WITH, Subquery
│   └── dispatches to _plan_select() for SELECT nodes
│
├── _plan_select(node, ...)      # Plans a single SELECT in execution order
│   └── calls _subqueries_in_expr() for embedded subqueries
│
├── _subqueries_in_expr(expr, ...)  # Finds & expands subqueries in any expression
│   └── handles: IN, NOT IN, EXISTS, NOT EXISTS, scalar, ANY, ALL, CASE WHEN
│
├── _scan(source, ...)           # Emits SCAN / SUBQUERY-SCAN / LATERAL-SCAN
├── _join_type(join)             # Determines join type label (INNER, LEFT, CROSS, ...)
├── _join_strategy(join)         # Determines execution strategy (HASH vs NESTED-LOOP)
│
├── _collect_windows(exprs)      # Finds window functions in SELECT list
├── _collect_aggs_shallow(exprs) # Finds aggregate functions (shallow, no subquery crossing)
│
└── _render(steps)               # Formats step list into numbered text output
```

---

## Bug Fixes Applied

This version includes 15 fixes over the original implementation:

| Fix | Description |
|-----|-------------|
| F1 | Scalar subqueries in SELECT list were never expanded — now walked |
| F2 | Subqueries inside JOIN ON conditions were silently buried — now walked |
| F3 | Subqueries inside ORDER BY expressions were not expanded — now walked |
| F4 | Window functions (ROW_NUMBER, RANK, etc.) get an explicit WINDOW step |
| F5 | Comma-joins (`FROM a, b`) correctly identified (sqlglot emits them as CROSS) |
| F6 | `INTERSECT ALL` / `EXCEPT ALL` now distinguished from `INTERSECT` / `EXCEPT` |
| F7 | `NOT IN` expressed as `exp.Not(exp.In(...))` now labelled correctly |
| F8 | LATERAL subqueries / LATERAL JOINs handled explicitly |
| F9 | CASE expressions with embedded subqueries now expanded |
| F10 | Subqueries inside GROUP BY / HAVING expressions fully walked |
| F11 | Arg key `from` → `from_`, `with` → `with_` to match sqlglot version |
| F12 | LIMIT / OFFSET read from `.expression` (not `.this`) per sqlglot version |
| F13 | CTEs on UNION / INTERSECT / EXCEPT nodes now emitted before set-op steps |
| F14 | JOIN ON: pass ON expression directly to `_subqueries_in_expr` |
| F15 | CASE WHEN IF node: condition is in `.this`, not `.condition` |


