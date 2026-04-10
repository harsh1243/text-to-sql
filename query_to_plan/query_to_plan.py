"""
SQL Query Planner — fully corrected version using sqlglot
==========================================================
pip install sqlglot

Handles all Spider 1.0 query categories including hard / extra-hard.

Fixes applied vs. previous version
------------------------------------
F1  Scalar subqueries in SELECT list were never expanded — now walked.
F2  Subqueries inside JOIN ON conditions were silently buried — now walked.
F3  Subqueries inside ORDER BY expressions were not expanded — now walked.
F4  Window functions (ROW_NUMBER, RANK, etc.) get an explicit WINDOW step.
F5  Comma-joins (FROM a, b) correctly identified (sqlglot emits them as CROSS).
F6  INTERSECT ALL / EXCEPT ALL now distinguished from INTERSECT / EXCEPT.
F7  NOT IN expressed as exp.Not(exp.In(...)) now labelled correctly.
F8  LATERAL subqueries / LATERAL JOINs handled explicitly.
F9  CASE expressions with embedded subqueries now expanded.
F10 Subqueries inside GROUP BY / HAVING expressions fully walked.
F11 Arg key "from" → "from_", "with" → "with_" to match this sqlglot version.
F12 LIMIT / OFFSET read from .expression (not .this) per this sqlglot version.
F13 CTEs on UNION / INTERSECT / EXCEPT nodes now emitted before set-op steps.
F14 JOIN ON: pass the ON expression directly to _subqueries_in_expr.
F15 CASE WHEN IF node: condition is in .this, not .condition.
"""

import sqlglot
import sqlglot.expressions as exp


# ── Public API ────────────────────────────────────────────────────────────────

def generate_plan(sql: str) -> str:
    ast   = sqlglot.parse_one(sql, dialect="sqlite")
    steps = []
    _walk(ast, steps, depth=0)
    return _render(steps)


# ── Main walker ───────────────────────────────────────────────────────────────

def _walk(node, steps, depth=0):
    if node is None:
        return

    # WITH / CTE node (emitted by both Select and set-ops)
    if isinstance(node, exp.With):
        for cte in node.expressions:
            steps.append((depth, "CTE", f"Materialise CTE '{cte.alias}'"))
            _walk(cte.this, steps, depth + 1)

    # UNION / UNION ALL
    elif isinstance(node, exp.Union):
        # F13: CTE attached to the UNION node itself
        _maybe_walk_cte(node, steps, depth)

        is_all  = node.args.get("distinct") is False
        op_name = "UNION ALL" if is_all else "UNION"
        steps.append((depth, "SET-OP-LEFT",  f"Evaluate LEFT side of {op_name}"))
        _walk(node.left,  steps, depth + 1)
        steps.append((depth, "SET-OP-RIGHT", f"Evaluate RIGHT side of {op_name}"))
        _walk(node.right, steps, depth + 1)
        steps.append((depth, op_name,
                      "Merge both result sets" + (" (keep duplicates)" if is_all
                                                   else " (remove duplicates)")))

    # INTERSECT / INTERSECT ALL
    elif isinstance(node, exp.Intersect):
        _maybe_walk_cte(node, steps, depth)                    # F13
        is_all  = node.args.get("distinct") is False           # F6
        op_name = "INTERSECT ALL" if is_all else "INTERSECT"
        steps.append((depth, "SET-OP-LEFT",  f"Evaluate LEFT side of {op_name}"))
        _walk(node.this,       steps, depth + 1)
        steps.append((depth, "SET-OP-RIGHT", f"Evaluate RIGHT side of {op_name}"))
        _walk(node.expression, steps, depth + 1)
        steps.append((depth, op_name,
                      "Keep rows present in BOTH results"
                      + (" (keep duplicates)" if is_all else " (remove duplicates)")))

    # EXCEPT / EXCEPT ALL
    elif isinstance(node, exp.Except):
        _maybe_walk_cte(node, steps, depth)                    # F13
        is_all  = node.args.get("distinct") is False           # F6
        op_name = "EXCEPT ALL" if is_all else "EXCEPT"
        steps.append((depth, "SET-OP-LEFT",  f"Evaluate LEFT side of {op_name}"))
        _walk(node.this,       steps, depth + 1)
        steps.append((depth, "SET-OP-RIGHT", f"Evaluate RIGHT side of {op_name}"))
        _walk(node.expression, steps, depth + 1)
        steps.append((depth, op_name,
                      "Keep rows in LEFT not present in RIGHT"
                      + (" (keep duplicates)" if is_all else "")))

    # Subquery wrapper — unwrap and walk inner SELECT
    elif isinstance(node, exp.Subquery):
        _walk(node.this, steps, depth)

    # SELECT
    elif isinstance(node, exp.Select):
        _plan_select(node, steps, depth)


def _maybe_walk_cte(node, steps, depth):
    """Emit CTE steps for CTEs attached directly to a set-op or select node."""
    with_ = node.args.get("with_")           # F11: key is "with_" not "with"
    if with_:
        _walk(with_, steps, depth)


# ── SELECT planner ────────────────────────────────────────────────────────────

def _plan_select(node: exp.Select, steps, depth):

    # 1. WITH attached to this SELECT  (F11: key is "with_")
    _maybe_walk_cte(node, steps, depth)

    # 2. FROM  (F11: key is "from_")
    from_ = node.args.get("from_")
    if from_:
        _scan(from_.this, steps, depth)
    else:
        steps.append((depth, "DUAL", "No FROM clause — evaluate constant expression"))

    # 3. JOINs
    for join in (node.args.get("joins") or []):
        _scan(join.this, steps, depth)

        jtype  = _join_type(join)
        jstrat = _join_strategy(join)
        on     = join.args.get("on")
        using  = join.args.get("using")

        if on:
            cond = f" ON {on.sql()}"
        elif using:
            cond = f" USING ({', '.join(c.sql() for c in using)})"
        else:
            cond = ""

        right = join.this
        if isinstance(right, exp.Table):
            tname = right.alias or right.name
        elif isinstance(right, (exp.Subquery, exp.Lateral)):
            tname = right.alias or "subquery"
        else:
            tname = right.alias or right.sql()

        steps.append((depth, jstrat, f"{jtype} JOIN {tname}{cond}"))

        # F2: expand subqueries inside the ON condition
        # Pass `on` directly — it IS the predicate expression (e.g. EQ node)  F14
        if on:
            _subqueries_in_expr(on, steps, depth)

    # 4. WHERE — expand subqueries first, then emit FILTER
    where = node.args.get("where")
    if where:
        _subqueries_in_expr(where.this, steps, depth)
        steps.append((depth, "FILTER", f"Apply WHERE: {where.this.sql()}"))

    # 5. GROUP BY + aggregates
    group  = node.args.get("group")
    having = node.args.get("having")
    aggs   = _collect_aggs_shallow(node.expressions)
    if not aggs and having:
        aggs = _collect_aggs_shallow([having.this])

    if group:
        # F10: expand subqueries inside GROUP BY expressions
        for ge in group.expressions:
            _subqueries_in_expr(ge, steps, depth)
        cols = ", ".join(e.sql() for e in group.expressions)
        desc = f"GROUP BY [{cols}]"
        if aggs:
            desc += f"  ->  compute {', '.join(aggs)}"
        steps.append((depth, "AGGREGATE", desc))
    elif aggs:
        steps.append((depth, "AGGREGATE",
                      f"Scalar aggregate (no GROUP BY)  ->  compute {', '.join(aggs)}"))

    # 6. HAVING
    if having:
        # F10: expand subqueries inside HAVING
        _subqueries_in_expr(having.this, steps, depth)
        steps.append((depth, "HAVING-FILTER", f"Apply HAVING: {having.this.sql()}"))

    # 7. Window functions (F4)
    windows = _collect_windows(node.expressions)
    for w_sql in windows:
        steps.append((depth, "WINDOW", f"Compute window function: {w_sql}"))

    # 8. PROJECT — F1: expand subqueries in SELECT list BEFORE emitting PROJECT
    for expr in node.expressions:
        _subqueries_in_expr(expr, steps, depth)
    cols = ", ".join(e.sql() for e in node.expressions)
    steps.append((depth, "PROJECT", f"SELECT {cols}"))

    # 9. DISTINCT
    if node.args.get("distinct"):
        steps.append((depth, "DISTINCT", "Remove duplicate rows"))

    # 10. ORDER BY — F3: expand subqueries in ORDER BY expressions
    order = node.args.get("order")
    if order:
        for o in order.expressions:
            # o is exp.Ordered; o.this is the actual sort expression
            _subqueries_in_expr(o.this, steps, depth)
        cols = ", ".join(e.sql() for e in order.expressions)
        steps.append((depth, "SORT", f"Sort by: {cols}"))

    # 11. OFFSET / LIMIT  (F12: value is in .expression, not .this)
    offset = node.args.get("offset")
    limit  = node.args.get("limit")
    if offset:
        val = offset.args.get("expression") or offset.args.get("this")
        steps.append((depth, "OFFSET", f"Skip {val.sql()} rows"))
    if limit:
        val = limit.args.get("expression") or limit.args.get("this")
        steps.append((depth, "LIMIT", f"Return at most {val.sql()} rows"))


# ── Subquery finder ───────────────────────────────────────────────────────────

def _subqueries_in_expr(expr, steps, depth):
    """
    Recursively walk any expression and emit SUBQUERY plan steps for every
    embedded subquery. Does not cross subquery boundaries (avoids double-count).
    Handles: IN, NOT IN, EXISTS, NOT EXISTS, scalar, ANY, ALL, CASE WHEN,
             and scalar subquery on either side of any comparison operator.
    """
    if expr is None:
        return

    # IN (subquery) / NOT IN (subquery)
    if isinstance(expr, exp.In):
        query = expr.args.get("query")
        if query:
            neg = "NOT IN" if (expr.args.get("negate") or expr.args.get("not")) else "IN"
            steps.append((depth, "SUBQUERY", f"Evaluate {neg} subquery"))
            _walk(query, steps, depth + 1)
            return

    # F7: NOT IN as exp.Not wrapping exp.In
    if isinstance(expr, exp.Not) and isinstance(expr.this, exp.In):
        inner = expr.this
        query = inner.args.get("query")
        if query:
            steps.append((depth, "SUBQUERY", "Evaluate NOT IN subquery"))
            _walk(query, steps, depth + 1)
            return

    # NOT EXISTS — check before plain EXISTS
    if isinstance(expr, exp.Not) and isinstance(expr.this, exp.Exists):
        steps.append((depth, "SUBQUERY", "Evaluate NOT EXISTS subquery"))
        _walk(expr.this.this, steps, depth + 1)
        return

    # EXISTS
    if isinstance(expr, exp.Exists):
        steps.append((depth, "SUBQUERY", "Evaluate EXISTS subquery"))
        _walk(expr.this, steps, depth + 1)
        return

    # Scalar subquery  e.g. (SELECT AVG(...) FROM ...)
    if isinstance(expr, exp.Subquery):
        steps.append((depth, "SUBQUERY", "Evaluate scalar subquery"))
        _walk(expr.this, steps, depth + 1)
        return

    # ANY subquery
    if isinstance(expr, exp.Any):
        steps.append((depth, "SUBQUERY", "Evaluate ANY subquery"))
        _walk(expr.this, steps, depth + 1)
        return

    # ALL subquery (class name varies across sqlglot versions)
    for _cls_name in ("All", "Every"):
        _cls = getattr(exp, _cls_name, None)
        if _cls and isinstance(expr, _cls):
            steps.append((depth, "SUBQUERY", "Evaluate ALL subquery"))
            _walk(expr.this, steps, depth + 1)
            return

    # F9: CASE WHEN — recurse into WHEN/THEN/ELSE branches
    # IF node args: .this = condition, .true = THEN result   (F15)
    if isinstance(expr, exp.Case):
        for ifs in (expr.args.get("ifs") or []):
            _subqueries_in_expr(ifs.args.get("this"),  steps, depth)  # WHEN condition
            _subqueries_in_expr(ifs.args.get("true"),  steps, depth)  # THEN result
        _subqueries_in_expr(expr.args.get("default"), steps, depth)   # ELSE
        return

    # General: recurse into all children
    # Handles AND / OR / NOT / comparisons / arithmetic / COALESCE / etc.
    for child in expr.args.values():
        if isinstance(child, exp.Expression):
            _subqueries_in_expr(child, steps, depth)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, exp.Expression):
                    _subqueries_in_expr(item, steps, depth)


# ── Table / subquery scan ─────────────────────────────────────────────────────

def _scan(source, steps, depth):
    # F8: LATERAL subquery
    if isinstance(source, exp.Lateral):
        alias = source.alias or "lateral"
        steps.append((depth, "LATERAL-SCAN",
                      f"Materialise LATERAL subquery '{alias}'"))
        inner = source.this
        if isinstance(inner, (exp.Subquery, exp.Select)):
            _walk(inner, steps, depth + 1)
        return

    if isinstance(source, exp.Subquery):
        alias = source.alias or "subquery"
        steps.append((depth, "SUBQUERY-SCAN",
                      f"Materialise derived table '{alias}'"))
        _walk(source.this, steps, depth + 1)

    elif isinstance(source, exp.Table):
        name  = source.name
        alias = source.alias
        label = f"'{name}'"
        if alias and alias != name:
            label += f" AS '{alias}'"
        steps.append((depth, "SCAN", f"Sequential scan on table {label}"))

    else:
        steps.append((depth, "SCAN", f"Scan: {source.sql()}"))


# ── Join helpers ──────────────────────────────────────────────────────────────

def _join_type(join: exp.Join) -> str:
    if join.args.get("cross"):   return "CROSS"
    if join.args.get("natural"): return "NATURAL"
    kind = join.args.get("kind")
    if kind:
        k = str(kind).upper()
        # F5: sqlglot emits comma-joins as kind="CROSS" — they are implicit INNER
        # We keep the label CROSS because logically they ARE cross-products filtered
        # by WHERE, which is equivalent. No rename needed.
        return k
    return "INNER"


def _join_strategy(join: exp.Join) -> str:
    if join.args.get("natural"): return "HASH-JOIN"
    if join.args.get("cross"):   return "NESTED-LOOP-JOIN"
    if join.args.get("using"):   return "HASH-JOIN"
    on = join.args.get("on")
    if on:
        # find() always returns the node directly — version-safe
        if on.find(exp.EQ):
            return "HASH-JOIN"
    return "NESTED-LOOP-JOIN"


# ── Window function collector ─────────────────────────────────────────────────

def _collect_windows(expressions) -> list:
    """Return sql() strings for every window function in the SELECT list. (F4)"""
    windows = []

    def _visit(node):
        if node is None:
            return
        if isinstance(node, exp.Subquery):
            return                   # don't cross subquery boundary
        if isinstance(node, exp.Window):
            windows.append(node.sql())
            return                   # don't recurse inside window spec
        for child in node.args.values():
            if isinstance(child, exp.Expression):
                _visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, exp.Expression):
                        _visit(item)

    for expr in expressions:
        _visit(expr)

    return list(dict.fromkeys(windows))


# ── Aggregate collector (shallow — does not cross subquery boundaries) ────────

def _collect_aggs_shallow(expressions) -> list:
    aggs = []

    def _visit(node):
        if node is None:
            return
        if isinstance(node, exp.Subquery):
            return
        if isinstance(node, exp.Window):
            return                   # window funcs handled separately
        if isinstance(node, exp.AggFunc):
            aggs.append(node.sql())
            return
        for child in node.args.values():
            if isinstance(child, exp.Expression):
                _visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, exp.Expression):
                        _visit(item)

    for expr in expressions:
        _visit(expr)

    return list(dict.fromkeys(aggs))


# ── Renderer ──────────────────────────────────────────────────────────────────

def _render(steps) -> str:
    lines    = []
    counters = {}

    for depth, op, desc in steps:
        pad = "    " * depth

        if op in ("SET-OP-LEFT", "SET-OP-RIGHT"):
            lines.append(f"{pad}  -- {desc}")
            continue

        for d in [d for d in list(counters) if d > depth]:
            del counters[d]

        n = counters.get(depth, 1)
        counters[depth] = n + 1

        lines.append(f"{pad}Step {n:02d}  [{op}]")
        lines.append(f"{pad}         {desc}")
        lines.append("")

    return "\n".join(lines)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = {
        "Simple SELECT + WHERE": """
            SELECT name FROM students WHERE age > 20
        """,
        "JOIN + GROUP BY + HAVING + ORDER + LIMIT": """
            SELECT d.name, COUNT(e.id) AS cnt
            FROM departments d
            JOIN employees e ON e.dept_id = d.id
            WHERE d.active = 1
            GROUP BY d.name
            HAVING COUNT(e.id) > 5
            ORDER BY cnt DESC
            LIMIT 10
        """,
        "IN subquery": """
            SELECT name FROM students
            WHERE id IN (SELECT student_id FROM enrolled WHERE course = 'Math')
        """,
        "NOT IN subquery": """
            SELECT name FROM students
            WHERE id NOT IN (SELECT student_id FROM enrolled WHERE course = 'Math')
        """,
        "Correlated NOT EXISTS": """
            SELECT name FROM employees e
            WHERE NOT EXISTS (
                SELECT 1 FROM projects p WHERE p.lead_id = e.id
            )
        """,
        "Scalar subquery in SELECT list (F1)": """
            SELECT name,
                   (SELECT COUNT(*) FROM orders o WHERE o.cust_id = c.id) AS order_cnt
            FROM customers c
        """,
        "Subquery in JOIN ON (F2)": """
            SELECT t1.name, t2.val
            FROM t1
            JOIN t2 ON t2.id = (SELECT MAX(id) FROM t3)
        """,
        "Subquery in ORDER BY (F3)": """
            SELECT name FROM customers c
            ORDER BY (SELECT COUNT(*) FROM orders o WHERE o.cust_id = c.id) DESC
        """,
        "Window function RANK (F4)": """
            SELECT name, salary,
                   RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
            FROM employees
        """,
        "Comma-join implicit INNER (F5)": """
            SELECT s.name, c.title
            FROM students s, courses c
            WHERE s.course_id = c.id
        """,
        "INTERSECT ALL (F6)": """
            SELECT name FROM a
            INTERSECT ALL
            SELECT name FROM b
        """,
        "EXCEPT (F6)": """
            SELECT name FROM a
            EXCEPT
            SELECT name FROM b
        """,
        "CTE + UNION (F13)": """
            WITH ranked AS (
                SELECT name, score FROM tests WHERE subject = 'Math'
            )
            SELECT name FROM ranked WHERE score > 80
            UNION
            SELECT name FROM ranked WHERE score < 40
        """,
        "CASE WHEN with subquery (F9)": """
            SELECT name,
                CASE WHEN id IN (SELECT vip_id FROM vip_list) THEN 'VIP' ELSE 'Regular' END
            FROM customers
        """,
        "Derived table in FROM": """
            SELECT sub.name FROM
              (SELECT name, MAX(score) AS top FROM grades GROUP BY name) AS sub
            WHERE sub.top > 90
        """,
        "OFFSET + LIMIT": """
            SELECT name FROM students ORDER BY gpa DESC LIMIT 5 OFFSET 10
        """,
        "Extra-hard: CTE + correlated subquery + window + IN + ORDER": """
            WITH base AS (
                SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id
            )
            SELECT e.name,
                   e.salary,
                   RANK() OVER (PARTITION BY e.dept_id ORDER BY e.salary DESC) AS rnk
            FROM employees e
            JOIN base b ON b.dept_id = e.dept_id
            WHERE e.salary > (SELECT AVG(salary) FROM employees)
              AND e.dept_id IN (SELECT dept_id FROM departments WHERE active = 1)
            ORDER BY rnk
        """,
    }

    for title, sql in tests.items():
        print("=" * 72)
        print(f"TEST: {title}")
        print("-" * 72)
        try:
            print(generate_plan(sql.strip()))
        except Exception as exc:
            import traceback; traceback.print_exc()
        print()
