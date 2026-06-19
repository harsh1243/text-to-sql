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

    dispatch = {
        exp.With:      _walk_cte_node,
        exp.Union:     _walk_union,
        exp.Intersect: _walk_intersect,
        exp.Except:    _walk_except,
        exp.Subquery:  _walk_subquery,
        exp.Select:    _walk_select_node,
    }

    handler = dispatch.get(type(node))
    if handler:
        handler(node, steps, depth)


# ── Node-type-specific handlers ───────────────────────────────────────────────

def _walk_cte_node(node, steps, depth):
    """Handle WITH / CTE node (emitted by both Select and set-ops)."""
    for cte in node.expressions:
        steps.append((depth, "CTE", f"Materialise CTE '{cte.alias}'"))
        _walk(cte.this, steps, depth + 1)


def _walk_union(node, steps, depth):
    """Handle UNION / UNION ALL."""
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


def _walk_intersect(node, steps, depth):
    """Handle INTERSECT / INTERSECT ALL."""
    _maybe_walk_cte(node, steps, depth)
    is_all  = node.args.get("distinct") is False
    op_name = "INTERSECT ALL" if is_all else "INTERSECT"
    steps.append((depth, "SET-OP-LEFT",  f"Evaluate LEFT side of {op_name}"))
    _walk(node.this,       steps, depth + 1)
    steps.append((depth, "SET-OP-RIGHT", f"Evaluate RIGHT side of {op_name}"))
    _walk(node.expression, steps, depth + 1)
    steps.append((depth, op_name,
                  "Keep rows present in BOTH results"
                  + (" (keep duplicates)" if is_all else " (remove duplicates)")))


def _walk_except(node, steps, depth):
    """Handle EXCEPT / EXCEPT ALL."""
    _maybe_walk_cte(node, steps, depth)
    is_all  = node.args.get("distinct") is False
    op_name = "EXCEPT ALL" if is_all else "EXCEPT"
    steps.append((depth, "SET-OP-LEFT",  f"Evaluate LEFT side of {op_name}"))
    _walk(node.this,       steps, depth + 1)
    steps.append((depth, "SET-OP-RIGHT", f"Evaluate RIGHT side of {op_name}"))
    _walk(node.expression, steps, depth + 1)
    steps.append((depth, op_name,
                  "Keep rows in LEFT not present in RIGHT"
                  + (" (keep duplicates)" if is_all else "")))


def _walk_subquery(node, steps, depth):
    """Handle Subquery wrapper — unwrap and walk inner SELECT."""
    _walk(node.this, steps, depth)


def _walk_select_node(node, steps, depth):
    """Handle SELECT node."""
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
            cond = f"ON {on.sql(dialect='sqlite')}"
        elif using:
            cols = ", ".join(c.name for c in using)
            cond = f"USING ({cols})"
        else:
            cond = "(cross / natural)"

        steps.append((depth, f"{jtype} JOIN",
                      f"{jstrat} join — {cond}"))

    # 4. WHERE
    where = node.args.get("where")
    if where:
        _walk_where(where, steps, depth)

    # 5. GROUP BY
    group = node.args.get("group")
    if group:
        cols = ", ".join(e.sql(dialect="sqlite") for e in group.expressions)
        steps.append((depth, "GROUP BY", f"Group rows by: {cols}"))

    # 6. HAVING
    having = node.args.get("having")
    if having:
        steps.append((depth, "HAVING",
                      f"Filter groups: {having.this.sql(dialect='sqlite')}"))

    # 7. SELECT list (projections + subqueries in expressions)
    _walk_select_list(node, steps, depth)

    # 8. DISTINCT
    if node.args.get("distinct"):
        steps.append((depth, "DISTINCT", "Remove duplicate rows"))

    # 9. ORDER BY
    order = node.args.get("order")
    if order:
        cols = ", ".join(
            f"{e.this.sql(dialect='sqlite')} {'DESC' if e.args.get('desc') else 'ASC'}"
            for e in order.expressions
        )
        steps.append((depth, "ORDER BY", f"Sort rows by: {cols}"))

    # 10. LIMIT / OFFSET
    limit  = node.args.get("limit")
    offset = node.args.get("offset")
    if limit:
        lval = limit.this.sql(dialect="sqlite")
        oval = offset.this.sql(dialect="sqlite") if offset else "0"
        steps.append((depth, "LIMIT",
                      f"Return at most {lval} rows, skip {oval}"))


# ── SELECT list walker ────────────────────────────────────────────────────────

def _walk_select_list(node: exp.Select, steps, depth):
    """Emit projection step and recurse into any subqueries in the SELECT list."""
    exprs = node.expressions
    if not exprs:
        return

    cols = ", ".join(e.sql(dialect="sqlite") for e in exprs)
    steps.append((depth, "PROJECT", f"Compute columns: {cols}"))

    for expr in exprs:
        for sub in _subqueries_in_expr(expr):
            steps.append((depth, "SCALAR-SUBQUERY",
                          f"Evaluate scalar subquery: {sub.sql(dialect='sqlite')}"))
            _walk(sub.this, steps, depth + 1)


# ── WHERE walker ──────────────────────────────────────────────────────────────

def _walk_where(where, steps, depth):
    """Emit WHERE step and recurse into any subqueries in the predicate."""
    steps.append((depth, "FILTER",
                  f"Apply WHERE: {where.this.sql(dialect='sqlite')}"))
    for sub in _subqueries_in_expr(where.this):
        steps.append((depth, "SUBQUERY-FILTER",
                      f"Evaluate subquery in WHERE: {sub.sql(dialect='sqlite')}"))
        _walk(sub.this, steps, depth + 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan(source, steps, depth):
    """Emit a scan/read step for a FROM source, recursing into subqueries."""
    if isinstance(source, exp.Subquery):
        alias = source.alias or "<subquery>"
        steps.append((depth, "SUBQUERY", f"Evaluate subquery as '{alias}'"))
        _walk(source.this, steps, depth + 1)
    elif isinstance(source, exp.Table):
        tname = source.name
        alias = source.alias
        label = f"Scan table '{tname}'" + (f" AS {alias}" if alias else "")
        steps.append((depth, "SCAN", label))
    else:
        steps.append((depth, "SOURCE", str(source)))


def _subqueries_in_expr(expr):
    """Yield all Subquery nodes that are direct children of expr."""
    if expr is None:
        return
    for node in expr.walk():
        if isinstance(node, exp.Subquery):
            yield node


def _join_type(join) -> str:
    if join.args.get("cross"):
        return "CROSS"
    if join.args.get("left"):
        return "LEFT"
    if join.args.get("right"):
        return "RIGHT"
    if join.args.get("full"):
        return "FULL OUTER"
    return "INNER"


def _join_strategy(join) -> str:
    if join.args.get("hash"):
        return "Hash"
    if join.args.get("merge"):
        return "Merge"
    return "Nested-loop"


def _visit(node, steps, depth):
    """Visit a node, used for direct dispatch from external callers."""
    _walk(node, steps, depth)


# ── Renderer ──────────────────────────────────────────────────────────────────

def _render(steps) -> str:
    lines = []
    for depth, tag, description in steps:
        indent = "  " * depth
        lines.append(f"{indent}[{tag}] {description}")
    return "\n".join(lines)