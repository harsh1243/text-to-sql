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
    with_ = node.args.get("with_")           # key is "with_" not "with"
    if with_:
        _walk(with_, steps, depth)


# ── SELECT planner (decomposed) ───────────────────────────────────────────────

def _plan_select(node: exp.Select, steps, depth):
    """Coordinate all SELECT clause planning by delegating to focused helpers."""
    _plan_ctes(node, steps, depth)
    _plan_from_joins(node, steps, depth)
    _plan_predicates(node, steps, depth)
    _plan_groupby(node, steps, depth)
    _plan_having(node, steps, depth)
    _plan_projections(node, steps, depth)
    _plan_distinct(node, steps, depth)
    _plan_orderby(node, steps, depth)
    _plan_limit_offset(node, steps, depth)


def _plan_ctes(node: exp.Select, steps, depth):
    """Step 0: Walk any CTEs attached to this SELECT."""
    _maybe_walk_cte(node, steps, depth)


def _plan_from_joins(node: exp.Select, steps, depth):
    """Step 1 & 2: Emit SCAN/JOIN steps for FROM clause and joins."""
    from_ = node.args.get("from")
    if not from_:
        return

    _scan(from_.this, steps, depth)

    for join in node.args.get("joins") or []:
        join_type = (join.args.get("kind") or "INNER").upper()
        join_expr = join.args.get("on") or join.args.get("using")
        cond_sql  = join_expr.sql() if join_expr else "(cross)"
        tbl       = join.this
        tbl_name  = tbl.alias_or_name if hasattr(tbl, "alias_or_name") else tbl.sql()
        steps.append((depth, f"{join_type} JOIN",
                      f"Join with {tbl_name} ON {cond_sql}"))
        _scan(tbl, steps, depth + 1)

        # Subqueries inside JOIN ON / USING
        if join_expr:
            for sq_steps in _subqueries_in_expr(join_expr, depth + 1):
                steps.extend(sq_steps)


def _plan_predicates(node: exp.Select, steps, depth):
    """Step 3: Emit FILTER step for WHERE clause."""
    where = node.args.get("where")
    if not where:
        return

    steps.append((depth, "FILTER", f"WHERE {where.this.sql()}"))
    for sq_steps in _subqueries_in_expr(where.this, depth + 1):
        steps.extend(sq_steps)


def _plan_groupby(node: exp.Select, steps, depth):
    """Step 4: Emit GROUP-BY step."""
    group = node.args.get("group")
    if not group:
        return

    cols = ", ".join(e.sql() for e in group.expressions)
    steps.append((depth, "GROUP BY", f"Group by {cols}"))


def _plan_having(node: exp.Select, steps, depth):
    """Step 5: Emit HAVING step."""
    having = node.args.get("having")
    if not having:
        return

    steps.append((depth, "HAVING", f"Having {having.this.sql()}"))
    for sq_steps in _subqueries_in_expr(having.this, depth + 1):
        steps.extend(sq_steps)


def _plan_projections(node: exp.Select, steps, depth):
    """Step 6: Emit PROJECT step and handle subqueries in SELECT list."""
    exprs = node.expressions
    if not exprs:
        return

    cols = ", ".join(e.sql() for e in exprs)
    steps.append((depth, "PROJECT", f"Select {cols}"))
    for expr in exprs:
        for sq_steps in _subqueries_in_expr(expr, depth + 1):
            steps.extend(sq_steps)


def _plan_distinct(node: exp.Select, steps, depth):
    """Step 7: Emit DISTINCT step if applicable."""
    if node.args.get("distinct"):
        steps.append((depth, "DISTINCT", "Remove duplicate rows"))


def _plan_orderby(node: exp.Select, steps, depth):
    """Step 8: Emit SORT step for ORDER BY clause."""
    order = node.args.get("order")
    if not order:
        return

    cols = ", ".join(
        e.sql()
        for e in order.expressions
    )
    steps.append((depth, "SORT", f"Order by {cols}"))


def _plan_limit_offset(node: exp.Select, steps, depth):
    """Step 9: Handle LIMIT and OFFSET clauses."""
    limit  = node.args.get("limit")
    offset = node.args.get("offset")
    if limit:
        limit_val  = limit.this.sql()
        offset_val = offset.this.sql() if offset else "0"
        steps.append((depth, "LIMIT/OFFSET",
                      f"Return at most {limit_val} rows, skip {offset_val}"))


# ── Scan helper ───────────────────────────────────────────────────────────────

def _scan(table_node, steps, depth):
    """Emit a SCAN or subquery step for a FROM target."""
    if table_node is None:
        return

    if isinstance(table_node, exp.Subquery):
        alias = table_node.alias or "<subquery>"
        steps.append((depth, "SUBQUERY", f"Evaluate subquery '{alias}'"))
        _walk(table_node.this, steps, depth + 1)
    elif isinstance(table_node, exp.Table):
        name = table_node.alias_or_name
        steps.append((depth, "SCAN", f"Scan table '{name}'"))
    else:
        steps.append((depth, "SCAN", f"Scan {table_node.sql()}"))


# ── Subquery extraction (decomposed) ─────────────────────────────────────────

def _subqueries_in_expr(expr, depth):
    """
    Yield lists of steps for every subquery found within *expr*.

    Delegates to focused helpers for each expression category.
    """
    if expr is None:
        return

    yield from _visit_expr(expr, depth)


def _visit_expr(expr, depth):
    """Dispatch a single expression node to the appropriate visitor."""
    if isinstance(expr, (exp.Subquery, exp.Select)):
        yield from _visit_subquery_node(expr, depth)
    elif isinstance(expr, exp.Exists):
        yield from _visit_exists(expr, depth)
    elif isinstance(expr, (exp.In, exp.Any, exp.All)):
        yield from _visit_in_any_all(expr, depth)
    else:
        yield from _visit_generic(expr, depth)


def _visit_subquery_node(expr, depth):
    """Handle a Subquery or bare Select node — collect its steps."""
    inner = expr.this if isinstance(expr, exp.Subquery) else expr
    sq_steps = []
    _walk(inner, sq_steps, depth)
    if sq_steps:
        yield sq_steps


def _visit_exists(expr, depth):
    """Handle EXISTS(subquery) expressions."""
    inner = expr.this
    if inner is not None:
        yield from _visit_expr(inner, depth)


def _visit_in_any_all(expr, depth):
    """Handle IN / ANY / ALL subquery expressions."""
    # The subquery is in expr.query or expr.this depending on expression type
    query = expr.args.get("query") or (
        expr.this if isinstance(expr.this, (exp.Subquery, exp.Select)) else None
    )
    if query is not None:
        yield from _visit_expr(query, depth)
    else:
        # Fall back to scanning all child args
        yield from _visit_children(expr, depth)


def _visit_generic(expr, depth):
    """Recursively visit all child nodes of a generic expression."""
    yield from _visit_children(expr, depth)


def _visit_children(expr, depth):
    """Iterate over all child expression nodes and visit each."""
    for child in expr.args.values():
        if isinstance(child, exp.Expression):
            yield from _visit_expr(child, depth)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, exp.Expression):
                    yield from _visit_expr(item, depth)


# ── Renderer ──────────────────────────────────────────────────────────────────

def _render(steps) -> str:
    """Convert the list of (depth, tag, description) tuples to a readable string."""
    if not steps:
        return "(no steps)"

    lines = []
    for depth, tag, description in steps:
        indent = "  " * depth
        lines.append(f"{indent}[{tag}] {description}")
    return "\n".join(lines)