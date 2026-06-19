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
    """Step 1: Handle WITH / CTEs attached to this SELECT."""
    _maybe_walk_cte(node, steps, depth)


def _plan_from_joins(node: exp.Select, steps, depth):
    """Step 2: Handle FROM clause and JOINs."""
    from_ = node.args.get("from_")
    if from_:
        _scan(from_.this, steps, depth)
    else:
        steps.append((depth, "DUAL", "No FROM clause — evaluate constant expression"))
        return

    for join in node.args.get("joins") or []:
        join_type = (join.args.get("kind") or "INNER").upper()
        join_tbl  = join.this
        condition = join.args.get("on")
        tbl_name  = _table_name(join_tbl)
        cond_sql  = condition.sql() if condition else "NATURAL / USING"
        steps.append((depth, f"{join_type} JOIN",
                      f"Join with {tbl_name} ON {cond_sql}"))
        _subqueries_in_expr(join_tbl, steps, depth + 1)
        if condition:
            _subqueries_in_expr(condition, steps, depth + 1)


def _plan_predicates(node: exp.Select, steps, depth):
    """Step 3: Handle WHERE clause predicates."""
    where = node.args.get("where")
    if where:
        steps.append((depth, "FILTER (WHERE)", where.this.sql()))
        _subqueries_in_expr(where.this, steps, depth + 1)


def _plan_groupby(node: exp.Select, steps, depth):
    """Step 4: Handle GROUP BY clause."""
    group = node.args.get("group")
    if group:
        cols = ", ".join(e.sql() for e in group.expressions)
        steps.append((depth, "GROUP BY", f"Group rows by {cols}"))


def _plan_having(node: exp.Select, steps, depth):
    """Step 5: Handle HAVING clause."""
    having = node.args.get("having")
    if having:
        steps.append((depth, "FILTER (HAVING)", having.this.sql()))
        _subqueries_in_expr(having.this, steps, depth + 1)


def _plan_projections(node: exp.Select, steps, depth):
    """Step 6: Handle SELECT projections / expressions."""
    expressions = node.expressions
    if expressions:
        cols = ", ".join(e.sql() for e in expressions)
        steps.append((depth, "PROJECT", f"Compute columns: {cols}"))
        for expr in expressions:
            _subqueries_in_expr(expr, steps, depth + 1)


def _plan_distinct(node: exp.Select, steps, depth):
    """Step 7: Handle DISTINCT modifier."""
    if node.args.get("distinct"):
        steps.append((depth, "DISTINCT", "Remove duplicate rows"))


def _plan_orderby(node: exp.Select, steps, depth):
    """Step 8: Handle ORDER BY clause."""
    order = node.args.get("order")
    if order:
        cols = ", ".join(
            f"{e.this.sql()} {'DESC' if e.args.get('desc') else 'ASC'}"
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
    if isinstance(table_node, exp.Subquery):
        alias = table_node.alias or "<subquery>"
        steps.append((depth, "SUBQUERY", f"Evaluate subquery as '{alias}'"))
        _walk(table_node.this, steps, depth + 1)
    elif isinstance(table_node, exp.Table):
        steps.append((depth, "SCAN", f"Full scan of table '{table_node.name}'"))
    else:
        steps.append((depth, "SCAN", f"Scan: {table_node.sql()}"))


# ── Subquery detector ─────────────────────────────────────────────────────────

def _subqueries_in_expr(expr, steps, depth):
    """Walk an expression tree and plan any embedded subqueries."""
    if expr is None:
        return
    for subq in expr.find_all(exp.Subquery):
        alias = subq.alias or "<subquery>"
        steps.append((depth, "SUBQUERY", f"Evaluate subquery as '{alias}'"))
        _walk(subq.this, steps, depth + 1)


# ── Utility ───────────────────────────────────────────────────────────────────

def _table_name(node) -> str:
    """Return a human-readable name for a join target."""
    if isinstance(node, exp.Table):
        return node.name
    if isinstance(node, exp.Subquery):
        return node.alias or "<subquery>"
    return node.sql()


def _visit(node, steps, depth):
    """Generic visitor — delegates back to _walk."""
    _walk(node, steps, depth)


# ── Renderer ──────────────────────────────────────────────────────────────────

def _render(steps) -> str:
    lines = []
    for depth, tag, description in steps:
        indent = "  " * depth
        lines.append(f"{indent}[{tag}] {description}")
    return "\n".join(lines)