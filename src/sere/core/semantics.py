from typing import Any, Dict, Optional, Set, Tuple, List
from dataclasses import dataclass, field
import re
from sere.pddl.grounding import ground_literal
from sere.pddl.sexpr import parse_one, to_string, SExprError
from .world_state import WorldState

Predicate = Tuple[str, Tuple[str, ...]]

NUM_CMP = re.compile(
    r"^\(\s*(<=|>=|<|>|=)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([+-]?\d+(?:\.\d+)?)\s*\)$"
)
NUM_EFF = re.compile(
    r"^\(\s*(increase|decrease|assign)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+(.+)\s*\)$"
)



def _bind_args(argstr: Optional[str], bind: Dict[str, str]) -> Tuple[str, ...]:
    toks = argstr.split() if argstr else []
    return tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in toks)

def eval_num_pre(world: WorldState, expr: str, bind: Dict[str, str]) -> bool:
    m = NUM_CMP.match(expr.strip())
    if not m:
        raise ValueError(f"Bad num_pre: {expr}")
    op, fname, argstr, rhs = m.groups()
    args = _bind_args(argstr, bind)
    val = world.get_fluent(fname, args)
    rhsf = float(rhs)
    if op == "<":  return val < rhsf
    if op == "<=": return val <= rhsf
    if op == ">":  return val > rhsf
    if op == ">=": return val >= rhsf
    return abs(val - rhsf) < 1e-9

def _eval_rhs_token(rhs: str, bind: dict, world: Optional[WorldState] = None) -> float:
    """
    Evaluate RHS as one of:
      - NUMBER
      - ?var
      - NUMBER*<term> | <term>*NUMBER | ?var*?var | <fluent>*NUMBER | NUMBER*<fluent>
      - (fluent args...)  e.g., (water-temp ?k)

    where <term> recursively is NUMBER | ?var | (fluent ...).

    NOTE: when evaluating a fluent, `world` must be provided.
    """
    rhs = rhs.strip()

    # 1) Plain number
    try:
        return float(rhs)
    except ValueError:
        pass

    # 2) Product: allow a*b*c*... (left-assoc)
    if "*" in rhs:
        parts = [p.strip() for p in rhs.split("*")]
        val = _eval_rhs_token(parts[0], bind, world)
        for p in parts[1:]:
            val *= _eval_rhs_token(p, bind, world)
        return val


    # 3) Fluent read: "(name args...)"
    if rhs.startswith("(") and rhs.endswith(")"):
        if world is None:
            raise ValueError("World is required to evaluate fluent RHS.")
        inner = rhs[1:-1].strip()
        if not inner:
            raise ValueError(f"Unsupported numeric RHS: {rhs!r}")
        parts = inner.split()
        fname = parts[0]
        raw_args = parts[1:]
        # bind ?vars in args
        args = tuple(bind.get(a[1:], a) if a.startswith("?") else a for a in raw_args)
        return float(world.get_fluent(fname, args))

    # 4) Variable like "?n"
    if rhs.startswith("?"):
        v = bind.get(rhs[1:], rhs[1:])
        return float(v)

    raise ValueError(f"Unsupported numeric RHS: {rhs!r}")


def apply_num_eff(world: WorldState, expr: str, bind: Dict[str,str], info: Dict[str,Any]):
    m = NUM_EFF.match(expr.strip())
    if not m:
        raise ValueError(f"Bad num_eff: {expr}")
    op, fname, argstr, rhs = m.groups()
    args = _bind_args(argstr, bind)
    d = _eval_rhs_token(rhs, bind, world)  # now supports (fluent ...) and products
    if op == "assign":
        world.set_fluent(fname, args, d)
    elif op == "increase":
        world.set_fluent(fname, args, world.get_fluent(fname, args) + d)
    elif op == "decrease":
        world.set_fluent(fname, args, world.get_fluent(fname, args) - d)
    else:
        raise ValueError(f"Unsupported numeric op: {op}")

# --- Clause evaluator with support for (or ...) and (not ...) ---
def _split_top_level(expr: str) -> List[str]:
    # splits "(or X Y Z)" children into ["X","Y","Z"] using a light parser
    try:
        parsed = parse_one(expr)
    except SExprError as exc:
        raise AssertionError(str(exc)) from exc
    if not isinstance(parsed, list) or not parsed:
        return []
    parts: List[str] = []
    for child in parsed[1:]:
        if isinstance(child, list):
            parts.append(to_string(child))
        else:
            parts.append(str(child))
    return parts

def eval_clause(world: WorldState, static_facts: Set[Predicate], s: str, bind: Dict[str, str], *, enable_numeric: bool = True) -> bool:
    s = s.strip()
    # numeric guard?
    if enable_numeric and NUM_CMP.match(s):
        return eval_num_pre(world, s, bind)

    # built-in: (distinct a b [c ...]) — true iff all bound tokens are pairwise different
    if s.startswith("(distinct"):
        try:
            assert s.endswith(")")
            inner = s[1:-1].strip()        # drop surrounding parens
            parts = inner.split()          # ["distinct", "X", "Y", ...]
            if len(parts) < 3:             # need at least two terms after 'distinct'
                return False
            terms = parts[1:]
            vals = []
            for t in terms:
                v = bind.get(t[1:], t) if t.startswith("?") else t
                vals.append(v)
            return len(vals) == len(set(vals))
        except Exception:
            return False

    # negation
    if s.startswith("(not"):
        # "(not X)" -> extract X
        assert s.endswith(")")
        inner = s[4:-1].strip()
        return not eval_clause(world, static_facts, inner, bind, enable_numeric=enable_numeric)

    # conjunction
    if s.startswith("(and"):
        children = _split_top_level(s)
        return all(
            eval_clause(world, static_facts, c, bind, enable_numeric=enable_numeric)
            for c in children
        )

    # disjunction
    if s.startswith("(or"):
        children = _split_top_level(s)
        return any(
            eval_clause(world, static_facts, c, bind, enable_numeric=enable_numeric)
            for c in children
        )

    # plain literal
    try:
        is_neg, litp = ground_literal(s, bind)
    except Exception:
        # be permissive: unknown pattern -> false
        return False
    h = (world.holds(litp) or (litp in static_facts))
    return (not h) if is_neg else h


# =========================
#  Explanatory evaluation
# =========================

@dataclass
class EvalNode:
    """Explanation node for a (sub)clause."""
    expr: str                       # original (possibly schematic) s-expr
    satisfied: bool                 # evaluation result
    kind: str                       # "lit" | "not" | "or" | "num" | "distinct"
    grounded: Optional[str] = None  # concrete s-expr like "(at r1 kitchen)" when applicable
    children: List["EvalNode"] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)   # e.g., {"op":">=","fluent":"energy","args":("r1",),"current":0,"rhs":1}

def _bind_infix(expr: str, bind: Dict[str, str]) -> str:
    """Lightweight replacement of ?vars in a clause string (best-effort, safe for display)."""
    def _sub(m):
        v = m.group(1)
        return bind.get(v, v)
    return re.sub(r"\?([A-Za-z0-9_\-]+)", _sub, expr)

def trace_clause(world: WorldState, static_facts: Set[Predicate], s: str, bind: Dict[str, str], *, enable_numeric: bool = True) -> EvalNode:
    s = s.strip()
    # numeric comparison
    if enable_numeric and NUM_CMP.match(s):
        m = NUM_CMP.match(s)
        assert m
        op, fname, argstr, rhs = m.groups()
        args = _bind_args(argstr, bind)
        current = world.get_fluent(fname, args)
        rhsf = float(rhs)
        sat = False
        if op == "<":   sat = current < rhsf
        elif op == "<=": sat = current <= rhsf
        elif op == ">":  sat = current > rhsf
        elif op == ">=": sat = current >= rhsf
        else:            sat = abs(current - rhsf) < 1e-9
        return EvalNode(
            expr=_bind_infix(s, bind),
            satisfied=sat,
            kind="num",
            grounded=f"({op} ({fname}{'' if not args else ' ' + ' '.join(args)}) {rhs})",
            details={"op": op, "fluent": fname, "args": args, "current": current, "rhs": rhsf},
        )

    # distinct
    if s.startswith("(distinct"):
        # (distinct a b c ...)
        inner = s[1:-1].strip().split()
        terms = inner[1:]
        bound = [bind.get(t[1:], t) if t.startswith("?") else t for t in terms]
        sat = len(bound) == len(set(bound))
        dupes = sorted(list({x for x in bound if bound.count(x) > 1}))
        return EvalNode(
            expr=_bind_infix(s, bind),
            satisfied=sat,
            kind="distinct",
            grounded=f"(distinct {' '.join(bound)})",
            details={"duplicates": dupes},
        )

    # negation
    if s.startswith("(not"):
        inner = s[4:-1].strip()
        child = trace_clause(world, static_facts, inner, bind, enable_numeric=enable_numeric)
        return EvalNode(
            expr=_bind_infix(s, bind),
            satisfied=not child.satisfied,
            kind="not",
            grounded=None,
            children=[child],
        )

    # conjunction
    if s.startswith("(and"):
        parts = _split_top_level(s)
        kids = [trace_clause(world, static_facts, c, bind, enable_numeric=enable_numeric) for c in parts]
        sat = all(k.satisfied for k in kids)
        return EvalNode(
            expr=_bind_infix(s, bind),
            satisfied=sat,
            kind="and",
            children=kids,
        )

    # disjunction
    if s.startswith("(or"):
        parts = _split_top_level(s)
        kids = [trace_clause(world, static_facts, c, bind, enable_numeric=enable_numeric) for c in parts]
        sat = any(k.satisfied for k in kids)
        return EvalNode(
            expr=_bind_infix(s, bind),
            satisfied=sat,
            kind="or",
            children=kids,
        )


    # plain literal (positive or negated via ground_literal)
    try:
        is_neg, litp = ground_literal(s, bind)
    except Exception:
        # unparsable; treat as false
        return EvalNode(expr=_bind_infix(s, bind), satisfied=False, kind="lit", grounded=None)

    truth = (world.holds(litp) or (litp in static_facts))
    sat = (not truth) if is_neg else truth

    details: Dict[str, Any] = {}
    # helpful hint for co-located
    if litp[0] == "co-located" and len(litp[1]) == 2:
        x, y = litp[1]
        # mimic the logic in WorldState.holds for transparency
        lx = world.locations_of(x)
        ly = world.locations_of(y)
        details["locs_x"] = sorted(lx)
        details["locs_y"] = sorted(ly)

    return EvalNode(
        expr=_bind_infix(s, bind),
        satisfied=sat,
        kind="lit",
        grounded=f"({litp[0]}{' ' if litp[1] else ''}{' '.join(litp[1])})",
        details=details,
    )
