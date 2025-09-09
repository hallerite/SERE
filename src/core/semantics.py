from typing import Any, Dict, Optional, Set, Tuple, List
import re
from ..pddl.grounding import ground_literal
from .world_state import WorldState

Predicate = Tuple[str, Tuple[str, ...]]

NUM_CMP = re.compile(
    r"^\(\s*(<=|>=|<|>|=)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([+-]?\d+(?:\.\d+)?)\s*\)$"
)
NUM_EFF = re.compile(
    r"^\(\s*(increase|decrease|assign|cost)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+([+-]?\d+(?:\.\d+)?)\s*\)$"
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

def apply_num_eff(world: WorldState, expr: str, bind: Dict[str,str], info: Dict[str,Any]):
    m = NUM_EFF.match(expr.strip())
    if not m: 
        raise ValueError(f"Bad num_eff: {expr}")
    op, fname, argstr, delta = m.groups()
    args  = _bind_args(argstr, bind)
    d = float(delta)
    if op == "assign":
        world.set_fluent(fname, args, d)
    elif op == "increase":
        world.set_fluent(fname, args, world.get_fluent(fname, args) + d)
    elif op == "decrease":
        world.set_fluent(fname, args, world.get_fluent(fname, args) - d)
    elif op == "cost":
        info["action_cost"] = info.get("action_cost", 0.0) + d

# --- Clause evaluator with support for (or ...) and (not ...) ---
def _split_top_level(expr: str) -> List[str]:
    # splits "(or X Y Z)" children into ["X","Y","Z"] without parsing deeply
    expr = expr.strip()
    assert expr.startswith("(") and expr.endswith(")")
    inner = expr[1:-1].strip()
    # remove leading "or" or "not"
    # when called for or: inner startswith "or "
    parts = []
    depth = 0
    cur = []
    # skip head symbol
    tokens = list(inner)
    i = 0
    # skip head
    while i < len(tokens) and not tokens[i].isspace():
        i += 1
    # skip spaces after head
    while i < len(tokens) and tokens[i].isspace():
        i += 1
    # now parse children s-exprs
    while i < len(tokens):
        ch = tokens[i]
        cur.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                parts.append("".join(cur).strip())
                cur = []
                # skip spaces between children
                i += 1
                while i < len(tokens) and tokens[i].isspace():
                    i += 1
                continue
        i += 1
    # also allow bare atoms inside or (rare)
    if cur:
        s = "".join(cur).strip()
        if s:
            parts.append(s)
    return parts

def eval_clause(world: WorldState, static_facts: Set[Predicate], s: str, bind: Dict[str, str], *, enable_numeric: bool = True) -> bool:
    s = s.strip()
    # numeric guard?
    if enable_numeric and NUM_CMP.match(s):
        return eval_num_pre(world, s, bind)
    # negation?
    if s.startswith("(not"):
        # "(not X)" -> extract X
        assert s.endswith(")")
        inner = s[4:-1].strip()
        return not eval_clause(world, static_facts, inner, bind, enable_numeric=enable_numeric)
    # disjunction?
    if s.startswith("(or"):
        children = _split_top_level(s)
        return any(eval_clause(world, static_facts, c, bind, enable_numeric=enable_numeric) for c in children)
    # plain literal
    try:
        is_neg, litp = ground_literal(s, bind)
    except Exception:
        # be permissive: unknown pattern -> false
        return False
    h = (world.holds(litp) or (litp in static_facts))
    return (not h) if is_neg else h
