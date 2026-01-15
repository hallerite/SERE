from typing import Any, Dict, Optional, Set, Tuple, List, Iterable
from dataclasses import dataclass, field
import logging
import re
from sere.pddl.grounding import ground_literal
from sere.pddl.sexpr import parse_one, to_string, SExprError
from .world_state import WorldState

logger = logging.getLogger(__name__)

Predicate = Tuple[str, Tuple[str, ...]]

NUM_CMP = re.compile(
    r"^\(\s*(<=|>=|<|>|=)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+(.+)\s*\)$"
)
NUM_EFF = re.compile(
    r"^\(\s*(increase|decrease|assign)\s*\(\s*([^\s()]+)(?:\s+([^)]+))?\)\s+(.+)\s*\)$",
    re.IGNORECASE,
)

NUM_HEADS = {">", "<", ">=", "<=", "="}
LOGICAL_HEADS = {"and", "or", "not"}



def _bind_args(argstr: Optional[str], bind: Dict[str, str]) -> Tuple[str, ...]:
    toks = argstr.split() if argstr else []
    return tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in toks)

def eval_num_pre(world: WorldState, expr: str, bind: Dict[str, str]) -> bool:
    m = NUM_CMP.match(expr.strip())
    if not m:
        raise ValueError(f"Bad num_pre: {expr}")
    op, fname, argstr, rhs = m.groups()
    fname = fname.lower()
    args = _bind_args(argstr, bind)
    val = world.get_fluent(fname, args)
    try:
        rhsf = _eval_rhs_token(rhs.strip(), bind, world)
    except Exception:
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
        fname = parts[0].lower()
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
    op = op.lower()
    fname = fname.lower()
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

def eval_clause(
    world: WorldState,
    static_facts: Set[Predicate],
    s: str,
    bind: Dict[str, str],
    *,
    enable_numeric: bool = True,
    derived_cache: Optional[Dict[Tuple[str, Tuple[str, ...]], bool]] = None,
) -> bool:
    return trace_clause(
        world,
        static_facts,
        s,
        bind,
        enable_numeric=enable_numeric,
        _derived_cache=derived_cache,
    ).satisfied


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


def _parse_quantifier_vars(varlist: list) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    if not isinstance(varlist, list):
        return out
    i = 0
    while i < len(varlist):
        tok = varlist[i]
        if not isinstance(tok, str) or not tok.startswith("?"):
            return []
        var = tok[1:]
        typ: Optional[str] = None
        if i + 2 < len(varlist) and varlist[i + 1] == "-" and isinstance(varlist[i + 2], str):
            typ = str(varlist[i + 2]).lower()
            i += 3
        else:
            i += 1
        out.append((var, typ))
    return out


def _iter_quantifier_bindings(
    world: WorldState,
    varspecs: List[Tuple[str, Optional[str]]],
) -> Iterable[Dict[str, str]]:
    if not varspecs:
        yield {}
        return

    def _candidates(typ: Optional[str]) -> List[str]:
        if not typ:
            return list(world.objects.keys())
        t = typ.lower()
        return [sym for sym, tys in world.objects.items() if t in {str(x).lower() for x in tys}]

    def _rec(i: int, bind: Dict[str, str]):
        if i >= len(varspecs):
            yield dict(bind)
            return
        var, typ = varspecs[i]
        for sym in _candidates(typ):
            bind[var] = sym
            yield from _rec(i + 1, bind)
        bind.pop(var, None)

    yield from _rec(0, {})

def trace_clause(
    world: WorldState,
    static_facts: Set[Predicate],
    s: str,
    bind: Dict[str, str],
    *,
    enable_numeric: bool = True,
    _derived_cache: Optional[Dict[Tuple[str, Tuple[str, ...]], bool]] = None,
    _derived_stack: Optional[Set[Tuple[str, Tuple[str, ...]]]] = None,
) -> EvalNode:
    s = s.strip()
    if _derived_cache is None:
        _derived_cache = {}
    if _derived_stack is None:
        _derived_stack = set()
    # numeric comparison
    if enable_numeric and NUM_CMP.match(s):
        m = NUM_CMP.match(s)
        assert m
        op, fname, argstr, rhs = m.groups()
        fname = fname.lower()
        args = _bind_args(argstr, bind)
        current = world.get_fluent(fname, args)
        try:
            rhsf = _eval_rhs_token(rhs.strip(), bind, world)
        except Exception:
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
            grounded=f"({op} ({fname}{'' if not args else ' ' + ' '.join(args)}) {rhs.strip()})",
            details={"op": op, "fluent": fname, "args": args, "current": current, "rhs": rhsf},
        )

    # distinct
    if s.lstrip().lower().startswith("(distinct"):
        try:
            parsed = parse_one(s)
            if not (isinstance(parsed, list) and parsed and str(parsed[0]).lower() == "distinct"):
                raise SExprError("not distinct")
            terms = [str(x) for x in parsed[1:]]
            if len(terms) < 2:
                raise SExprError("distinct requires at least two terms")
        except SExprError:
            return EvalNode(
                expr=_bind_infix(s, bind),
                satisfied=False,
                kind="distinct",
                grounded=None,
                details={"duplicates": []},
            )
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

    # parse once to detect logical heads
    try:
        parsed = parse_one(s)
    except SExprError:
        parsed = None

    if isinstance(parsed, list) and parsed:
        head = str(parsed[0]).lower()

        # quantifiers
        if head in {"forall", "exists"}:
            if len(parsed) < 3:
                return EvalNode(
                    expr=_bind_infix(s, bind),
                    satisfied=False,
                    kind=head,
                    grounded=None,
                    children=[],
                )
            varspecs = _parse_quantifier_vars(parsed[1]) if isinstance(parsed[1], list) else []
            if not varspecs:
                return EvalNode(
                    expr=_bind_infix(s, bind),
                    satisfied=False,
                    kind=head,
                    grounded=None,
                    children=[],
                )
            body_parts = parsed[2:]
            if len(body_parts) == 1:
                body_expr = to_string(body_parts[0]) if isinstance(body_parts[0], list) else str(body_parts[0])
            else:
                body_expr = to_string(["and"] + body_parts)

            children: List[EvalNode] = []
            details: Dict[str, Any] = {}

            if head == "forall":
                satisfied = True
                for b in _iter_quantifier_bindings(world, varspecs):
                    new_bind = dict(bind)
                    new_bind.update(b)
                    child = trace_clause(
                        world,
                        static_facts,
                        body_expr,
                        new_bind,
                        enable_numeric=enable_numeric,
                        _derived_cache=_derived_cache,
                        _derived_stack=_derived_stack,
                    )
                    if not child.satisfied:
                        satisfied = False
                        children = [child]
                        details = {"binding": b}
                        break
                return EvalNode(
                    expr=_bind_infix(s, bind),
                    satisfied=satisfied,
                    kind=head,
                    grounded=None,
                    children=children,
                    details=details,
                )

            # exists
            satisfied = False
            first_child: Optional[EvalNode] = None
            first_bind: Optional[Dict[str, str]] = None
            for b in _iter_quantifier_bindings(world, varspecs):
                new_bind = dict(bind)
                new_bind.update(b)
                child = trace_clause(
                    world,
                    static_facts,
                    body_expr,
                    new_bind,
                    enable_numeric=enable_numeric,
                    _derived_cache=_derived_cache,
                    _derived_stack=_derived_stack,
                )
                if child.satisfied:
                    satisfied = True
                    children = [child]
                    details = {"binding": b}
                    break
                if first_child is None:
                    first_child = child
                    first_bind = b
            if not satisfied and first_child is not None:
                children = [first_child]
                details = {"binding": first_bind}
            return EvalNode(
                expr=_bind_infix(s, bind),
                satisfied=satisfied,
                kind=head,
                grounded=None,
                children=children,
                details=details,
            )

        # negation
        if head == "not":
            if len(parsed) != 2:
                return EvalNode(
                    expr=_bind_infix(s, bind),
                    satisfied=False,
                    kind="not",
                    grounded=None,
                    children=[],
                )
            child_expr = to_string(parsed[1]) if isinstance(parsed[1], list) else str(parsed[1])
            child = trace_clause(
                world,
                static_facts,
                child_expr,
                bind,
                enable_numeric=enable_numeric,
                _derived_cache=_derived_cache,
                _derived_stack=_derived_stack,
            )
            return EvalNode(
                expr=_bind_infix(s, bind),
                satisfied=not child.satisfied,
                kind="not",
                grounded=None,
                children=[child],
            )

        # conjunction
        if head == "and":
            kids = [
                trace_clause(
                    world,
                    static_facts,
                    to_string(c) if isinstance(c, list) else str(c),
                    bind,
                    enable_numeric=enable_numeric,
                    _derived_cache=_derived_cache,
                    _derived_stack=_derived_stack,
                )
                for c in parsed[1:]
            ]
            sat = all(k.satisfied for k in kids)
            return EvalNode(
                expr=_bind_infix(s, bind),
                satisfied=sat,
                kind="and",
                children=kids,
            )

        # disjunction
        if head == "or":
            kids = [
                trace_clause(
                    world,
                    static_facts,
                    to_string(c) if isinstance(c, list) else str(c),
                    bind,
                    enable_numeric=enable_numeric,
                    _derived_cache=_derived_cache,
                    _derived_stack=_derived_stack,
                )
                for c in parsed[1:]
            ]
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
    except (SExprError, ValueError, KeyError) as e:
        # Expected: malformed S-expression, unbound variable, or invalid binding
        logger.debug(f"Failed to ground literal '{s}' with bindings {bind}: {e}")
        return EvalNode(expr=_bind_infix(s, bind), satisfied=False, kind="lit", grounded=None)

    truth = _holds_literal(
        world,
        static_facts,
        litp,
        bind=bind,
        enable_numeric=enable_numeric,
        derived_cache=_derived_cache,
        derived_stack=_derived_stack,
    )
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


def _holds_literal(
    world: WorldState,
    static_facts: Set[Predicate],
    litp: Predicate,
    *,
    bind: Dict[str, str],
    enable_numeric: bool,
    derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool],
    derived_stack: Set[Tuple[str, Tuple[str, ...]]],
) -> bool:
    if litp in world.facts or litp in static_facts:
        return True
    name, args = litp
    rules = getattr(world.domain, "derived", {}).get(name, [])
    if not rules:
        return False
    key = (name, args)
    if key in derived_cache:
        return derived_cache[key]
    if key in derived_stack:
        return False

    derived_stack.add(key)
    try:
        for rule in rules:
            bind_rule = _bind_derived_head(rule.head_terms, args)
            if bind_rule is None:
                continue
            if _derived_rule_holds(
                world,
                static_facts,
                rule,
                bind_rule,
                enable_numeric=enable_numeric,
                derived_cache=derived_cache,
                derived_stack=derived_stack,
            ):
                derived_cache[key] = True
                return True
        derived_cache[key] = False
        return False
    finally:
        derived_stack.discard(key)


def _bind_derived_head(head_terms: List[str], args: Tuple[str, ...]) -> Optional[Dict[str, str]]:
    if len(head_terms) != len(args):
        return None
    bind: Dict[str, str] = {}
    for term, arg in zip(head_terms, args):
        if term.startswith("?"):
            v = term[1:]
            if v in bind and bind[v] != arg:
                return None
            bind[v] = arg
        else:
            if term != arg:
                return None
    return bind


def _derived_rule_holds(
    world: WorldState,
    static_facts: Set[Predicate],
    rule,
    bind: Dict[str, str],
    *,
    enable_numeric: bool,
    derived_cache: Dict[Tuple[str, Tuple[str, ...]], bool],
    derived_stack: Set[Tuple[str, Tuple[str, ...]]],
) -> bool:
    vars_all = set(bind.keys())
    rule_vars, rule_constraints = _rule_meta(rule, world)
    vars_all |= rule_vars
    extra_vars = sorted(vars_all - set(bind.keys()))
    if not extra_vars:
        return all(
            trace_clause(
                world,
                static_facts,
                c,
                bind,
                enable_numeric=enable_numeric,
                _derived_cache=derived_cache,
                _derived_stack=derived_stack,
            ).satisfied
            for c in (rule.when or [])
        )

    candidates = _candidate_map(world, extra_vars, rule_constraints)
    if any(not v for v in candidates.values()):
        return False

    def _iter_assignments(
        vars_list: List[str],
        base: Dict[str, str],
    ) -> Iterable[Dict[str, str]]:
        if not vars_list:
            yield dict(base)
            return
        v = vars_list[0]
        for cand in candidates.get(v, []):
            base[v] = cand
            yield from _iter_assignments(vars_list[1:], base)
        base.pop(v, None)

    # smallest domains first
    ordered = sorted(extra_vars, key=lambda v: len(candidates.get(v, [])))

    for assignment in _iter_assignments(ordered, dict(bind)):
        if all(
            trace_clause(
                world,
                static_facts,
                c,
                assignment,
                enable_numeric=enable_numeric,
                _derived_cache=derived_cache,
                _derived_stack=derived_stack,
            ).satisfied
            for c in (rule.when or [])
        ):
            return True
    return False


def _vars_in_expr(expr: str) -> Set[str]:
    try:
        parsed = parse_one(expr)
    except SExprError as e:
        logger.debug(f"Failed to parse expression '{expr}' for variable extraction: {e}")
        return set()

    return _vars_in_parsed(parsed)


def _vars_in_parsed(parsed: Any) -> Set[str]:
    out: Set[str] = set()

    def _walk(node):
        if isinstance(node, list):
            for x in node:
                _walk(x)
        else:
            tok = str(node)
            if tok.startswith("?") and len(tok) > 1:
                out.add(tok[1:])

    _walk(parsed)
    return out


def _candidate_map(
    world: WorldState,
    vars_needed: List[str],
    constraints: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    all_objs = sorted(world.objects.keys())
    candidates: Dict[str, List[str]] = {}
    for v in vars_needed:
        reqs = constraints.get(v) or set()
        if "number" in reqs:
            candidates[v] = []
            continue
        if not reqs:
            candidates[v] = list(all_objs)
            continue
        matches = []
        for sym, tys in world.objects.items():
            if not tys:
                continue
            ok = True
            for req in reqs:
                if not any(world.domain.is_subtype(t, req) for t in tys):
                    ok = False
                    break
            if ok:
                matches.append(sym)
        candidates[v] = sorted(matches)
    return candidates


def _collect_type_constraints(
    parsed: Any,
    world: WorldState,
    out: Dict[str, Set[str]],
) -> None:
    if not isinstance(parsed, list) or not parsed:
        return
    head = str(parsed[0]).lower()
    if head in LOGICAL_HEADS:
        for x in parsed[1:]:
            _collect_type_constraints(x, world, out)
        return
    if head == "distinct":
        return
    if head in NUM_HEADS:
        if len(parsed) < 2:
            return
        term = parsed[1]
        if isinstance(term, list) and term:
            fname = str(term[0]).lower()
            spec = world.domain.fluents.get(fname)
            if not spec:
                return
            for arg, (_var, typ) in zip(term[1:], spec.args):
                if isinstance(arg, str) and arg.startswith("?"):
                    out.setdefault(arg[1:], set()).add(str(typ).lower())
        return
    spec = world.domain.predicates.get(head)
    if not spec:
        return
    for arg, (_var, typ) in zip(parsed[1:], spec.args):
        if isinstance(arg, str) and arg.startswith("?"):
            out.setdefault(arg[1:], set()).add(str(typ).lower())


def _rule_meta(rule, world: WorldState) -> Tuple[Set[str], Dict[str, Set[str]]]:
    vars_cached = getattr(rule, "vars_in_when", None)
    cons_cached = getattr(rule, "constraints", None)
    if vars_cached is not None and cons_cached is not None:
        return vars_cached, cons_cached

    vars_all: Set[str] = set()
    constraints: Dict[str, Set[str]] = {}
    for clause in (rule.when or []):
        try:
            parsed = parse_one(clause)
        except SExprError as e:
            logger.debug(f"Failed to parse clause '{clause}' in derived rule: {e}")
            continue
        vars_all |= _vars_in_parsed(parsed)
        _collect_type_constraints(parsed, world, constraints)

    # Cache on rule object if it's mutable (some rule objects may be frozen)
    if hasattr(rule, '__dict__'):
        try:
            rule.vars_in_when = vars_all
            rule.constraints = constraints
        except AttributeError:
            # Rule object is frozen/immutable, can't cache
            # This is OK, we'll recompute next time
            pass

    return vars_all, constraints
