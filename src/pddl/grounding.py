import re
from typing import Dict, Tuple, List
from .domain_spec import ActionSpec, Predicate

SEXPR = re.compile(r"^\(([^()\s]+)(?:\s+([^()]+))?\)$")
NOT_RE = re.compile(r"^\(\s*not\s*\(([^()\s]+)\s+([^)]+)\)\s*\)$")

def parse_grounded(expr: str) -> Tuple[str, Tuple[str, ...]]:
    m = SEXPR.match(expr.strip())
    if not m:
        raise ValueError("Expected single S-expression: '(name arg1 arg2 ...)'")
    name = m.group(1).lower()
    args = tuple(m.group(2).split()) if m.group(2) else tuple()
    return name, args

def substitute(template: str, bind: Dict[str, str]) -> Predicate:
    toks = template.replace("(", " ").replace(")", " ").split()
    name = toks[0]
    args = tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in toks[1:])
    return (name, args)

def ground_literal(template: str, bind: Dict[str, str]) -> Tuple[bool, Predicate]:
    s = template.strip()
    m = NOT_RE.match(s)
    if m:
        name, argstr = m.group(1), m.group(2)
        args = tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in argstr.split())
        return True, (name, args)
    m = SEXPR.match(s)
    if not m:
        raise ValueError(f"Bad literal: {s}")
    name = m.group(1)
    argstr = (m.group(2) or "").strip()
    args = tuple(argstr.split()) if argstr else tuple()
    args = tuple(bind.get(t[1:], t) if t.startswith("?") else t for t in args)
    return False, (name, args)

def instantiate(domain, act: ActionSpec, args: Tuple[str, ...]) -> Tuple[Tuple[str, Tuple[str, ...]], List[Predicate], List[Predicate]]:
    """
    Ground an action's effects.
    Returns: ((action-name, grounded-args), add-list, delete-list)
    """
    if len(args) != len(act.params):
        raise ValueError(f"Arity mismatch for action '{act.name}': expected {len(act.params)}, got {len(args)}")

    bind: Dict[str, str] = {var: val for (var, _), val in zip(act.params, args)}

    add: List[Predicate] = [substitute(t, bind) for t in (act.add or [])]
    dele: List[Predicate] = [substitute(t, bind) for t in (act.delete or [])]

    return (act.name, args), add, dele