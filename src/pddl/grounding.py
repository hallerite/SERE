import re
from typing import Dict, Tuple
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
    toks = template.replace("("," ").replace(")"," ").split()
    name = toks[0]
    args = tuple(bind[t[1:]] if t.startswith("?") else t for t in toks[1:])
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
