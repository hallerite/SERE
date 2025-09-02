import re
from typing import Dict, List, Tuple
from .domain_spec import DomainSpec, ActionSpec, Predicate

SEXPR = re.compile(r"^\(([^()\s]+)(?:\s+([^()]+))?\)$")

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

def instantiate(dom: DomainSpec, act: ActionSpec, args: Tuple[str, ...]):
    if len(args) != len(act.params):
        raise ValueError(f"Arity mismatch for {act.name}")
    bind = {var: val for (var,_), val in zip(act.params, args)}
    pre  = [substitute(s, bind) for s in act.pre]
    add  = [substitute(s, bind) for s in act.add]
    dele = [substitute(s, bind) for s in act.delete]
    return pre, add, dele
