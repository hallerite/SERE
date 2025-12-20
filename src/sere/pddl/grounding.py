from typing import Dict, Tuple, List
from .domain_spec import ActionSpec, Predicate
from .sexpr import parse_one, SExprError

def parse_grounded(expr: str) -> Tuple[str, Tuple[str, ...]]:
    try:
        parsed = parse_one(expr)
    except SExprError as exc:
        raise ValueError("Expected single S-expression: '(name arg1 arg2 ...)'") from exc
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
        raise ValueError("Expected single S-expression: '(name arg1 arg2 ...)'")
    if any(isinstance(x, list) for x in parsed[1:]):
        raise ValueError("Expected grounded action: '(name arg1 arg2 ...)'")
    name = str(parsed[0]).lower()
    args = tuple(str(a) for a in parsed[1:])
    return name, args

def substitute(template: str, bind: Dict[str, str]) -> Predicate:
    try:
        parsed = parse_one(template)
    except SExprError as exc:
        raise ValueError(f"Bad literal: {template}") from exc
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
        raise ValueError(f"Bad literal: {template}")
    if any(isinstance(x, list) for x in parsed[1:]):
        raise ValueError(f"Bad literal (nested expr): {template}")
    name = str(parsed[0]).lower()
    args = tuple(
        bind.get(t[1:], t) if t.startswith("?") else t for t in (str(x) for x in parsed[1:])
    )
    return (name, args)

def ground_literal(template: str, bind: Dict[str, str]) -> Tuple[bool, Predicate]:
    try:
        parsed = parse_one(template)
    except SExprError as exc:
        raise ValueError(f"Bad literal: {template}") from exc

    if isinstance(parsed, list) and parsed and str(parsed[0]).lower() == "not":
        if len(parsed) != 2 or not isinstance(parsed[1], list) or not parsed[1]:
            raise ValueError(f"Bad literal: {template}")
        inner = parsed[1]
        if not isinstance(inner[0], str) or any(isinstance(x, list) for x in inner[1:]):
            raise ValueError(f"Bad literal: {template}")
        name = str(inner[0]).lower()
        args = tuple(
            bind.get(t[1:], t) if t.startswith("?") else t
            for t in (str(x) for x in inner[1:])
        )
        return True, (name, args)

    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
        raise ValueError(f"Bad literal: {template}")
    if any(isinstance(x, list) for x in parsed[1:]):
        raise ValueError(f"Bad literal: {template}")
    name = str(parsed[0]).lower()
    args = tuple(
        bind.get(t[1:], t) if t.startswith("?") else t for t in (str(x) for x in parsed[1:])
    )
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
