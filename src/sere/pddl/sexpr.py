from __future__ import annotations

from typing import List, Union

SExpr = Union[str, List["SExpr"]]


class SExprError(ValueError):
    pass


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(text or "")
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("(", ")"):
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < n and (not text[j].isspace()) and text[j] not in ("(", ")"):
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def parse_many(text: str) -> List[SExpr]:
    tokens = tokenize(text)
    stack: List[List[SExpr]] = []
    cur: List[SExpr] = []

    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            if not stack:
                raise SExprError("Unbalanced ')'")
            expr = cur
            cur = stack.pop()
            cur.append(expr)
        else:
            cur.append(tok)

    if stack:
        raise SExprError("Unbalanced '('")
    return cur


def parse_one(text: str) -> SExpr:
    exprs = parse_many(text)
    if len(exprs) != 1:
        raise SExprError("Expected single S-expression")
    return exprs[0]


def to_string(expr: SExpr) -> str:
    if isinstance(expr, list):
        return "(" + " ".join(to_string(x) for x in expr) + ")"
    return str(expr)
