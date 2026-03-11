"""
PDDL parser: reads standard .pddl domain and problem files into SERE data structures.

Supports PDDL 1.0-2.2 features:
  - :strips, :typing, :equality, :adl
  - :conditional-effects (when, forall)
  - :numeric-fluents (increase, decrease, assign, :functions)
  - :derived-predicates
  - :action-costs
  - :negative-preconditions
  - :disjunctive-preconditions
  - :constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .domain_spec import (
    ActionSpec,
    ConditionalBlock,
    DerivedRule,
    DomainSpec,
    FluentSpec,
    OutcomeSpec,
    PredicateSpec,
    _as_nl_list,
    _parse_derived as _parse_derived_rules,
    _validate_static_effects,
)
from .sexpr import SExpr, SExprError, parse_many, parse_one, to_string


# ---------------------------------------------------------------------------
#  Intermediate representations (raw PDDL parse output)
# ---------------------------------------------------------------------------

@dataclass
class PDDLAction:
    name: str
    params: List[Tuple[str, str]]
    pre: List[str]
    add: List[str]
    delete: List[str]
    num_eff: List[str] = field(default_factory=list)
    cond: List[PDDLConditionalEffect] = field(default_factory=list)


@dataclass
class PDDLConditionalEffect:
    forall: List[Tuple[str, str]]
    when: List[str]
    add: List[str]
    delete: List[str]
    num_eff: List[str] = field(default_factory=list)


@dataclass
class PDDLDerived:
    head: str
    body: str


@dataclass
class PDDLDomain:
    name: str
    requirements: List[str]
    types: Dict[str, str]
    constants: List[Tuple[str, str]]
    predicates: Dict[str, List[Tuple[str, str]]]
    functions: Dict[str, List[Tuple[str, str]]]
    actions: Dict[str, PDDLAction]
    derived: List[PDDLDerived]


@dataclass
class PDDLProblem:
    name: str
    domain_name: str
    objects: List[Tuple[str, str]]
    init_facts: List[Tuple[str, Tuple[str, ...]]]
    init_fluents: List[Tuple[str, Tuple[str, ...], float]]
    goal: str


# ---------------------------------------------------------------------------
#  Comment / whitespace helpers
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove PDDL line-comments (;)."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
#  Typed-list parsing   ?v1 ?v2 - type ?v3 - type2 ...
# ---------------------------------------------------------------------------

def _flatten(nodes: list) -> List[str]:
    """Flatten nested lists to a flat list of string tokens."""
    out: List[str] = []
    for n in nodes:
        if isinstance(n, list):
            out.extend(_flatten(n))
        else:
            out.append(str(n))
    return out


def _parse_typed_list(tokens: list, *, strip_question: bool = False) -> List[Tuple[str, str]]:
    """
    Parse PDDL typed-list:  ``a b - T  c - T2``  ->  ``[('a','T'), ('b','T'), ('c','T2')]``

    If strip_question is True, leading '?' is removed from names (for action params).
    """
    flat = _flatten(tokens)
    result: List[Tuple[str, str]] = []
    buf: List[str] = []
    i = 0
    while i < len(flat):
        tok = flat[i]
        if tok == "-":
            if i + 1 >= len(flat):
                break
            typ = flat[i + 1].lower()
            for name in buf:
                n = name.lstrip("?") if strip_question else name
                result.append((n, typ))
            buf = []
            i += 2
        else:
            buf.append(tok)
            i += 1
    # untyped remainder -> type "object"
    for name in buf:
        if name == "-":
            continue
        n = name.lstrip("?") if strip_question else name
        result.append((n, "object"))
    return result


# ---------------------------------------------------------------------------
#  Section dispatch
# ---------------------------------------------------------------------------

def _section_key(node) -> Optional[str]:
    """Return the keyword (lowered) of a section like [':types', ...], or None."""
    if isinstance(node, list) and node and isinstance(node[0], str):
        k = node[0].lower()
        if k.startswith(":"):
            return k
    return None


# ---------------------------------------------------------------------------
#  Type parsing
# ---------------------------------------------------------------------------

def _parse_types_section(section: list) -> Dict[str, str]:
    """Parse (:types ...) -> {child: parent}."""
    # section[0] is ':types', rest are tokens or sublists
    tokens = _flatten(section[1:])
    result: Dict[str, str] = {}
    buf: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            if i + 1 >= len(tokens):
                break
            parent = tokens[i + 1].lower()
            for child in buf:
                result[child.lower()] = parent
            buf = []
            i += 2
        else:
            buf.append(tok)
            i += 1
    for child in buf:
        if child != "-":
            result[child.lower()] = ""
    return result


# ---------------------------------------------------------------------------
#  Predicate / function parsing
# ---------------------------------------------------------------------------

def _parse_predicates_section(section: list) -> Dict[str, List[Tuple[str, str]]]:
    """Parse (:predicates ...) -> {name: [(var, type), ...]}."""
    result: Dict[str, List[Tuple[str, str]]] = {}
    for child in section[1:]:
        if not isinstance(child, list) or not child:
            continue
        name = str(child[0]).lower()
        args = _parse_typed_list(child[1:], strip_question=True)
        result[name] = args
    return result


def _parse_functions_section(section: list) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse (:functions ...) -> {name: [(var, type), ...]}.
    Handles return-type annotations like ``(cost) - number``.
    """
    result: Dict[str, List[Tuple[str, str]]] = {}
    # Collect function definitions (sublists), skip bare tokens used for return types
    for child in section[1:]:
        if isinstance(child, list) and child:
            name = str(child[0]).lower()
            args = _parse_typed_list(child[1:], strip_question=True)
            result[name] = args
    return result


# ---------------------------------------------------------------------------
#  Precondition decomposition
# ---------------------------------------------------------------------------

def _precondition_to_list(pre: Optional[SExpr]) -> List[str]:
    """Unwrap top-level (and ...) into a list of S-expression strings."""
    if pre is None:
        return []
    if isinstance(pre, str):
        return [pre]
    if isinstance(pre, list) and pre:
        head = str(pre[0]).lower()
        if head == "and":
            return [to_string(child) for child in pre[1:]]
    return [to_string(pre)]


# ---------------------------------------------------------------------------
#  Effect decomposition (the hard part)
# ---------------------------------------------------------------------------

_NUMERIC_HEADS = {"increase", "decrease", "assign", "scale-up", "scale-down"}


def _decompose_effect(
    effect: Optional[SExpr],
) -> Tuple[List[str], List[str], List[str], List[PDDLConditionalEffect]]:
    """Decompose a PDDL :effect into (add, delete, num_eff, conditional_blocks)."""
    add: List[str] = []
    delete: List[str] = []
    num_eff: List[str] = []
    cond: List[PDDLConditionalEffect] = []
    if effect is not None:
        _walk_effect(effect, add, delete, num_eff, cond)
    return add, delete, num_eff, cond


def _walk_effect(
    node: SExpr,
    add: List[str],
    delete: List[str],
    num_eff: List[str],
    cond: List[PDDLConditionalEffect],
) -> None:
    if not isinstance(node, list) or not node:
        return

    head = str(node[0]).lower()

    if head == "and":
        for child in node[1:]:
            _walk_effect(child, add, delete, num_eff, cond)

    elif head == "not":
        if len(node) >= 2:
            delete.append(to_string(node[1]))

    elif head in _NUMERIC_HEADS:
        num_eff.append(to_string(node))

    elif head == "when":
        if len(node) >= 3:
            condition = node[1]
            eff_part = node[2]
            when_list = _precondition_to_list(condition)
            inner_add: List[str] = []
            inner_del: List[str] = []
            inner_num: List[str] = []
            _walk_effect_simple(eff_part, inner_add, inner_del, inner_num)
            cond.append(PDDLConditionalEffect(
                forall=[], when=when_list,
                add=inner_add, delete=inner_del, num_eff=inner_num,
            ))

    elif head == "forall":
        if len(node) >= 3:
            qvars = _parse_typed_list(node[1], strip_question=True)
            body = node[2]
            if isinstance(body, list) and body and str(body[0]).lower() == "when":
                # (forall (?v - T) (when <cond> <eff>))
                if len(body) >= 3:
                    when_list = _precondition_to_list(body[1])
                    inner_add2: List[str] = []
                    inner_del2: List[str] = []
                    inner_num2: List[str] = []
                    _walk_effect_simple(body[2], inner_add2, inner_del2, inner_num2)
                    cond.append(PDDLConditionalEffect(
                        forall=qvars, when=when_list,
                        add=inner_add2, delete=inner_del2, num_eff=inner_num2,
                    ))
            elif isinstance(body, list) and body and str(body[0]).lower() == "and":
                # (forall (?v - T) (and ...)) - might contain when blocks inside
                inner_cond: List[PDDLConditionalEffect] = []
                inner_add3: List[str] = []
                inner_del3: List[str] = []
                inner_num3: List[str] = []
                for child in body[1:]:
                    if isinstance(child, list) and child and str(child[0]).lower() == "when":
                        if len(child) >= 3:
                            wl = _precondition_to_list(child[1])
                            ia: List[str] = []
                            id_: List[str] = []
                            inu: List[str] = []
                            _walk_effect_simple(child[2], ia, id_, inu)
                            cond.append(PDDLConditionalEffect(
                                forall=qvars, when=wl,
                                add=ia, delete=id_, num_eff=inu,
                            ))
                    else:
                        _walk_effect_simple(child, inner_add3, inner_del3, inner_num3)
                if inner_add3 or inner_del3 or inner_num3:
                    cond.append(PDDLConditionalEffect(
                        forall=qvars, when=[],
                        add=inner_add3, delete=inner_del3, num_eff=inner_num3,
                    ))
            else:
                # (forall (?v - T) <simple-effect>)
                inner_add4: List[str] = []
                inner_del4: List[str] = []
                inner_num4: List[str] = []
                _walk_effect_simple(body, inner_add4, inner_del4, inner_num4)
                cond.append(PDDLConditionalEffect(
                    forall=qvars, when=[],
                    add=inner_add4, delete=inner_del4, num_eff=inner_num4,
                ))
    else:
        # Positive literal
        add.append(to_string(node))


def _walk_effect_simple(
    node: SExpr,
    add: List[str],
    delete: List[str],
    num_eff: List[str],
) -> None:
    """Walk effect without handling forall/when (for inner effects)."""
    if not isinstance(node, list) or not node:
        return
    head = str(node[0]).lower()
    if head == "and":
        for child in node[1:]:
            _walk_effect_simple(child, add, delete, num_eff)
    elif head == "not":
        if len(node) >= 2:
            delete.append(to_string(node[1]))
    elif head in _NUMERIC_HEADS:
        num_eff.append(to_string(node))
    else:
        add.append(to_string(node))


# ---------------------------------------------------------------------------
#  Action parsing
# ---------------------------------------------------------------------------

def _parse_action_section(section: list) -> PDDLAction:
    """Parse (:action name :parameters (...) :precondition (...) :effect (...))."""
    name = str(section[1]).lower()
    params: List[Tuple[str, str]] = []
    precondition = None
    effect = None

    i = 2
    while i < len(section):
        key = str(section[i]).lower() if isinstance(section[i], str) else ""
        if key == ":parameters" and i + 1 < len(section):
            raw = section[i + 1]
            if isinstance(raw, list):
                params = _parse_typed_list(raw, strip_question=True)
            i += 2
        elif key == ":precondition" and i + 1 < len(section):
            precondition = section[i + 1]
            i += 2
        elif key == ":effect" and i + 1 < len(section):
            effect = section[i + 1]
            i += 2
        else:
            i += 1

    pre_list = _precondition_to_list(precondition)
    add, delete, num_eff, cond_blocks = _decompose_effect(effect)

    return PDDLAction(
        name=name, params=params, pre=pre_list,
        add=add, delete=delete, num_eff=num_eff, cond=cond_blocks,
    )


# ---------------------------------------------------------------------------
#  Derived predicate parsing
# ---------------------------------------------------------------------------

def _parse_derived_section(section: list) -> PDDLDerived:
    """Parse (:derived (head ?x) (body))."""
    head = to_string(section[1])
    body = to_string(section[2]) if len(section) > 2 else ""
    return PDDLDerived(head=head, body=body)


# ---------------------------------------------------------------------------
#  Top-level domain parser
# ---------------------------------------------------------------------------

def parse_domain(text: str) -> PDDLDomain:
    """Parse a PDDL domain string."""
    cleaned = _strip_comments(text)
    parsed = parse_one(cleaned)

    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("Not a valid PDDL domain definition")
    if str(parsed[0]).lower() != "define":
        raise ValueError("Domain must start with (define ...)")
    if not isinstance(parsed[1], list) or str(parsed[1][0]).lower() != "domain":
        raise ValueError("Expected (domain <name>) as first section")

    name = str(parsed[1][1]) if len(parsed[1]) > 1 else "unnamed"
    requirements: List[str] = []
    types: Dict[str, str] = {}
    constants: List[Tuple[str, str]] = []
    predicates: Dict[str, List[Tuple[str, str]]] = {}
    functions: Dict[str, List[Tuple[str, str]]] = {}
    actions: Dict[str, PDDLAction] = {}
    derived: List[PDDLDerived] = []

    for section in parsed[2:]:
        key = _section_key(section)
        if key is None:
            continue

        if key == ":requirements":
            requirements = [str(r).lower() for r in section[1:] if isinstance(r, str)]

        elif key == ":types":
            types = _parse_types_section(section)

        elif key == ":constants":
            constants = _parse_typed_list(section[1:], strip_question=False)

        elif key == ":predicates":
            predicates = _parse_predicates_section(section)

        elif key == ":functions":
            functions = _parse_functions_section(section)

        elif key == ":action":
            act = _parse_action_section(section)
            actions[act.name] = act

        elif key == ":derived":
            derived.append(_parse_derived_section(section))

    return PDDLDomain(
        name=name, requirements=requirements, types=types,
        constants=constants, predicates=predicates, functions=functions,
        actions=actions, derived=derived,
    )


def parse_domain_file(path: str | Path) -> PDDLDomain:
    """Read and parse a .pddl domain file."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_domain(f.read())


# ---------------------------------------------------------------------------
#  Problem parser
# ---------------------------------------------------------------------------

def parse_problem(text: str) -> PDDLProblem:
    """Parse a PDDL problem string."""
    cleaned = _strip_comments(text)
    parsed = parse_one(cleaned)

    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("Not a valid PDDL problem definition")
    if str(parsed[0]).lower() != "define":
        raise ValueError("Problem must start with (define ...)")

    name = ""
    domain_name = ""
    objects: List[Tuple[str, str]] = []
    init_facts: List[Tuple[str, Tuple[str, ...]]] = []
    init_fluents: List[Tuple[str, Tuple[str, ...], float]] = []
    goal = ""

    for section in parsed[1:]:
        key = _section_key(section)
        if key is None:
            continue

        if key == "problem" and len(section) > 1:
            name = str(section[1])

        elif key == ":domain" and len(section) > 1:
            domain_name = str(section[1])

        elif key == ":objects":
            objects = _parse_typed_list(section[1:], strip_question=False)

        elif key == ":init":
            for item in section[1:]:
                if not isinstance(item, list) or not item:
                    continue
                head = str(item[0]).lower()
                if head == "=":
                    # (= (func args...) value)
                    if len(item) >= 3 and isinstance(item[1], list):
                        func_part = item[1]
                        fname = str(func_part[0]).lower()
                        fargs = tuple(str(a) for a in func_part[1:])
                        try:
                            val = float(str(item[2]))
                        except ValueError:
                            val = 0.0
                        init_fluents.append((fname, fargs, val))
                else:
                    args = tuple(str(a) for a in item[1:])
                    init_facts.append((head, args))

        elif key == ":goal":
            if len(section) > 1:
                goal = to_string(section[1])

    return PDDLProblem(
        name=name, domain_name=domain_name, objects=objects,
        init_facts=init_facts, init_fluents=init_fluents, goal=goal,
    )


def parse_problem_file(path: str | Path) -> PDDLProblem:
    """Read and parse a .pddl problem file."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_problem(f.read())


# ---------------------------------------------------------------------------
#  NL auto-generation
# ---------------------------------------------------------------------------

def _humanize(name: str) -> str:
    """Convert snake/kebab case to space-separated: 'on_board' -> 'on board'."""
    return name.replace("_", " ").replace("-", " ")


def _auto_pred_nl(name: str, args: List[Tuple[str, str]]) -> str:
    h = _humanize(name)
    if not args:
        return h
    return h + " " + " ".join(f"{{{a[0]}}}" for a in args)


def _auto_action_nl(name: str, params: List[Tuple[str, str]]) -> str:
    h = _humanize(name)
    if not params:
        return h
    return h + " " + " ".join(f"{{{p[0]}}}" for p in params)


def _auto_fluent_nl(name: str, args: List[Tuple[str, str]]) -> str:
    h = _humanize(name)
    if not args:
        return h
    return h + " of " + " ".join(f"{{{a[0]}}}" for a in args)


# ---------------------------------------------------------------------------
#  Static predicate inference
# ---------------------------------------------------------------------------

def infer_static_predicates(domain: PDDLDomain) -> Set[str]:
    """Identify predicates never modified by any action effect."""
    modified: Set[str] = set()

    def _scan_effects(effects: List[str]) -> None:
        for eff_str in effects:
            try:
                parsed = parse_one(eff_str)
                if isinstance(parsed, list) and parsed:
                    modified.add(str(parsed[0]).lower())
            except SExprError:
                pass

    for act in domain.actions.values():
        _scan_effects(act.add)
        _scan_effects(act.delete)
        _scan_effects(act.num_eff)
        for cb in act.cond:
            _scan_effects(cb.add)
            _scan_effects(cb.delete)
            _scan_effects(cb.num_eff)

    return set(domain.predicates.keys()) - modified


# ---------------------------------------------------------------------------
#  Conversion: PDDLDomain -> DomainSpec
# ---------------------------------------------------------------------------

def domain_to_spec(
    domain: PDDLDomain,
    *,
    nl_overrides: Optional[Dict[str, Any]] = None,
    static_overrides: Optional[Set[str]] = None,
    outcome_overrides: Optional[Dict[str, list]] = None,
) -> DomainSpec:
    """
    Convert a parsed PDDLDomain into a SERE DomainSpec.

    Parameters
    ----------
    nl_overrides : dict, optional
        Nested dict with keys 'predicates', 'actions', 'fluents' mapping
        name -> {nl: ..., static: ...}.
    static_overrides : set, optional
        Predicate names to mark as static. If None, inferred automatically.
    outcome_overrides : dict, optional
        action_name -> list of outcome dicts.
    """
    nl = nl_overrides or {}
    nl_preds = nl.get("predicates", {})
    nl_acts = nl.get("actions", {})
    nl_fls = nl.get("fluents", {})

    # Determine static predicates
    if static_overrides is not None:
        statics = static_overrides
    else:
        inferred = infer_static_predicates(domain)
        # Merge with explicit overrides from nl
        statics = set(inferred)
        for pname, pinfo in nl_preds.items():
            if isinstance(pinfo, dict) and pinfo.get("static") is True:
                statics.add(pname.lower())
            elif isinstance(pinfo, dict) and pinfo.get("static") is False:
                statics.discard(pname.lower())

    # Types
    types = dict(domain.types)

    # Predicates
    predicates: Dict[str, PredicateSpec] = {}
    for pname, pargs in domain.predicates.items():
        pinfo = nl_preds.get(pname, {})
        if isinstance(pinfo, str):
            pinfo = {"nl": pinfo}
        raw_nl = pinfo.get("nl") if isinstance(pinfo, dict) else None
        predicates[pname] = PredicateSpec(
            name=pname,
            args=pargs,
            nl=_as_nl_list(raw_nl, _auto_pred_nl(pname, pargs)),
            static=pname in statics,
        )

    # Fluents
    fluents: Dict[str, FluentSpec] = {}
    for fname, fargs in domain.functions.items():
        finfo = nl_fls.get(fname, {})
        if isinstance(finfo, str):
            finfo = {"nl": finfo}
        raw_nl = finfo.get("nl") if isinstance(finfo, dict) else None
        fluents[fname] = FluentSpec(
            name=fname,
            args=fargs,
            nl=_as_nl_list(raw_nl, _auto_fluent_nl(fname, fargs)),
        )

    # Actions
    actions: Dict[str, ActionSpec] = {}
    oc_overrides = outcome_overrides or {}
    for aname, pddl_act in domain.actions.items():
        ainfo = nl_acts.get(aname, {})
        if isinstance(ainfo, str):
            ainfo = {"nl": ainfo}

        # Conditional blocks
        cond_blocks: List[ConditionalBlock] = []
        for cb in pddl_act.cond:
            cond_blocks.append(ConditionalBlock(
                forall=cb.forall,
                when=cb.when,
                add=cb.add,
                delete=cb.delete,
                num_eff=cb.num_eff,
            ))

        # Outcomes
        outcomes: List[OutcomeSpec] = []
        raw_ocs = oc_overrides.get(aname) or (ainfo.get("outcomes") if isinstance(ainfo, dict) else None)
        if raw_ocs:
            for oc in raw_ocs:
                outcomes.append(OutcomeSpec(
                    name=oc.get("name", "outcome"),
                    p=float(oc.get("p", 1.0)),
                    status=oc.get("status"),
                    add=oc.get("add", []) or [],
                    delete=oc.get("delete", oc.get("del", [])) or [],
                    num_eff=oc.get("num_eff", []) or [],
                    when=oc.get("when", []) or [],
                    messages=oc.get("messages", []) or [],
                ))

        raw_nl = ainfo.get("nl") if isinstance(ainfo, dict) else None

        actions[aname] = ActionSpec(
            name=aname,
            params=pddl_act.params,
            pre=pddl_act.pre,
            add=pddl_act.add,
            delete=pddl_act.delete,
            nl=_as_nl_list(raw_nl, _auto_action_nl(aname, pddl_act.params)),
            num_eff=pddl_act.num_eff,
            cond=cond_blocks,
            duration=ainfo.get("duration") if isinstance(ainfo, dict) else None,
            duration_var=ainfo.get("duration_var") if isinstance(ainfo, dict) else None,
            duration_unit=ainfo.get("duration_unit") if isinstance(ainfo, dict) else None,
            messages=ainfo.get("messages", []) if isinstance(ainfo, dict) else [],
            outcomes=outcomes,
        )

    _validate_static_effects(actions, predicates)

    # Derived predicates - convert from PDDLDerived to the format _parse_derived_rules expects
    derived_items: List[Dict[str, Any]] = []
    for d in domain.derived:
        derived_items.append({"head": d.head, "when": d.body})
    # Also merge any derived rules from extensions
    ext_derived = nl.get("derived", [])
    if ext_derived:
        derived_items.extend(ext_derived)
    derived = _parse_derived_rules(derived_items, predicates) if derived_items else {}

    return DomainSpec(
        name=domain.name,
        types=types,
        predicates=predicates,
        actions=actions,
        fluents=fluents,
        derived=derived,
    )
