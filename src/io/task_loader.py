import yaml
from typing import Tuple, Set
from ..pddl.domain_spec import DomainSpec
from ..core.world_state import WorldState
from ..core.pddl_env import PDDLEnv

def _parse_lit(s: str):
    s = s.strip()
    assert s[0]=="(" and s[-1]==")", f"Bad literal: {s}"
    toks = s[1:-1].split()
    return toks[0], tuple(toks[1:])

def load_task(domain_path: str, task_path: str, plugins=None, **env_kwargs) -> Tuple[PDDLEnv, dict]:
    dom = DomainSpec.from_yaml(domain_path)
    y = yaml.safe_load(open(task_path, "r"))

    w = WorldState(dom)
    # objects
    for typ, ids in y["objects"].items():
        for sym in ids: w.add_object(sym, typ)

    # static facts
    static_facts: Set[tuple] = set(_parse_lit(x) for x in y.get("static_facts", []))
    # init
    for fact in y.get("init", []): w.facts.add(_parse_lit(fact))
    # goals
    goals = [_parse_lit(g) for g in y["goal"]]

    env = PDDLEnv(dom, w, static_facts, goals, plugins=plugins or [], **env_kwargs)
    meta = {"id": y["id"], "name": y.get("name", y["id"])}
    return env, meta
