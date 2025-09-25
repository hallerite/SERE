import random
from sere.pddl.domain_spec import DomainSpec
from sere.pddl.nl_mapper import NLMapper

def make_domain():
    # minimal fake domain with variants
    from dataclasses import dataclass
    from typing import List, Tuple, Dict

    # build via your DomainSpec.from_yaml in real tests; inline here for brevity
    preds = {
        "at": type("P", (), {"name":"at","args":[("r","robot"),("l","location")],"nl":["{r} is at {l}","{r} currently at {l}"]})()
    }
    acts = {
        "move": type("A", (), {
            "name":"move",
            "params":[("r","robot"),("from","location"),("to","location")],
            "nl":["Move {r} from {from} to {to}","{r} goes {from}→{to}"]
        })()
    }
    fls = {
        "energy": type("F", (), {"name":"energy","args":[("r","robot")],"nl":["Energy of {r}","{r}'s energy"]})()
    }
    return type("D", (), {"name":"test","predicates":preds,"actions":acts,"fluents":fls})()

def test_deterministic_variants():
    d = make_domain()
    nl = NLMapper(d, stochastic=False)
    assert nl.pred_to_text(("at", ("r1","A"))) == "r1 is at A"
    assert nl.act_to_text("move", ("r1","A","B")) == "Move r1 from A to B"
    # fluent first template
    assert nl.fluent_to_text("energy", ("r1",), 7.0).startswith("Energy of r1")

def test_stochastic_variants_reproducible():
    d = make_domain()
    rng = random.Random(123)
    nl1 = NLMapper(d, stochastic=True, rng=rng)
    out1 = [nl1.pred_to_text(("at", ("r1","A"))) for _ in range(5)]

    rng = random.Random(123)
    nl2 = NLMapper(d, stochastic=True, rng=rng)
    out2 = [nl2.pred_to_text(("at", ("r1","A"))) for _ in range(5)]

    assert out1 == out2  # same seed → same choices

def test_stochastic_variants_variety():
    d = make_domain()
    nl = NLMapper(d, stochastic=True, rng=random.Random(42))
    samples = {nl.act_to_text("move", ("r1","A","B")) for _ in range(50)}
    assert len(samples) >= 2  # both variants should appear
