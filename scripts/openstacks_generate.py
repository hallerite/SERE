#!/usr/bin/env python3
"""
Generate SERE Openstacks tasks (domain + task YAML) with a clean-room generator.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import json


Order = str
Product = str


def _sample_weighted_without_replacement(
    items: Sequence[str],
    weights: Sequence[float],
    k: int,
    rng: random.Random,
) -> List[str]:
    pool = list(items)
    w = [float(x) for x in weights]
    chosen: List[str] = []
    k = min(k, len(pool))
    for _ in range(k):
        total = sum(w)
        if total <= 0.0:
            idx = rng.randrange(len(pool))
        else:
            roll = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for i, wi in enumerate(w):
                acc += wi
                if roll <= acc:
                    idx = i
                    break
        chosen.append(pool.pop(idx))
        w.pop(idx)
    return chosen


def _product_popularity_weights(num_products: int, rng: random.Random, skew: float) -> List[float]:
    if skew <= 0.0:
        return [1.0] * num_products
    ranks = list(range(1, num_products + 1))
    rng.shuffle(ranks)
    return [1.0 / (r ** skew) for r in ranks]


def _band_weights(order_idx: int, num_orders: int, num_products: int, band_width: float) -> List[float]:
    if band_width <= 0.0:
        return [1.0] * num_products
    sigma = band_width * num_products if band_width <= 1.0 else band_width
    sigma = max(sigma, 1e-6)
    center = (order_idx + 0.5) * (num_products / max(1, num_orders)) - 0.5
    weights = []
    for i in range(num_products):
        dist = i - center
        weights.append(math.exp(-(dist * dist) / (2.0 * sigma * sigma)))
    return weights


def _draw_order_size(
    num_products: int,
    *,
    density: int,
    size_min: Optional[int],
    size_max: Optional[int],
    rng: random.Random,
) -> int:
    if size_min is not None or size_max is not None:
        lo = size_min if size_min is not None else size_max
        hi = size_max if size_max is not None else size_min
        lo = max(1, min(int(lo), num_products))
        hi = max(1, min(int(hi), num_products))
        if lo > hi:
            lo, hi = hi, lo
        return rng.randint(lo, hi)
    p = density / 100.0
    count = sum(1 for _ in range(num_products) if rng.random() < p)
    return max(1, count)


def _gen_requirements(
    *,
    num_products: int,
    num_orders: int,
    density: int,
    rng: random.Random,
    corr_mode: str,
    band_width: float,
    size_min: Optional[int],
    size_max: Optional[int],
    product_skew: float,
) -> Tuple[List[Product], List[Order], Dict[Order, Set[Product]]]:
    if not (0 <= density <= 100):
        raise ValueError("density must be in [0, 100]")

    products = [f"p{i+1}" for i in range(num_products)]
    orders = [f"o{i+1}" for i in range(num_orders)]

    pop_weights = _product_popularity_weights(num_products, rng, product_skew)

    req: Dict[Order, Set[Product]] = {o: set() for o in orders}
    for oi, o in enumerate(orders):
        size = _draw_order_size(
            num_products,
            density=density,
            size_min=size_min,
            size_max=size_max,
            rng=rng,
        )
        if corr_mode == "banded":
            corr = _band_weights(oi, num_orders, num_products, band_width)
        else:
            corr = [1.0] * num_products
        weights = [p * c for p, c in zip(pop_weights, corr)]
        chosen = _sample_weighted_without_replacement(products, weights, size, rng)
        req[o].update(chosen)

    # Ensure each order has at least one product.
    for o in orders:
        if not req[o]:
            req[o].add(rng.choice(products))

    # Ensure each product is required by at least one order.
    for p in products:
        if all(p not in req[o] for o in orders):
            smallest = min(orders, key=lambda x: len(req[x]))
            req[smallest].add(p)

    return products, orders, req


def _can_open(o: Order, req: Dict[Order, Set[Product]], made: Set[Product], open_count: int, limit: int) -> bool:
    if open_count >= limit:
        return False
    # Enforce openstacks semantics: open only before any required product is made.
    return not any(p in made for p in req[o])


def _can_make(
    p: Product,
    orders: Sequence[Order],
    req: Dict[Order, Set[Product]],
    open_set: Set[Order],
    shipped_set: Set[Order],
) -> bool:
    for o in orders:
        if p in req[o] and o not in open_set and o not in shipped_set:
            return False
    return True


def _plan_bfs(
    products: Sequence[Product],
    orders: Sequence[Order],
    req: Dict[Order, Set[Product]],
    stack_limit: int,
) -> Optional[List[Tuple[str, str]]]:
    """Return a plan as a list of (action, target) or None if not found."""
    orders_set = set(orders)
    start = (frozenset(), frozenset(), frozenset())  # open, shipped, made
    q = deque([(start, [])])
    seen = {start}

    while q:
        (open_set, shipped_set, made_set), plan = q.popleft()
        open_list = set(open_set)
        shipped_list = set(shipped_set)
        made_list = set(made_set)

        if shipped_list == orders_set:
            return plan

        open_count = len(open_list)

        # Option 1: open a new order.
        for o in orders:
            if o in open_list or o in shipped_list:
                continue
            if not _can_open(o, req, made_list, open_count, stack_limit):
                continue
            new_open = set(open_list)
            new_open.add(o)
            new_state = (frozenset(new_open), frozenset(shipped_list), frozenset(made_list))
            if new_state not in seen:
                seen.add(new_state)
                q.append((new_state, plan + [("open", o)]))

        # Option 2: make a product (auto-ships any completed orders).
        for p in products:
            if p in made_list:
                continue
            if not _can_make(p, orders, req, open_list, shipped_list):
                continue
            new_made = set(made_list)
            new_made.add(p)
            new_open = set(open_list)
            new_shipped = set(shipped_list)
            for o in list(new_open):
                if req[o].issubset(new_made):
                    new_open.remove(o)
                    new_shipped.add(o)
            new_state = (frozenset(new_open), frozenset(new_shipped), frozenset(new_made))
            if new_state not in seen:
                seen.add(new_state)
                q.append((new_state, plan + [("make", p)]))

    return None


def _build_domain_yaml() -> Dict:
    predicates = [
        {"name": "open", "args": [{"name": "o", "type": "order"}], "nl": "{o} is open"},
        {"name": "shipped", "args": [{"name": "o", "type": "order"}], "nl": "{o} is shipped"},
        {
            "name": "requires",
            "args": [{"name": "o", "type": "order"}, {"name": "p", "type": "product"}],
            "static": True,
            "nl": "{o} requires {p}",
        },
        {"name": "made", "args": [{"name": "p", "type": "product"}], "nl": "{p} is made"},
    ]

    fluents = [
        {"name": "open-count", "args": [], "nl": "Number of open orders"},
        {"name": "stack-limit", "args": [], "nl": "Maximum allowed open orders"},
        {"name": "remaining", "args": [{"name": "o", "type": "order"}], "nl": "Remaining products for {o}"},
    ]

    open_pre = [
        "(not (open ?o))",
        "(not (shipped ?o))",
        "(< (open-count) (stack-limit))",
        "(forall (?p - product) (or (not (requires ?o ?p)) (not (made ?p))))",
    ]

    open_action = {
        "name": "open-order",
        "params": [{"r": "robot"}, {"o": "order"}],
        "pre": open_pre,
        "add": ["(open ?o)"],
        "num_eff": ["(increase (open-count) 1)"],
        "nl": "Open order {o}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0},
            {"name": "fail", "status": "fail", "p": 0.0},
        ],
    }

    make_action = {
        "name": "make-product",
        "params": [{"r": "robot"}, {"p": "product"}],
        "pre": [
            "(not (made ?p))",
            "(forall (?o - order) (or (not (requires ?o ?p)) (open ?o) (shipped ?o)))",
        ],
        "add": ["(made ?p)"],
        "nl": "Make product {p}",
        "outcomes": [
            {"name": "success", "status": "success", "p": 1.0},
            {"name": "fail", "status": "fail", "p": 0.0},
        ],
    }

    make_action["cond"] = [
        {
            "forall": [{"o": "order"}],
            "when": ["(requires ?o ?p)", "(open ?o)", "(= (remaining ?o) 1)"],
            "add": ["(shipped ?o)"],
            "del": ["(open ?o)"],
            "num_eff": ["(decrease (open-count) 1)", "(decrease (remaining ?o) 1)"],
        },
        {
            "forall": [{"o": "order"}],
            "when": ["(requires ?o ?p)", "(open ?o)", "(> (remaining ?o) 1)"],
            "num_eff": ["(decrease (remaining ?o) 1)"],
        },
    ]

    return {
        "domain": "openstacks",
        "requirements": [
            ":strips",
            ":typing",
            ":negative-preconditions",
            ":conditional-effects",
            ":fluents",
            ":adl",
            ":quantified-preconditions",
        ],
        "types": [{"name": "robot"}, {"name": "order"}, {"name": "product"}],
        "predicates": predicates,
        "fluents": fluents,
        "actions": [open_action, make_action],
    }


def _build_task_yaml(
    *,
    task_id: str,
    name: str,
    description: str,
    products: Sequence[Product],
    orders: Sequence[Order],
    req: Dict[Order, Set[Product]],
    stack_limit: int,
    plan: List[Tuple[str, str]],
) -> Dict:
    init_fluents = [
        ["open-count", [], 0.0],
        ["stack-limit", [], float(stack_limit)],
    ]
    for o in orders:
        init_fluents.append(["remaining", [o], float(len(req[o]))])

    static_facts = [f"(requires {o} {p})" for o in orders for p in sorted(req[o])]
    goal = "(forall (?o - order) (shipped ?o))"

    reference_plan = []
    for act, arg in plan:
        if act == "open":
            reference_plan.append(f"(open-order r1 {arg})")
        elif act == "make":
            reference_plan.append(f"(make-product r1 {arg})")
        elif act == "ship":
            reference_plan.append(f"(ship-order r1 {arg})")
        else:
            raise ValueError(f"Unknown action in plan: {act}")

    max_steps = max(5, len(reference_plan) + 5)

    return {
        "id": task_id,
        "name": name,
        "description": description,
        "meta": {
            "domain": "openstacks",
            "enable_numeric": True,
            "enable_conditional": True,
            "enable_durations": False,
            "enable_stochastic": False,
            "max_steps": max_steps,
            "init_fluents": init_fluents,
        },
        "objects": {"r1": "robot", **{o: "order" for o in orders}, **{p: "product" for p in products}},
        "static_facts": static_facts,
        "init": [],
        "termination": [{"name": "goal", "when": goal, "outcome": "success", "reward": 1.0}],
        "reference_plan": reference_plan,
    }


def _dump_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # JSON is valid YAML; avoid external YAML dependencies.
        f.write(json.dumps(payload, indent=2, sort_keys=False))
        f.write("\n")


def generate_task(
    *,
    task_id: str,
    name: str,
    description: str,
    num_products: int,
    num_orders: int,
    density: int,
    stack_limit: int,
    seed: int,
    task_dir: Path,
    domain_path: Path,
    max_tries: int,
    corr_mode: str,
    band_width: float,
    size_min: Optional[int],
    size_max: Optional[int],
    product_skew: float,
) -> None:
    rng = random.Random(seed)
    plan = None
    products: List[Product] = []
    orders: List[Order] = []
    req: Dict[Order, Set[Product]] = {}

    for _ in range(max_tries):
        products, orders, req = _gen_requirements(
            num_products=num_products,
            num_orders=num_orders,
            density=density,
            rng=rng,
            corr_mode=corr_mode,
            band_width=band_width,
            size_min=size_min,
            size_max=size_max,
            product_skew=product_skew,
        )
        plan = _plan_bfs(products, orders, req, stack_limit)
        if plan:
            break

    if not plan:
        raise RuntimeError("Failed to generate a solvable instance; try increasing stack_limit or max_tries.")

    domain_yaml = _build_domain_yaml()
    task_yaml = _build_task_yaml(
        task_id=task_id,
        name=name,
        description=description,
        products=products,
        orders=orders,
        req=req,
        stack_limit=stack_limit,
        plan=plan,
    )

    _dump_yaml(domain_path, domain_yaml)
    _dump_yaml(task_dir / f"{task_id}.yaml", task_yaml)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a SERE Openstacks task.")
    ap.add_argument("--task-id", required=True, help="Task id, e.g. t01_openstacks_small")
    ap.add_argument("--name", default=None, help="Human-readable task name")
    ap.add_argument("--rationale", default=None, help="Short rationale appended to the description")
    ap.add_argument("--products", type=int, required=True, help="Number of products")
    ap.add_argument("--orders", type=int, required=True, help="Number of orders")
    ap.add_argument("--density", type=int, default=50, help="Density in percent")
    ap.add_argument("--stack-limit", type=int, default=0, help="Max open orders (0 = auto)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--max-tries", type=int, default=50, help="Max regeneration attempts")
    ap.add_argument(
        "--corr-mode",
        choices=["uniform", "banded"],
        default="banded",
        help="Order/product correlation mode",
    )
    ap.add_argument(
        "--band-width",
        type=float,
        default=0.35,
        help="Banded correlation width (fraction of products if <=1, else absolute)",
    )
    ap.add_argument(
        "--size-min",
        type=int,
        default=None,
        help="Minimum products per order (enables size range if set)",
    )
    ap.add_argument(
        "--size-max",
        type=int,
        default=None,
        help="Maximum products per order (enables size range if set)",
    )
    ap.add_argument(
        "--product-skew",
        type=float,
        default=0.0,
        help="Power-law skew for product popularity (0 = uniform)",
    )
    ap.add_argument(
        "--task-dir",
        default="src/sere/assets/tasks/openstacks",
        help="Output directory for task YAML",
    )
    ap.add_argument(
        "--domain-path",
        default="src/sere/assets/domain/openstacks.yaml",
        help="Output path for the domain YAML",
    )
    args = ap.parse_args()

    limit = args.stack_limit
    if limit <= 0:
        limit = max(1, (args.orders + 1) // 2)

    name = args.name or f"Openstacks {args.task_id}"
    desc = (
        "Open orders and sequence product production without exceeding the stack limit. "
        f"Settings: orders={args.orders}, products={args.products}, "
        f"stack-limit={limit}, density={args.density}%, "
        f"corr={args.corr_mode}, band-width={args.band_width}, "
        f"size-range={args.size_min}-{args.size_max}, product-skew={args.product_skew}."
    )
    if args.rationale:
        desc = desc.rstrip(".") + f". Rationale: {args.rationale}"

    generate_task(
        task_id=args.task_id,
        name=name,
        description=desc,
        num_products=args.products,
        num_orders=args.orders,
        density=args.density,
        stack_limit=limit,
        seed=args.seed,
        task_dir=Path(args.task_dir),
        domain_path=Path(args.domain_path),
        max_tries=args.max_tries,
        corr_mode=args.corr_mode,
        band_width=args.band_width,
        size_min=args.size_min,
        size_max=args.size_max,
        product_skew=args.product_skew,
    )

    print(f"Generated {args.task_id} (products={args.products}, orders={args.orders}, limit={limit}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
