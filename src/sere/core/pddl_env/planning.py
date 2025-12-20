from typing import List, Tuple
from .engine import step_one
from sere.core.pddl_env.run_mode import RunMode
from sere.pddl.sexpr import parse_many, SExprError

def _dbg(env, *args):
    """Call env._dbg if available and enabled."""
    if getattr(env, "debug", False) and hasattr(env, "_dbg"):
        env._dbg(*args)

def parse_actions(block: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Parse a block containing one or more grounded actions:
      (op arg1 arg2 ...)(op2 arg1 ...)
    Returns: [(name, (args...)), ...]
    Raises ValueError if no actions are found.
    """
    try:
        exprs = parse_many(block or "")
    except SExprError as exc:
        raise ValueError(f"{exc}") from exc
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for expr in exprs:
        if not isinstance(expr, list) or not expr:
            continue
        head = expr[0]
        if not isinstance(head, str):
            raise ValueError("Bad action expression: non-atom head.")
        if any(isinstance(x, list) for x in expr[1:]):
            raise ValueError("Bad action expression: nested list.")
        name = head.lower()
        args = tuple(str(x) for x in expr[1:])
        out.append((name, args))
    if not out:
        raise ValueError("No actions found.")
    return out

def execute_plan(env, plan, *, atomic: bool = False):
    """
    Execute a parsed plan [(name, args), ...] sequentially.
    - Accumulates total reward (step_penalty + shaping + any terminal reward).
    - Returns the last obs/info; adds plan_trace, steps_executed, shaping_bonus_total.
    - If atomic=True and an invalid move occurs, roll back to the pre-plan snapshot,
      but DO NOT call env._illegal(). Preserve the original info/outcome from the
      failing step and annotate plan_trace.
    """
    _dbg(env, f"[PLAN-START] atomic={atomic} len={len(plan)}")

    snap = None
    if atomic:
        snap = (
            set(env.world.facts),
            dict(env.world.fluents),
            float(env.time),
            int(env.steps),
            list(env.messages),
            list(env._step_messages),
            bool(env.done),
        )

    plan_trace = []
    total_reward = 0.0
    total_shaping = 0.0
    steps_executed = 0
    final_obs, final_info, terminal = "", {}, False

    # DO NOT reset env._retries_used_this_turn here. Retries accumulate across
    # calls until a valid action executes.

    for idx, (name, args) in enumerate(plan, start=1):
        # Short-circuit if env already terminal (e.g., prior action timed out)
        if env.done:
            terminal = True
            break

        obs, rew, done, info = step_one(env, name, args)

        # Collect tracing + reward rollup
        plan_trace.append({
            "i": idx,
            "action": name,
            "args": list(args),
            "outcome": info.get("outcome"),
            "error": info.get("error"),
            "messages": info.get("messages", []),
        })
        total_reward += float(rew)
        step_shape = float(info.get("shaping_bonus", 0.0))
        total_shaping += step_shape
        steps_executed += 1

        _dbg(env, f"[PLAN] i={idx} action={name}{args} "
                  f"step_rew={float(rew):.2f} step_shape={step_shape:.2f} "
                  f"acc_total={total_reward:.2f} acc_shape={total_shaping:.2f} "
                  f"outcome={info.get('outcome')}")

        final_obs, final_info, terminal = obs, info, bool(done)

        # ---- STOP / ROLLBACK LOGIC ----
        if info.get("invalid_move"):
            if atomic and snap:
                # roll back only for invalids; preserve failing step's info
                env.world.facts, env.world.fluents = snap[0], snap[1]
                env.time, env.steps = snap[2], snap[3]
                env.messages, env._step_messages, env.done = snap[4], snap[5], snap[6]

                # annotate final_info rather than reclassifying via _illegal()
                final_info = dict(final_info)
                final_info.setdefault("outcome", "invalid_move")
                final_info["plan_trace"] = plan_trace
                final_info["steps_executed"] = steps_executed
                final_info["shaping_bonus_total"] = total_shaping
                _dbg(env, f"[PLAN-END] atomic={atomic} EARLY-INVALID steps={steps_executed} "
                          f"total_r={total_reward:.2f} total_shape={total_shaping:.2f}")
                return final_obs, total_reward, True, final_info
            break  # non-atomic: just stop

        if done:
            # terminal but not invalid → keep state and stop
            break

    # --- classify open-loop cutoff as failed so metrics sum to 100% ---
    if env.run_mode == RunMode.OPEN_LOOP and not (terminal or env.done):
        env.done = True
        final_info = dict(final_info)
        final_info["outcome"] = "failed"
        final_info["reason"] = "open_loop_end"
        terminal = True
    # ----------------------------------------------------------------------

    final_info = dict(final_info)
    final_info["plan_trace"] = plan_trace
    final_info["steps_executed"] = steps_executed
    final_info["shaping_bonus_total"] = total_shaping
    _dbg(env, f"[PLAN-END] atomic={atomic} steps={steps_executed} "
              f"total_r={total_reward:.2f} total_shape={total_shaping:.2f} "
              f"final_outcome={final_info.get('outcome')}")
    return final_obs, total_reward, terminal, final_info
