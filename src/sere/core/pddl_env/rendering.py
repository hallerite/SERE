from typing import List

def push_msg(env, s: str) -> None:
    env._step_messages.append(s)
    env.messages.append(s)

def obs(env) -> str:
    aff = env.formatter.generate_affordances(
        env.world, env.static_facts, enable_numeric=env.enable_numeric
    )
    step_msgs: List[str] = env._step_messages[:]
    text = env.formatter.format_obs(
        world=env.world,
        steps=env.steps,
        max_steps=env.max_steps,
        time_val=env.time,
        durations_on=env.enable_durations,
        messages=step_msgs,
        affordances=aff,
        time_limit=env.time_limit,
        termination_rules=env.termination_rules,
    )
    env._step_messages.clear()
    return text
