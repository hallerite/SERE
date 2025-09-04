from typing import Tuple
from .task_loader import load_task
from ..core.pddl_env import PDDLEnv
from ..core.invariants import KitchenInvariants

def load_kitchen(task_yaml: str, domain_yaml: str = "domain/kitchen.yaml", **env_kwargs) -> Tuple[PDDLEnv, dict]:
    return load_task(domain_yaml, task_yaml, plugins=[KitchenInvariants()], **env_kwargs)
