"""SERE integrations with external frameworks."""

# Lazy imports to avoid requiring optional dependencies
__all__ = [
    "SereLudicEnv",
    "make_ludic_env",
    "pddl_action_parser",
    "pddl_action_tag_parser",
    "load_environment",
]


def __getattr__(name):
    """Lazy import integration modules."""
    if name in ("SereLudicEnv", "make_ludic_env"):
        from sere.integrations.ludic_env import SereLudicEnv, make_ludic_env
        return SereLudicEnv if name == "SereLudicEnv" else make_ludic_env
    elif name in ("pddl_action_parser", "pddl_action_tag_parser"):
        from sere.integrations.ludic_parser import pddl_action_parser, pddl_action_tag_parser
        return pddl_action_parser if name == "pddl_action_parser" else pddl_action_tag_parser
    elif name == "load_environment":
        from sere.integrations.verifiers import load_environment
        return load_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
