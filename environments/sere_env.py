"""SERE environment module for vf-eval.

Usage:
    PRIME_API_KEY=pit_xxx vf-eval sere_env -m openai/gpt-4.1 -n 10 -r 1 -p ./environments
"""

from integrations.verifiers.vf_sere import load_environment

__all__ = ["load_environment"]
