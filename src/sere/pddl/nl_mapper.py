from __future__ import annotations

import random
from typing import Any, Iterable, Tuple

from .domain_spec import DomainSpec, Predicate


class NLMapper:
    """
    Natural-language mapper for grounded predicates/actions/fluents.

    - Domain YAML may provide `nl` as either a string or a list of strings.
    - Deterministic by default: picks the FIRST template in the list.
    - If constructed with stochastic=True, randomly samples a template
      when multiple are available (independent of action outcome sampling).
    """

    def __init__(self, domain: DomainSpec, *, stochastic: bool = False, rng: random.Random | None = None):
        self.domain = domain
        self.stochastic = bool(stochastic)
        # Allow caller to pass a seeded RNG for reproducibility; else, use module random
        self._rng = rng if rng is not None else random

    # -------- internals --------
    def _as_templates(self, nl_field: Any) -> list[str]:
        """Normalize `nl` (str or list[str]) to a non-empty list of strings."""
        if isinstance(nl_field, list):
            return [str(x) for x in nl_field if str(x).strip()]
        if nl_field is None:
            return []
        s = str(nl_field)
        return [s] if s.strip() else []

    def _choose_tpl(self, templates: list[str]) -> str:
        """
        Choose a template according to mode:
          - stochastic=False → first template (deterministic)
          - stochastic=True  → random choice among templates (if >1)
        Returns "" if templates is empty.
        """
        if not templates:
            return ""
        if self.stochastic and len(templates) > 1:
            return self._rng.choice(templates)
        return templates[0]

    def _format_with_fallbacks(self, templates: Iterable[str], mapping: dict, raw_fallback: str) -> str:
        """
        Try each template in order until one formats. If all fail, return raw_fallback.
        This lets a list contain alternative phrasings with different placeholders.
        """
        for tpl in templates:
            try:
                return tpl.format(**mapping)
            except Exception:
                continue
        return raw_fallback

    # -------- public API --------
    def pred_to_text(self, pred: Predicate) -> str:
        """
        Map a grounded predicate to NL. Supports nl as str OR list[str] in the domain.
        """
        name, args = pred
        spec = self.domain.predicates.get(name)
        if not spec:
            return f"({name} {' '.join(args)})"

        # Build mapping by declared arg order (truncate if domain lists fewer)
        mapping = {spec.args[i][0]: args[i] for i in range(min(len(spec.args), len(args)))}

        templates = self._as_templates(getattr(spec, "nl", name))
        if not templates:
            return f"({name} {' '.join(args)})"

        # Deterministic/stochastic choice of *primary* template, but still try all as fallbacks
        primary = self._choose_tpl(templates)
        ordered = [primary] + [t for t in templates if t is not primary]

        return self._format_with_fallbacks(ordered, mapping, raw_fallback=f"({name} {' '.join(args)})")

    def act_to_text(self, act_name: str, args: Tuple[str, ...]) -> str:
        """
        Map a grounded action to NL. Supports nl as str OR list[str] in the domain.
        Note: callers often pass args like ('{r}', '{from}', ... ) when rendering the
        action catalog; those braces survive formatting as expected.
        """
        spec = self.domain.actions.get(act_name)
        if not spec:
            return f"({act_name} {' '.join(args)})"

        mapping = {spec.params[i][0]: args[i] for i in range(min(len(spec.params), len(args)))}

        templates = self._as_templates(getattr(spec, "nl", act_name))
        if not templates:
            return f"({act_name} {' '.join(args)})"

        primary = self._choose_tpl(templates)
        ordered = [primary] + [t for t in templates if t is not primary]

        return self._format_with_fallbacks(ordered, mapping, raw_fallback=f"({act_name} {' '.join(args)})")

    def fluent_to_text(self, name: str, args: Tuple[str, ...], value: float) -> str:
        spec = self.domain.fluents.get(name)
        if not spec:
            return f"({name}{'' if not args else ' ' + ' '.join(args)})={value:g}"

        # spec.nl is List[str] after your DomainSpec change
        templates = self._as_templates(getattr(spec, "nl", name))
        mapping = {}
        if spec.args:
            mapping = {spec.args[i][0]: args[i] for i in range(min(len(spec.args), len(args)))}

        if templates:
            primary = self._choose_tpl(templates)
            ordered = [primary] + [t for t in templates if t is not primary]
            try:
                base = self._format_with_fallbacks(ordered, mapping, raw_fallback=name)
            except Exception:
                base = name
        else:
            base = name

        return f"{base} is {value:.2f}"
