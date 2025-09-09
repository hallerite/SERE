from .domain_spec import DomainSpec, Predicate

class NLMapper:
    def __init__(self, domain: DomainSpec):
        self.domain = domain

    def pred_to_text(self, pred: Predicate) -> str:
        name, args = pred
        spec = self.domain.predicates[name]
        mapping = {spec.args[i][0]: args[i] for i in range(len(spec.args))}
        # Defensive: tolerate missing/short mappings
        try:
            return spec.nl.format(**mapping)
        except Exception:
            # Fallback to raw
            return f"({name} {' '.join(args)})"

    def act_to_text(self, act_name: str, args: tuple) -> str:
        spec = self.domain.actions[act_name]
        mapping = {spec.params[i][0]: args[i] for i in range(len(spec.params))}
        try:
            return spec.nl.format(**mapping)
        except Exception:
            return f"({act_name} {' '.join(args)})"

    def fluent_to_text(self, name: str, args: tuple, value: float) -> str:
        spec = self.domain.fluents.get(name)
        if not spec:
            return f"({name}{'' if not args else ' ' + ' '.join(args)})={value:g}"
        mapping = {spec.args[i][0]: args[i] for i in range(len(spec.args))}
        try:
            base = spec.nl.format(**mapping)
        except Exception:
            base = f"{name}{'' if not args else ' ' + ' '.join(args)}"
        # natural-ish: "Quality score of asm1 is 0.95"
        return f"{base} is {value:.2f}"
