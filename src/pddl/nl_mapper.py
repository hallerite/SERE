from .domain_spec import DomainSpec, Predicate

class NLMapper:
    def __init__(self, domain: DomainSpec):
        self.domain = domain

    def pred_to_text(self, pred: Predicate) -> str:
        name, args = pred
        spec = self.domain.predicates[name]
        mapping = {spec.args[i][0]: args[i] for i in range(len(spec.args))}
        return spec.nl.format(**mapping)

    def act_to_text(self, act_name: str, args: tuple) -> str:
        spec = self.domain.actions[act_name]
        mapping = {spec.params[i][0]: args[i] for i in range(len(spec.params))}
        return spec.nl.format(**mapping)
