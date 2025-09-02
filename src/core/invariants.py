from typing import Protocol, List
from .world_state import WorldState

class InvariantPlugin(Protocol):
    def validate(self, world: WorldState) -> List[str]: ...

class KitchenInvariants:
    def validate(self, w: WorldState) -> List[str]:
        # Add kitchen-specific checks here if needed
        return []
