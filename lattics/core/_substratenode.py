from dataclasses import dataclass
from typing import Any


@dataclass
class SubstrateInfo:
    type: str
    concentration: float = 0.0
    passive_rate: float = 0.0
    uptake_rate: float = 0.0
    release_rate: float = 0.0


class SubstrateNode:
    def __init__(self) -> None:
        """Constructor method.
        """
        self._attributes = dict()
        self.set_attribute('substrate_info', dict())

    def set_attribute(self, name: str, value: Any) -> None:
        self._attributes[name] = value

    def get_attribute(self, name: str) -> Any:
        return self._attributes[name]
