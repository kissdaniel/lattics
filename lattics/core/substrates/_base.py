from abc import ABC, abstractmethod
from lattics.core import SubstrateNode


class BaseSubstrateField(ABC):
    def __init__(self,
                 domain,
                 substrate_name: str,
                 diffusion_coefficient: float = 0.0,
                 decay_coefficient: float = 0.0
                 ) -> None:
        self._domain = domain
        self._substrate_name = substrate_name
        self._diffusion_coefficient = diffusion_coefficient
        self._decay_coefficient = decay_coefficient
        self._static_nodes = list()
        self._dynamic_nodes = list()

    def add_static_substrate_node(self, substrate_node: SubstrateNode) -> None:
        self._static_nodes.append(substrate_node)

    def add_dynamic_substrate_node(self, substrate_node: SubstrateNode) -> None:
        self._dynamic_nodes.append(substrate_node)

    def clear_dynamic_nodes(self) -> None:
        self._dynamic_nodes.clear()

    @abstractmethod
    def get_concentration(self) -> float:
        pass

    @abstractmethod
    def update(self, dt) -> None:
        pass
