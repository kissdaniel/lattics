from dataclasses import dataclass
from typing import Any

from lattics._units import convert_volume


@dataclass
class SubstrateInfo:
    type: str
    concentration: float = 0.0
    passive_rate: float = 0.0
    uptake_rate: float = 0.0
    release_rate: float = 0.0


class SubstrateNode:
    """A source or sink of substrates that is not an agent, for example the
    blood vessels or an organ acting as a drug reservoir. It is added to the
    simulation with :meth:`~lattics.Simulation.add_substrate_node`.

    Parameters
    ----------
    volume : tuple[float, str], optional
        The (virtual) volume of the node, for example ``(0.8, 'mm3')``
    transport : dict[str, SubstrateInfo], optional
        The exchange with the substrate fields, by substrate name
    position : tuple[int, ...], optional
        The position of the node in spaces with spatial structure
    """

    def __init__(
        self,
        volume: tuple[float, str] = None,
        transport: dict[str, SubstrateInfo] = None,
        position: tuple[int, ...] = None,
    ) -> None:
        self._attributes = dict()
        self.set_attribute('substrate.transport', dict(transport) if transport else dict())
        if volume is not None:
            self.set_attribute('space.volume', convert_volume(volume[0], volume[1], 'um3'))
        if position is not None:
            self.set_attribute('space.position', position)

    def has_attribute(self, name: str) -> bool:
        return name in self._attributes

    def set_attribute(self, name: str, value: Any) -> None:
        self._attributes[name] = value

    def get_attribute(self, name: str) -> Any:
        return self._attributes[name]
