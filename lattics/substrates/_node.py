from dataclasses import dataclass, field
from typing import Any

from lattics._units import convert_flow_rate, convert_volume

_RATES = ('passive_rate', 'uptake_rate', 'release_rate')


@dataclass
class SubstrateInfo:
    """The exchange of a substrate between an agent (or a substrate node) and
    the concentration field.

    The rates are volume flow rates given as ``(value, unit)`` tuples, for
    example ``(0.5, 'um3/sec')``; they are converted to um3/ms when the
    object is created. A number without a unit is taken in um3/ms, and the
    simulation warns about it when the agent or node is added.

    Parameters
    ----------
    type : str
        ``'flux'`` for exchange driven by the rates, ``'fixed'`` for a
        constant concentration imposed on the field
    concentration : float
        The concentration inside the agent or node
    passive_rate : tuple[float, str]
        Passive transport, driven by the concentration difference
    uptake_rate : tuple[float, str]
        Active uptake from the field
    release_rate : tuple[float, str]
        Active release into the field
    """

    type: str
    concentration: float = 0.0
    passive_rate: tuple[float, str] | float = 0.0
    uptake_rate: tuple[float, str] | float = 0.0
    release_rate: tuple[float, str] | float = 0.0
    unitless_rates: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        unitless = []
        for name in _RATES:
            value = getattr(self, name)
            if isinstance(value, tuple):
                setattr(self, name, convert_flow_rate(value[0], value[1], 'um3/ms'))
            elif value != 0:
                unitless.append(name)
        self.unitless_rates = tuple(unitless)


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
