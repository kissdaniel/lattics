from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lattics._context import UpdateContext

from ._node import SubstrateNode
from ._substrate import Substrate

if TYPE_CHECKING:
    from lattics.spaces._base import BaseSpace


class BaseSubstrateField(ABC):
    """The concentration field of a substrate in a space. Fields are created by
    the space (:meth:`~lattics.spaces.BaseSpace.create_field`), not by the user.

    The agents and substrate nodes whose ``substrate.transport`` attribute has
    an entry for the substrate exchange it with the field.

    Parameters
    ----------
    substrate : Substrate
        The definition of the substrate
    space : BaseSpace
        The space the field belongs to
    nodes : list[SubstrateNode]
        The substrate nodes of the simulation; nodes added later are seen too
    """

    def __init__(self, substrate: Substrate, space: 'BaseSpace', nodes: list[SubstrateNode]) -> None:
        self._substrate = substrate
        self._space = space
        self._nodes = nodes

    @property
    def substrate(self) -> Substrate:
        """The definition of the substrate."""
        return self._substrate

    @property
    @abstractmethod
    def concentration(self):
        """The concentration of the field: a number in a homogeneous field, an array in a spatial one."""

    @abstractmethod
    def concentration_at(self, position=None) -> float:
        """Returns the concentration at the given position.

        Parameters
        ----------
        position : tuple[int, ...], optional
            The position; homogeneous fields ignore it

        Returns
        -------
        float
            The concentration
        """

    @abstractmethod
    def update(self, ctx: UpdateContext) -> None:
        """Advances the field from ``ctx.time`` to ``ctx.time + ctx.dt``: the
        exchange with the agents and nodes, the diffusion and the kinetics.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update
        """

    def exchanging_nodes(self) -> list:
        """Returns the substrate nodes and then the agents that exchange the substrate, in this order.

        Returns
        -------
        list
            The nodes and agents
        """
        name = self._substrate.name
        nodes = [n for n in self._nodes if name in n.get_attribute('substrate.transport')]
        for agent in self._space._population:
            if agent.has_attribute('substrate.transport') and name in agent.get_attribute('substrate.transport'):
                nodes.append(agent)
        return nodes
