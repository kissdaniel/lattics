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

    def collect_notes(self) -> list[str]:
        """Returns the notes for the run summary, for example about nodes the
        field treats in a special way.

        Returns
        -------
        list[str]
            The notes
        """
        return []

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


def exchange(info, v_n: float, c_f: float, v_f: float, dt: float) -> tuple[float, float]:
    """Exchange between a node (or an agent) and a compartment of a field over a time step, solved semi-implicitly.
    The total mass is conserved.

    Parameters
    ----------
    info : SubstrateInfo
        The transport parameters and the concentration of the node
    v_n : float
        The volume of the node, in cubic micrometers
    c_f : float
        The concentration of the compartment
    v_f : float
        The volume of the compartment, in cubic micrometers
    dt : float
        The time step, in milliseconds

    Returns
    -------
    tuple[float, float]
        The new concentrations of the node and of the compartment
    """
    k_p = info.passive_rate
    k_u = info.uptake_rate
    k_r = info.release_rate
    c_n_cur = info.concentration
    m_t = c_n_cur * v_n + c_f * v_f
    c_n_new = (c_n_cur + (dt / v_n) * (m_t / v_f) * (k_p + k_u)) / (
        1 + (dt / v_n) * ((1 + (v_n / v_f)) * k_p + (v_n / v_f) * k_u + k_r)
    )
    c_f_new = (m_t - c_n_new * v_n) / v_f
    return c_n_new, c_f_new
