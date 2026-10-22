import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._population import Population
from lattics._units import convert_length, convert_time, convert_volume, from_quantity

if TYPE_CHECKING:
    from lattics.substrates import BaseSubstrateField, Substrate, SubstrateNode


class BaseSpace(ABC):
    """Base class for the simulation spaces.

    All spaces accept the same keyword arguments and compute the geometry of
    the domain in the same way, so a simulation can switch between spaces by
    replacing only the space class. Arguments a space does not need are
    accepted and ignored.

    The domain is given either by ``size`` (and ``thickness``) or by
    ``volume``, but not by both. Spaces with a spatial structure need ``size``.

    The simulation updates the agents of the space (:meth:`update_agents`)
    every ``interval``. The space also creates the concentration fields of
    the substrates (:meth:`create_field`), of the type that suits it.

    The class attributes list the names of the agent attributes the space
    uses: ``reads`` the attributes it needs, ``optional_reads`` the ones it
    uses only when an agent has them, and ``writes`` the ones it sets.

    Parameters
    ----------
    size : tuple[tuple[float, str], ...], optional
        The size of the domain along each axis, for example
        ``((750, 'um'), (750, 'um'))``
    volume : tuple[float, str], optional
        The volume of the domain, for example ``(8.4375, 'mm3')``; an
        alternative to ``size`` for spaces without spatial structure
    thickness : tuple[float, str], optional
        The thickness of a two-dimensional domain. Defaults to ``grid_spacing``.
    grid_spacing : tuple[float, str], optional
        The distance between neighboring lattice sites. Defaults to 20 um.
    interval : tuple[float, str], optional
        The time between two updates of the agents, for example ``(30, 'min')``
    """

    reads: tuple[str, ...] = ()
    optional_reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __init__(
        self,
        size: tuple[tuple[float, str], ...] = None,
        volume: tuple[float, str] = None,
        thickness: tuple[float, str] = None,
        grid_spacing: tuple[float, str] = (20, 'um'),
        interval: tuple[float, str] = None,
    ) -> None:
        if size is not None and volume is not None:
            raise ValueError('Give either the size or the volume of the domain, not both.')
        self._grid_spacing = from_quantity(grid_spacing, convert_length, 'um', 'grid spacing', "(20, 'um')")
        self._thickness = (
            from_quantity(thickness, convert_length, 'um', 'thickness', "(15, 'um')")
            if thickness
            else self._grid_spacing
        )
        self._size = (
            tuple(
                from_quantity(length, convert_length, 'um', 'size', "((750, 'um'), (750, 'um'))")
                for length in size
            )
            if size
            else None
        )
        if self._size:
            self._volume = math.prod(self._size) * self._thickness
        elif volume is not None:
            self._volume = from_quantity(volume, convert_volume, 'um3', 'volume', "(8.4, 'mm3')")
        else:
            self._volume = None
        self._population = None
        self._rng = None
        self._notes = list()
        self.interval = (
            from_quantity(interval, convert_time, 'ms', 'interval', "(30, 'min')") if interval else None
        )

    @property
    def volume(self) -> float | None:
        """The volume of the domain in cubic micrometers, or ``None`` if neither size nor volume is given."""
        return self._volume

    def _validate_volume(self, agent: Agent) -> None:
        if not agent.has_attribute('space.volume'):
            raise ValueError('The agent must have an attribute named \'space.volume\'.')

    def collect_notes(self) -> list[str]:
        """Returns the notes for the run summary collected since the previous
        run, for example about agents placed automatically, and clears them.

        Returns
        -------
        list[str]
            The notes
        """
        notes, self._notes = self._notes, list()
        return notes

    def _bind(self, population: Population, rng: np.random.Generator) -> None:
        """Connects the space to the population and the random number generator of the simulation it is set to."""
        if self._population is not None:
            raise RuntimeError('The space is already set to a simulation.')
        self._population = population
        self._rng = rng

    @abstractmethod
    def add_agent(self, agent: Agent) -> None:
        """Places the specified agent, which is already part of the population,
        in the space.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        pass

    @abstractmethod
    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the space.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        pass

    @abstractmethod
    def create_field(self, substrate: 'Substrate', nodes: list['SubstrateNode']) -> 'BaseSubstrateField':
        """Creates the concentration field of the substrate in this space.
        Called by the simulation.

        Parameters
        ----------
        substrate : Substrate
            The definition of the substrate
        nodes : list[SubstrateNode]
            The substrate nodes of the simulation

        Returns
        -------
        BaseSubstrateField
            The field
        """

    def update_agents(self, ctx: UpdateContext) -> None:
        """Processes the agents: removal and division requests and the
        space-specific behaviour (for example movement).

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update; the agents advance from ``ctx.time`` to
            ``ctx.time + ctx.dt``
        """
        self._update_agents(ctx, ctx.dt)

    def interval_warnings(self, period: float) -> list[str]:
        """Checks whether the agents can be updated with the given period
        without losing validity, for example because a probability per update
        would exceed 1. Subclasses override this method.

        Parameters
        ----------
        period : float
            The effective update period of the agents, in milliseconds

        Returns
        -------
        list[str]
            Warning messages; empty if the period is valid
        """
        return []

    @abstractmethod
    def _update_agents(self, ctx: UpdateContext, dt: float) -> None:
        """Processes the division and removal requests and the space-specific
        behaviour of the agents (for example movement).

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update
        dt : float
            The time step of the update, in milliseconds
        """
        pass
