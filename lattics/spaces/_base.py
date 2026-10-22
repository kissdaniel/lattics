import math
from abc import ABC, abstractmethod

import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._population import Population
from lattics._units import convert_length, convert_time, convert_volume


class BaseSpace(ABC):
    """Base class for the simulation spaces.

    All spaces accept the same keyword arguments and compute the geometry of
    the domain in the same way, so a simulation can switch between spaces by
    replacing only the space class. Arguments a space does not need are
    accepted and ignored.

    The domain is given either by ``size`` (and ``thickness``) or by
    ``volume``, but not by both. Spaces with a spatial structure need ``size``.

    The simulation schedules two parts of the space separately: the agents
    (:meth:`update_agents`, every ``interval``) and the substrate fields
    (:meth:`update_substrates`, every ``substrate_interval``).

    The class attributes ``reads`` and ``writes`` list the names of the agent
    attributes the space uses and sets.

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
    substrate_interval : tuple[float, str], optional
        The time between two updates of the substrate fields
    """

    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __init__(
        self,
        size: tuple[tuple[float, str], ...] = None,
        volume: tuple[float, str] = None,
        thickness: tuple[float, str] = None,
        grid_spacing: tuple[float, str] = (20, 'um'),
        interval: tuple[float, str] = None,
        substrate_interval: tuple[float, str] = None,
    ) -> None:
        if size is not None and volume is not None:
            raise ValueError('Give either the size or the volume of the domain, not both.')
        self._grid_spacing = convert_length(grid_spacing[0], grid_spacing[1], 'um')
        self._thickness = convert_length(thickness[0], thickness[1], 'um') if thickness else self._grid_spacing
        self._size = tuple(convert_length(value, unit, 'um') for value, unit in size) if size else None
        if self._size:
            self._volume = math.prod(self._size) * self._thickness
        elif volume is not None:
            self._volume = convert_volume(volume[0], volume[1], 'um3')
        else:
            self._volume = None
        self._population = None
        self._rng = None
        self._substrates = dict()
        self.interval = convert_time(interval[0], interval[1], 'ms') if interval else None
        self.substrate_interval = (
            convert_time(substrate_interval[0], substrate_interval[1], 'ms') if substrate_interval else None
        )

    @property
    def volume(self) -> float | None:
        """The volume of the domain in cubic micrometers, or ``None`` if neither size nor volume is given."""
        return self._volume

    def _validate_volume(self, agent: Agent) -> None:
        if not agent.has_attribute('space.volume'):
            raise ValueError('The agent must have an attribute named \'space.volume\'.')

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
    def add_substrate(
        self,
        name: str,
        diffusion_coefficient: float = 0.0,
        decay_coefficient: float = 0.0,
        decay_kinetics: str = 'first-order',
    ) -> None:
        pass

    def update_agents(self, ctx: UpdateContext) -> None:
        """Processes the agents: division and removal requests and the
        space-specific behaviour (for example movement), then registers the
        agents exchanging substrates.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update; the agents advance from ``ctx.time`` to
            ``ctx.time + ctx.dt``
        """
        self._update_agents(ctx, ctx.dt)
        self._register_substrate_nodes()

    def update_substrates(self, ctx: UpdateContext) -> None:
        """Advances the substrate fields from ``ctx.time`` to ``ctx.time + ctx.dt``.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update
        """
        for s in self._substrates.values():
            s.update(ctx.dt)

    def has_substrates(self) -> bool:
        """Indicates whether the space has substrate fields to update."""
        return bool(self._substrates)

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

    def _register_substrate_nodes(self) -> None:
        for s in self._substrates.values():
            s.clear_dynamic_nodes()
        for a in self._population:
            if a.has_attribute('substrate.transport'):
                for name in a.get_attribute('substrate.transport').keys():
                    self._substrates[name].add_dynamic_substrate_node(a)
