from abc import ABC, abstractmethod

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._population import Population
from lattics._updateinfo import UpdateInfo


class BaseSpace(ABC):
    """Base class for the simulation spaces.

    The simulation calls :meth:`update` in every step. The base class keeps
    track of the update intervals, lets the subclass process the agents
    (:meth:`_update_agents`), registers the agents exchanging substrates, and
    updates the substrate fields.

    Parameters
    ----------
    dt_agent : tuple[float, str], optional
        The time between two updates of the agents, for example ``(30, 'min')``
    dt_substrate : tuple[float, str], optional
        The time between two updates of the substrate fields
    """

    def __init__(self, dt_agent: tuple[float, str] = None, dt_substrate: tuple[float, str] = None) -> None:
        self._population = None
        self._substrates = dict()
        self._agent_update_info = UpdateInfo(dt_agent)
        self._substrate_update_info = UpdateInfo(dt_substrate)

    def _bind(self, population: Population) -> None:
        """Connects the space to the population of the simulation it is set to."""
        if self._population is not None:
            raise RuntimeError('The space is already set to a simulation.')
        self._population = population

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

    def update(self, ctx: UpdateContext) -> None:
        """Updates the space.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update; ``ctx.dt`` is the time elapsed since the
            previous call, in milliseconds
        """
        self._agent_update_info.increase_time(ctx.dt)
        self._substrate_update_info.increase_time(ctx.dt)

        if self._agent_update_info.update_needed():
            self._update_agents(ctx, self._agent_update_info.elapsed_time)
            self._register_substrate_nodes()
            self._agent_update_info.reset_time()

        if self._substrate_update_info.update_needed():
            for s in self._substrates.values():
                s.update(self._substrate_update_info.elapsed_time)
            self._substrate_update_info.reset_time()

    @abstractmethod
    def _update_agents(self, ctx: UpdateContext, dt: float) -> None:
        """Processes the division and removal requests and the space-specific
        behaviour of the agents (for example movement).

        Parameters
        ----------
        ctx : UpdateContext
            The context of the update
        dt : float
            The time elapsed since the previous update of the agents, in milliseconds
        """
        pass

    def _register_substrate_nodes(self) -> None:
        for s in self._substrates.values():
            s.clear_dynamic_nodes()
        for a in self._population:
            if a.has_attribute('substrate.transport'):
                for name in a.get_attribute('substrate.transport').keys():
                    self._substrates[name].add_dynamic_substrate_node(a)
