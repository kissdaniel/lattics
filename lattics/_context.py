from typing import TYPE_CHECKING

import numpy as np

from ._scheduler import EPSILON

if TYPE_CHECKING:
    from ._agent import Agent
    from ._population import Population
    from ._simulation import Simulation
    from .spaces._base import BaseSpace
    from .substrates._base import BaseSubstrateField


class UpdateContext:
    """The information a component (a model or a space) receives when it is
    updated. A new context is created for every update call and is valid only
    during that call, so it should not be stored.

    Parameters
    ----------
    time : float
        The current time of the simulation, in milliseconds
    dt : float
        The time step of the update, in milliseconds: the component advances
        from ``time`` to ``time + dt``. It equals the update interval of the
        component, rounded up to a multiple of the simulation time step.
    population : Population, optional
        The agents of the simulation
    space : BaseSpace, optional
        The space of the simulation
    simulation : Simulation, optional
        The simulation that adds and removes agents on behalf of
        :meth:`add_agent` and :meth:`remove_agent`. It may be omitted, for
        example when a model is tested on its own.
    last_update_time : float, optional
        The time of the previous update of the component, in milliseconds;
        ``None`` before its first update
    rng : numpy.random.Generator, optional
        The random number generator of the simulation. Without it the
        context creates an unseeded generator.
    """

    __slots__ = ('_time', '_dt', '_population', '_space', '_simulation', '_last_update_time', '_rng')

    def __init__(
        self,
        time: float,
        dt: float,
        population: 'Population' = None,
        space: 'BaseSpace' = None,
        simulation: 'Simulation' = None,
        last_update_time: float = None,
        rng: np.random.Generator = None,
    ) -> None:
        self._time = time
        self._dt = dt
        self._population = population
        self._space = space
        self._simulation = simulation
        self._last_update_time = last_update_time
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def time(self) -> float:
        """The current time of the simulation, in milliseconds."""
        return self._time

    @property
    def dt(self) -> float:
        """The time step of the update, in milliseconds: the component advances from ``time`` to ``time + dt``."""
        return self._dt

    @property
    def last_update_time(self) -> float | None:
        """The time of the previous update of the component, in milliseconds; ``None`` before its first update."""
        return self._last_update_time

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator of the simulation. Components should draw
        every random number from it, so that a seeded simulation is reproducible.
        """
        return self._rng

    @property
    def population(self) -> 'Population':
        """The agents of the simulation."""
        return self._population

    @property
    def space(self) -> 'BaseSpace':
        """The space of the simulation."""
        return self._space

    def divided_since_last_update(self, agent: 'Agent') -> bool:
        """Tells whether the space divided the agent since the previous update
        of the component. Both the mother and the daughter agent count as
        divided. Before the first update of the component it is ``False``.

        Parameters
        ----------
        agent : Agent
            The agent to be checked

        Returns
        -------
        bool
            ``True`` if the agent was divided since the previous update
        """
        if self._last_update_time is None or not agent.has_attribute('space.last_division_time'):
            return False
        division_time = agent.get_attribute('space.last_division_time')
        # The space is updated after the models at the same time point, so a division at the time of the
        # previous update happened after that update
        return division_time is not None and division_time >= self._last_update_time - EPSILON

    def substrate(self, name: str) -> 'BaseSubstrateField':
        """Returns the concentration field of the substrate.

        Parameters
        ----------
        name : str
            The name of the substrate

        Returns
        -------
        BaseSubstrateField
            The field
        """
        return self._require_simulation().substrate_field(name)

    def add_agent(self, agent: 'Agent') -> None:
        """Adds a new agent to the simulation, for example the daughter cell of a division.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        self._require_simulation()._add_agent(agent)

    def remove_agent(self, agent: 'Agent') -> None:
        """Removes an agent from the simulation.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        self._require_simulation().remove_agent(agent)

    def _require_simulation(self) -> 'Simulation':
        if self._simulation is None:
            raise RuntimeError('This context is not attached to a simulation.')
        return self._simulation
