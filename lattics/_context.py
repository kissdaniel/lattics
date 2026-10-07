from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._agent import Agent
    from ._population import Population
    from ._simulation import Simulation
    from .spaces._base import BaseSpace


class UpdateContext:
    """The information a component (a model or a space) receives when it is
    updated. A new context is created for every update call and is valid only
    during that call, so it should not be stored.

    Parameters
    ----------
    time : float
        The current time of the simulation, in milliseconds
    dt : float
        The time elapsed since the previous update of the component, in milliseconds
    population : Population, optional
        The agents of the simulation
    space : BaseSpace, optional
        The space of the simulation
    simulation : Simulation, optional
        The simulation that adds and removes agents on behalf of
        :meth:`add_agent` and :meth:`remove_agent`. It may be omitted, for
        example when a model is tested on its own.
    """

    __slots__ = ('_time', '_dt', '_population', '_space', '_simulation')

    def __init__(
        self,
        time: float,
        dt: float,
        population: 'Population' = None,
        space: 'BaseSpace' = None,
        simulation: 'Simulation' = None,
    ) -> None:
        self._time = time
        self._dt = dt
        self._population = population
        self._space = space
        self._simulation = simulation

    @property
    def time(self) -> float:
        """The current time of the simulation, in milliseconds."""
        return self._time

    @property
    def dt(self) -> float:
        """The time elapsed since the previous update of the component, in milliseconds."""
        return self._dt

    @property
    def population(self) -> 'Population':
        """The agents of the simulation."""
        return self._population

    @property
    def space(self) -> 'BaseSpace':
        """The space of the simulation."""
        return self._space

    def add_agent(self, agent: 'Agent') -> None:
        """Adds a new agent to the simulation, for example the daughter cell of a division.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        self._require_simulation().add_agent(agent)

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
            raise RuntimeError('This context is not attached to a simulation, so it cannot add or remove agents.')
        return self._simulation
