
from .agent import Agent
from .simulation import Simulation
import numpy as np
import warnings


class SimulationDomain:
    """Acts as a base class for different simulation domains.
    """
    def __init__(self,
                 simulation: Simulation
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        simulation : Simulation
            The Simulation instance containing the domain
        """
        self._simulation = simulation

    def initialize(self) -> None:
        """Initializes the domain.
        """
        pass

    def add_agent(self, agent: Agent) -> None:
        """Adds the specified agent to the structure representing the internal
        storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        pass

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        pass

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        pass


class UnstructuredSimulationDomain(SimulationDomain):
    """Represents a perfectly mixed simulation domain with no spatial structure
    or localized interactions, where each agent has an equal probability of
    interacting with any other agent in the population. Agents are stored in a
    list.
    """
    def __init__(self,
                 simulation: Simulation,
                 capacity: int = None
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        simulation : Simulation
            The Simulation instance containing the domain
        capacity : int, optional
            The maximum number of agents in the population. Defaults to None,
            meaning no limit (infinite population).
        """
        super().__init__(simulation)
        self._capacity = capacity
        self._agents = list()

    def initialize(self) -> None:
        """Initializes the domain.
        """
        pass

    def add_agent(self, agent: Agent) -> None:
        """Adds the specified agent to the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        if self._capacity < len(self._agents):
            warnings.warn('The number of agents exceeded the capacity '
                          'of the domain.')
        self._agents.append(agent)
        self.initialize_agent_status_flags(agent)

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        self._agents.remove(agent)

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration. Agents
        labeled as ``division_pending`` are duplicated based on the available
        space in the domain, constrained by ``capacity``.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        for a in self._agents:
            if a.get_status_flag('division_pending'):
                self._perform_cell_division(a)

    def initialize_agent_status_flags(self, agent) -> None:
        """Initializes the status flags used by the domain. The domain tracks
        two flags: ``division_pending`` and ``division_completed``. The
        ``division_pending`` flag can be set by cell-level functions to indicate
        that an agent is ready to divide and that the domain should handle the
        division event. Once division has occurred, the ``division_completed``
        flag is set to ``True``.
        """
        if not agent.has_status_flag('division_pending'):
            agent.initialize_status_flag('division_pending', False)
        if not agent.has_status_flag('division_completed'):
            agent.initialize_status_flag('division_completed', False)

    def _perform_cell_division(self, agent: Agent) -> None:
        if np.random.random() < self._get_division_probability():
            agent.set_status_flag('division_pending', False)
            agent.set_status_flag('division_completed', True)
            new_agent = agent.clone()
            self._simulation.add_agent(new_agent)

    def _get_division_probability(self) -> float:
        if self._capacity:
            return 1.0 - len(self._agents) / self._capacity
        else:
            return 1.0


class Structured2DSimulationDomain(SimulationDomain):
    pass
