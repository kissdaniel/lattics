"""The main component of the LattiCS framework containing functionalities to
set up and execute a simulation.
"""

from .agent import Agent
import warnings


class Simulation:
    """Represents a simulation instance. This object manages the participating
    agents (:class:`Agent`), the environment (:class:`SimulationSpace`), and
    the various chemical substances (:class:`Substrate`) present within it.
    The class provides high-level access to configure and execute a simulation.
    """
    def __init__(self, id=None):
        """Constructor method.

        Parameters
        ----------
        id : str
            The identifier for the simulation instance. If not provided, a
            random identifier will be generated.
        """
        self._id = self._get_id(id)
        self._simulation_space = None
        self._agents = list()
        self._substrates = list()

    @property
    def agents(self) -> list[Agent]:
        """Get the collection of agents currently present in the simulation.
        The order of agents in this collection is maintained throughout the
        simulation, with new agent instances always being added at the end.

        Returns
        -------
        list[agent.Agent]
            Collection of the agents
        """
        return self._agents

    def add_agent(self, agent: Agent) -> None:
        """Adds the specified agent to the simulation. The agent will be added
        to the collection of all agents and, if a simulation space is defined,
        will also be placed within the simulation space.

        Parameters
        ----------
        agent : agent.Agent
            The agent to be added
        """
        self._agents.append(agent)
        if self._simulation_space:
            self._simulation_space.add_agent(agent)
        else:
            warnings.warn('No simulation space has been defined.'
                          'You can proceed without one, but this may'
                          'lead to unexpected consequences.')

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the simulation. The agent will be
        removed from the collection of all agents and, if applicable, will also
        be removed from the simulation space.

        Parameters
        ----------
        agent : agent.Agent
            The agent to be removed
        """
        self._agents.remove(agent)
        if self._simulation_space:
            self._simulation_space.remove_agent(agent)

    def initialize(self):
        pass

    def run(self, steps):
        for t in range(steps):
            for a in self._agents:
                a.update_models(1)

    def _get_id(self, identifier):
        return identifier if identifier else id(self)
