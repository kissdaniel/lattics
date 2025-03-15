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
        self._time = None

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

    @property
    def time(self) -> int:
        """Get the current internal time of the simulation, representing the elapsed time since the simulation started.

        Returns
        -------
        int
            Internal time of the simulation, in milliseconds.
        """

        return self._time

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

    def initialize(self) -> None:
        """Initializes the simulation before start. It initializes the
        simulation settings to default values and registers the necessary
        connections with the utilized sub-modules. Call this function only once.
        """
        self._time = 0

    def run(self, time, dt) -> None:
        """Runs the simulation from the current state for the specified
        duration using the given time step.

        Parameters
        ----------
        time : int
            The duration to be simulated, in milliseconds
        dt : _type_
            Time step, in milliseconds
        """
        steps = int(round(time / dt, 0))
        for t in range(steps):
            for a in self._agents:
                a.update_models(dt)
            self._time = self._time + dt

    def _get_id(self, identifier):
        return identifier if identifier else id(self)
