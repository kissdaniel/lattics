"""The main component of the LattiCS framework containing functionalities to
set up and execute a simulation.
"""

from .agent import Agent
import math
import warnings


class Simulation:
    """Represents a simulation instance. This object manages the participating
    agents (:class:`Agent`), the environment (:class:`SimulationDomain`), and
    the various chemical substances (:class:`Substrate`) present within it.
    The class provides high-level access to configure and execute a simulation.
    """
    def __init__(self, id=None) -> None:
        """Constructor method.

        Parameters
        ----------
        id : str
            The identifier for the simulation instance. If not provided, a
            random identifier will be generated.
        """
        self._id = self._get_id(id)
        self._simulation_domain = None
        self._agents = list()
        self._substrates = list()
        self._models = list()
        self._time = 0

    @property
    def agents(self) -> list[Agent]:
        """Get the collection of agents currently present in the simulation.
        The order of agents in this collection is maintained throughout the
        simulation, with new agent instances always being added at the end.

        Returns
        -------
        list[Agent]
            Collection of the agents
        """
        return self._agents

    @property
    def time(self) -> int:
        """Get the current internal time of the simulation, representing the
        elapsed time since the simulation started.

        Returns
        -------
        int
            Internal time of the simulation, in milliseconds.
        """

        return self._time

    def add_agent(self, agent: Agent, **kwargs) -> None:
        """Adds the specified agent to the simulation. The agent will be added
        to the collection of all agents and, if a simulation domain is defined,
        will also be placed within the simulation domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        self._agents.append(agent)
        for m in self._models:
            m.initialize_attributes(agent)
        if self._simulation_domain:
            self._simulation_domain.add_agent(agent, **kwargs)
        else:
            warnings.warn('No simulation domain has been defined. '
                          'You can proceed without one, but this may '
                          'lead to unexpected consequences.')

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the simulation. The agent will be
        removed from the collection of all agents and, if applicable, will also
        be removed from the simulation domain.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        self._agents.remove(agent)
        if self._simulation_domain:
            self._simulation_domain.remove_agent(agent)

    def add_simulation_domain(self, domain: 'domains.SimulationDomain') -> None:
        """Sets the simulation domain to the instance passed as a parameter.

        Parameters
        ----------
        domain : SimulationDomain
            The simulation domain instance to be used
        """
        if self._simulation_domain:
            raise AttributeError('Simulation domain is already set and cannot be modified.')
        self._simulation_domain = domain
        self._simulation_domain.initialize()

    def add_model(self, model: 'cellfunction.CellFunctionModel') -> None:
        """Adds the provided model instance to the agent's collection of cell
        function models and invokes the model's initialization method.

        Parameters
        ----------
        model : CellFunctionModel
            A subclass of the ``CellFunctionModel`` abstract base class.
        """
        self._models.append(model)
        for a in self._agents:
            model.initialize_agent_attributes(a)

    def update_models(self, agent: Agent, dt: int) -> None:
        """Sequentially updates all models associated with the agent. If
        multiple sub-models exist within the same category, they are updated
        in the order they appear in their respective collection.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update, in milliseconds
        agent : Agent
            TODO
        """
        for m in self._cell_function_models:
            m.update(dt)

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
        steps = int(math.ceil(time / dt))
        for t in range(steps):
            for m in self._models:
                if m.update_needed():
                    for a in self._agents:
                        m.update_attributes(a, dt)
                    m.reset_time()
                m.increase_time(dt)
            if self._simulation_domain:
                self._simulation_domain.update(dt)
            self._time += dt

    def _get_id(self, identifier):
        return identifier if identifier else id(self)
