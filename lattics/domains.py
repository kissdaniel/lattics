
from .agent import Agent
from .simulation import Simulation
from . import numba_functions
import copy
import numpy as np
import numpy.typing as npt
from scipy import ndimage
import warnings


class SimulationDomain:
    """Base class for different simulation domains.
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

    def add_agent(self, agent: Agent, **kwargs) -> None:
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

    def add_agent(self, agent: Agent, **kwargs) -> None:
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
                self._perform_cell_division(a, dt)

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

    def _perform_cell_division(self, agent: Agent, dt: int) -> None:
        if len(self._agents) < self._capacity:
            agent.set_status_flag('division_pending', False)
            agent.set_status_flag('division_completed', True)
            new_agent = agent.clone()
            self._simulation.add_agent(new_agent)


class Structured2DSimulationDomain(SimulationDomain):
    def __init__(self,
                 simulation: Simulation,
                 dimensions: tuple[int, int]
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
        self._agents = None
        self._agent_layer = None
        self._dimensions = dimensions
        self.initialize()

    def initialize(self) -> None:
        """Initializes the domain.
        """
        self._agents = list()
        self._agent_layer = np.empty(self._dimensions, dtype='object')

    def add_agent(self, agent: Agent, position: tuple[int, int]) -> None:
        """Adds the specified agent to the specified index of the internal
        array of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        position : tuple[int, int]
            The index describing the agent's position
        """
        # if 'position' not in kwargs:
        #     raise ValueError('Position is needed.')
        # position = kwargs['position']
        if not self._is_valid_position(position):
            raise ValueError(f'Position {position} is out of the bounds of the domain.')
        if not self._is_empty_position(position):
            warnings.warn(f'Position {position} already occupied, existing '
                          'agent overwritten by new agent.')
        self.initialize_agent_status_flags(agent)
        self._agents.append(agent)
        self._agent_layer[tuple(position)] = agent
        agent.set_status_flag('position', position)

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        position = agent.get_status_flag('position')
        self._agents.remove(agent)
        self._agent_layer[tuple(position)] = None

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration. Agents
        labeled as ``division_pending`` are duplicated based on the available
        space in the domain, constrained by ``capacity``.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        agents_temp = copy.copy(self._agents)
        np.random.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_status_flag('division_pending'):
                self._perform_cell_division(a, dt)

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
        if not agent.has_status_flag('position'):
            agent.initialize_status_flag('position', (None, None))

    def _perform_cell_division(self, agent: Agent, dt: int) -> None:
        current_position = agent.get_status_flag('position')
        displacement_limit = 100
        coverage_mask = np.where(self._agent_layer, float('inf'), 0)
        if np.any(coverage_mask == 0):
            source_map = np.ones(self._agent_layer.shape)
            source_map[tuple(current_position)] = 0
            distance_map = ndimage.distance_transform_edt(source_map)
            distance_map = distance_map + coverage_mask
            min_distance = np.min(distance_map)
            if min_distance <= displacement_limit:
                target_sites = np.argwhere(distance_map == min_distance)
                target_position = target_sites[np.random.randint(target_sites.shape[0])]
                x1, y1 = current_position[:2]
                x2, y2 = target_position[:2]
                path = numba_functions.bresenham_2d(x1, y1, x2, y2)
                if path.shape[0] > 2:
                    for i in range(path.shape[0] - 2, 0, -1):
                        a_old_x, a_old_y = path[i, :2]
                        a_new_x, a_new_y = path[i + 1, :2]
                        agent_to_move = self._agent_layer[a_old_x, a_old_y]
                        self._agent_layer[a_new_x, a_new_y] = agent_to_move
                        agent_to_move.set_status_flag('position', path[i + 1])
                clone_position = path[1]
                self._agent_layer[tuple(clone_position)] = None
                agent.set_status_flag('division_pending', False)
                agent.set_status_flag('division_completed', True)
                new_agent = agent.clone()
                gen = agent.get_status_flag('generation')
                new_agent.set_status_flag('generation', gen + 1)
                self._simulation.add_agent(new_agent, position=clone_position)

    def _is_valid_position(self, position: tuple[int, int]) -> bool:
        return np.all(np.zeros(2) <= np.array(position)) and np.all(np.array(position) < self._agent_layer.shape)

    def _is_empty_position(self, position: tuple[int, int]) -> bool:
        return self._agent_layer[tuple(position)] is None
