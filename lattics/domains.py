
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
        self.initialize_agent_attributes(agent)

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
            if a.get_attribute('division_pending'):
                self._perform_cell_division(a, dt)

    def initialize_agent_attributes(self, agent) -> None:
        """Initializes the attributes used by the domain. The domain tracks
        two attributes: ``division_pending`` and ``division_completed``. The
        ``division_pending`` attribute can be set by cell-level functions to
        indicate that an agent is ready to divide and that the domain should
        handle the division event. Once division has occurred, the
        ``division_completed`` attribute is set to ``True``.
        """
        if not agent.has_attribute('division_pending'):
            agent.initialize_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.initialize_attribute('division_completed', False)

    def _perform_cell_division(self, agent: Agent, dt: int) -> None:
        if len(self._agents) < self._capacity:
            agent.set_attribute('division_pending', False)
            agent.set_attribute('division_completed', True)
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
        dimensions : tuple[int, int]
            Width and height of the lattice on which agents are located, given
            as the number of columns and rows
        """
        super().__init__(simulation)
        self._agents = None
        self._agent_layer = None
        self._dimensions = dimensions
        self._dx = 1    # TODO
        self.initialize()

    def initialize(self) -> None:
        """Initializes the domain.
        """
        self._agents = list()
        self._agent_layer = np.empty(self._dimensions, dtype='object')

    def add_agent(self,
                  agent: Agent,
                  position: tuple[int, int],
                  motility: int = 0,
                  binding_affinity: int = 0,
                  displacement_limit: int = 1) -> None:
        """Adds the specified agent to the specified index of the internal
        array of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        position : tuple[int, int]
            The column and row index describing the agent's position
        motility : int, optional
            Characteristic velocity of the agent, expressed in
            micrometers per millisecond, by default 0
        binding_affinity : int, optional
            Scale factor that determines the binding strength between
            the agent and other agents, by default 0

        Raises
        ------
        ValueError
            If the position of the agent is out of the bounds of the domain
        """
        if not self.is_valid_position(position):
            raise ValueError(f'Position {position} is out of the bounds of the domain.')
        if not self.is_empty_position(position):
            warnings.warn(f'Position {position} already occupied, existing '
                          'agent overwritten by new agent.')
        self.initialize_agent_attributes(agent)
        self._agents.append(agent)
        self._agent_layer[tuple(position)] = agent
        agent.set_attribute('position', position)
        agent.set_attribute('motility', motility)
        agent.set_attribute('binding_affinity', binding_affinity)
        agent.set_attribute('displacement_limit', displacement_limit)

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
        """Updates the domain according to the specified time duration.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        self._displacement_trials(dt)
        self._cell_division_trials(dt)

    def initialize_agent_attributes(self, agent) -> None:
        """Initializes the attributes used by the domain. The domain uses
        the following attributes: ``division_pending``, ``division_completed``,
        ``motility``, ``binding_affinity``, and ``displacement_limit``. Agents
        are displaced to adjacent grid points based on their ``motility`` and
        ``binding_affinity`` values. Agents labeled as ``division_pending``
        are duplicated depending on the number of available grid points and the
        ``displacement_limit``, which specifies how far an agent can push others
        to create space for a daughter cell. Once division has occurred, the
        ``division_completed`` attribute is set to ``True``.
        """
        if not agent.has_attribute('division_pending'):
            agent.initialize_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.initialize_attribute('division_completed', False)
        if not agent.has_attribute('position'):
            agent.initialize_attribute('position', (None, None))
        if not agent.has_attribute('motility'):
            agent.initialize_attribute('motility', None)
        if not agent.has_attribute('binding_affinity'):
            agent.initialize_attribute('binding_affinity', None)
        if not agent.has_attribute('displacement_limit'):
            agent.initialize_attribute('displacement_limit', None)

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """Indicates whether the given position lies within the bounds
        of the domain.

        Parameters
        ----------
        position : tuple[int, int]
            The column and row index describing the position

        Returns
        -------
        bool
            ``True`` if the position exists within the domain, ``False`` otherwise
        """
        return np.all(np.zeros(2) <= np.array(position)) and np.all(np.array(position) < self._agent_layer.shape)

    def is_empty_position(self, position: tuple[int, int]) -> bool:
        """Indicates whether the specified lattice point is unoccupied
        (i.e., not containing any agent instance).

        Parameters
        ----------
        position : tuple[int, int]
            The column and row index describing the position

        Returns
        -------
        bool
            ``True`` if the lattice point is empty, ``False`` otherwise
        """
        return self._agent_layer[tuple(position)] is None

    def _displacement_trials(self, dt: int) -> None:
        if self._agents:
            agents = copy.copy(self._agents)
            np.random.shuffle(agents)
            positions = np.array([a.get_attribute('position') for a in agents], dtype='int32')
            disp_probs = np.array([a.get_attribute('motility') * dt / self._dx for a in agents], dtype='float32')
            binding_affs = np.array([a.get_attribute('binding_affinity') for a in agents], dtype='float32')
            # 3D array containing identifiers (idx) at those elements occupied by agents
            idx_array = np.full((self._agent_layer.shape[0], self._agent_layer.shape[1]), -1, dtype='int32')
            for i, a in enumerate(agents):
                position = a.get_attribute('position')
                idx_array[tuple(position)] = np.int32(i)
            # Indicates whether a certain agent changed its position during the MC trial
            change_flags = np.full(len(agents), False, dtype='bool8')

            for i, dprob in enumerate(disp_probs):
                # Check if trial is needed based on the displacement probability
                if np.random.random() < dprob:
                    numba_functions.displacement_trial_2d(i, positions, binding_affs, idx_array, change_flags)

            for i, cflag in enumerate(change_flags):
                if cflag:
                    pos_new = positions[i]
                    pos_old = agents[i].get_attribute('position')
                    agent_new = self._agent_layer[pos_new[0], pos_new[1]]
                    agent_old = self._agent_layer[pos_old[0], pos_old[1]]
                    self._agent_layer[tuple(pos_new)] = agent_old
                    self._agent_layer[tuple(pos_old)] = agent_new
                    if agent_new:
                        agent_new.set_attribute('position', pos_old)
                    if agent_old:
                        agent_old.set_attribute('position', pos_new)

    def _cell_division_trials(self, dt: int) -> None:
        agents_temp = copy.copy(self._agents)
        np.random.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('division_pending'):
                self._perform_cell_division(a, dt)

    def _perform_cell_division(self, agent: Agent, dt: int) -> None:
        current_position = agent.get_attribute('position')
        displacement_limit = agent.get_attribute('displacement_limit')
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
                        agent_to_move.set_attribute('position', path[i + 1])
                clone_position = path[1]
                self._agent_layer[tuple(clone_position)] = None
                agent.set_attribute('division_pending', False)
                agent.set_attribute('division_completed', True)
                new_agent = agent.clone()
                gen = agent.get_attribute('generation')
                new_agent.set_attribute('generation', gen + 1)
                self._simulation.add_agent(new_agent, position=clone_position)
