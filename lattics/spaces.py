from .core import Agent
from .core import Simulation
from .core import UpdateInfo
from .numba_functions import bresenham_2d, displacement_trial_2d
from .substrates import HomogeneousSubstrateField, Lattice2DSubstrateField

from abc import ABC, abstractmethod
import copy
import numpy as np
import numpy.typing as npt
from scipy import ndimage
import warnings


class BaseSpace(ABC):
    """Base class for different simulation domains.
    """
    def __init__(self,
                 simulation: Simulation,
                 dt_agent: tuple[float, str] = None,
                 dt_substrate: tuple[float, str] = None,
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        simulation : Simulation
            The Simulation instance containing the domain
        """
        self._simulation = simulation
        self._update_infos = dict()
        self._agent_update_info = UpdateInfo(dt_agent)
        self._substrate_update_info = UpdateInfo(dt_substrate)

    @abstractmethod
    def add_agent(self, agent: Agent, **kwargs) -> None:
        """Adds the specified agent to the structure representing the internal
        storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        pass

    @abstractmethod
    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        pass

    @abstractmethod
    def add_substrate(self,
                      name: str,
                      diffusion_coefficient: float = 0.0,
                      decay_coefficient: float = 0.0
                      ) -> None:
        pass

    @abstractmethod
    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        pass


class HomogeneousSpace(BaseSpace):
    """Represents a perfectly mixed simulation domain with no spatial structure
    or localized interactions, where each agent has an equal probability of
    interacting with any other agent in the population. Agents are stored in a
    list.
    """
    def __init__(self,
                 simulation: Simulation,
                 volume: int = None,
                 dt_agent: tuple[float, str] = None,
                 dt_substrate: tuple[float, str] = None
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        simulation : Simulation
            The Simulation instance containing the domain
        volume : int, optional
            TODO: The maximum number of agents in the population. Defaults to None,
            meaning no limit (infinite population).
        """
        super().__init__(simulation=simulation, dt_agent=dt_agent, dt_substrate=dt_substrate)
        self._volume = volume
        self._agents = list()
        self._substrates = dict()
        self._has_free_volume = True

    def add_agent(self, agent: Agent, volume: int = 0, **params) -> None:
        """Adds the specified agent to the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        sum_volumes = self._get_total_agent_volume()
        if self._volume < sum_volumes + volume:
            warnings.warn('The total volume of agents exceeded the capacity '
                          'of the domain.')
        self._agents.append(agent)
        self._initialize_attributes(agent, volume)

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        self._agents.remove(agent)
        sum_volumes = self._get_total_agent_volume()
        if sum_volumes <= self._volume:
            self._has_free_volume = True

    def add_substrate(self,
                      name: str,
                      diffusion_coefficient: float = 0.0,
                      decay_coefficient: float = 0.0
                      ) -> None:
        substrate = HomogeneousSubstrateField(domain=self,
                                              substrate_name=name,
                                              diffusion_coefficient=diffusion_coefficient,
                                              decay_coefficient=decay_coefficient
                                              )
        self._substrates[name] = substrate

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration. Agents
        labeled as ``division_pending`` are duplicated based on the available
        space in the domain, constrained by ``capacity``.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        self._agent_update_info.increase_time(dt)
        self._substrate_update_info.increase_time(dt)

        if self._agent_update_info.update_needed():
            self._clear_substrate_dynamic_nodes()
            for a in list(self._agents):
                if a.get_attribute('division_pending'):
                    self._process_agent_division(a)
                if a.get_attribute('remove_pending'):
                    self._simulation.remove_agent(a)

                info = a.get_attribute('substrate_info')
                for i in info.keys():
                    self._substrates[i].add_dynamic_substrate_node(a)
            self._agent_update_info.reset_time()

        if self._substrate_update_info.update_needed():
            for s in self._substrates.values():
                s.update(self._substrate_update_info.elapsed_time)
            self._substrate_update_info.reset_time()

    def _initialize_attributes(self, agent, volume) -> None:
        """Initializes the attributes used by the domain. The domain tracks
        two attributes: ``division_pending`` and ``division_completed``. The
        ``division_pending`` attribute can be set by cell-level functions to
        indicate that an agent is ready to divide and that the domain should
        handle the division event. Once division has occurred, the
        ``division_completed`` attribute is set to ``True``.
        TODO: volume
        """
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', False)
        if not agent.has_attribute('remove_pending'):
            agent.set_attribute('remove_pending', False)
        if not agent.has_attribute('volume'):
            agent.set_attribute('volume', volume)

    def _process_substrates(self, dt) -> None:
        pass

    def _clear_substrate_dynamic_nodes(self):
        for s in self._substrates.values():
            s.clear_dynamic_nodes()

    def _process_agent_division(self, agent: Agent) -> None:
        if not self._has_free_volume:
            return
        sum_volumes = self._get_total_agent_volume()
        agent_volume = agent.get_attribute('volume')
        if sum_volumes + agent_volume <= self._volume:
            agent.set_attribute('division_pending', False)
            agent.set_attribute('division_completed', True)
            new_agent = agent.clone()
            self._simulation.add_agent(new_agent)
        else:
            self._has_free_volume = False

    def _process_agent_removal(self) -> None:
        pass

    def _get_total_agent_volume(self) -> int:
        return sum([a.get_attribute('volume') for a in self._agents])


class Lattice2DSpace(BaseSpace):
    def __init__(
            self,
            simulation: Simulation,
            dimensions: tuple[int, int],
            grid_spacing: int,
            dt_agent: tuple[float, str] = None,
            dt_substrate: tuple[float, str] = None) -> None:
        super().__init__(
            simulation=simulation,
            dt_agent=dt_agent,
            dt_substrate=dt_substrate)
        self._dimensions = dimensions
        self._grid_spacing = grid_spacing
        self._agents = list()
        self._agent_layer = np.empty(self._dimensions, dtype='object')
        self._substrates = dict()

    def add_agent(self,
                  agent: Agent,
                  position: tuple[int, int],
                  volume: int = 0,
                  **params) -> None:
        """Adds the specified agent to the specified index of the internal
        array of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        volume: int
            TODO
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
            raise ValueError(f'Position {position} already occupied.')
        if self.get_remaining_volume(position) < volume:
            warnings.warn('The agent\'s volume is larger than the volume assigned to '
                          'the domain chunks. This may lead to unexpected behavior.')
        self._agents.append(agent)
        self._agent_layer[tuple(position)] = agent
        self._initialize_attributes(agent, position, volume, **params)

    def _initialize_attributes(self,
                               agent: Agent,
                               position: tuple[int, int],
                               volume: int,
                               motility: int = 0,
                               binding_affinity: int = 0,
                               displacement_limit: int = 1,
                               **params) -> None:
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', False)
        if not agent.has_attribute('remove_pending'):
            agent.set_attribute('remove_pending', False)
        if not agent.has_attribute('volume'):
            agent.set_attribute('volume', volume)
        if not agent.has_attribute('position'):
            agent.set_attribute('position', position)
        if not agent.has_attribute('motility'):
            agent.set_attribute('motility', motility)
        if not agent.has_attribute('binding_affinity'):
            agent.set_attribute('binding_affinity', binding_affinity)
        if not agent.has_attribute('displacement_limit'):
            agent.set_attribute('displacement_limit', displacement_limit)

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal list of the agents.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        position = agent.get_attribute('position')
        self._agents.remove(agent)
        self._agent_layer[tuple(position)] = None

    def add_substrate(self,
                      name: str,
                      diffusion_coefficient: float = 0.0,
                      decay_coefficient: float = 0.0
                      ) -> None:
        substrate = Lattice2DSubstrateField(domain=self,
                                            substrate_name=name,
                                            diffusion_coefficient=diffusion_coefficient,
                                            decay_coefficient=decay_coefficient
                                            )
        self._substrates[name] = substrate

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """

        self._agent_update_info.increase_time(dt)
        self._substrate_update_info.increase_time(dt)

        if self._agent_update_info.update_needed():
            self._displacement_trials(self._agent_update_info.elapsed_time)
            self._cell_division_trials(self._agent_update_info.elapsed_time)
            self._cell_removal_trials(self._agent_update_info.elapsed_time)

            self._clear_substrate_dynamic_nodes()
            for a in self._agents:
                info = a.get_attribute('substrate_info')
                for i in info.keys():
                    self._substrates[i].add_dynamic_substrate_node(a)
            self._agent_update_info.reset_time()

        if self._substrate_update_info.update_needed():
            for s in self._substrates.values():
                s.update(self._substrate_update_info.elapsed_time)
            self._substrate_update_info.reset_time()

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

    def get_remaining_volume(self, position: tuple[int, int]) -> int:
        if self.is_empty_position(position):
            return self._grid_spacing**2
        return self._grid_spacing**2 - self._agent_layer[tuple(position)].get_attribute('volume')

    def _displacement_trials(self, dt: int) -> None:
        if self._agents:
            agents = copy.copy(self._agents)
            np.random.shuffle(agents)
            positions = np.array([a.get_attribute('position') for a in agents], dtype='int32')
            disp_probs = np.array([a.get_attribute('motility') * dt / self._grid_spacing for a in agents], dtype='float32')
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
                    displacement_trial_2d(i, positions, binding_affs, idx_array, change_flags)

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

    def _cell_removal_trials(self, dt: int) -> None:
        agents_temp = copy.copy(self._agents)
        np.random.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('remove_pending'):
                self._simulation.remove_agent(a)

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
                path = bresenham_2d(x1, y1, x2, y2)
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
                new_agent.set_attribute('position', clone_position)
                self._simulation.add_agent(new_agent, position=clone_position)

    def _clear_substrate_dynamic_nodes(self):
        for s in self._substrates.values():
            s.clear_dynamic_nodes()
