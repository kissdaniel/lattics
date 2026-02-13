from ._base import BaseSpace
# from lattics.core.substrates import HomogeneousSubstrateField
from lattics.core import Agent
from lattics.core import Simulation
from lattics.core import UpdateInfo

from abc import ABC, abstractmethod
import copy
import numpy as np
import numpy.typing as npt
from scipy import ndimage
import warnings
import numba
from numba import void, float32, int32, boolean, types


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
        if not dimensions:
            raise ValueError('You must provide a valid dimension for the space size, for example (50, 70).')
        self._dimensions = dimensions
        self._grid_spacing = grid_spacing
        self._agents = list()
        self._agent_layer = np.empty(self._dimensions, dtype='object')
        self._substrates = dict()

    def add_agent(self, agent: Agent) -> None:
        """TODO

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
        self._validate_required_attributes(agent)
        self._validate_optional_attributes(agent)
        position = agent.get_attribute('position')
        self._agents.append(agent)
        self._agent_layer[tuple(position)] = agent

    def _validate_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('position'):
            raise ValueError('The agent must have an attribute named \'position\'.')
        if not agent.has_attribute('volume'):
            raise ValueError('The agent must have an attribute named \'volume\'.')
        position = agent.get_attribute('position')
        volume = agent.get_attribute('volume')
        if not self.is_valid_position(position):
            raise ValueError(f'Position {position} is out of the bounds of the domain.')
        if not self.is_empty_position(position):
            raise ValueError(f'Position {position} already occupied.')
        if self.get_remaining_volume(position) < volume:
            warnings.warn('The agent\'s volume is larger than the volume assigned to '
                          'the domain chunks. This may lead to unexpected behavior.')

    def _validate_optional_attributes(self, agent: Agent) -> None:
        motility_default_value = 0
        binding_affinity_default_value = 0
        displacement_limit_default_value = 1
        division_completed_default_value = False
        division_pending_default_value = False
        remove_pending_default_value = False
        if not agent.has_attribute('binding_affinity'):
            agent.set_attribute('binding_affinity', binding_affinity_default_value)
            warnings.warn('Agent attribute \'binding_affinity\' was not defined, a default value is used.')
        if not agent.has_attribute('displacement_limit'):
            agent.set_attribute('displacement_limit', displacement_limit_default_value)
            warnings.warn('Agent attribute \'displacement_limit\' was not defined, a default value is used.')
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', division_completed_default_value)
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', division_pending_default_value)
        if not agent.has_attribute('motility'):
            agent.set_attribute('motility', motility_default_value)
            warnings.warn('Agent attribute \'motility\' was not defined, a default value is used.')
        if not agent.has_attribute('remove_pending'):
            agent.set_attribute('remove_pending', remove_pending_default_value)

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
        # substrate = Lattice2DSubstrateField(domain=self,
        #                                     substrate_name=name,
        #                                     diffusion_coefficient=diffusion_coefficient,
        #                                     decay_coefficient=decay_coefficient
        #                                     )
        # self._substrates[name] = substrate
        pass

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
            change_flags = np.full(len(agents), False, dtype='bool')

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


@numba.njit(int32[:, :](int32, int32, int32, int32), cache=True)
def bresenham_2d(x1, y1, x2, y2):
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    size = np.max(np.array([dx, dy], dtype='int32')) + 1
    path = np.zeros((size, 2), dtype='int32')
    idx = 0
    x = x1
    y = y1

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dy < dx:
        error = dx // 2
        while x != x2:
            path[idx] = [x, y]
            idx += 1
            error = error - dy
            if error < 0:
                y = y + sy
                error = error + dx
            x = x + sx
    else:
        error = dy // 2
        while y != y2:
            path[idx] = [x, y]
            idx += 1
            error = error - dx
            if error < 0:
                x = x + sx
                error = error + dy
            y = y + sy
    path[idx] = [x2, y2]
    return path


@numba.njit(int32[:, :](types.unicode_type), cache=True)
def get_neighborhood_2d(type):
    if type == 'von_neumann':
        neighborhood = np.array(
            [
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1]
            ],
            dtype=np.int32
        )
        return neighborhood
    elif type == 'moore':
        neighborhood = np.array(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1]
            ],
            dtype=np.int32
        )
        return neighborhood
    else:
        raise ValueError('Argument must be either \'von_neumann\' or \'moore\'.')


@numba.njit(void(int32[:, :], int32, int32[:], int32[:, :]), cache=True)
def displace_agent_2d(positions, idx, new_position, agent_idx_array):
    old_position = positions[idx]
    agent_idx_array[old_position[0], old_position[1]] = -1
    agent_idx_array[new_position[0], new_position[1]] = idx
    positions[idx] = new_position


@numba.njit(float32(int32[:], float32, int32[:], float32), cache=True)
def pairwise_interaction_energy_2d(position_one, bindig_aff_one, position_two, binding_aff_two):
    """Computes pairwise interaction energies of two agents.

    Args:
        position_one (array of ints): the coordinates of agent one
        bindig_aff_one (float): the binding affinity of agent one
        position_two (array of ints): the coordinates of agent two
        binding_aff_two (float): the binding affinity of agent two

    Returns:
        float: the interaction energy
    """
    distance = abs(position_one[0] - position_two[0]) + abs(position_one[1] - position_two[1])
    if distance == 0:
        return np.inf
    elif distance == 1:
        return -np.sqrt(bindig_aff_one * binding_aff_two)
    else:
        return np.float32(0)


@numba.njit(float32(int32, int32[:], float32[:], int32[:, :]), cache=True)
def total_interaction_energy_2d(idx, agent_pos, bind_affs, agent_idx_array):
    """Computes the sum of pairwise interaction energies of a given agent.

    Args:
        idx (int): identifier (index) of the selected agent
        agent_pos (array of ints): the position of the selected agent
        bind_affs (array of floats): 1D array of agent binding affinities
        agent_idx_array (array of ints): 2D array containing identifiers (indexes)
            of the agents based on their positions

    Returns:
        float: the total interaction energy of the agent
    """
    neighborhood = get_neighborhood_2d('moore')
    n_size = neighborhood.shape[0]
    agent_bind = bind_affs[idx]
    size_x = agent_idx_array.shape[0]
    size_y = agent_idx_array.shape[1]
    energy = np.float32(0.0)
    for i in range(n_size):
        neighbor_pos = np.add(agent_pos, neighborhood[i])
        nx = neighbor_pos[0]
        ny = neighbor_pos[1]
        # Avoid positions outside the boundaries of the array
        if 0 <= nx < size_x and 0 <= ny < size_y:
            # Check if the position is occupied by an agent
            if not agent_idx_array[nx, ny] == -1:
                nidx = agent_idx_array[nx, ny]
                neighbor_bind = bind_affs[nidx]
                energy += pairwise_interaction_energy_2d(agent_pos, agent_bind, neighbor_pos, neighbor_bind)
    return energy


@numba.njit(void(int32, int32[:, :], float32[:], int32[:, :], boolean[:]), cache=True)
def displacement_trial_2d(idx, positions, binding_affs, agent_idx_array, change_flags):
    """Performs a displacement trial with a given agent assuming 2D coordinates.

    Args:
        idx (int):
        positions (array of ints): 2D array of the positions of all agents
        binding_affs (array of float): the binding affinities of all agents
        agent_idx_array (2D array of ints): identifiers (indexes) of the agents
            based on their positions
        change_flags (array of bools): indicating whether an agent was
            relocated during the trial
    """
    current_pos = np.copy(positions[idx])
    current_energy = total_interaction_energy_2d(idx, current_pos, binding_affs, agent_idx_array)
    neighborhood = get_neighborhood_2d('von_neumann')
    n_idx = np.random.randint(neighborhood.shape[0])
    target_pos = np.add(positions[idx], neighborhood[n_idx])
    tx = target_pos[0]
    ty = target_pos[1]
    size_x = agent_idx_array.shape[0]
    size_y = agent_idx_array.shape[1]
    if 0 <= tx < size_x and 0 <= ty < size_y:
        target_idx = agent_idx_array[tx, ty]
        if target_idx == -1:
            displace_agent_2d(positions, idx, target_pos, agent_idx_array)
            target_energy = total_interaction_energy_2d(idx, target_pos, binding_affs, agent_idx_array)
            if not np.random.random() < np.exp(-(target_energy - current_energy)):
                displace_agent_2d(positions, idx, current_pos, agent_idx_array)
            else:
                change_flags[idx] = bool(True)
