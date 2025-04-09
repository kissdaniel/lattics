import numpy as np
import numba
from numba import void, float32, int32, boolean, types


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


@numba.njit(int32[:, :](types.unicode_type), cache=True)
def get_neighborhood_3d(type):
    if type == 'von_neumann':
        neighborhood = np.array(
            [
                [-1, 0, 0], [ 1, 0, 0],
                [ 0,-1, 0], [ 0, 1, 0],
                [ 0, 0,-1], [ 0, 0, 1]
            ],
            dtype=np.int32
        )
        return neighborhood
    elif type == 'moore':
        neighborhood = np.array(
            [
                [-1,-1,-1], [-1,-1, 0], [-1,-1, 1],
                [-1, 0,-1], [-1, 0, 0], [-1, 0, 1],
                [-1, 1,-1], [-1, 1, 0], [-1, 1, 1],
                [ 0,-1,-1], [ 0,-1, 0], [ 0,-1, 1],
                [ 0, 0,-1],             [ 0, 0, 1],
                [ 0, 1,-1], [ 0, 1, 0], [ 0, 1, 1],
                [ 1,-1,-1], [ 1,-1, 0], [ 1,-1, 1],
                [ 1, 0,-1], [ 1, 0, 0], [ 1, 0, 1],
                [ 1, 1,-1], [ 1, 1, 0], [ 1, 1, 1]
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
        return np.Inf
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
                change_flags[idx] = np.bool8(True)
