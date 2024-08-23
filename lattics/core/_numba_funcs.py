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
    path[idx] = [x1, y1]
    idx += 1
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    error = dx + dy

    while (x1 != x2 and y1 != y2):
        e = 2 * error
        if e >= dy:
            if x1 != x2:
                error = error + dy
                x1 = x1 + xs
        if e <= dx:
            if y1 != y2:
                error = error + dx
                y1 = y1 + ys
        path[idx] = [x1, y1]
        idx += 1
    return path


@numba.njit(int32[:, :](int32, int32, int32, int32, int32, int32), cache=True)
def bresenham_3d(x1, y1, z1, x2, y2, z2):
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    dz = np.abs(z2 - z1)
    size = np.max(np.array([dx, dy, dz], dtype='int32')) + 1
    path = np.zeros((size, 3), dtype='int32')
    idx = 0
    path[idx] = [x1, y1, z1]
    idx += 1
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            path[idx] = [x1, y1, z1]
            idx += 1

    # Driving axis is Y-axis
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            path[idx] = [x1, y1, z1]
            idx += 1

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            path[idx] = [x1, y1, z1]
            idx += 1
    return path


@numba.jit(float32[:](float32[:],float32[:],float32[:],float32[:]),nopython=True, cache=True)
def TDMA_solver(sub, diag, sup, const):
    N = len(diag)
    sol = np.zeros(N, dtype='float32')
    for i in range(1, N):
        w = sub[i] / diag[i - 1]
        diag[i] = diag[i] - w * sup[i - 1]
        const[i] = const[i] - w * const[i - 1]
    sol[N - 1] = const[N - 1] / diag[N - 1]
    for i in range(N - 2, -1, -1):
        sol[i] = (const[i] - sup[i] * sol[i + 1]) / diag[i]
    return sol


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


@numba.njit(void(int32[:, :], int32, int32[:], int32[:, :, :]), cache=True)
def displace_agent(positions, idx, new_position, agent_idx_array):
    old_position = positions[idx]
    agent_idx_array[old_position[0], old_position[1], old_position[2]] = -1
    agent_idx_array[new_position[0], new_position[1], new_position[2]] = idx
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

@numba.njit(float32(int32[:], float32, int32[:], float32), cache=True)
def pairwise_interaction_energy_3d(position_one, bindig_aff_one, position_two, binding_aff_two):
    """Computes pairwise interaction energies of two agents.

    Args:
        position_one (array of ints): the coordinates of agent one
        bindig_aff_one (float): the binding affinity of agent one
        position_two (array of ints): the coordinates of agent two
        binding_aff_two (float): the binding affinity of agent two

    Returns:
        float: the interaction energy
    """
    distance = abs(position_one[0] - position_two[0]) + abs(position_one[1] - position_two[1]) + abs(position_one[2] - position_two[2])
    if distance == 0:
        return np.Inf
    elif distance == 1:
        return -np.sqrt(bindig_aff_one * binding_aff_two)
    else:
        return np.float32(0)

@numba.njit(float32(int32, int32[:], float32[:], int32[:, :, :]), cache=True)
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


@numba.njit(float32(int32, int32[:], float32[:], int32[:, :, :]), cache=True)
def total_interaction_energy_3d(idx, agent_pos, bind_affs, agent_idx_array):
    """Computes the sum of pairwise interaction energies of a given agent.

    Args:
        idx (int): identifier (index) of the selected agent
        agent_pos (array of ints): the position of the selected agent
        bind_affs (array of floats): 1D array of agent binding affinities
        agent_idx_array (array of ints): 3D array containing identifiers (indexes)
            of the agents based on their positions

    Returns:
        float: the total interaction energy of the agent
    """
    neighborhood = get_neighborhood_3d('moore')
    n_size = neighborhood.shape[0]
    agent_bind = bind_affs[idx]
    size_x = agent_idx_array.shape[0]
    size_y = agent_idx_array.shape[1]
    size_z = agent_idx_array.shape[2]
    energy = np.float32(0.0)
    for i in range(n_size):
        neighbor_pos = np.add(agent_pos, neighborhood[i])
        nx = neighbor_pos[0]
        ny = neighbor_pos[1]
        nz = neighbor_pos[2]
        # Avoid positions outside the boundaries of the array
        if 0 <= nx < size_x and 0 <= ny < size_y and 0 <= nz < size_z:
            # Check if the position is occupied by an agent
            if not agent_idx_array[nx, ny, nz] == -1:
                nidx = agent_idx_array[nx, ny, nz]
                neighbor_bind = bind_affs[nidx]
                energy += pairwise_interaction_energy_3d(agent_pos, agent_bind, neighbor_pos, neighbor_bind)
    return energy


@numba.njit(void(int32, int32[:, :], float32[:], int32[:, :, :], boolean[:]), cache=True)
def displacement_trial_2d(idx, positions, binding_affs, agent_idx_array, change_flags):
    """Performs a displacement trial with a given agent assuming 2D coordinates.

    Args:
        idx (int): identifier (index) of the selected agent
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
            displace_agent(positions, idx, target_pos, agent_idx_array)
            target_energy = total_interaction_energy_2d(idx, target_pos, binding_affs, agent_idx_array)
            if not np.random.random() < np.exp(-(target_energy - current_energy)):
                displace_agent(positions, idx, current_pos, agent_idx_array)
            else:
                change_flags[idx] = np.bool8(True)


@numba.njit(void(int32, int32[:, :], float32[:], int32[:, :, :], boolean[:]), cache=True)
def displacement_trial_3d(idx, positions, binding_affs, agent_idx_array, change_flags):
    """Performs a displacement trial with a given agent.

    Args:
        idx (int): identifier (index) of the selected agent
        positions (array of ints): 2D array of the positions of all agents
        binding_affs (array of float): the binding affinities of all agents
        agent_idx_array (3D array of ints): identifiers (indexes) of the agents
            based on their positions
        change_flags (array of bools): indicating whether an agent was
            relocated during the trial
    """
    current_pos = np.copy(positions[idx])
    current_energy = total_interaction_energy_3d(idx, current_pos, binding_affs, agent_idx_array)
    neighborhood = get_neighborhood_3d('von_neumann')
    n_idx = np.random.randint(neighborhood.shape[0])
    target_pos = np.add(positions[idx], neighborhood[n_idx])
    tx = target_pos[0]
    ty = target_pos[1]
    tz = target_pos[2]
    size_x = agent_idx_array.shape[0]
    size_y = agent_idx_array.shape[1]
    size_z = agent_idx_array.shape[2]
    if 0 <= tx < size_x and 0 <= ty < size_y and 0 <= tz < size_z:
        target_idx = agent_idx_array[tx, ty, tz]
        if target_idx == -1:
            displace_agent(positions, idx, target_pos, agent_idx_array)
            target_energy = total_interaction_energy_3d(idx, target_pos, binding_affs, agent_idx_array)
            if not np.random.random() < np.exp(-(target_energy - current_energy)):
                displace_agent(positions, idx, current_pos, agent_idx_array)
            else:
                change_flags[idx] = np.bool8(True)
