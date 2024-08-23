"""LattiCS module containing the models for simulation space consists of agent layer and substrate layers.
"""

from .mechanics import MonteCarloMechanics2D
from .mechanics import MonteCarloMechanics3D
import numpy as np
from scipy import ndimage
import copy
from . import _numba_funcs

class SimulationSpace2D:
    def __init__(self,
                 simulation=None,
                 dimensions=None,
                 substrates=None,
                 agent_layer_dx=10,
                 substrate_layer_dx=10,
                 masstransport='2D-ADI',
                 mechanics='2D-MC'
                 ):
        # general members
        self._simulation = simulation
        self._dimensions = np.array(dimensions)
        self._substrates = substrates

        # members related to agents
        self._a_dx = agent_layer_dx
        self._agent_layer = None

        # members related to substrates
        self._s_dx = substrate_layer_dx
        self._substrate_layer = None
        self._sl_permanent = None
        self._sl_agentlocs = None

        # models
        self._masstransport_model = None
        self._mechanics_model = None
        if masstransport == '2D-ADI':
            pass
        if mechanics == '2D-MC':
            self._mechanics_model = MonteCarloMechanics2D()

        # self._update_flags = None

    @property
    def dimensions(self):
        """The spatial dimensions (width, height) of the simulation space expressed in micrometers."""
        return self._dimensions

    @property
    def agent_layer_dx(self):
        """The spatial resolution of the grid containing the agents, expressed in micrometers."""
        return self._a_dx

    @property
    def substrate_layer_dx(self):
        """The spatial resolution of the grid containing the substrate concentrations, expressed in micrometers."""
        return self.a_dx

    @property
    def agent_layer_shape(self):
        """The shape property of the underlying Numpy array."""
        return self._agent_layer.shape

    def is_valid_position(self, position):
        return np.all(np.zeros(2) <= np.array(position)) and np.all(np.array(position) < self._agent_layer.shape)

    def is_empty_position(self, position):
        return self._agent_layer[position[0], position[1]] is None

    def add_agent(self, agent):
        if (agent.position is None or
            not agent.position.shape == self.dimensions.shape):
            raise ValueError('Agent must have a 2D position defined.')
        if not self.is_valid_position(agent.position):
            raise ValueError('Invaid position was given. Use positions between zero and the maximum size of the simulation space.')
        self._simulation.agents.append(agent)
        x, y = agent.position[:2]
        self._agent_layer[x, y] = agent

    def remove_agent(self, agent):
        self._simulation.agents.remove(agent)
        x, y = agent.position[:2]
        self._agent_layer[x, y] = None

    def update_masstransport(self, dt):
        pass

    def update_mechanics(self, dt):
        self._mechanics_model.update(dt)

    def update_divisions(self):
        agents = copy.copy(self._simulation.agents)
        np.random.shuffle(agents)
        for a in agents:
            if a.get_status_flag('division_ready'):
                self.division_trial(a)

    def division_trial(self, agent):
        coverage_mask = np.where(self._agent_layer, float('inf'), 0)
        if np.any(coverage_mask == 0):
            source_map = np.ones(self.agent_layer_shape)
            source_map[tuple(agent.position)] = 0
            distance_map = ndimage.distance_transform_edt(source_map)
            distance_map = distance_map + coverage_mask
            min_distance = np.min(distance_map)
            if min_distance <= agent.displacement_limit:
                target_sites = np.argwhere(distance_map == min_distance)
                target = target_sites[np.random.randint(target_sites.shape[0])]
                x1, y1 = agent.position[:2]
                x2, y2 = target[:2]
                path = _numba_funcs.bresenham_2d(x1, y1, x2, y2)
                if path.shape[0] > 2:
                    for i in range(path.shape[0] - 2, 0, -1):
                        a_old_x, a_old_y = path[i, :2]
                        a_new_x, a_new_y = path[i + 1, :2]
                        agent_to_move = self._agent_layer[a_old_x, a_old_y]
                        self._agent_layer[a_new_x, a_new_y] = agent_to_move
                        agent_to_move.position = path[i + 1]
                clone_pos = path[1]
                agent.cellcycle_model.reset()
                clone = agent.clone()
                clone.position = clone_pos
                self.add_agent(clone)

    def initialize(self):
        dim_agent_x = int(np.ceil(self._dimensions[0] / self._a_dx))
        dim_agent_y = int(np.ceil(self._dimensions[1] / self._a_dx))
        self._agent_layer = np.empty((dim_agent_x, dim_agent_y), dtype='object')
        self._mechanics_model.initialize(self._simulation)

    def get_neighbors(self, position):
        neighbors = list()
        neighborhood = _numba_funcs.get_neighborhood_2d('von_neumann')
        for n in neighborhood:
            n_pos = np.add(position, n)
            if self.is_valid_position(n_pos):
                x, y = n_pos[:2]
                if self._agent_layer[x, y]:
                    neighbors.append(self._agent_layer[x, y])
        return neighbors

    def _pos_to_agent_idx(self, position):
        return (np.array(position) / self._a_dx).astype(int)

    def _pos_to_substrate_idx(self, position):
        return (np.array(position) / self._s_dx).astype(int)
