"""LattiCS module containing the models for cell-cell mechanical interactions and motion.
"""

import copy
import numpy as np
from . import _numba_funcs


class MonteCarloMechanics2D:
    """Class implementing on-lattice Monte Carlo cell motility in 2D.

    The model is a simplified version of the Cellular Potts Model (CPM) with
    one single grid point representing a biological cell.
    """
    def __init__(self):
        self._simulation = None
        self._dx = None
        self._space_shape = None
        self._agent_layer = None

    def initialize(self, simulation):
        self._simulation = simulation
        self._dx = simulation._simulation_space.agent_layer_dx
        self._space_shape = simulation._simulation_space.agent_layer_shape
        self._agent_layer = simulation._simulation_space._agent_layer

    def update(self, dt):
        """Updates positions by performing an MC trial on the agents.

        Upon an MC trial, all agents are selected in random order and
        a displacement trial is executed depending on the motility
        (charactersitic velocity) of the individual.

        Args:
            dt (int): time step in milliseconds
        """
        if self._simulation.agents:
            agents = copy.copy(self._simulation.agents)
            np.random.shuffle(agents)
            positions = np.array([a.position for a in agents], dtype='int32')
            disp_probs = np.array([a.velocity * dt / self._dx for a in agents], dtype='float32')
            binding_affs = np.array([a.binding_affinity for a in agents], dtype='float32')
            # 3D array containing identifiers (idx) at those elements occupied by agents
            idx_array = np.full((self._space_shape[0], self._space_shape[1], self._space_shape[2]), -1, dtype='int32')
            for i, a in enumerate(agents):
                idx_array[a.position[0], a.position[1]] = np.int32(i)
            # Indicates whether a certain agent changed its position during the MC trial
            change_flags = np.full(len(agents), False, dtype='bool8')

            monte_carlo_trial_2d(positions, disp_probs, binding_affs, idx_array, change_flags)

            for i, cflag in enumerate(change_flags):
                if cflag:
                    pos_new = positions[i]
                    pos_old = agents[i].position
                    agent_new = self._agent_layer[pos_new[0], pos_new[1]]
                    agent_old = self._agent_layer[pos_old[0], pos_old[1]]
                    self._agent_layer[pos_new[0], pos_new[1]] = agent_old
                    self._agent_layer[pos_old[0], pos_old[1]] = agent_new
                    if agent_new:
                        agent_new.position = pos_old
                    if agent_old:
                        agent_old.position = pos_new


class MonteCarloMechanics3D:
    """Class implementing on-lattice Monte Carlo cell motility in 3D.

    The model is a simplified version of the Cellular Potts Model (CPM) with
    one single grid point representing a biological cell.
    """
    def __init__(self):
        self._simulation = None
        self._dx = None
        self._space_shape = None
        self._agent_layer = None

    def initialize(self, simulation):
        self._simulation = simulation
        self._dx = simulation._simulation_space.agent_layer_dx
        self._space_shape = simulation._simulation_space.agent_layer_shape
        self._agent_layer = simulation._simulation_space._agent_layer

    def update(self, dt):
        """Updates positions by performing an MC trial on the agents.

        Upon an MC trial, all agents are selected in random order and
        a displacement trial is executed depending on the charactersitic
        velocity of the individual.

        Args:
            dt (int): time step in milliseconds
        """
        if self._simulation.agents:
            agents = copy.copy(self._simulation.agents)
            np.random.shuffle(agents)
            positions = np.array([a.position for a in agents], dtype='int32')
            disp_probs = np.array([a.velocity * dt / self._dx for a in agents], dtype='float32')
            binding_affs = np.array([a.binding_affinity for a in agents], dtype='float32')
            # 3D array containing identifiers (idx) at those elements occupied by agents
            idx_array = np.full((self._space_shape[0], self._space_shape[1], self._space_shape[2]), -1, dtype='int32')
            for i, a in enumerate(agents):
                idx_array[a.position[0], a.position[1], a.position[2]] = np.int32(i)
            # Indicates whether a certain agent changed its position during the MC trial
            change_flags = np.full(len(agents), False, dtype='bool8')

            monte_carlo_trial_3d(positions, disp_probs, binding_affs, idx_array, change_flags)

            for i, cflag in enumerate(change_flags):
                if cflag:
                    pos_new = positions[i]
                    pos_old = agents[i].position
                    agent_new = self._agent_layer[pos_new[0], pos_new[1], pos_new[2]]
                    agent_old = self._agent_layer[pos_old[0], pos_old[1], pos_old[2]]
                    self._agent_layer[pos_new[0], pos_new[1], pos_new[2]] = agent_old
                    self._agent_layer[pos_old[0], pos_old[1], pos_old[2]] = agent_new
                    if agent_new:
                        agent_new.position = pos_old
                    if agent_old:
                        agent_old.position = pos_new


def monte_carlo_trial_2d(positions, disp_probs, binding_affs, agent_idx_array, change_flags):
    for i, dprob in enumerate(disp_probs):
        # Check if trial is needed based on the displacement probability
        if np.random.random() < dprob:
            _numba_funcs.displacement_trial_2d(i, positions, binding_affs, agent_idx_array, change_flags)


def monte_carlo_trial_3d(positions, disp_probs, binding_affs, agent_idx_array, change_flags):
    for i, dprob in enumerate(disp_probs):
        # Check if trial is needed based on the displacement probability
        if np.random.random() < dprob:
            _numba_funcs.displacement_trial_3d(i, positions, binding_affs, agent_idx_array, change_flags)