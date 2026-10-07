import warnings

import numpy as np
from scipy import ndimage

from lattics._agent import Agent
from lattics._kernels.lattice import bresenham_2d, displacement_trial_2d
from lattics._simulation import Simulation
from lattics.attributes import default_of

from ._base import BaseSpace


class Lattice2DSpace(BaseSpace):
    def __init__(
        self,
        simulation: Simulation,
        dimensions: tuple[int, int],
        grid_spacing: int,
        dt_agent: tuple[float, str] = None,
        dt_substrate: tuple[float, str] = None,
    ) -> None:
        super().__init__(simulation=simulation, dt_agent=dt_agent, dt_substrate=dt_substrate)
        if not dimensions:
            raise ValueError('You must provide a valid dimension for the space size, for example (50, 70).')
        self._dimensions = dimensions
        self._grid_spacing = grid_spacing
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
        position = agent.get_attribute('space.position')
        self._agent_layer[tuple(position)] = agent

    def _validate_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('space.position'):
            raise ValueError('The agent must have an attribute named \'space.position\'.')
        if not agent.has_attribute('space.volume'):
            raise ValueError('The agent must have an attribute named \'space.volume\'.')
        position = agent.get_attribute('space.position')
        volume = agent.get_attribute('space.volume')
        if not self.is_valid_position(position):
            raise ValueError(f'Position {position} is out of the bounds of the domain.')
        if not self.is_empty_position(position):
            raise ValueError(f'Position {position} already occupied.')
        if self.get_remaining_volume(position) < volume:
            warnings.warn(
                'The agent\'s volume is larger than the volume assigned to '
                'the domain chunks. This may lead to unexpected behavior.'
            )

    def _validate_optional_attributes(self, agent: Agent) -> None:
        for name in ('mechanics.binding_affinity', 'mechanics.displacement_limit'):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))
                warnings.warn(f'Agent attribute \'{name}\' was not defined, a default value is used.')
        for name in ('space.division_completed', 'space.division_requested'):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))
        if not agent.has_attribute('mechanics.motility'):
            agent.set_attribute('mechanics.motility', default_of('mechanics.motility'))
            warnings.warn('Agent attribute \'mechanics.motility\' was not defined, a default value is used.')
        for name in ('space.removal_requested', 'space.local_density'):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the lattice.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        position = agent.get_attribute('space.position')
        self._agent_layer[tuple(position)] = None

    def add_substrate(self, name: str, diffusion_coefficient: float = 0.0, decay_coefficient: float = 0.0) -> None:
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
            self._update_local_densities()

            self._clear_substrate_dynamic_nodes()
            for a in self._simulation.population:
                if a.has_attribute('substrate.transport'):
                    info = a.get_attribute('substrate.transport')
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
        return self._grid_spacing**2 - self._agent_layer[tuple(position)].get_attribute('space.volume')

    def _displacement_trials(self, dt: int) -> None:
        if len(self._simulation.population):
            agents = list(self._simulation.population)
            np.random.shuffle(agents)
            positions = np.array([a.get_attribute('space.position') for a in agents], dtype='int32')
            disp_probs = np.array(
                [a.get_attribute('mechanics.motility') * dt / self._grid_spacing for a in agents],
                dtype='float32',
            )
            binding_affs = np.array(
                [a.get_attribute('mechanics.binding_affinity') for a in agents], dtype='float32'
            )
            # 3D array containing identifiers (idx) at those elements occupied by agents
            idx_array = np.full((self._agent_layer.shape[0], self._agent_layer.shape[1]), -1, dtype='int32')
            for i, a in enumerate(agents):
                position = a.get_attribute('space.position')
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
                    pos_old = agents[i].get_attribute('space.position')
                    agent_new = self._agent_layer[pos_new[0], pos_new[1]]
                    agent_old = self._agent_layer[pos_old[0], pos_old[1]]
                    self._agent_layer[tuple(pos_new)] = agent_old
                    self._agent_layer[tuple(pos_old)] = agent_new
                    if agent_new:
                        agent_new.set_attribute('space.position', pos_old)
                    if agent_old:
                        agent_old.set_attribute('space.position', pos_new)

    def _cell_removal_trials(self, dt: int) -> None:
        agents_temp = list(self._simulation.population)
        # TODO: check if shuffle is needed here
        np.random.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('space.removal_requested'):
                self._simulation.remove_agent(a)

    def _cell_division_trials(self, dt: int) -> None:
        agents_temp = list(self._simulation.population)
        np.random.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('space.division_requested'):
                self._perform_cell_division(a, dt)

    def _perform_cell_division(self, agent: Agent, dt: int) -> None:
        current_position = agent.get_attribute('space.position')
        displacement_limit = agent.get_attribute('mechanics.displacement_limit')
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
                        agent_to_move.set_attribute('space.position', path[i + 1])
                clone_position = path[1]
                self._agent_layer[tuple(clone_position)] = None
                agent.set_attribute('space.division_requested', False)
                agent.set_attribute('space.division_completed', True)
                new_agent = agent.clone()
                new_agent.set_attribute('space.position', clone_position)
                self._simulation.add_agent(new_agent)

    def _clear_substrate_dynamic_nodes(self):
        for s in self._substrates.values():
            s.clear_dynamic_nodes()

    def _update_local_densities(self) -> None:
        coverage_map = np.where(self._agent_layer, 1, 0)
        i_indices, j_indices = np.ogrid[: coverage_map.shape[0], : coverage_map.shape[1]]
        for agent in self._simulation.population:
            current_position = agent.get_attribute('space.position')
            displacement_limit = agent.get_attribute('mechanics.displacement_limit')
            mask = (i_indices - current_position[0]) ** 2 + (
                j_indices - current_position[1]
            ) ** 2 <= displacement_limit**2
            mask_covered = (mask) & (coverage_map == 1)
            density = (np.sum(mask_covered) - 1) / (np.sum(mask) - 1)
            agent.set_attribute('space.local_density', density)
