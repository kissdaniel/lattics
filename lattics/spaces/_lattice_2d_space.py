import math
import warnings

import numpy as np
from scipy import ndimage

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._kernels.lattice import bresenham_2d, displacement_trial_2d
from lattics.attributes import default_of

from ._base import BaseSpace


class Lattice2DSpace(BaseSpace):
    """A two-dimensional square lattice: every lattice site holds at most one
    agent. Agents move by Monte Carlo displacement trials, push their
    neighbors when they divide, and sense the local density of their
    neighborhood.

    The number of lattice sites along each axis is ``size`` divided by
    ``grid_spacing``; every size must be a multiple of the grid spacing. The
    volume of a lattice site is ``grid_spacing**2 * thickness``. Agents
    without a ``space.position`` attribute are placed on a random free site.

    Parameters
    ----------
    size : tuple[tuple[float, str], tuple[float, str]]
        The size of the domain along the two axes, for example
        ``((750, 'um'), (750, 'um'))``
    volume : tuple[float, str], optional
        Not supported: a lattice space needs ``size``. Accepted only to keep
        the arguments of all spaces the same.
    thickness : tuple[float, str], optional
        The thickness of the domain. Defaults to ``grid_spacing``.
    grid_spacing : tuple[float, str], optional
        The distance between neighboring lattice sites. Defaults to 20 um.
    interval : tuple[float, str], optional
        The time between two updates of the agents, for example ``(30, 'min')``
    substrate_interval : tuple[float, str], optional
        The time between two updates of the substrate fields

    Raises
    ------
    ValueError
        If the size is missing or not a multiple of the grid spacing, or the
        volume is given
    """

    reads = (
        'space.position',
        'space.volume',
        'space.division_requested',
        'space.removal_requested',
        'mechanics.motility',
        'mechanics.binding_affinity',
        'mechanics.displacement_limit',
        'substrate.transport',
    )
    writes = (
        'space.position',
        'space.local_density',
        'space.division_requested',
        'space.last_division_time',
        'space.removal_requested',
        'mechanics.motility',
        'mechanics.binding_affinity',
        'mechanics.displacement_limit',
    )

    def __init__(
        self,
        size: tuple[tuple[float, str], ...] = None,
        volume: tuple[float, str] = None,
        thickness: tuple[float, str] = None,
        grid_spacing: tuple[float, str] = (20, 'um'),
        interval: tuple[float, str] = None,
        substrate_interval: tuple[float, str] = None,
    ) -> None:
        super().__init__(
            size=size,
            volume=volume,
            thickness=thickness,
            grid_spacing=grid_spacing,
            interval=interval,
            substrate_interval=substrate_interval,
        )
        if volume is not None:
            raise ValueError('A lattice space needs the size of the domain, the volume alone is not enough.')
        if size is None or len(size) != 2:
            raise ValueError("You must provide the size of the domain, for example ((750, 'um'), (750, 'um')).")
        dimensions = []
        for (value, unit), length in zip(size, self._size):
            n_sites = length / self._grid_spacing
            if not math.isclose(n_sites, round(n_sites)):
                raise ValueError(
                    f'The domain size {value} {unit} is not a multiple of the grid spacing '
                    f'({self._grid_spacing:g} um).'
                )
            dimensions.append(round(n_sites))
        self._dimensions = tuple(dimensions)
        self._agent_layer = np.empty(self._dimensions, dtype='object')
        self._site_volume = self._grid_spacing**2 * self._thickness

    def add_agent(self, agent: Agent) -> None:
        """Places the specified agent, which is already part of the population,
        on the lattice. Agents without a ``space.position`` attribute are placed
        on a random free site.

        Parameters
        ----------
        agent : Agent
            The agent to be added

        Raises
        ------
        ValueError
            If the agent has no ``space.volume`` attribute, its position is out
            of the bounds of the domain or already occupied, or the lattice is full
        """
        self._validate_volume(agent)
        if not agent.has_attribute('space.position'):
            agent.set_attribute('space.position', self._random_free_site())
        self._validate_position(agent)
        self._validate_optional_attributes(agent)
        position = agent.get_attribute('space.position')
        self._agent_layer[tuple(position)] = agent

    def _validate_position(self, agent: Agent) -> None:
        position = agent.get_attribute('space.position')
        volume = agent.get_attribute('space.volume')
        if not self.is_valid_position(position):
            raise ValueError(f'Position {position} is out of the bounds of the domain.')
        if not self.is_empty_position(position):
            raise ValueError(f'Position {position} already occupied.')
        if self._site_volume < volume:
            warnings.warn(
                'The agent\'s volume is larger than the volume of a lattice site. '
                'This may lead to unexpected behavior.'
            )

    def _random_free_site(self) -> np.ndarray:
        # Random draws are fast while the lattice is sparse; fall back to listing the free sites
        for _ in range(100):
            site = np.array([self._rng.integers(n) for n in self._dimensions], dtype=np.int32)
            if self._agent_layer[tuple(site)] is None:
                return site
        free_sites = np.argwhere(np.equal(self._agent_layer, None))
        if len(free_sites) == 0:
            raise ValueError('The lattice is full, there is no free site for the agent.')
        return free_sites[self._rng.integers(len(free_sites))].astype(np.int32)

    def _validate_optional_attributes(self, agent: Agent) -> None:
        for name in ('mechanics.binding_affinity', 'mechanics.displacement_limit'):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))
                warnings.warn(f'Agent attribute \'{name}\' was not defined, a default value is used.')
        for name in ('space.last_division_time', 'space.division_requested'):
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

    def interval_warnings(self, period: float) -> list[str]:
        if not len(self._population):
            return []
        max_motility = max(a.get_attribute('mechanics.motility') for a in self._population)
        probability = max_motility * period / self._grid_spacing
        if probability <= 1:
            return []
        return [
            f'The displacement probability per update is {probability:.2f} > 1, so the motility of the agents is '
            'truncated to one lattice step per update. Use a shorter update interval for the agents.'
        ]

    def _update_agents(self, ctx: UpdateContext, dt: float) -> None:
        self._displacement_trials(dt)
        self._cell_removal_trials(ctx)
        self._cell_division_trials(ctx)
        self._update_local_densities()

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

    def get_remaining_volume(self, position: tuple[int, int]) -> float:
        if self.is_empty_position(position):
            return self._site_volume
        return self._site_volume - self._agent_layer[tuple(position)].get_attribute('space.volume')

    def _displacement_trials(self, dt: int) -> None:
        if len(self._population):
            agents = list(self._population)
            self._rng.shuffle(agents)
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
                if self._rng.random() < dprob:
                    displacement_trial_2d(i, positions, binding_affs, idx_array, change_flags, self._rng)

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

    def _cell_removal_trials(self, ctx: UpdateContext) -> None:
        agents_temp = list(self._population)
        # TODO: check if shuffle is needed here
        self._rng.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('space.removal_requested'):
                ctx.remove_agent(a)

    def _cell_division_trials(self, ctx: UpdateContext) -> None:
        agents_temp = list(self._population)
        self._rng.shuffle(agents_temp)
        for a in agents_temp:
            if a.get_attribute('space.division_requested'):
                self._perform_cell_division(a, ctx)

    def _perform_cell_division(self, agent: Agent, ctx: UpdateContext) -> None:
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
                target_position = target_sites[self._rng.integers(target_sites.shape[0])]
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
                agent.set_attribute('space.last_division_time', ctx.time)
                new_agent = agent.clone()
                new_agent.set_attribute('space.position', clone_position)
                ctx.add_agent(new_agent)

    def _update_local_densities(self) -> None:
        coverage_map = np.where(self._agent_layer, 1, 0)
        i_indices, j_indices = np.ogrid[: coverage_map.shape[0], : coverage_map.shape[1]]
        for agent in self._population:
            current_position = agent.get_attribute('space.position')
            displacement_limit = agent.get_attribute('mechanics.displacement_limit')
            mask = (i_indices - current_position[0]) ** 2 + (
                j_indices - current_position[1]
            ) ** 2 <= displacement_limit**2
            mask_covered = (mask) & (coverage_map == 1)
            density = (np.sum(mask_covered) - 1) / (np.sum(mask) - 1)
            agent.set_attribute('space.local_density', density)
