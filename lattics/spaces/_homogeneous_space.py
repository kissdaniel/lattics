import warnings

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics.attributes import default_of
from lattics.substrates import HomogeneousSubstrateField, Substrate, SubstrateNode

from ._base import BaseSpace


class HomogeneousSpace(BaseSpace):
    """Represents a perfectly mixed simulation domain with no spatial structure
    or localized interactions, where each agent has an equal probability of
    interacting with any other agent in the population.

    The capacity of the domain is its volume, given directly by ``volume`` or
    computed from ``size`` and ``thickness``; without either the capacity is
    unlimited. ``grid_spacing`` only serves as the default thickness. The local density of an agent is the
    fraction of the free volume (without the agent itself) occupied by the
    other agents.

    The keyword arguments are the same as those of every space, see
    :class:`~lattics.spaces.Lattice2DSpace`.
    """

    reads = (
        'space.volume',
        'space.division_requested',
        'space.removal_requested',
    )
    optional_reads = ('substrate.transport',)
    writes = (
        'space.local_density',
        'space.division_requested',
        'space.last_division_time',
        'space.removal_requested',
    )

    def __init__(
        self,
        size: tuple[tuple[float, str], ...] = None,
        volume: tuple[float, str] = None,
        thickness: tuple[float, str] = None,
        grid_spacing: tuple[float, str] = (20, 'um'),
        interval: tuple[float, str] = None,
    ) -> None:
        super().__init__(
            size=size,
            volume=volume,
            thickness=thickness,
            grid_spacing=grid_spacing,
            interval=interval,
        )
        self._has_free_volume = True
        # The total volume of the agents while the divisions of an update are processed, None otherwise
        self._division_total = None

    def add_agent(self, agent: Agent) -> None:
        """Initializes the attributes of the specified agent, which is already
        part of the simulation's population.

        Parameters
        ----------
        agent : Agent
            The agent to be added

        Raises
        ------
        ValueError
            If the agent has no ``space.volume`` attribute
        """
        self._validate_volume(agent)
        self._initialize_attributes(agent)
        if self._division_total is not None:
            # A daughter: it is the last agent of the population, so this equals the sum over all agents
            self._division_total += agent.get_attribute('space.volume')
            total = self._division_total
        else:
            total = self._get_total_agent_volume()
        if not self._fits(total):
            warnings.warn('The total volume of agents exceeded the capacity of the domain.')

    def remove_agent(self, agent: Agent) -> None:
        """Updates the free capacity after the specified agent has been removed
        from the simulation's population.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        if self._fits(self._get_total_agent_volume()):
            self._has_free_volume = True

    def create_field(self, substrate: Substrate, nodes: list[SubstrateNode]) -> HomogeneousSubstrateField:
        if self._volume is None:
            raise ValueError('A substrate needs a domain with a volume: set the size or the volume of the space.')
        if substrate.diffusion_coefficient:
            self._notes.append(f'the diffusion coefficient of \'{substrate.name}\' is not used')
        return HomogeneousSubstrateField(substrate, self, nodes)

    def _update_agents(self, ctx: UpdateContext, dt: float) -> None:
        """Agents labeled as ``space.removal_requested`` are removed first. Then
        the agents labeled as ``space.division_requested`` are duplicated based
        on the available space in the domain, constrained by its volume.
        Finally the local densities are updated.
        """
        for a in self._population:
            if a.get_attribute('space.removal_requested'):
                ctx.remove_agent(a)
        # Only the divisions change the population from here on, so the total volume is kept up to date
        # instead of being summed again for every division
        self._division_total = self._get_total_agent_volume()
        try:
            for a in self._population:
                if a.get_attribute('space.division_requested'):
                    self._process_agent_division(a, ctx)
        finally:
            self._division_total = None
        self._update_local_densities()

    def _initialize_attributes(self, agent: Agent) -> None:
        for name in (
            'space.division_requested',
            'space.last_division_time',
            'space.removal_requested',
            'space.local_density',
        ):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))

    def _process_agent_division(self, agent: Agent, ctx: UpdateContext) -> None:
        if not self._has_free_volume:
            return
        sum_volumes = self._division_total
        agent_volume = agent.get_attribute('space.volume')
        if self._fits(sum_volumes + agent_volume):
            agent.set_attribute('space.division_requested', False)
            agent.set_attribute('space.last_division_time', ctx.time)
            new_agent = agent.clone()
            ctx.add_agent(new_agent)
        else:
            self._has_free_volume = False

    def _update_local_densities(self) -> None:
        total_volume = self._get_total_agent_volume()
        for agent in self._population:
            density = 0.0
            if self._volume is not None:
                own_volume = agent.get_attribute('space.volume')
                free_volume = self._volume - own_volume
                density = (total_volume - own_volume) / free_volume if free_volume > 0 else 1.0
            agent.set_attribute('space.local_density', density)

    def _get_total_agent_volume(self) -> int:
        return sum([a.get_attribute('space.volume') for a in self._population])

    def _fits(self, total_volume: float) -> bool:
        return self._volume is None or total_volume <= self._volume
