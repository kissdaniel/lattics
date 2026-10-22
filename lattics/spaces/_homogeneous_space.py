import warnings

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics.attributes import default_of
from lattics.substrates import HomogeneousSubstrateField

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
        'substrate.transport',
    )
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
        self._has_free_volume = True

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
        if not self._fits(self._get_total_agent_volume()):
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

    # TODO: create a Substrate class to store the data below (name, diffusion_coeff, etc.)
    def add_substrate(
        self,
        name: str,
        diffusion_coefficient: float = 0.0,
        decay_coefficient: float = 0.0,
        decay_kinetics: str = 'first-order',
        mm_constant: float = None,
    ) -> None:
        if self._volume is None:
            raise ValueError('A substrate needs a domain with a volume: set the size or the volume of the space.')
        substrate = HomogeneousSubstrateField(
            domain=self,
            substrate_name=name,
            diffusion_coefficient=diffusion_coefficient,
            decay_coefficient=decay_coefficient,
            decay_kinetics=decay_kinetics,
            mm_constant=mm_constant,
        )
        self._substrates[name] = substrate

    def _update_agents(self, ctx: UpdateContext, dt: float) -> None:
        """Agents labeled as ``space.removal_requested`` are removed first. Then
        the agents labeled as ``space.division_requested`` are duplicated based
        on the available space in the domain, constrained by its volume.
        Finally the local densities are updated.
        """
        for a in self._population:
            if a.get_attribute('space.removal_requested'):
                ctx.remove_agent(a)
        for a in self._population:
            if a.get_attribute('space.division_requested'):
                self._process_agent_division(a, ctx)
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
        sum_volumes = self._get_total_agent_volume()
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
