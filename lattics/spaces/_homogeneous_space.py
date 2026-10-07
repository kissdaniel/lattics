import warnings

from lattics._agent import Agent
from lattics._simulation import Simulation
from lattics.attributes import default_of
from lattics.substrates import HomogeneousSubstrateField

from ._base import BaseSpace


class HomogeneousSpace(BaseSpace):
    """Represents a perfectly mixed simulation domain with no spatial structure
    or localized interactions, where each agent has an equal probability of
    interacting with any other agent in the population. Agents are stored in a
    list.
    """

    def __init__(
        self,
        simulation: Simulation,
        volume: int = None,
        dt_agent: tuple[float, str] = None,
        dt_substrate: tuple[float, str] = None,
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
            warnings.warn('The total volume of agents exceeded the capacity ' 'of the domain.')
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

    # TODO: create a Substrate class to store the data below (name, diffusion_coeff, etc.)
    def add_substrate(
        self,
        name: str,
        diffusion_coefficient: float = 0.0,
        decay_coefficient: float = 0.0,
        decay_kinetics: str = 'first-order',
        mm_constant: float = None,
    ) -> None:
        substrate = HomogeneousSubstrateField(
            domain=self,
            substrate_name=name,
            diffusion_coefficient=diffusion_coefficient,
            decay_coefficient=decay_coefficient,
            decay_kinetics=decay_kinetics,
            mm_constant=mm_constant,
        )
        self._substrates[name] = substrate

    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration. Agents
        labeled as ``space.division_requested`` are duplicated based on the available
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
                if a.get_attribute('space.division_requested'):
                    self._process_agent_division(a)
                if a.get_attribute('space.removal_requested'):
                    self._simulation.remove_agent(a)

                if a.has_attribute('substrate.transport'):
                    info = a.get_attribute('substrate.transport')
                    for i in info.keys():
                        self._substrates[i].add_dynamic_substrate_node(a)
            self._agent_update_info.reset_time()

        if self._substrate_update_info.update_needed():
            for s in self._substrates.values():
                s.update(self._substrate_update_info.elapsed_time)
            self._substrate_update_info.reset_time()

    def _initialize_attributes(self, agent, volume) -> None:
        """Initializes the attributes used by the domain. The domain tracks
        two attributes: ``space.division_requested`` and ``space.division_completed``. The
        ``space.division_requested`` attribute can be set by cell-level functions to
        indicate that an agent is ready to divide and that the domain should
        handle the division event. Once division has occurred, the
        ``space.division_completed`` attribute is set to ``True``.
        TODO: volume
        """
        for name in (
            'space.division_requested',
            'space.division_completed',
            'space.removal_requested',
        ):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))
        if not agent.has_attribute('space.volume'):
            agent.set_attribute('space.volume', volume)

    def _process_substrates(self, dt) -> None:
        pass

    def _clear_substrate_dynamic_nodes(self):
        for s in self._substrates.values():
            s.clear_dynamic_nodes()

    def _process_agent_division(self, agent: Agent) -> None:
        if not self._has_free_volume:
            return
        sum_volumes = self._get_total_agent_volume()
        agent_volume = agent.get_attribute('space.volume')
        if sum_volumes + agent_volume <= self._volume:
            agent.set_attribute('space.division_requested', False)
            agent.set_attribute('space.division_completed', True)
            new_agent = agent.clone()
            self._simulation.add_agent(new_agent)
        else:
            self._has_free_volume = False

    def _process_agent_removal(self) -> None:
        pass

    def _get_total_agent_volume(self) -> int:
        return sum([a.get_attribute('space.volume') for a in self._agents])
