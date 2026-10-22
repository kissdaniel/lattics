import abc
from collections.abc import Iterable

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._units import convert_time


class BaseModel(abc.ABC):
    """Base class for models that change the attributes of the agents.

    A custom model implements :meth:`update_agent`, which is called for every
    agent whenever the model is updated. Models that can process the whole
    population at once (for example with vectorized code) may override
    :meth:`update` instead.

    The class attributes ``reads`` and ``writes`` list the names of the agent
    attributes the model uses and sets.

    Models draw random numbers from ``ctx.rng``, the generator of the
    simulation, so that a seeded simulation is reproducible.

    Parameters
    ----------
    update_interval : tuple[float, str], optional
        The time between two updates of the model, for example ``(30, 'min')``.
        Defaults to every time step of the simulation.
    """

    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __init__(self, update_interval: tuple[float, str] = None) -> None:
        self.interval = convert_time(update_interval[0], update_interval[1], 'ms') if update_interval else None

    def initialize(self, agent: Agent, ctx: UpdateContext) -> None:
        """Initializes the attributes the model needs, when the agent or the
        model is added to the simulation.

        Parameters
        ----------
        agent : Agent
            The agent to be initialized
        ctx : UpdateContext
            The context of the simulation
        """
        self._initialize_required_attributes(agent, ctx)
        self._initialize_optional_attributes(agent, ctx)
        self._initialize_attributes_default_values(agent, ctx)

    def update(self, population: Iterable[Agent], ctx: UpdateContext) -> None:
        """Updates every agent of the population.

        Parameters
        ----------
        population : Iterable[Agent]
            The agents to be updated
        ctx : UpdateContext
            The context of the update
        """
        for agent in population:
            self.update_agent(agent, ctx)

    @abc.abstractmethod
    def update_agent(self, agent: Agent, ctx: UpdateContext) -> None:
        """Updates the attributes of a single agent.

        Parameters
        ----------
        agent : Agent
            The agent to be updated
        ctx : UpdateContext
            The context of the update: the model advances the agent from
            ``ctx.time`` to ``ctx.time + ctx.dt``, where ``ctx.dt`` is the update
            interval of the model
        """
        raise NotImplementedError

    def reset_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        pass

    def _initialize_required_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        pass

    def _initialize_optional_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        pass

    def _initialize_attributes_default_values(self, agent: Agent, ctx: UpdateContext) -> None:
        pass
