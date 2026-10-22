import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._units import convert_time
from lattics.attributes import default_of

from ._base import BaseModel


class TimeBasedCellCycleModel(BaseModel):
    """A timestep-based cell cycle model. Uses the length (time duration) of the cell cycle
    and an internal clock (counter) to track the current state of the cell cycle.

    =============================  ====  ========
    *space.division_requested*     bool  The model sets this to `True` once the cell cycle is complete, signaling
                                         that the agent is ready for replication. The space clears it when it
                                         divides the agent; until then the cell cycle waits.
    *space.last_division_time*     float The space sets this to the time of the division; the model then starts
                                         a new cycle and carries over the time the previous one overshot.
    cellcycle.is_active            bool  Signals if the agent is proliferating (i.e. proceeds through the cell
                                         cycle)
    cellcycle.mean_length          int   Az ágens sejtciklus hosszának átlagos értéke ms-ban kifejezve.
    cellcycle.length               int   .
    cellcycle.current_time         int   .
    cellcycle.random_initial       bool  .
    =============================  ====  ========

    Parameters
    ----------
    BaseModel : _type_
        _description_
    """

    reads = (
        'cellcycle.mean_length',
        'cellcycle.is_active',
        'space.division_requested',
        'space.last_division_time',
    )
    optional_reads = ('cellcycle.random_initial',)
    writes = (
        'cellcycle.length',
        'cellcycle.current_time',
        'space.division_requested',
    )

    def __init__(
        self, update_interval: tuple[float, str] = None, distribution: str = 'erlang', distribution_param: int = 4
    ) -> None:
        super().__init__(update_interval)
        self._distribution = distribution
        self._distribution_param = distribution_param

    def update_agent(self, agent: Agent, ctx: UpdateContext) -> None:
        if ctx.divided_since_last_update(agent):
            self.reset_attributes(agent, ctx)
        # An inactive cell does not progress; a completed cycle waits until the space divides the cell
        if not agent.get_attribute('cellcycle.is_active') or agent.get_attribute('space.division_requested'):
            return
        updated_time = agent.get_attribute('cellcycle.current_time') + ctx.dt
        agent.set_attribute('cellcycle.current_time', updated_time)
        if agent.get_attribute('cellcycle.length') <= updated_time:
            agent.set_attribute('space.division_requested', True)

    def reset_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        """Starts a new cycle after a division. The time by which the previous
        cycle overshot its length is carried over, so the update interval
        does not lengthen the cycles.
        """
        overshoot = agent.get_attribute('cellcycle.current_time') - agent.get_attribute('cellcycle.length')
        agent.set_attribute('cellcycle.current_time', max(0, overshoot))
        attr = agent.get_attribute('cellcycle.mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
        length = self._generate_cellcycle_length(mean_length, ctx.rng)
        agent.set_attribute('cellcycle.length', length)

    def _generate_cellcycle_length(self, mean_length: float, rng: np.random.Generator) -> float:
        if self._distribution == 'fixed':
            return mean_length
        if self._distribution == 'erlang':
            shape = self._distribution_param
            scale = mean_length / shape
            return rng.gamma(shape=shape, scale=scale)
        if self._distribution == 'normal':
            loc = mean_length
            scale = self._distribution_param
            return rng.normal(loc=loc, scale=scale)

    def _initialize_required_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        self._require_attributes(agent, 'cellcycle.mean_length')
        attr = agent.get_attribute('cellcycle.mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
        length = self._generate_cellcycle_length(mean_length, ctx.rng)
        agent.set_attribute('cellcycle.length', length)

    def _initialize_optional_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        if agent.has_attribute('cellcycle.random_initial'):
            length = agent.get_attribute('cellcycle.length')
            cc_time = ctx.rng.uniform(low=0, high=length)
            agent.set_attribute('cellcycle.current_time', cc_time)

    def _initialize_attributes_default_values(self, agent: Agent, ctx: UpdateContext) -> None:
        for name in (
            'cellcycle.is_active',
            'cellcycle.current_time',
            'space.division_requested',
            'space.last_division_time',
        ):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))


class MultiPhaseTimeBasedCellCycleModel(BaseModel):
    """This model implements a simple fixed timestep-based cell cycle by using the cell cycle's length (time
    duration) and an internal clock (counter) to track its current state.

    Parameters
    ----------
    BaseModel : _type_
        _description_
    """

    reads = (
        'cellcycle.phase_g1_mean',
        'cellcycle.phase_s_mean',
        'cellcycle.phase_g2_mean',
        'cellcycle.phase_m_mean',
        'cellcycle.inhibition_density_limit',
        'cellcycle.is_active',
        'space.local_density',
        'space.division_requested',
        'space.last_division_time',
    )
    optional_reads = (
        'cellcycle.phase_g1_sd',
        'cellcycle.phase_s_sd',
        'cellcycle.phase_g2_sd',
        'cellcycle.phase_m_sd',
        'cellcycle.random_initial',
    )
    writes = (
        'cellcycle.phase_g1_length',
        'cellcycle.phase_s_length',
        'cellcycle.phase_g2_length',
        'cellcycle.phase_m_length',
        'cellcycle.current_time',
        'cellcycle.current_phase',
        'cellcycle.inhibition_density_limit',
        'space.division_requested',
    )

    def __init__(self, update_interval: tuple[float, str] = None) -> None:
        super().__init__(update_interval)

    def update_agent(self, agent: Agent, ctx: UpdateContext) -> None:
        if ctx.divided_since_last_update(agent):
            self.reset_attributes(agent, ctx)
        # An inactive cell does not progress; a completed cycle waits until the space divides the cell
        if not agent.get_attribute('cellcycle.is_active') or agent.get_attribute('space.division_requested'):
            return

        limit = agent.get_attribute('cellcycle.inhibition_density_limit')
        if agent.get_attribute('cellcycle.current_phase') == 'G1':
            if agent.get_attribute('space.local_density') >= limit:
                agent.set_attribute('cellcycle.current_phase', 'G0')

        if agent.get_attribute('cellcycle.current_phase') == 'G0':
            if agent.get_attribute('space.local_density') < limit:
                agent.set_attribute('cellcycle.current_phase', 'G1')

        if not agent.get_attribute('cellcycle.current_phase') == 'G0':
            updated_time = agent.get_attribute('cellcycle.current_time') + ctx.dt
            agent.set_attribute('cellcycle.current_time', updated_time)
            self._update_cellcycle_phase(agent)

    def reset_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        """Starts a new cycle in G1 after a division. The time by which the
        previous cycle overshot its length is carried over, so the update
        interval does not lengthen the cycles.
        """
        length = sum(
            agent.get_attribute(name)
            for name in (
                'cellcycle.phase_g1_length',
                'cellcycle.phase_s_length',
                'cellcycle.phase_g2_length',
                'cellcycle.phase_m_length',
            )
        )
        overshoot = agent.get_attribute('cellcycle.current_time') - length
        agent.set_attribute('cellcycle.current_time', max(0, overshoot))
        agent.set_attribute('cellcycle.current_phase', 'G1')
        self._set_cellcycle_phase_lengths(agent, ctx.rng)

    def _set_cellcycle_phase_lengths(self, agent: Agent, rng: np.random.Generator) -> None:
        attributes = [
            ('cellcycle.phase_g1_mean', 'cellcycle.phase_g1_sd', 'cellcycle.phase_g1_length'),
            ('cellcycle.phase_s_mean', 'cellcycle.phase_s_sd', 'cellcycle.phase_s_length'),
            ('cellcycle.phase_g2_mean', 'cellcycle.phase_g2_sd', 'cellcycle.phase_g2_length'),
            ('cellcycle.phase_m_mean', 'cellcycle.phase_m_sd', 'cellcycle.phase_m_length'),
        ]
        for attr in attributes:
            sd_ms = 0
            if agent.has_attribute(attr[1]):
                sd = agent.get_attribute(attr[1])
                sd_ms = convert_time(sd[0], sd[1], 'ms')
            mean = agent.get_attribute(attr[0])
            mean_ms = convert_time(mean[0], mean[1], 'ms')
            length = self._generate_cellcycle_phase_length(mean_ms, sd_ms, rng)
            agent.set_attribute(attr[2], length)

    def _update_cellcycle_phase(self, agent: Agent) -> None:
        current_time = agent.get_attribute('cellcycle.current_time')
        g1_length = agent.get_attribute('cellcycle.phase_g1_length')
        s_length = agent.get_attribute('cellcycle.phase_s_length')
        g2_length = agent.get_attribute('cellcycle.phase_g2_length')
        m_length = agent.get_attribute('cellcycle.phase_m_length')
        s_checkpoint = g1_length
        g2_checkpoint = g1_length + s_length
        m_checkpoint = g1_length + s_length + g2_length
        div_checkpoint = g1_length + s_length + g2_length + m_length
        if s_checkpoint <= current_time < g2_checkpoint:
            agent.set_attribute('cellcycle.current_phase', 'S')
        if g2_checkpoint <= current_time < m_checkpoint:
            agent.set_attribute('cellcycle.current_phase', 'G2')
        if m_checkpoint <= current_time < div_checkpoint:
            agent.set_attribute('cellcycle.current_phase', 'M')
        if div_checkpoint <= current_time:
            agent.set_attribute('space.division_requested', True)

    def _generate_cellcycle_phase_length(
        self, length_mean: float, length_sd: float, rng: np.random.Generator
    ) -> float:
        if length_sd == 0.0:
            return length_mean
        shape = length_mean**2 / length_sd**2
        scale = length_sd**2 / length_mean
        return rng.gamma(shape=shape, scale=scale)

    def _initialize_required_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        self._require_attributes(
            agent,
            'cellcycle.phase_g1_mean',
            'cellcycle.phase_s_mean',
            'cellcycle.phase_g2_mean',
            'cellcycle.phase_m_mean',
        )
        self._set_cellcycle_phase_lengths(agent, ctx.rng)

    def _initialize_optional_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        if agent.has_attribute('cellcycle.random_initial') and agent.get_attribute('cellcycle.random_initial'):
            g1_length = agent.get_attribute('cellcycle.phase_g1_length')
            s_length = agent.get_attribute('cellcycle.phase_s_length')
            g2_length = agent.get_attribute('cellcycle.phase_g2_length')
            m_length = agent.get_attribute('cellcycle.phase_m_length')
            length = g1_length + s_length + g2_length + m_length
            cc_time = ctx.rng.uniform(low=0, high=length)
            agent.set_attribute('cellcycle.current_time', cc_time)
            self._update_cellcycle_phase(agent)

    def _initialize_attributes_default_values(self, agent: Agent, ctx: UpdateContext) -> None:
        for name in (
            'cellcycle.is_active',
            'cellcycle.current_time',
            'cellcycle.current_phase',
            'space.division_requested',
            'space.last_division_time',
            'cellcycle.inhibition_density_limit',
        ):
            if not agent.has_attribute(name):
                agent.set_attribute(name, default_of(name))
