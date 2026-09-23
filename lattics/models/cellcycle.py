import numpy as np

from lattics.core._agent import Agent
from lattics.core._convert import convert_time

from ._base import BaseModel


class TimeBasedCellCycleModel(BaseModel):
    """A timestep-based cell cycle model. Uses the length (time duration) of the cell cycle
    and an internal clock (counter) to track the current state of the cell cycle.

    =============================  ====  ========
    *space.duplication_needed*     bool  The model sets this to `True` once the cell cycle is complete, signaling
                                         that the agent is ready for replication.
    *space.duplication_completed*  bool  The simulation environment sets this to `True` once the agent has been
                                         successfully copied, indicating that the replication process is finished.
    cellcycle.is_active            bool  Signals if the agent is proliferating (i.e. proceeds through the cell cycle)
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

    def __init__(
        self, update_interval: tuple[float, str] = None, distribution: str = 'erlang', distribution_param: int = 4
    ) -> None:
        super().__init__(update_interval)
        self._distribution = distribution
        self._distribution_param = distribution_param

    def update_attributes(self, agent: Agent) -> None:
        if agent.get_attribute('space.duplication_completed'):
            self.reset_attributes(agent)
        if agent.get_attribute('cellcycle.is_active'):
            current_time = agent.get_attribute('cellcycle.current_time')
            time_since_last_update = self.update_info.elapsed_time
            updated_time = current_time + time_since_last_update
            agent.set_attribute('cellcycle.current_time', updated_time)
            if agent.get_attribute('cellcycle.length') <= updated_time:
                agent.set_attribute('space.duplication_needed', True)

    def reset_attributes(self, agent: Agent) -> None:
        agent.set_attribute('cellcycle.is_active', True)
        agent.set_attribute('cellcycle.current_time', 0)
        agent.set_attribute('space.duplication_needed', False)
        agent.set_attribute('space.duplication_completed', False)
        attr = agent.get_attribute('cellcycle.mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
        length = self._generate_cellcycle_length(mean_length)
        agent.set_attribute('cellcycle.length', length)

    def _generate_cellcycle_length(self, mean_length) -> int:
        if self._distribution == 'fixed':
            return mean_length
        if self._distribution == 'erlang':
            shape = self._distribution_param
            scale = mean_length / shape
            return np.random.gamma(shape=shape, scale=scale)
        if self._distribution == 'normal':
            loc = mean_length
            scale = self._distribution_param
            return np.random.normal(loc=loc, scale=scale)

    def _initialize_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle.mean_length'):
            raise AttributeError()
        attr = agent.get_attribute('cellcycle.mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
        length = self._generate_cellcycle_length(mean_length)
        agent.set_attribute('cellcycle.length', length)

    def _initialize_optional_attributes(self, agent: Agent) -> None:
        if agent.has_attribute('cellcycle.random_initial'):
            length = agent.get_attribute('cellcycle.length')
            cc_time = np.random.uniform(low=0, high=length)
            agent.set_attribute('cellcycle.current_time', cc_time)

    def _initialize_attributes_default_values(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle.is_active'):
            agent.set_attribute('cellcycle.is_active', True)
        if not agent.has_attribute('cellcycle.current_time'):
            agent.set_attribute('cellcycle.current_time', 0)
        if not agent.has_attribute('space.duplication_needed'):
            agent.set_attribute('space.duplication_needed', False)
        if not agent.has_attribute('space.duplication_completed'):
            agent.set_attribute('space.duplication_completed', False)


class MultiPhaseTimeBasedCellCycleModel(BaseModel):
    """This model implements a simple fixed timestep-based cell cycle by using the cell cycle's length (time
    duration) and an internal clock (counter) to track its current state.

    Parameters
    ----------
    BaseModel : _type_
        _description_
    """

    def __init__(self, update_interval: tuple[float, str] = None) -> None:
        super().__init__(update_interval)

    def update_attributes(self, agent: Agent) -> None:
        if agent.get_attribute('duplication_completed'):
            self.reset_attributes(agent)

        limit = agent.get_attribute('cellcycle.inhibition_density_limit')
        if agent.get_attribute('cellcycle.current_phase') == 'G1':
            if agent.get_attribute('mechanics.local_density') >= limit:
                agent.set_attribute('cellcycle.current_phase', 'G0')

        if agent.get_attribute('cellcycle.current_phase') == 'G0':
            if agent.get_attribute('mechanics.local_density') < limit:
                agent.set_attribute('cellcycle.current_phase', 'G1')

        if not agent.get_attribute('cellcycle.current_phase') == 'G0':
            current_time = agent.get_attribute('cellcycle.current_time')
            time_since_last_update = self.update_info.elapsed_time
            updated_time = current_time + time_since_last_update
            agent.set_attribute('cellcycle.current_time', updated_time)
            self._update_cellcycle_phase(agent)

    def reset_attributes(self, agent: Agent) -> None:
        # agent.set_attribute('cellcycle.is_proliferating', True)
        agent.set_attribute('cellcycle.current_time', 0)
        agent.set_attribute('cellcycle.current_phase', 'G1')
        agent.set_attribute('duplication_needed', False)
        agent.set_attribute('duplication_completed', False)
        self._set_cellcycle_phase_lengths(agent)

    def _set_cellcycle_phase_lengths(self, agent: Agent) -> None:
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
            length = self._generate_cellcycle_phase_length(mean_ms, sd_ms)
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
            agent.set_attribute('duplication_needed', True)

    def _generate_cellcycle_phase_length(self, length_mean, length_sd) -> int:
        if length_sd == 0.0:
            return length_mean
        shape = length_mean**2 / length_sd**2
        scale = length_sd**2 / length_mean
        return np.random.gamma(shape=shape, scale=scale)

    def _initialize_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle.phase_g1_mean'):
            raise AttributeError()
        if not agent.has_attribute('cellcycle.phase_s_mean'):
            raise AttributeError()
        if not agent.has_attribute('cellcycle.phase_g2_mean'):
            raise AttributeError()
        if not agent.has_attribute('cellcycle.phase_m_mean'):
            raise AttributeError()
        self._set_cellcycle_phase_lengths(agent)

    def _initialize_optional_attributes(self, agent: Agent) -> None:
        if agent.has_attribute('cellcycle.random_initial') and agent.get_attribute('cellcycle.random_initial'):
            g1_length = agent.get_attribute('cellcycle.phase_g1_length')
            s_length = agent.get_attribute('cellcycle.phase_s_length')
            g2_length = agent.get_attribute('cellcycle.phase_g2_length')
            m_length = agent.get_attribute('cellcycle.phase_m_length')
            length = g1_length + s_length + g2_length + m_length
            cc_time = np.random.uniform(low=0, high=length)
            agent.set_attribute('cellcycle.current_time', cc_time)
            self._update_cellcycle_phase(agent)

    def _initialize_attributes_default_values(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle.current_time'):
            agent.set_attribute('cellcycle.current_time', 0)
        if not agent.has_attribute('cellcycle.current_phase'):
            agent.set_attribute('cellcycle.current_phase', 'G1')
        if not agent.has_attribute('duplication_needed'):
            agent.set_attribute('duplication_needed', False)
        if not agent.has_attribute('duplication_completed'):
            agent.set_attribute('duplication_completed', False)
        if not agent.has_attribute('cellcycle.inhibition_density_limit'):
            agent.set_attribute('cellcycle.inhibition_density_limit', 1.0)
