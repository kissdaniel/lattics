from .core import Agent, UpdateInfo
from .utils import UnitConverter
import numpy as np


class BaseModel:
    def __init__(self,
                 update_interval: tuple[float, str] = None
                 ) -> None:
        self.update_info = UpdateInfo(update_interval=update_interval)

    def initialize_attributes(self, agent: Agent, **params) -> None:
        pass

    def update_attributes(self, agent: Agent) -> None:
        pass

    def reset_attributes(self, agent: Agent) -> None:
        pass


class FixedIncrementCellCycleModel(BaseModel):
    def __init__(self,
                 update_interval: tuple[float, str] = None,
                 distribution: str = 'erlang',
                 distribution_param: int = 4
                 ) -> None:
        super().__init__(update_interval)
        self._distribution = distribution
        self._distribution_param = distribution_param

    def initialize_attributes(self, agent: Agent, **params) -> None:
        if not agent.has_attribute('cellcycle_is_active'):
            agent.set_attribute('cellcycle_is_active', True)
        if not agent.has_attribute('cellcycle_mean_length'):
            agent.set_attribute('cellcycle_mean_length', None)
        if not agent.has_attribute('cellcycle_length'):
            agent.set_attribute('cellcycle_length', None)
        if not agent.has_attribute('cellcycle_current_time'):
            agent.set_attribute('cellcycle_current_time', 0)
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', False)
        if 'cellcycle_mean_length' in params:
            mean_length = UnitConverter.time_to_ms(params['cellcycle_mean_length'])
            agent.set_attribute('cellcycle_mean_length', mean_length)
            length = self._generate_cellcycle_length(mean_length)
            agent.set_attribute('cellcycle_length', length)
        if 'cellcycle_current_time' in params:
            agent.set_attribute('cellcycle_current_time', params['cellcycle_current_time'])
        if 'cellcycle_random_initial' in params:
            if params['cellcycle_random_initial']:
                length = agent.get_attribute('cellcycle_length')
                cc_time = np.random.uniform(low=0, high=length)
                agent.set_attribute('cellcycle_current_time', cc_time)

    def update_attributes(self, agent: Agent) -> None:
        if agent.get_attribute('division_completed'):
            self.reset_attributes(agent)
        if agent.get_attribute('cellcycle_is_active'):
            current_time = agent.get_attribute('cellcycle_current_time')
            time_since_last_update = self.update_info.elapsed_time
            updated_time = current_time + time_since_last_update
            agent.set_attribute('cellcycle_current_time', updated_time)
            if agent.get_attribute('cellcycle_length') <= updated_time:
                agent.set_attribute('division_pending', True)

    def reset_attributes(self, agent: Agent) -> None:
        agent.set_attribute('cellcycle_is_active', True)
        agent.set_attribute('cellcycle_current_time', 0)
        agent.set_attribute('division_pending', False)
        agent.set_attribute('division_completed', False)
        mean_length = agent.get_attribute('cellcycle_mean_length')
        length = self._generate_cellcycle_length(mean_length)
        agent.set_attribute('cellcycle_length', length)

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
