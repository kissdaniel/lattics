from typing import Any
from .core import Agent, UpdateInfo
from .utils import convert_time
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
            mean_length = convert_time(params['cellcycle_mean_length'][0], params['cellcycle_mean_length'][1], 'ms')
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


class StochasticTransitionModel(BaseModel):
    def __init__(self,
                 update_interval: tuple[float, str] = None,
                 condition: tuple[str, Any] = None,
                 end_states: dict[str, Any] = None,
                 rate: float = 0
                 ) -> None:
        super().__init__(update_interval)
        self._condition = condition
        self._end_states = end_states
        self._rate = rate / convert_time(1, 'day', 'ms')

    def update_attributes(self, agent: Agent) -> None:
        attr_name = self._condition[0]
        attr_value = self._condition[1]
        if agent.get_attribute(attr_name) == attr_value:
            dt = self.update_info.elapsed_time
            prob = 1 - np.exp(-self._rate * dt)
            if np.random.random() < prob:
                for s_name, s_value in self._end_states.items():
                    agent.set_attribute(s_name, s_value)


class ConcentrationDependentToxicityModel(BaseModel):
    def __init__(self,
                 update_interval: tuple[float, str],
                 substrate_name: str,
                 max_rate: float,
                 median_effective_concentration: float
                 ) -> None:
        super().__init__(update_interval)
        self._substrate_name = substrate_name
        self._max_rate = max_rate / convert_time(1, 'day', 'ms')
        self._ed50 = median_effective_concentration

    def update_attributes(self, agent) -> None:
        concentration = agent.get_attribute('substrate_info')[self._substrate_name].concentration
        rate = self.mm_fun(x=concentration, vmax=self._max_rate, ed50=self._ed50)
        dt = self.update_info.elapsed_time
        prob = 1 - np.exp(-rate * dt)
        if np.random.random() < prob:
            agent.set_attribute('state', 'necrotic')
            agent.set_attribute('cellcycle_is_active', False)
            agent.set_attribute('division_pending', False)
            agent.set_attribute('motility', 0)
            agent.set_attribute('binding_affinity', 0)

    def mm_fun(self, x, vmax, ed50):
        return vmax * (x / (ed50 + x))
