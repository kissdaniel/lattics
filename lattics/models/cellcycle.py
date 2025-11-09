import numpy as np

from ._base import BaseModel
from lattics.core._agent import Agent
from lattics.core._convert import convert_time


class FixedIncrementCellCycleModel(BaseModel):
    """This model implements a simple fixed timestep-based cell cycle by using the cell cycle's length (time
    duration) and an internal clock (counter) to track its current state.





    Parameters
    ----------
    BaseModel : _type_
        _description_
    """
    def __init__(self,
                 update_interval: tuple[float, str] = None,
                 distribution: str = 'erlang',
                 distribution_param: int = 4
                 ) -> None:
        super().__init__(update_interval)
        self._distribution = distribution
        self._distribution_param = distribution_param

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
        attr = agent.get_attribute('cellcycle_mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
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

    def _initialize_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle_mean_length'):
            raise AttributeError()
        attr = agent.get_attribute('cellcycle_mean_length')
        mean_length = convert_time(attr[0], attr[1], 'ms')
        length = self._generate_cellcycle_length(mean_length)
        agent.set_attribute('cellcycle_length', length)

    def _initialize_optional_attributes(self, agent: Agent) -> None:
        if agent.has_attribute('cellcycle_random_initial'):
            length = agent.get_attribute('cellcycle_length')
            cc_time = np.random.uniform(low=0, high=length)
            agent.set_attribute('cellcycle_current_time', cc_time)

    def _initialize_attributes_default_values(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle_is_active'):
            agent.set_attribute('cellcycle_is_active', True)
        if not agent.has_attribute('cellcycle_current_time'):
            agent.set_attribute('cellcycle_current_time', 0)
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', False)
