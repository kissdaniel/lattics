from .core import Agent, UpdateInfo
import numpy as np


class BaseModel:
    def __init__(self,
                 update_interval: int
                 ) -> None:
        self.update_info = UpdateInfo(update_interval=update_interval)

    def initialize_attributes(self, agent: Agent, **params) -> None:
        pass

    def update_attributes(self, agent: Agent) -> None:
        pass

    def reset_attributes(self, agent: Agent) -> None:
        pass


class FixedIncrementalCellCycleModel(BaseModel):
    def __init__(self,
                 update_interval: int
                 ) -> None:
        super().__init__(update_interval)

    def initialize_attributes(self, agent: Agent, **params) -> None:
        if not agent.has_attribute('cellcycle_is_active'):
            agent.set_attribute('cellcycle_is_active', True)
        if not agent.has_attribute('cellcycle_length'):
            agent.set_attribute('cellcycle_length', None)
        if not agent.has_attribute('cellcycle_current_time'):
            agent.set_attribute('cellcycle_current_time', 0)
        if not agent.has_attribute('division_pending'):
            agent.set_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.set_attribute('division_completed', False)
        if 'cellcycle_length' in params:
            agent.set_attribute('cellcycle_length', params['cellcycle_length'])
        if 'cellcycle_current_time' in params:
            agent.set_attribute('cellcycle_current_time', params['cellcycle_current_time'])
        if 'cellcycle_random_initial' in params:
            if params['cellcycle_random_initial']:
                cc_time = np.random.uniform(low=0, high=params['cellcycle_length'])
                agent.set_attribute('cellcycle_current_time', cc_time)

    def update_attributes(self, agent: Agent) -> None:
        if agent.get_attribute('division_completed'):
            self.reset_attributes(agent)
        if agent.get_attribute('cellcycle_is_active'):
            current_time = agent.get_attribute('cellcycle_current_time')
            time_since_last_update = self.update_info._time_since_last_update
            updated_time = current_time + time_since_last_update
            agent.set_attribute('cellcycle_current_time', updated_time)
            if agent.get_attribute('cellcycle_length') <= updated_time:
                agent.set_attribute('division_pending', True)

    def reset_attributes(self, agent: Agent) -> None:
        agent.set_attribute('cellcycle_is_active', True)
        agent.set_attribute('cellcycle_current_time', 0)
        agent.set_attribute('division_pending', False)
        agent.set_attribute('division_completed', False)
