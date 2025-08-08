
from .agent import Agent
import numpy as np
import copy
import warnings


class CellFunctionModel:
    """Acts as a base class for all cellular function models, such as
    metabolism, cell cycle regulation, and other regulatory processes.
    """
    def __init__(self,
                 update_interval: int
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        update_interval : int
            The time interval that passes between two update steps,
            in milliseconds
        """
        self._update_interval = update_interval
        self._time_since_last_update = 0

    def set_update_interval(self, update_interval: int) -> None:
        """Sets the time interval that passes between two update steps.

        Parameters
        ----------
        update_interval : int
            Time between two update steps, in milliseconds
        """
        self._update_interval = update_interval

    def initialize_attributes(self, agents: list[Agent]) -> None:
        pass

    def increase_time(self, dt: int) -> None:
        self._time_since_last_update += dt

    def reset_time(self) -> None:
        self._time_since_last_update = 0

    def update_needed(self) -> bool:
        return self._update_interval <= self._time_since_last_update

    def update_attributes(self, agent: Agent, dt: int) -> None:
        """The function that performs the actual update of the model's state.
        """
        pass

    def reset_attributes(self, agent: Agent, dt: int) -> None:
        pass


class FixedIncrementalCellCycleModel(CellFunctionModel):
    def __init__(self,
                 update_interval: int,
                 length: int,
                 initial_time: int = 0,
                 random_initial: bool = False
                 ) -> None:
        super().__init__(update_interval)
        self._length = length
        self._current_time = initial_time
        self._random_initial = random_initial

    def initialize_attributes(self, agent: Agent, **params) -> None:
        if not agent.has_attribute('cellcycle_length'):
            agent.initialize_attribute('cellcycle_length', self._length)
        if not agent.has_attribute('cellcycle_current_time'):
            agent.initialize_attribute('cellcycle_current_time', self._current_time)
        if not agent.has_attribute('division_pending'):
            agent.initialize_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.initialize_attribute('division_completed', False)
        if 'cellcycle_length' in params:
            agent.set_attribute('cellcycle_length', params['cellcycle_length'])
        if 'cellcycle_current_time' in params:
            agent.set_attribute('cellcycle_current_time', params['cellcycle_current_time'])
        if 'cellcycle_random_initial' in params:
            if params['cellcycle_random_initial']:
                cc_time = np.random.uniform(low=0, high=self._length)
                agent.set_attribute('cellcycle_current_time', cc_time)

    def update_attributes(self, agent: Agent, dt: int) -> None:
        if agent.get_attribute('division_completed'):
            self.reset_attributes(agent, dt)
        current_time = agent.get_attribute('cellcycle_current_time')
        time_since_last_update = self._time_since_last_update
        updated_time = current_time + time_since_last_update
        agent.set_attribute('cellcycle_current_time', updated_time)
        if agent.get_attribute('cellcycle_length') <= updated_time:
            agent.set_attribute('division_pending', True)

    def reset_attributes(self, agent: Agent, dt: int) -> None:
        agent.set_attribute('cellcycle_current_time', 0)
        agent.set_attribute('division_pending', False)
        agent.set_attribute('division_completed', False)
