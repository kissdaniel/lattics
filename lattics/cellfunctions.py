
from .agent import Agent
import numpy as np
import warnings


class CellFunctionModel:
    def __init__(self,
                 update_interval: int,
                 agent: Agent
                 ) -> None:
        """Constructor method.
        """
        self._update_interval = update_interval
        self._time_since_last_update = 0
        self._agent = agent

    def initialize_agent_state_flags(self) -> None:
        pass

    def update(self, dt: int) -> None:
        self._time_since_last_update += dt
        if self._update_interval <= self._time_since_last_update:
            self._update()
            self._time_since_last_update -= self._update_interval

    def _reset(self) -> None:
        pass

    def _update(self) -> None:
        pass


class FixedIncrementalCellCycle(CellFunctionModel):
    def __init__(self,
                 update_interval: int,
                 agent: Agent,
                 length: int,
                 initial_time: int = 0,
                 random_initial: bool = False
                 ):
        super().__init__(update_interval, agent)
        self._length = length
        self._current_time = initial_time
        if random_initial:
            self._current_time = np.random.uniform(low=0, high=length)

    @property
    def current_time(self) -> int:
        return self._current_time

    def initialize_agent_state_flags(self) -> None:
        self._agent.initialize_status_flag('division_pending', False)
        self._agent.initialize_status_flag('division_completed', False)

    def _reset(self) -> None:
        self._current_time = 0
        self._agent.set_status_flag('division_pending', False)
        self._agent.set_status_flag('division_completed', False)

    def _update(self) -> None:
        if self._agent.get_status_flag('division_completed'):
            self._reset()
        self._current_time += self._time_since_last_update
        if self._length <= self._current_time:
            self._agent.set_status_flag('division_pending', True)
