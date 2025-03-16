
from .agent import Agent
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

    def reset(self) -> None:
        pass

    def _update(self) -> None:
        pass


class FixedIncrementalCellCycle(CellFunctionModel):
    def __init__(self,
                 update_interval: int,
                 agent: Agent,
                 length: int
                 ):
        super().__init__(update_interval, agent)
        self._length = length
        self._current_time = 0

    def initialize_agent_state_flags(self) -> None:
        self._agent.initialize_status_flag('cell_cycle_complete')
        self._agent.set_status_flag('cell_cycle_complete', False)

    def reset(self):
        self._current_time = 0

    def _update(self):
        self._current_time += self._time_since_last_update
        if self._length <= self._current_time:
            self._agent.set_status_flag('cell_cycle_complete', True)
