
from .agent import Agent
import warnings


class CellFunctionModel:
    def __init__(self, update_interval: int) -> None:
        """Constructor method.
        """
        self._update_interval = update_interval
        self._time_since_last_update = 0

    def initialize_agent_state_flags(self):
        pass

    def update(self, dt):
        self._time_since_last_update += dt
        if self._update_interval <= self._time_since_last_update:
            self._update()
            self._time_since_last_update -= self._update_interval

    def _update(self):
        pass
