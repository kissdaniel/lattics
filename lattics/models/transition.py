import numpy as np
from typing import Any

from ._base import BaseModel
from lattics.core._agent import Agent
from lattics.core._convert import convert_time


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

    def initialize_attributes(self, agent: Agent, **params) -> None:
        pass

    def update_attributes(self, agent: Agent) -> None:
        attr_name = self._condition[0]
        attr_value = self._condition[1]
        if agent.get_attribute(attr_name) == attr_value:
            dt = self.update_info.elapsed_time
            prob = 1 - np.exp(-self._rate * dt)
            if np.random.random() < prob:
                for s_name, s_value in self._end_states.items():
                    agent.set_attribute(s_name, s_value)

    def reset_attributes(self, agent: Agent) -> None:
        pass
