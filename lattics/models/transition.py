from typing import Any

import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._units import convert_rate, from_quantity

from ._base import BaseModel


class StochasticTransitionModel(BaseModel):
    """Agents whose attribute matches the condition change to the end states
    with the given rate.

    Parameters
    ----------
    update_interval : tuple[float, str], optional
        The time between two updates of the model
    condition : tuple[str, Any]
        The name of an attribute and the value it must have
    end_states : dict[str, Any]
        The attributes and values set by the transition
    rate : tuple[float, str]
        The rate of the transition, for example ``(1.0, '1/day')``
    """

    def __init__(
        self,
        update_interval: tuple[float, str] = None,
        condition: tuple[str, Any] = None,
        end_states: dict[str, Any] = None,
        rate: tuple[float, str] = (0, '1/day'),
    ) -> None:
        super().__init__(update_interval)
        self._condition = condition
        self._end_states = end_states
        self._rate = from_quantity(rate, convert_rate, '1/ms', 'rate', "(1.0, '1/day')")
        self.reads = (condition[0],) if condition else ()
        self.writes = tuple(end_states) if end_states else ()

    def update_agent(self, agent: Agent, ctx: UpdateContext) -> None:
        attr_name = self._condition[0]
        attr_value = self._condition[1]
        if agent.get_attribute(attr_name) == attr_value:
            dt = ctx.dt
            prob = 1 - np.exp(-self._rate * dt)
            if ctx.rng.random() < prob:
                for s_name, s_value in self._end_states.items():
                    agent.set_attribute(s_name, s_value)

    def reset_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        pass
