
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

    def initialize_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('cellcycle_length'):
            agent.initialize_attribute('cellcycle_length', self._length)
        if not agent.has_attribute('cellcycle_current_time'):
            agent.initialize_attribute('cellcycle_current_time', self._current_time)
        if not agent.has_attribute('division_pending'):
            agent.initialize_attribute('division_pending', False)
        if not agent.has_attribute('division_completed'):
            agent.initialize_attribute('division_completed', False)

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








# class FixedIncrementalCellCycle(CellFunctionModel):
#     """A cell function model that implements a simple fixed-length cell cycle.
#     When the internal time counter reaches the specified length, the agent is
#     marked as ``division_pending``. If the simulation environment handles the
#     division and sets the agent's status to ``division_completed``, the
#     internal time counter resets, and the cell cycle begins again.
#     """
#     def __init__(self,
#                  update_interval: int,
#                  length: int = 0,
#                  initial_time: int = 0,
#                  random_initial: bool = False
#                  ):
#         """Constructor method.

#         Parameters
#         ----------
#         agent : Agent
#             The Agent to which the model is attached
#         update_interval : int
#             The time interval that passes between two update steps,
#             in milliseconds
#         length : int
#             The length of the cell cycle, in milliseconds
#         initial_time : int, optional
#             The initial state of the internal time counter, by default 0
#         random_initial : bool, optional
#             If True, ``initial_time`` is assigned a random value uniformly
#             distributed between 0 and ``length``, by default False
#         """
#         super().__init__(update_interval)
#         self._length = length
#         self._current_time = initial_time
#         if random_initial:
#             self._current_time = np.random.uniform(low=0, high=length)

#     @property
#     def current_time(self) -> int:
#         """Get the current internal time of the model, which represents the
#         elapsed time since the last division.

#         Returns
#         -------
#         int
#             Current internal time, in milliseconds
#         """
#         return self._current_time

#     def initialize_attributes(self, agent: Agent) -> None:
#         """Initializes the attributes required for the model. The model uses
#         two attributes: ``division_pending`` and ``division_completed``. For a
#         detailed description, refer to the :class:`FixedIncrementalCellCycle`
#         documentation.
#         """
#         if not agent.has_attribute('cellcycle_length'):
#             agent.initialize_attribute('cellcycle_length', self._length)
#         if not agent.has_attribute('cellcycle_current_time'):
#             agent.initialize_attribute('cellcycle_current_time', 0)
#         if not agent.has_attribute('division_pending'):
#             agent.initialize_attribute('division_pending', False)
#         if not agent.has_attribute('division_completed'):
#             agent.initialize_attribute('division_completed', False)

#     def _reset(self) -> None:
#         """Resets the model to its initial state by setting the current time
#         to zero and resetting the related state flags.
#         """
#         self._current_time = 0
#         self._agent.set_attribute('division_pending', False)
#         self._agent.set_attribute('division_completed', False)

#     def _update(self, agent, dt) -> None:
#         """Performs the actual update of the model's state.
#         """
#         if agent.get_attribute('division_completed'):
#             self._reset()
#         self._current_time += self._time_since_last_update
#         if self._length <= self._current_time:
#             self._agent.set_attribute('division_pending', True)
