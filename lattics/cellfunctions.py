
from .agent import Agent
import numpy as np
import copy
import warnings


class CellFunctionModel:
    """Acts as a base class for all cellular function models, such as
    metabolism, cell cycle regulation, and other regulatory processes.
    """
    def __init__(self
                 ) -> None:
        """Constructor method.
        """
        self._agent = None
        self._update_interval = None
        self._time_since_last_update = 0

    def set_agent(self, agent: Agent) -> None:
        """Sets the model's reference to the Agent to which it is attached.

        Parameters
        ----------
        agent : Agent
            The containing agent
        """
        self._agent = agent

    def set_update_interval(self, update_interval: int) -> None:
        """Sets the time interval that passes between two update steps.

        Parameters
        ----------
        update_interval : int
            Time between two update steps, in milliseconds
        """
        self._update_interval = update_interval

    def initialize_agent_state_flags(self) -> None:
        """Initializes the state flags required for the model.
        """
        pass

    def update(self, dt: int) -> None:
        """Determines whether an update step is needed based on the time
        elapsed since the last update. If the elapsed time meets or exceeds
        the ``update_interval`` parameter, the update is performed.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        self._status_check()
        self._time_since_last_update += dt
        if self._update_interval <= self._time_since_last_update:
            self._update()
            self._time_since_last_update -= self._update_interval

    def _status_check(self) -> None:
        """Checks if the model is initialized with the required parameters.

        Raises
        ------
        ValueError
            If there is no update interval set for the model
        ValueError
            If there is no Agent associated with the model
        """
        if not self._update_interval:
            raise ValueError('The update interval of the model is undefined. '
                             'Call \'set_update_interval\' before updating the model.')
        if not self._agent:
            raise ValueError('The containing agent of the model is undefined.'
                             'Call \'set_agent\' before updating the model.')

    def _reset(self) -> None:
        """Resets the model to its initial state.
        """
        pass

    def _update(self) -> None:
        """The function that performs the actual update of the model's state.
        """
        pass

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            setattr(new_instance, key, copy.deepcopy(value, memo))
        return new_instance


class FixedIncrementalCellCycle(CellFunctionModel):
    """A cell function model that implements a simple fixed-length cell cycle.
    When the internal time counter reaches the specified length, the agent is
    marked as ``division_pending``. If the simulation environment handles the
    division and sets the agent's status to ``division_completed``, the
    internal time counter resets, and the cell cycle begins again.
    """
    def __init__(self,
                 length: int,
                 initial_time: int = 0,
                 random_initial: bool = False
                 ):
        """Constructor method.

        Parameters
        ----------
        length : int
            The length of the cell cycle, in milliseconds
        initial_time : int, optional
            The initial state of the internal time counter, by default 0
        random_initial : bool, optional
            If True, ``initial_time`` is assigned a random value uniformly
            distributed between 0 and ``length``, by default False
        """
        super().__init__()
        self._length = length
        self._current_time = initial_time
        if random_initial:
            self._current_time = np.random.uniform(low=0, high=length)

    @property
    def current_time(self) -> int:
        """Get the current internal time of the model, which represents the
        elapsed time since the last division.

        Returns
        -------
        int
            Current internal time, in milliseconds
        """
        return self._current_time

    def initialize_agent_state_flags(self) -> None:
        """Initializes the state flags required for the model. The model uses
        two flags: ``division_pending`` and ``division_completed``. For a
        detailed description, refer to the :class:`FixedIncrementalCellCycle` documentation.
        """
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
