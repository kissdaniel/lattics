"""Basic unit of the LattiCS framework representing a single biological cell.
"""

from typing import Any
import copy
import warnings


class Agent:
    def __init__(self,
                 simulation: 'simulation.Simulation' = None
                 ):
        """Constructor method.
        """
        self._simulation = simulation
        self._status_flags = dict()
        self._metabolic_models = list()
        self._proliferation_models = list()
        self._state_transition_models = list()

    def __deepcopy__(self, memo):
        new_instance = Agent(
            self.simulation
        )
        new_instance._status_flags = copy.deepcopy(
            self._status_flags, memo
        )
        new_instance._metabolic_models = copy.deepcopy(
            self._metabolic_models, memo
        )
        new_instance._proliferation_models = copy.deepcopy(
            self._proliferation_models, memo
        )
        new_instance._state_transition_models = copy.deepcopy(
            self._state_transition_models, memo
        )
        return new_instance

    @property
    def simulation(self) -> 'simulation.Simulation':
        """Get the simulation instance associated with the agent.

        Returns
        -------
        Simulation
            An instance representing the simulation
        """
        return self._simulation

    def initialize_status_flag(self, identifier: str) -> None:
        """Initialize a new status flag for the agent instance with the
        specified identifier.

        Parameters
        ----------
        identifier : str
            The name of the status flag

        Examples
        --------
        >>> a = Agent()
        >>> a.initialize_status_flag('my_flag')
        """
        if identifier not in self._status_flags:
            self._status_flags[identifier] = None
        else:
            warnings.warn(f'Status flag \'{identifier}\' already in use.')

    def set_status_flag(self, identifier: str, value: Any) -> None:
        """Set the value of the specified status flag.

        Parameters
        ----------
        identifier : str
            The name of the status flag
        value : Any
            The new value to be set

        Raises
        ------
        ValueError
            If the specified identifier does not exist

        Examples
        --------
        >>> a.set_status_flag('my_flag', True)
        """
        if identifier in self._status_flags:
            self._status_flags[identifier] = value
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def get_status_flag(self, identifier: str) -> Any:
        """Returns the value of the specified status flag.

        Parameters
        ----------
        identifier : str
            The name of the status flag

        Returns
        -------
        any type
            The current value of the status flag

        Raises
        ------
        ValueError
            If the specified identifier does not exist

        Examples
        --------
        >>> print(a.get_status_flag('my_flag'))
        True
        """
        if identifier in self._status_flags:
            return self._status_flags[identifier]
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def update_models(self, dt: int) -> None:
        """Sequentially updates all models associated with the agent, including
        metabolic, proliferation, and state transition models, in the listed
        order. If multiple sub-models exist within the same category, they are
        updated in the order they appear in their respective collection.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update, in milliseconds
        """
        for m in self._metabolic_models:
            m.update(dt)
        for m in self._proliferation_models:
            m.update(dt)
        for m in self._state_transition_models:
            m.update(dt)
