"""Basic unit of the LattiCS framework representing a single biological cell.
"""

from typing import Any
import copy
import warnings


class Agent:
    def __init__(self,
                 simulation: 'simulation.Simulation' = None
                 ) -> None:
        """Constructor method.
        """
        self._simulation = simulation
        self._status_flags = dict()
        self._cell_function_models = list()

    @property
    def simulation(self) -> 'simulation.Simulation':
        """Get the simulation instance associated with the agent.

        Returns
        -------
        Simulation
            An instance representing the simulation
        """
        return self._simulation

    def clone(self) -> 'Agent':
        """Returns a deep copy instance of the agent.

        Returns
        -------
        Agent
            The cloned agent instance
        """

        cloned = Agent()
        cloned._simulation = self._simulation
        cloned._status_flags = copy.deepcopy(self._status_flags)
        cloned._cell_function_models = copy.deepcopy(self._cell_function_models)
        for m in cloned._cell_function_models:
            m.set_agent(cloned)
        return cloned

    def initialize_status_flag(self, identifier: str, value: Any = None) -> None:
        """Initialize a new status flag for the agent instance with the
        specified identifier.

        Parameters
        ----------
        identifier : str
            The name of the status flag
        value : Any, optional
            The initial value to be set, by default None

        Examples
        --------
        >>> a = Agent()
        >>> a.initialize_status_flag('my_flag_1')
        >>> a.initialize_status_flag('my_flag_2', 0)
        """
        if identifier not in self._status_flags:
            self._status_flags[identifier] = value
        else:
            warnings.warn(f'Status flag \'{identifier}\' already in use.')

    def has_status_flag(self, identifier: str) -> bool:
        """Returns whether the agent instance has a specific status flag initialized.

        Parameters
        ----------
        identifier : str
            The name of the status flag

        Returns
        -------
        bool
            True if the agent has the status flag, otherwise false

        Examples
        --------
        >>> a.initialize_status_flag('my_flag')
        >>> a.has_status_flag('my_flag')
        True
        >>> a.has_status_flag('nonexisting_flag')
        False
        """
        return identifier in self._status_flags

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
        >>> a.initialize_status_flag('my_flag_1')
        >>> a.set_status_flag('my_flag_1', True)
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
        >>> print(a.get_status_flag('my_flag_1'))
        True
        """
        if identifier in self._status_flags:
            return self._status_flags[identifier]
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def add_model(self, model: 'cellfunction.CellFunctionModel') -> None:
        """Adds the provided model instance to the agent's collection of cell
        function models and invokes the model's initialization method.

        Parameters
        ----------
        model : CellFunctionModel
            A subclass of the ``CellFunctionModel`` abstract base class.
        """
        self._cell_function_models.append(model)
        model.initialize_agent_status_flags()

    def update_models(self, dt: int) -> None:
        """Sequentially updates all models associated with the agent. If
        multiple sub-models exist within the same category, they are updated
        in the order they appear in their respective collection.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update, in milliseconds
        """
        for m in self._cell_function_models:
            m.update(dt)
