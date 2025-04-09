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
        self._attributes = dict()
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
        cloned._attributes = copy.deepcopy(self._attributes)
        cloned._cell_function_models = copy.deepcopy(self._cell_function_models)
        for m in cloned._cell_function_models:
            m.set_agent(cloned)
        return cloned

    def initialize_attribute(self, name: str, value: Any = None) -> None:
        """Initialize a new attribute for the agent instance with the
        specified name.

        Parameters
        ----------
        name : str
            The identifier of the status flag
        value : Any, optional
            The initial value to be set, by default None

        Examples
        --------
        >>> a = Agent()
        >>> a.initialize_attribute('my_attribute_1')
        >>> a.initialize_attribute('my_attribute_2', 0)
        >>> a.initialize_attribute('my_attribute_3', 'foo')
        """
        if name not in self._attributes:
            self._attributes[name] = value
        else:
            warnings.warn(f'Attribute name \'{name}\' already in use.')

    def has_attribute(self, name: str) -> bool:
        """Returns whether the agent instance has a specific attribute initialized.

        Parameters
        ----------
        name : str
            The identifier of the attribute

        Returns
        -------
        bool
            True if the agent has the attribute, otherwise false

        Examples
        --------
        >>> a.initialize_attribute('my_attribute')
        >>> a.has_attribute('my_attribute')
        True
        >>> a.has_attribute('nonexisting_attribute')
        False
        """
        return name in self._attributes

    def set_attribute(self, name: str, value: Any) -> None:
        """Set the value of the specified attribute.

        Parameters
        ----------
        name : str
            The identifier of the attribute
        value : Any
            The new value to be set

        Raises
        ------
        ValueError
            If the specified attribute does not exist

        Examples
        --------
        >>> a.initialize_attribute('my_attribute')
        >>> a.set_attribute('my_attribute', True)
        """
        if name in self._attributes:
            self._attributes[name] = value
        else:
            raise ValueError(f'Attribute \'{name}\' not available.')

    def get_attribute(self, name: str) -> Any:
        """Returns the value of the specified attribute.

        Parameters
        ----------
        name : str
            The identifier of the attribute

        Returns
        -------
        any type
            The current value of the attribute

        Raises
        ------
        ValueError
            If the specified attribute does not exist

        Examples
        --------
        >>> a.initialize_attribute('my_attribute', 0)
        >>> print(a.get_attribute(''my_attribute''))
        0
        """
        if name in self._attributes:
            return self._attributes[name]
        else:
            raise ValueError(f'Attribute name \'{name}\' not available.')

    def add_model(self, model: 'cellfunction.CellFunctionModel') -> None:
        """Adds the provided model instance to the agent's collection of cell
        function models and invokes the model's initialization method.

        Parameters
        ----------
        model : CellFunctionModel
            A subclass of the ``CellFunctionModel`` abstract base class.
        """
        self._cell_function_models.append(model)
        model.initialize_agent_attributes()

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
