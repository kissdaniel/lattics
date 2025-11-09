import copy
from typing import Any


class Agent:

    id_count = 0

    def __init__(self) -> None:
        """Constructor method.
        """
        self._attributes = dict()
        self._initialize_id()

    @property
    def id(self) -> int:
        return self._id

    def clone(self) -> 'Agent':
        """Returns a deep copy instance of the agent.

        Returns
        -------
        Agent
            The cloned agent instance
        """

        cloned = Agent()
        cloned._attributes = copy.deepcopy(self._attributes)
        return cloned

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

        Examples
        --------
        >>> a.initialize_attribute('my_attribute')
        >>> a.set_attribute('my_attribute', True)
        """
        self._attributes[name] = value

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

        Examples
        --------
        >>> a.initialize_attribute('my_attribute', 0)
        >>> print(a.get_attribute(''my_attribute''))
        0
        """
        return self._attributes[name]

    def _initialize_id(self) -> int:
        self._id = Agent.id_count
        Agent.id_count += 1
