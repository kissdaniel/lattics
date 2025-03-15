"""Basic unit of the LattiCS framework representing a single biological cell.
"""

import copy
import warnings


class Agent:
    def __init__(self,
                 simulation=None
                 ):
        self._simulation = simulation
        self._status_flags = dict()
        self._biochemical_models = list()
        self._cellcycle_model = None
        self._phenotype_transition_models = list()

    def __deepcopy__(self, memo):
        new_instance = Agent(
            self.simulation
        )
        new_instance._status_flags = copy.deepcopy(
            self._status_flags, memo
        )
        new_instance._biochemical_models = copy.deepcopy(
            self._biochemical_models, memo
        )
        new_instance._cellcycle_model = copy.deepcopy(
            self._cellcycle_model, memo
        )
        new_instance._phenotype_transition_models = copy.deepcopy(
            self._phenotype_transition_models, memo
        )
        return new_instance

    @property
    def simulation(self):
        """Get the simulation instance associated with the agent.

        :return: An instance representing the simulation
        :rtype: Simulation
        """
        return self._simulation

    def initialize_status_flag(self, identifier):
        """Initialize a new status flag for the agent instance with the specified identifier.

        :param identifier: The name of the status flag
        :type identifier: str
        """
        if identifier not in self._status_flags:
            self._status_flags[identifier] = None
        else:
            warnings.warn(f'Status flag \'{identifier}\' already in use.')

    def get_status_flag(self, identifier):
        """Returns the value of the specified status flag.

        :param identifier: The name of the status flag
        :type identifier: str
        :raises ValueError: If the specified :param identifier: does not exist
        :return: The current value of the status flag
        :rtype: any type
        """
        if identifier in self._status_flags:
            return self._status_flags[identifier]
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def set_status_flag(self, identifier, value):
        """Set the value of the specified status flag.

        :param identifier: The name of the status flag
        :type identifier: str
        :param value: The new value to set
        :type value: any type
        :raises ValueError: If the specified :param identifier: does not exist
        """
        if identifier in self._status_flags:
            self._status_flags[identifier] = value
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def update_models(self, dt):
        """Updates all sub-models attached to an agent.

        Args:
            dt (int): time elapsed since the last update step (milliseconds)
        """
        for m in self._biochemical_models:
            m.update(dt)
        for m in self._phenotype_transition_models:
            m.update(dt)
        if self._cellcycle_model:
            self._cellcycle_model.update(dt)

    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                if hasattr(self.cellcycle_model, key):
                    setattr(self.cellcycle_model, key, value)
                else:
                    raise AttributeError(f"Agent or any of its sub-models have no attribute '{key}'.")

    def clone(self):
        """Returns a deep copy of the source agent.

        Returns:
            Agent: a deep copy instance of the source agent
        """
        cloned = copy.deepcopy(self)
        cloned.cellcycle_model.set_owner(cloned)
        # TODO
        # for m in cloned.biochemical_models:
        #     pass
        for m in cloned.phenotype_transition_models:
            m.set_owner(cloned)
        return cloned
