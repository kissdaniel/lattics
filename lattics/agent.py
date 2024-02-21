"""Basic unit of the LattiCS framework representing a single biological cell.
"""

import copy

class Agent:
    def __init__(self):        
        self._type_id = None
        self._position = None
        self._velocity = None
        self._biochemical_models = list()
        self._cellcycle_model = None
        self._status_flags = dict()

    @property
    def type_id(self):
        """The type identifier of the agent."""
        return self._type_id
    
    @type_id.setter
    def type_id(self, value):
        self._type_id = value
    
    @property
    def position(self):
        """The coordinates of the agent."""
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
    
    @property
    def velocity(self):
        """The characteristic velocity (motility) of the agent in um/h."""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        self._velocity = value    
    
    def attach_biochemical_model(self, model):
        """Helper function to attach the specified biochemical model instance to the agent.

        Args:
            model (BiochemicalModel): the model instance to attach
        """
        if model not in self._biochemical_models:
            self._biochemical_models.append(model)
        else:
            raise ValueError('Model instance already attached.')

    def attach_cellcycle_model(self, model):
        """Helper function to attach the specified cell cycle model instance to the agent.

        Args:
            model (CellCycleModel): the model instance to attach
        """
        if self._cellcycle_model is None:
            self._cellcycle_model = model
        else:
            raise ValueError('A cell cycle model is already attached to the agent. Only one cell cycle model can be used at a time.')
        
    def initialize_status_flag(self, identifier, value):
        """Helper function to create and set a new status flag associated with the agent.

        Args:
            identifier (str): the identifier of the status flag
            value (bool): the value of the status flag
        """
        if identifier not in self._status_flags:
            self._status_flags[identifier] = value
        else:
            raise ValueError(f'Status flag name {identifier} is already in use.')
        
    def get_status_flag(self, identifier):
        """Helper function to get the value of a status flag associated with the agent.

        Args:
            identifier (str): the identifier of the status flag        

        Returns:
            bool: the value of the status flag
        """        
        if identifier in self._status_flags:
            return self._status_flags[identifier]
        else:
            raise ValueError(f'No status flag with name {identifier} is available.')

    def set_status_flag(self, identifier, value):
        """Helper function to set the value of a status flag associated with the agent.

        Args:
            identifier (str): the name of the status flag
            value (bool): the value of the status flag
        """
        if identifier in self._status_flags:
            self._status_flags[identifier] = value
        else:
            raise ValueError(f'No status flag with name {identifier} is available.')

    def update_models(self, dt):
        """Updates all sub-models attached to an agent.

        Args:
            dt (int): time elapsed since the last update step (milliseconds)
        """
        for m in self.biochemical_models:
            m.update(dt)
        self.cellcycle_model.update(dt)

    def clone_from(source_agent):
        """Returns a deep copy of the source agent.

        Args:
            source_agent (Agent): the agent instance to be cloned

        Returns:
            Agent: a deep copy instance of the source agent
        """
        cloned = copy.deepcopy(source_agent)
        return cloned