"""Basic unit of the LattiCS framework representing a single biological cell.
"""

import copy
import numpy as np
import warnings


class Agent:
    def __init__(self,
                 simulation,
                 position=None,
                 motility=0.0,
                 binding_affinity=0.0,
                 displacement_limit=1,
                 cellcycle_model=None
                 ):
        self._simulation = simulation
        self._position = position
        self._motility = motility
        self._binding_affinity = binding_affinity
        self._displacement_limit = displacement_limit
        self._biochemical_models = list()
        self._cellcycle_model = cellcycle_model
        self._phenotype_transition_models = list()
        self._status_flags = dict()

    def __deepcopy__(self, memo):
        new_instance = Agent(
            self.simulation,
            copy.deepcopy(self.position, memo),
            copy.deepcopy(self.motility, memo),
            copy.deepcopy(self.binding_affinity, memo),
            copy.deepcopy(self.displacement_limit, memo),
            copy.deepcopy(self.cellcycle_model, memo)
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
        new_instance._status_flags = copy.deepcopy(
            self._status_flags, memo
        )
        return new_instance

    @property
    def binding_affinity(self):
        """The strength of adhesion to other agents (dimensionless)."""
        return self._binding_affinity

    @binding_affinity.setter
    def binding_affinity(self, value):
        self._binding_affinity = value

    @property
    def displacement_limit(self):
        """The maximal number of other agents to be relocated (pushed) upon division."""
        return self._displacement_limit

    @displacement_limit.setter
    def displacement_limit(self, value):
        self._displacement_limit = value

    @property
    def position(self):
        """The coordinates of the agent."""
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.array(value)

    @property
    def simulation(self):
        """The simulation instance associated with the agent."""
        return self._simulation

    @property
    def motility(self):
        """The characteristic velocity of the agent in um/h."""
        return self._motility

    @motility.setter
    def motility(self, value):
        if value < 0:
            raise ValueError('Motility of an agent must be non-negative.')
        self._motility = value

    @property
    def cellcycle_model(self):
        return self._cellcycle_model

    @cellcycle_model.setter
    def cellcycle_model(self, value):
        if self._cellcycle_model:
            raise ValueError('A cell cycle model is already attached to the agent. Only one cell cycle model can be used at a time.')
        model_instance = value(self)
        model_instance.initialize()
        self._cellcycle_model = model_instance

    @property
    def phenotype_transition_models(self):
        return self._phenotype_transition_models

    @phenotype_transition_models.setter
    def phenotype_transition_models(self, values):
        if isinstance(values, tuple) or isinstance(values, list):
            for mi in values:
                model_instance = mi(self)
                self._phenotype_transition_models.append(model_instance)
        else:
            model_instance = values(self)
            self._phenotype_transition_models.append(model_instance)

    def attach_biochemical_model(self, model):
        """Helper function attachimg model instance to the agent.

        Args:
            model (BiochemicalModel): the model instance to attach
        """
        if model not in self._biochemical_models:
            self._biochemical_models.append(model)
        else:
            raise ValueError('Model instance already attached.')

    def attach_cellcycle_model(self, model):
        """Helper function attachimg model instance to the agent.

        Args:
            model (CellCycleModel): the model instance to attach
        """
        if self._cellcycle_model is None:
            self._cellcycle_model = model
            model.attach_agent(self)
        else:
            raise ValueError('A cell cycle model is already attached to the agent. Only one cell cycle model can be used at a time.')

    def attach_phenotype_transition_model(self, model):
        """Helper function attachimg model instance to the agent.

        Args:
            model (PhenotypeTransitionModel): the model instance to attach
        """
        if model not in self._phenotype_transition_models:
            self._phenotype_transition_models.append(model)
        else:
            raise ValueError('Model instance already attached.')

    def initialize_status_flag(self, identifier):
        if identifier not in self._status_flags:
            self._status_flags[identifier] = None
        else:
            warnings.warn(f'Status flag \'{identifier}\' already in use.')

    def get_status_flag(self, identifier):
        if identifier in self._status_flags:
            return self._status_flags[identifier]
        else:
            raise ValueError(f'Status flag \'{identifier}\' not available.')

    def set_status_flag(self, identifier, value):
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
