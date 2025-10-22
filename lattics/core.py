"""The main component of the LattiCS framework containing functionalities to
set up and execute a simulation.
"""

import copy
import math
import warnings
import pickle
import tqdm
import uuid
from typing import Any
from .utils import UnitConverter


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


class Event:
    def __init__(self,
                 time: tuple[float, str],
                 handler: Any,
                 **params: dict
                 ) -> None:
        self._time = UnitConverter.time_to_ms(time)
        self._handler = handler
        self._params = params

    def is_ready(self, current_time: int) -> bool:
        return self._time <= current_time

    def execute(self, simulation):
        self._handler(simulation, **self._params)


class Simulation:
    """Represents a simulation instance. This object manages the participating
    agents (:class:`Agent`), the environment (:class:`SimulationDomain`), and
    the various chemical substances (:class:`Substrate`) present within it.
    The class provides high-level access to configure and execute a simulation.
    """
    def __init__(self, id=None) -> None:
        """Constructor method.

        Parameters
        ----------
        id : str
            The identifier for the simulation instance. If not provided, a
            random identifier will be generated.
        """
        self._id = self._get_id(id)
        self._agents = list()
        self._space = None
        self._events = list()
        self._models = list()
        self._history = list()
        self._time = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_events']
        del state['_models']
        del state['_history']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._events = list()
        self._models = list()
        self._history = list()

    @property
    def agents(self) -> list[Agent]:
        """Get the collection of agents currently present in the simulation.
        The order of agents in this collection is maintained throughout the
        simulation, with new agent instances always being added at the end.

        Returns
        -------
        list[Agent]
            Collection of the agents
        """
        return self._agents

    @property
    def time(self) -> int:
        """Get the current internal time of the simulation, representing the
        elapsed time since the simulation started.

        Returns
        -------
        int
            Internal time of the simulation, in milliseconds.
        """

        return self._time

    def add_agent(self, agent: Agent, **params) -> None:
        """Adds the specified agent to the simulation. The agent will be added
        to the collection of all agents and, if a simulation domain is defined,
        will also be placed within the simulation domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        self._agents.append(agent)
        if self._space:
            self._space.add_agent(agent, **params)
        else:
            warnings.warn('No simulation domain has been defined. '
                          'You can proceed without one, but this may '
                          'lead to unexpected consequences.')
        for m in self._models:
            m.initialize_attributes(agent, **params)
        for par_name, par_value in params.items():
            if not agent.has_attribute(par_name):
                agent.set_attribute(par_name, par_value)

    def add_space(self, space: 'spaces.BaseSpace') -> None:
        """Sets the simulation space to the instance passed as a parameter.

        Parameters
        ----------
        space : BaseSpace
            The simulation space instance to be used
        """
        if self._space:
            raise AttributeError('Simulation space is already set and cannot be modified.')
        self._space = space

    def add_event(self, event) -> None:
        self._events.append(event)

    def add_model(self, model: 'cellfunction.CellFunctionModel') -> None:
        """Adds the provided model instance to the agent's collection of cell
        function models and invokes the model's initialization method.

        Parameters
        ----------
        model : CellFunctionModel
            A subclass of the ``CellFunctionModel`` abstract base class.
        """
        self._models.append(model)

    def add_substrate(self, name: str, substrate) -> None:
        if not self._space:
            raise AttributeError('A simulation domain has to be set to add substrates.')
        # self._domain.add_substrate_field(substrate)

    def remove_agent(self, agent: Agent) -> None:
        self._agents.remove(agent)
        if self._space:
            self._space.remove_agent(agent)

    def run(self, time: tuple[float, str] , dt: tuple[float, str], dt_history: tuple[float, str] = None, save_mode: str = 'always') -> None:
        """Runs the simulation from the current state for the specified
        duration using the given time step.

        Parameters
        ----------
        time : int
            The duration to be simulated, in milliseconds
        dt : _type_
            Time step, in milliseconds
        """
        time_ms = UnitConverter.time_to_ms(time)
        dt_ms = UnitConverter.time_to_ms(dt)
        if dt_history:
            history_ui = UpdateInfo(update_interval=dt_history)
            self._make_history_entry(save_mode)

        steps = int(math.ceil(time_ms / dt_ms)) + 1

        progressbar_format = "{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]"
        progressbar = tqdm.tqdm(total=steps, mininterval=1.0, colour='#d2de32', bar_format=progressbar_format)
        progressbar.set_description(f'ID={self._id}')
        PROGRESSBAR_SCALER = 100

        for i in range(steps):
            self._update_events(self._time)
            self._update_models(dt_ms)
            if self._space:
                self._space.update(dt_ms)
            if dt_history:
                if history_ui.update_needed():
                    self._make_history_entry(save_mode)
                    history_ui.reset_time()
                history_ui.increase_time(dt_ms)
            if i % PROGRESSBAR_SCALER == 0:
                days = UnitConverter.ms_to_days(self.time)
                progressbar.set_postfix(T=f'{days:.2f}', N=f'{len(self.agents)}')
                progressbar.update(PROGRESSBAR_SCALER)
            self._time += dt_ms

        progressbar.n = steps

        if save_mode == 'on_completion':
            self._save_history()

    def _make_history_entry(self, save_mode):
        state = pickle.dumps(self)
        self._history.append(state)
        if save_mode == 'always':
            self._save_history()

    def _save_history(self):
        filename = f'{self._id}.lsd'
        with open(filename, 'wb') as f:
            pickle.dump(self._history, f)

    def _get_id(self, identifier):
        return identifier if identifier else str(uuid.uuid4())

    def _update_events(self, time: int) -> None:
        for e in list(self._events):
            if e.is_ready(time):
                e.execute(self)
                self._events.remove(e)

    def _update_models(self, dt: int) -> None:
        """Sequentially updates all models associated with the agent. If
        multiple sub-models exist within the same category, they are updated
        in the order they appear in their respective collection.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update, in milliseconds
        """
        for m in self._models:
            if m.update_info.update_needed():
                for a in self._agents:
                    m.update_attributes(a)
                m.update_info.reset_time()
            m.update_info.increase_time(dt)


class UpdateInfo:
    def __init__(self,
                 update_interval: tuple[float, str]
                 ) -> None:
        if update_interval:
            self._update_interval = UnitConverter.time_to_ms(update_interval)
        else:
            self._update_interval = 0
        self._elapsed_time = 0

    @property
    def update_interval(self) -> int:
        return self._update_interval

    @property
    def elapsed_time(self) -> int:
        return self._elapsed_time

    def update_needed(self) -> bool:
        return self._update_interval <= self._elapsed_time

    def increase_time(self, msecs) -> None:
        self._elapsed_time += msecs

    def reset_time(self) -> None:
        self._elapsed_time = 0
