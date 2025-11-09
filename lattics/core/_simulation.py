import copy
import math
import warnings
import pickle
import tqdm
import uuid
from typing import Any
from ._agent import Agent
from ._convert import convert_time


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

    def run(self, time: tuple[float, str] , dt: tuple[float, str], dt_history: tuple[float, str] = None, save_mode: str = 'always', verbosity: int = 1) -> None:
        """Runs the simulation from the current state for the specified
        duration using the given time step.

        Parameters
        ----------
        time : int
            The duration to be simulated, in milliseconds
        dt : _type_
            Time step, in milliseconds
        """
        time_ms = convert_time(time[0], time[1], 'ms')
        dt_ms = convert_time(dt[0], dt[1], 'ms')
        if dt_history:
            history_ui = UpdateInfo(update_interval=dt_history)
            self._make_history_entry(save_mode)

        steps = int(math.ceil(time_ms / dt_ms)) + 1

        if verbosity == 1:
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
            if verbosity == 1:
                if i % PROGRESSBAR_SCALER == 0:
                    days = convert_time(self.time, 'ms', 'day')
                    progressbar.set_postfix(T=f'{days:.2f}', N=f'{len(self.agents)}')
                    progressbar.update(PROGRESSBAR_SCALER)
            self._time += dt_ms

        if verbosity == 1:
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


def create_simulation(id: str = None):
    return Simulation(id=id)
