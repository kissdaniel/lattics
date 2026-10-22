import heapq
import math
import pickle
import uuid
import warnings
from typing import TYPE_CHECKING

import numpy as np
import tqdm

from ._agent import Agent
from ._context import UpdateContext
from ._event import Event
from ._population import Population
from ._scheduler import EPSILON, ScheduledUpdate, effective_period, next_grid_point
from ._units import convert_time

if TYPE_CHECKING:
    from .models._base import BaseModel
    from .spaces._base import BaseSpace


class Simulation:
    """Represents a simulation instance. This object manages the participating
    agents (:class:`Agent`), the environment (:class:`SimulationDomain`), and
    the various chemical substances (:class:`Substrate`) present within it.
    The class provides high-level access to configure and execute a simulation.
    """

    def __init__(self, id: str = None, seed: int = None) -> None:
        """Constructor method.

        Parameters
        ----------
        id : str
            The identifier for the simulation instance. If not provided, a
            random identifier will be generated.
        seed : int
            The seed of the random number generator of the simulation. Every
            random number of the models and the space is drawn from this
            generator, so the same seed gives the same results. If not
            provided, a random seed is generated; it can be read from
            :attr:`seed` to repeat the simulation.
        """
        self._id = self._get_id(id)
        self._seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._rng = np.random.default_rng(self._seed)
        self._population = Population()
        self._space = None
        self._events = list()
        self._event_count = 0
        self._models = list()
        self._history = list()
        self._schedule_state = dict()
        self._removed_agents = list()
        self._time = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_events']
        del state['_models']
        del state['_history']
        del state['_schedule_state']
        del state['_removed_agents']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._events = list()
        self._models = list()
        self._history = list()
        self._schedule_state = dict()
        self._removed_agents = list()

    @property
    def agents(self) -> list[Agent]:
        """Get the agents currently present in the simulation, as a new list.
        The order of agents is maintained throughout the simulation, with new
        agent instances always being added at the end.

        Returns
        -------
        list[Agent]
            Collection of the agents
        """
        return list(self._population)

    @property
    def population(self) -> Population:
        """Get the population, the single collection that stores the agents.

        Returns
        -------
        Population
            The population of the simulation
        """
        return self._population

    @property
    def seed(self) -> int:
        """Get the seed of the random number generator of the simulation.

        Returns
        -------
        int
            The seed given to the constructor, or the generated one
        """
        return self._seed

    @property
    def rng(self) -> np.random.Generator:
        """Get the random number generator of the simulation, for example to
        draw random numbers in event handlers.

        Returns
        -------
        numpy.random.Generator
            The random number generator
        """
        return self._rng

    @property
    def removed_agents(self) -> list[Agent]:
        """Get the agents removed at the latest time point of the simulation,
        in the order of their removal. The list is renewed at every time
        point, after the history is recorded.

        Returns
        -------
        list[Agent]
            The removed agents
        """
        return list(self._removed_agents)

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

    def add_agent(self, agent: Agent) -> None:
        """Adds the specified agent to the simulation. The agent will be added
        to the collection of all agents and, if a simulation domain is defined,
        will also be placed within the simulation domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        self._population.add(agent)
        if self._space:
            self._space.add_agent(agent)
        else:
            warnings.warn(
                'No simulation domain has been defined. '
                'You can proceed without one, but this may '
                'lead to unexpected consequences.'
            )
        ctx = self._context(dt=0)
        for m in self._models:
            m.initialize(agent, ctx)

    def set_space(self, space: 'BaseSpace') -> None:
        """Sets the simulation space to the instance passed as a parameter.
        Agents added before are placed in the space now.

        Parameters
        ----------
        space : BaseSpace
            The simulation space instance to be used
        """
        if self._space:
            raise AttributeError('Simulation space is already set and cannot be modified.')
        space._bind(self._population, self._rng)
        self._space = space
        for agent in self._population:
            space.add_agent(agent)

    def add_event(self, event: Event) -> None:
        """Schedules the event. Events are executed at their exact time;
        events with the same time are executed in the order they were added.

        Parameters
        ----------
        event : Event
            The event to be scheduled
        """
        heapq.heappush(self._events, (event.time, self._event_count, event))
        self._event_count += 1

    def add_model(self, model: 'BaseModel') -> None:
        """Adds the provided model to the simulation and initializes the
        agents already present with it.

        Parameters
        ----------
        model : BaseModel
            A subclass of the ``BaseModel`` abstract base class.
        """
        self._models.append(model)
        ctx = self._context(dt=0)
        for a in self._population:
            model.initialize(a, ctx)

    def add_substrate(self, name: str, substrate) -> None:
        if not self._space:
            raise AttributeError('A simulation domain has to be set to add substrates.')
        self._domain.add_substrate_field(substrate)

    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the simulation and from its space.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        self._population.remove(agent)
        self._removed_agents.append(agent)
        if self._space:
            self._space.remove_agent(agent)

    def run(
        self,
        time: tuple[float, str],
        dt: tuple[float, str],
        dt_history: tuple[float, str] = None,
        save_mode: str = 'never',
        verbosity: int = 1,
    ) -> None:
        """Runs the simulation from the current state for the specified duration.

        The simulation advances on a grid of time points spaced by ``dt``. Every
        model, the agents of the space and the substrate fields are updated with
        their own interval, rounded up to a multiple of ``dt``: ``dt`` is the
        finest time step, so a larger ``dt`` gives a faster but less accurate
        run. Events are executed at their exact time.

        At each time point the history is recorded first, then the due events
        are executed, then the due models, the agents of the space and the
        substrate fields are updated, in this order. The space removes the
        agents that request it before it divides the agents.

        Parameters
        ----------
        time : tuple[float, str]
            The duration to be simulated, for example ``(5, 'day')``. It is
            rounded up to a multiple of ``dt``.
        dt : tuple[float, str]
            The time step of the simulation
        dt_history : tuple[float, str], optional
            The time between two history entries
        save_mode : str, optional
            When to write the history to a file: ``'never'`` (default),
            ``'always'`` or ``'on_completion'``
        verbosity : int, optional
            ``1`` (default) prints the update intervals and shows a progress
            bar, ``0`` is silent
        """
        time_ms = convert_time(time[0], time[1], 'ms')
        dt_ms = convert_time(dt[0], dt[1], 'ms')
        steps = int(math.ceil(time_ms / dt_ms - EPSILON))
        t0 = self._time
        t_end = t0 + steps * dt_ms

        updates = [self._prepare_update(update, dt_ms, t0) for update in self._scheduled_updates()]
        recorder = None
        if dt_history:
            recorder = ScheduledUpdate(
                'history',
                'History',
                convert_time(dt_history[0], dt_history[1], 'ms'),
                lambda now, period, last: self._make_history_entry(save_mode),
            )
            self._prepare_update(recorder, dt_ms, t0)

        if verbosity >= 1:
            self._print_summary(time_ms, dt_ms, updates + ([recorder] if recorder else []))
        self._warn_invalid_intervals(updates)

        if verbosity >= 1:
            progressbar_format = "{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]"
            progressbar = tqdm.tqdm(total=steps, mininterval=1.0, colour='#d2de32', bar_format=progressbar_format)
            progressbar.set_description(f'ID={self._id}')

        now = t0
        while now < t_end - EPSILON:
            self._time = now
            if recorder and recorder.is_due(now):
                recorder.run(now)
            self._removed_agents = list()
            self._execute_events(now)
            for update in updates:
                if update.is_due(now):
                    update.run(now)
            next_points = [next_grid_point(update.next_due, t0, dt_ms) for update in updates]
            if recorder:
                next_points.append(next_grid_point(recorder.next_due, t0, dt_ms))
            if self._events:
                next_points.append(self._events[0][0])
            next_time = min(next_points + [t_end])
            if verbosity >= 1:
                progressbar.update(round((next_time - now) / dt_ms))
                days = convert_time(next_time, 'ms', 'day')
                progressbar.set_postfix(T=f'{days:.2f}', N=f'{len(self._population)}', refresh=False)
            now = next_time

        self._time = t_end
        if recorder and recorder.is_due(t_end):
            recorder.run(t_end)
        for update in updates + ([recorder] if recorder else []):
            self._schedule_state[update.key] = (update.next_due, update.last_run)
        if verbosity >= 1:
            progressbar.close()

        if save_mode == 'on_completion':
            self._save_history()

    def _scheduled_updates(self) -> list[ScheduledUpdate]:
        """The periodically updated parts of the simulation, in the order of their execution."""
        updates = [
            ScheduledUpdate(
                ('model', model),
                type(model).__name__,
                model.interval,
                lambda now, period, last, model=model: model.update(self._population, self._context(period, last)),
            )
            for model in self._models
        ]
        if self._space:
            space_name = type(self._space).__name__
            updates.append(
                ScheduledUpdate(
                    ('agents', self._space),
                    f'{space_name} (agents)',
                    self._space.interval,
                    lambda now, period, last: self._space.update_agents(self._context(period, last)),
                )
            )
            if self._space.has_substrates():
                updates.append(
                    ScheduledUpdate(
                        ('substrates', self._space),
                        f'{space_name} (substrates)',
                        self._space.substrate_interval,
                        lambda now, period, last: self._space.update_substrates(self._context(period, last)),
                    )
                )
        return updates

    def _prepare_update(self, update: ScheduledUpdate, dt: float, start: float) -> ScheduledUpdate:
        update.period = effective_period(update.interval, dt)
        update.next_due, update.last_run = self._schedule_state.get(update.key, (start, None))
        return update

    def _print_summary(self, duration: float, dt: float, updates: list[ScheduledUpdate]) -> None:
        print(
            f'Simulation {self._id}: {_format_duration(duration)}, time step {_format_duration(dt)}, '
            f'seed {self._seed}'
        )
        width = max(len(update.name) for update in updates) if updates else 0
        for update in updates:
            if update.interval is None:
                interval = f'every time step ({_format_duration(update.period)})'
            elif abs(update.period - update.interval) <= EPSILON:
                interval = f'every {_format_duration(update.period)}'
            else:
                interval = (
                    f'every {_format_duration(update.period)} '
                    f'(requested {_format_duration(update.interval)}, rounded to the time step)'
                )
            print(f'  {update.name:<{width}}  {interval}')

    def _warn_invalid_intervals(self, updates: list[ScheduledUpdate]) -> None:
        for update in updates:
            if update.key[0] == 'agents':
                for message in self._space.interval_warnings(update.period):
                    warnings.warn(message)

    def _execute_events(self, now: float) -> None:
        while self._events and self._events[0][0] <= now + EPSILON:
            _, _, event = heapq.heappop(self._events)
            event.execute(self)

    def _make_history_entry(self, save_mode):
        state = pickle.dumps(self)
        self._history.append(state)
        if save_mode == 'always':
            self._save_history()

    def _save_history(self):
        filename = f'{self._id}.lsd'
        with open(filename, 'wb') as f:
            pickle.dump(self._history, f)

    def _context(self, dt: float, last_update_time: float = None) -> UpdateContext:
        return UpdateContext(
            time=self._time,
            dt=dt,
            population=self._population,
            space=self._space,
            simulation=self,
            last_update_time=last_update_time,
            rng=self._rng,
        )

    def _get_id(self, identifier):
        return identifier if identifier else str(uuid.uuid4())


def _format_duration(milliseconds: float) -> str:
    for unit in ('week', 'day', 'hour', 'min', 'sec'):
        value = convert_time(milliseconds, 'ms', unit)
        if value >= 1:
            return f'{value:g} {unit}'
    return f'{milliseconds:g} ms'
