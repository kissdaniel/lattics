"""Recorders: the data a simulation records while it runs.

A recorder is added with :meth:`lattics.Simulation.add_recorder` and records at its own interval, at the start
of the run and then every ``interval``, and once more at the end of the run if it is due. At every time point
it also receives the agents removed since the previous time point, so it can account for them even if it
records less often.

``AgentCount`` and ``SubstrateLevel`` record a single quantity as a :class:`~lattics.utils.time_series.TimeSeries`.
``Snapshot`` records the state of the agents and the substrate fields for later analysis and visualization.
"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._units import convert_time
from lattics.utils.time_series import TimeSeries


class Recorder(ABC):
    """Base class of the recorders. A custom recorder implements :meth:`record`
    and, if it needs the removed agents, :meth:`collect`.

    Parameters
    ----------
    interval : tuple[float, str], optional
        The time between two records, for example ``(1, 'hour')``. Defaults
        to every time step of the simulation.
    """

    def __init__(self, interval: tuple[float, str] = None) -> None:
        self.interval = convert_time(interval[0], interval[1], 'ms') if interval else None

    def collect(self, ctx: UpdateContext, removed: list[Agent]) -> None:
        """Called at every time point with the agents removed since the previous time point, before
        :meth:`record`, even if the recorder is not due.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the simulation
        removed : list[Agent]
            The removed agents
        """

    @abstractmethod
    def record(self, ctx: UpdateContext) -> None:
        """Records the current state.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the simulation; ``ctx.time`` is the time of the record
        """
        raise NotImplementedError


def _matcher(where) -> Callable[[Agent], bool]:
    if where is None:
        return lambda agent: True
    if callable(where):
        return where
    name, value = where
    return lambda agent: agent.has_attribute(name) and agent.get_attribute(name) == value


class AgentCount(Recorder):
    """Records the number of agents, optionally only those that match a condition.

    Parameters
    ----------
    interval : tuple[float, str], optional
        The time between two records
    where : tuple[str, Any] or Callable[[Agent], bool], optional
        An attribute name and the value it must have, for example
        ``('toxicity.state', 'necrotic')``, or a function of the agent

    Attributes
    ----------
    series : TimeSeries
        The number of matching agents at each record
    removed : TimeSeries
        The number of matching agents removed since the previous record
        (at the first record: since the start)
    """

    def __init__(
        self, interval: tuple[float, str] = None, where: tuple[str, Any] | Callable[[Agent], bool] = None
    ) -> None:
        super().__init__(interval)
        self._where = where
        self.series = TimeSeries([], [])
        self.removed = TimeSeries([], [])
        self._removed_count = 0

    def collect(self, ctx: UpdateContext, removed: list[Agent]) -> None:
        matches = _matcher(self._where)
        self._removed_count += sum(1 for agent in removed if matches(agent))

    def record(self, ctx: UpdateContext) -> None:
        matches = _matcher(self._where)
        self.series.append(ctx.time, sum(1 for agent in ctx.population if matches(agent)))
        self.removed.append(ctx.time, self._removed_count)
        self._removed_count = 0


_STATISTICS = {'mean': np.mean, 'min': np.min, 'max': np.max, 'sum': np.sum}


class SubstrateLevel(Recorder):
    """Records a statistic of the concentration field of a substrate.

    Parameters
    ----------
    substrate : str
        The name of the substrate
    interval : tuple[float, str], optional
        The time between two records
    statistic : str or Callable, optional
        ``'mean'`` (default), ``'min'``, ``'max'``, ``'sum'`` or a function of
        the concentration (a number or an array)
    position : tuple[int, ...], optional
        Record the concentration at this position instead of a statistic

    Attributes
    ----------
    series : TimeSeries
        The recorded values
    """

    def __init__(
        self,
        substrate: str,
        interval: tuple[float, str] = None,
        statistic: str | Callable = 'mean',
        position: tuple[int, ...] = None,
    ) -> None:
        super().__init__(interval)
        if isinstance(statistic, str) and statistic not in _STATISTICS:
            raise ValueError(
                f'Unknown statistic \'{statistic}\'. Use one of {", ".join(_STATISTICS)} or a function.'
            )
        self._substrate = substrate
        self._statistic = statistic
        self._position = position
        self.series = TimeSeries([], [])

    def record(self, ctx: UpdateContext) -> None:
        field = ctx.substrate(self._substrate)
        if self._position is not None:
            value = field.concentration_at(self._position)
        else:
            statistic = _STATISTICS[self._statistic] if isinstance(self._statistic, str) else self._statistic
            value = statistic(field.concentration)
        self.series.append(ctx.time, float(value))


class Frame:
    """The state of the agents and substrate fields at one time point, recorded by a :class:`Snapshot`.

    Attributes
    ----------
    time : float
        The time of the record, in milliseconds
    agents : list[dict[str, Any]]
        One dictionary per agent: ``'id'`` and the recorded attributes
    removed : list[dict[str, Any]]
        The agents removed since the previous frame, as they were at their removal
    fields : dict[str, Any]
        The concentration of each recorded substrate (a number or an array)
    """

    def __init__(self, time: float, data: bytes) -> None:
        self.time = time
        self._data = data

    def _content(self) -> dict:
        return pickle.loads(self._data)

    @property
    def agents(self) -> list[dict[str, Any]]:
        return self._content()['agents']

    @property
    def removed(self) -> list[dict[str, Any]]:
        return self._content()['removed']

    @property
    def fields(self) -> dict[str, Any]:
        return self._content()['fields']


class Snapshot(Recorder):
    """Records the state of the agents and the substrate fields: every
    attribute of every agent and every field, unless narrowed down. The data
    is copied (and stored compactly), so later changes of the simulation do
    not affect it.

    Parameters
    ----------
    interval : tuple[float, str], optional
        The time between two records
    attributes : Iterable[str], optional
        Record only these agent attributes (and the identifier)
    substrates : Iterable[str], optional
        Record only the fields of these substrates

    Attributes
    ----------
    frames : list[Frame]
        The recorded frames, in time order
    dimensions : tuple[int, ...] or None
        The number of lattice sites along each axis, for spaces with a lattice
    """

    def __init__(
        self,
        interval: tuple[float, str] = None,
        attributes: Iterable[str] = None,
        substrates: Iterable[str] = None,
    ) -> None:
        super().__init__(interval)
        self._attributes = tuple(attributes) if attributes is not None else None
        self._substrates = tuple(substrates) if substrates is not None else None
        self._removed = []
        self.frames = []
        self.dimensions = None

    def collect(self, ctx: UpdateContext, removed: list[Agent]) -> None:
        self._removed += [self._agent_record(agent) for agent in removed]

    def record(self, ctx: UpdateContext) -> None:
        if self.dimensions is None and hasattr(ctx.space, '_dimensions'):
            self.dimensions = tuple(ctx.space._dimensions)
        names = self._substrates if self._substrates is not None else ctx.substrate_names()
        content = {
            'agents': [self._agent_record(agent) for agent in ctx.population],
            'removed': self._removed,
            'fields': {name: ctx.substrate(name).concentration for name in names},
        }
        self.frames.append(Frame(ctx.time, pickle.dumps(content, protocol=pickle.HIGHEST_PROTOCOL)))
        self._removed = []

    def frame_at(self, time: tuple[float, str]) -> Frame:
        """Returns the frame recorded at the given time.

        Parameters
        ----------
        time : tuple[float, str]
            The time, for example ``(6, 'hour')``

        Returns
        -------
        Frame
            The frame

        Raises
        ------
        KeyError
            If no frame was recorded at that time
        """
        time_ms = convert_time(time[0], time[1], 'ms')
        for frame in self.frames:
            if abs(frame.time - time_ms) <= 1e-6:
                return frame
        raise KeyError(f'No frame was recorded at {time[0]} {time[1]}.')

    def save(self, path: str) -> None:
        """Saves the frames to a file, to be read with :meth:`load`.

        Parameters
        ----------
        path : str
            The path of the file
        """
        with open(path, 'wb') as f:
            pickle.dump({'format': 'lattics-snapshot', 'dimensions': self.dimensions, 'frames': self.frames}, f)

    @staticmethod
    def load(path: str) -> 'Snapshot':
        """Reads frames saved with :meth:`save`. Only load files from trusted
        sources: reading a file may execute code stored in it.

        Parameters
        ----------
        path : str
            The path of the file

        Returns
        -------
        Snapshot
            A snapshot holding the frames
        """
        with open(path, 'rb') as f:
            content = pickle.load(f)
        if not isinstance(content, dict) or content.get('format') != 'lattics-snapshot':
            raise ValueError(f'{path} is not a LattiCS snapshot file.')
        snapshot = Snapshot()
        snapshot.frames = content['frames']
        snapshot.dimensions = content['dimensions']
        return snapshot

    def _agent_record(self, agent: Agent) -> dict[str, Any]:
        attributes = agent._attributes
        if self._attributes is not None:
            attributes = {name: attributes[name] for name in self._attributes if name in attributes}
        return {'id': agent.id, **attributes}
