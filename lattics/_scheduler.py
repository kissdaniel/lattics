import math
from typing import Callable

# Tolerance for comparing times, in milliseconds
EPSILON = 1e-6


class ScheduledUpdate:
    """A periodically updated part of the simulation: a model, the agents or
    the substrate fields of the space, or the history recorder.

    The simulation advances on a grid of time points (the multiples of ``dt``
    from the start of a run). An update with a requested ``interval`` runs every
    ``period`` time, where ``period`` is ``interval`` rounded up to a multiple
    of ``dt`` (and at least ``dt``). When the update runs at time ``t``, it
    advances its part of the simulation from ``t`` to ``t + period``.

    Parameters
    ----------
    key : object
        Identifies the update across consecutive runs of the simulation
    name : str
        The name shown in the run summary
    interval : float or None
        The requested time between two updates, in milliseconds; ``None``
        means every time step
    action : Callable[[float, float], None]
        Called with the current time and the period when the update runs
    """

    __slots__ = ('key', 'name', 'interval', 'action', 'period', 'next_due')

    def __init__(
        self, key: object, name: str, interval: float | None, action: Callable[[float, float], None]
    ) -> None:
        self.key = key
        self.name = name
        self.interval = interval
        self.action = action
        self.period = None
        self.next_due = None

    def is_due(self, time: float) -> bool:
        return self.next_due <= time + EPSILON

    def run(self, time: float) -> None:
        self.action(time, self.period)
        self.next_due = time + self.period


def effective_period(interval: float | None, dt: float) -> float:
    """Rounds the requested interval up to a multiple of the time step (at least one time step)."""
    if not interval or interval <= dt + EPSILON:
        return dt
    return dt * math.ceil(interval / dt - EPSILON)


def next_grid_point(time: float, origin: float, dt: float) -> float:
    """Returns the first time point of the grid ``origin + k * dt`` that is not before ``time``."""
    return origin + dt * math.ceil((time - origin) / dt - EPSILON)
