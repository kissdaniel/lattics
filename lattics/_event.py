from typing import Any

from ._units import convert_time


class Event:
    def __init__(self, time: tuple[float, str], handler: Any, **params: dict) -> None:
        self._time = convert_time(time[0], time[1], 'ms')
        self._handler = handler
        self._params = params

    @property
    def time(self) -> float:
        """The time of the event, in milliseconds."""
        return self._time

    def is_ready(self, current_time: int) -> bool:
        return self._time <= current_time

    def execute(self, simulation):
        self._handler(simulation, **self._params)
