"""Events: actions executed at an exact time of the simulation, for example a treatment.

An event is added with :meth:`lattics.Simulation.add_event`. Events are executed at their exact time, after
the recorders and before the models of that time point; events with the same time are executed in the order
they were added.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._context import UpdateContext
from ._units import convert_time

if TYPE_CHECKING:
    from .substrates import SubstrateNode


class Event:
    """An event that calls a function. The function receives the context of
    the simulation and the keyword arguments of the event, for example
    ``handler(ctx, dose=5.0)``. Through the context it can add and remove
    agents, reach the substrate fields and draw random numbers.

    To save the simulation with :meth:`~lattics.Simulation.save`, the handler
    must be a function defined at module level (not a lambda or a nested
    function).

    Parameters
    ----------
    time : tuple[float, str]
        The time of the event, for example ``(2, 'day')``
    handler : Callable[..., None]
        The function to be called
    **params
        Keyword arguments passed to the handler
    """

    def __init__(self, time: tuple[float, str], handler: Callable[..., None] = None, **params: Any) -> None:
        self._time = convert_time(time[0], time[1], 'ms')
        self._handler = handler
        self._params = params

    @property
    def time(self) -> float:
        """The time of the event, in milliseconds."""
        return self._time

    def execute(self, ctx: UpdateContext) -> None:
        """Executes the event.

        Parameters
        ----------
        ctx : UpdateContext
            The context of the simulation
        """
        self._handler(ctx, **self._params)


class SetConcentration(Event):
    """Sets the concentration of a substrate, for example to give a dose.

    With a ``node``, the concentration inside the node is set (for example in
    the node representing the blood vessels). Without a node, the whole
    concentration field is set to the given value.

    Parameters
    ----------
    time : tuple[float, str]
        The time of the event
    substrate : str
        The name of the substrate
    concentration : float
        The new concentration
    node : SubstrateNode, optional
        The node whose concentration is set
    """

    def __init__(
        self, time: tuple[float, str], substrate: str, concentration: float, node: 'SubstrateNode' = None
    ) -> None:
        super().__init__(time)
        self._substrate = substrate
        self._concentration = concentration
        self._node = node

    def execute(self, ctx: UpdateContext) -> None:
        if self._node is not None:
            self._node.get_attribute('substrate.transport')[self._substrate].concentration = self._concentration
        else:
            ctx.substrate(self._substrate).set_concentration(self._concentration)
