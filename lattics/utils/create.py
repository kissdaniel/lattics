from lattics.core import Agent
from lattics.core import Simulation
from typing import Any


def create_agent(attributes: dict[str, Any]) -> Agent:
    """Creates an :class:`~lattics.core.Agent` object and initializes it with the provided attributes.

    Parameters
    ----------
    attributes : dict[str, Any]
        The agent's attributes that we want to store

    Returns
    -------
    :class:`~lattics.core.Agent`
        An agent object

    Examples
    --------
    >>> agent = lattics.create_agent(
    ...     attributes={
    ...         'volume': 2000,
    ...         'position': (0, 0),
    ...         'my_custom_attribute': 'High'
    ...         }
    ...     )
    """
    agent = Agent()
    for par_name, par_value in attributes.items():
        if not agent.has_attribute(par_name):
            agent.set_attribute(par_name, par_value)
    return agent


def create_population(size: int, attributes: dict[str, Any]) -> list[Agent]:
    """Creates the given number of :class:`~lattics.core.Agent` objects and initializes them with the provided attributes.

    Parameters
    ----------
    size : int
        The number of agents we want to create
    attributes : dict[str, Any]
        The agent's attributes that we want to store

    Returns
    -------
    list[:class:`~lattics.core.Agent`]
        A list of agent objects

    Examples
    --------
    >>> agents = lattics.create_population(
    ...     size=100,
    ...     attributes={'volume': 2000}
    ...     )
    """
    agents = []
    for _ in range(size):
        a = create_agent(attributes=attributes)
        agents.append(a)
    return agents


def create_simulation(id: str = None, agents: list[Agent] = None):
    sim = Simulation(id=id)
    if agents:
        for a in agents:
            sim.add_agent(a)
    return sim
