import random
from typing import Any

from lattics.core import Agent, Simulation
from lattics.core._convert import convert_length
from lattics.core.spaces import Lattice2DSpace


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


def create_simulation_2d(
    id: str = None,
    dimensions: tuple[tuple[float, str], tuple[float, str]] = None,
    grid_spacing: int = 20,
    dt_agent: tuple = (30, 'sec'),
    dt_substrate: tuple = (1, 'sec'),
) -> Simulation:
    dimensions_um = convert_length(dimensions_um)
    simulation = Simulation(id=id)
    space = Lattice2DSpace(
        simulation=simulation,
        dimensions=dimensions,
        grid_spacing=grid_spacing,
        dt_agent=dt_agent,
        dt_substrate=dt_substrate,
    )
    simulation.add_space(space=space)
    return simulation


def populate_agents(
    simulation: Simulation,
    method: str = 'count',
    count: int = 1,
    density: float = 0.0,
    map: str = None,
    attributes: dict[str, Any] = None,
) -> None:
    if not attributes:
        attributes = {}
    if method == 'count':
        rows, cols = simulation._space._agent_layer.shape
        all_coordinates = [(r, c) for r in range(rows) for c in range(cols)]
        target_coordinates = random.sample(all_coordinates, k=count)
        for tc in target_coordinates:
            agent_attributes = attributes | {'position': tc}
            agent = create_agent(attributes=agent_attributes)
            simulation.add_agent(agent=agent)
    if method == 'density':
        pass
    if method == 'map':
        pass
