from typing import Any

from lattics._agent import Agent
from lattics._simulation import Simulation
from lattics.spaces import Lattice2DSpace


def create_agent(attributes: dict[str, Any]) -> Agent:
    """Creates an :class:`~lattics.Agent` object and initializes it with the provided attributes.

    Parameters
    ----------
    attributes : dict[str, Any]
        The agent's attributes that we want to store

    Returns
    -------
    :class:`~lattics.Agent`
        An agent object

    Examples
    --------
    >>> agent = lattics.create_agent(
    ...     attributes={
    ...         'space.volume': 2000,
    ...         'space.position': (0, 0),
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
    """Creates the given number of :class:`~lattics.Agent` objects and initializes them with the provided
    attributes.

    Parameters
    ----------
    size : int
        The number of agents we want to create
    attributes : dict[str, Any]
        The agent's attributes that we want to store

    Returns
    -------
    list[:class:`~lattics.Agent`]
        A list of agent objects

    Examples
    --------
    >>> agents = lattics.create_population(
    ...     size=100,
    ...     attributes={'space.volume': 2000}
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
    seed: int = None,
) -> Simulation:
    """Creates a :class:`~lattics.Simulation` with a :class:`~lattics.Lattice2DSpace` of the given physical size.

    Parameters
    ----------
    dimensions : tuple[tuple[float, str], tuple[float, str]]
        The size of the domain along the two axes, for example ``((1, 'mm'), (750, 'um'))``. Each size must be a
        multiple of ``grid_spacing``.
    grid_spacing : int
        The distance between neighboring lattice sites, in micrometers
    seed : int, optional
        The seed of the random number generator of the simulation

    Raises
    ------
    ValueError
        If the size is missing or not a multiple of the grid spacing
    """
    simulation = Simulation(id=id, seed=seed)
    space = Lattice2DSpace(
        size=dimensions,
        grid_spacing=(grid_spacing, 'um'),
        interval=dt_agent,
        substrate_interval=dt_substrate,
    )
    simulation.set_space(space)
    return simulation


def populate_agents(
    simulation: Simulation,
    method: str = 'count',
    count: int = 1,
    density: float = 0.0,
    map: str = None,
    attributes: dict[str, Any] = None,
) -> None:
    """Adds new agents with the given attributes to the simulation. The space
    places them: a lattice space puts them on random free sites.

    Parameters
    ----------
    simulation : Simulation
        The simulation to be populated
    method : str
        Only ``'count'`` is supported yet: adds ``count`` agents
    count : int
        The number of agents to be added
    attributes : dict[str, Any]
        The attributes of the new agents
    """
    if not attributes:
        attributes = {}
    if method == 'count':
        for _ in range(count):
            simulation.add_agent(agent=create_agent(attributes=attributes))
    if method == 'density':
        pass
    if method == 'map':
        pass
