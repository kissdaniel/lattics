import warnings

import pytest


def test_ids_are_consecutive_within_a_population():
    from lattics import Agent, Population

    population = Population()
    agents = [Agent() for _ in range(3)]
    for a in agents:
        population.add(a)
    expected = [0, 1, 2]
    actual = [a.id for a in agents]
    assert actual == expected


def test_each_simulation_numbers_its_agents_from_zero():
    from lattics import Agent, Simulation

    sims = [Simulation(), Simulation()]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for sim in sims:
            sim.add_agent(Agent())
    expected = [0, 0]
    actual = [sim.agents[0].id for sim in sims]
    assert actual == expected


def test_removal_keeps_the_order_of_the_others():
    from lattics import Agent, Population

    population = Population()
    agents = [Agent() for _ in range(4)]
    for a in agents:
        population.add(a)
    population.remove(agents[1])
    expected = [agents[0], agents[2], agents[3]]
    actual = list(population)
    assert actual == expected and agents[1] not in population


def test_agent_cannot_be_added_twice():
    from lattics import Agent, Population

    population = Population()
    agent = Agent()
    population.add(agent)
    with pytest.raises(ValueError):
        population.add(agent)


def test_homogeneous_space_without_volume_limit():
    from lattics import Agent, HomogeneousSpace, Simulation

    sim = Simulation()
    sim.add_space(HomogeneousSpace(sim, dt_agent=(1, 'hour')))
    agent = Agent()
    agent.set_attribute('space.volume', 1000)
    agent.set_attribute('space.division_requested', True)
    sim.add_agent(agent)
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    expected = 2
    actual = len(sim.agents)
    assert actual == expected
