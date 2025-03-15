import pytest


def test_default_instantiation():
    from lattics import simulation
    sim = simulation.Simulation()
    assert sim is not None


def test_agent_addition_no_space_gives_warning():
    from lattics import simulation
    from lattics import agent
    sim = simulation.Simulation()
    a = agent.Agent()
    with pytest.warns(Warning):
        sim.add_agent(a)


def test_agent_addition_no_space():
    from lattics import simulation
    from lattics import agent
    sim = simulation.Simulation()
    with pytest.warns(Warning):
        a = agent.Agent()
        sim.add_agent(a)
        expected = 1
        actual = len(sim.agents)
        assert actual == expected


def test_agent_removal_no_space():
    from lattics import simulation
    from lattics import agent
    sim = simulation.Simulation()
    with pytest.warns(Warning):
        a1 = agent.Agent()
        a2 = agent.Agent()
        sim.add_agent(a1)
        sim.add_agent(a2)
        sim.remove_agent(a1)
        expected = a2
        actual = sim.agents[0]
        assert actual == expected
