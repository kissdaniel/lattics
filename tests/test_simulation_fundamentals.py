import pytest


def test_default_instantiation():
    from lattics import Simulation

    sim = Simulation()
    assert sim is not None


def test_agent_addition_no_space_gives_warning():
    from lattics import Agent, Simulation

    sim = Simulation()
    a = Agent()
    with pytest.warns(Warning):
        sim.add_agent(a)


def test_agent_addition_no_space():
    from lattics import Agent, Simulation

    sim = Simulation()
    with pytest.warns(Warning):
        a = Agent()
        sim.add_agent(a)
        expected = 1
        actual = len(sim.agents)
        assert actual == expected


def test_agent_removal_no_space():
    from lattics import Agent, Simulation

    sim = Simulation()
    with pytest.warns(Warning):
        a1 = Agent()
        a2 = Agent()
        sim.add_agent(a1)
        sim.add_agent(a2)
        sim.remove_agent(a1)
        expected = a2
        actual = sim.agents[0]
        assert actual == expected


@pytest.mark.parametrize('time, dt, expected', [(10, 2, 10), (5, 3, 6), (5, 7, 7)])
def test_simulation_run_elapsed_time(time, dt, expected):
    from lattics import Simulation

    sim = Simulation()
    sim.run(time=(time, 'ms'), dt=(dt, 'ms'), verbosity=0)
    actual = sim.time
    assert actual == expected


def test_removed_agents_are_kept_for_the_latest_time_point():
    from lattics import Agent, Event, HomogeneousSpace, Simulation

    sim = Simulation()
    sim.set_space(HomogeneousSpace())
    agents = [Agent() for _ in range(2)]
    for agent in agents:
        agent.set_attribute('space.volume', (1, 'um3'))
        sim.add_agent(agent)
    sim.add_event(Event(time=(1, 'hour'), handler=lambda simulation: simulation.remove_agent(agents[0])))
    sim.add_event(Event(time=(2, 'hour'), handler=lambda simulation: simulation.remove_agent(agents[1])))
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert sim.removed_agents == [agents[0]]
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert sim.removed_agents == [agents[1]]
