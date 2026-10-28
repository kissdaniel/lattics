import warnings


def test_time_based_cell_cycle_divides_in_homogeneous_space():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.set_space(HomogeneousSpace(interval=(1, 'hour')))
    sim.add_model(TimeBasedCellCycleModel(update_interval=(1, 'hour'), distribution='fixed'))
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    agent.set_attribute('cellcycle.mean_length', (10, 'hour'))
    sim.add_agent(agent)
    sim.run(time=(3, 'day'), dt=(1, 'hour'), verbosity=0)
    # 72 hours with a 10-hour cycle: six or seven completed generations
    actual = len(sim.agents)
    assert 2**6 <= actual <= 2**7


def test_toxicity_stops_the_cell_cycle():
    from lattics import Agent, SubstrateInfo, UpdateContext
    from lattics.models import ConcentrationDependentToxicityModel

    model = ConcentrationDependentToxicityModel(
        update_interval=(1, 'hour'),
        substrate_name='drug',
        max_rate=(1e9, '1/day'),
        median_effective_concentration=1.0,
    )
    agent = Agent()
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', concentration=1e6)})
    agent.set_attribute('cellcycle.is_active', True)
    agent.set_attribute('space.division_requested', True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.update_agent(agent, UpdateContext(time=0, dt=3600000))
    expected = ('necrotic', False, False)
    actual = (
        agent.get_attribute('toxicity.state'),
        agent.get_attribute('cellcycle.is_active'),
        agent.get_attribute('space.division_requested'),
    )
    assert actual == expected


def _cell_cycle_simulation(mean_length, space_volume=None):
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.set_space(HomogeneousSpace(volume=space_volume, interval=(1, 'hour')))
    sim.add_model(TimeBasedCellCycleModel(update_interval=(1, 'hour'), distribution='fixed'))
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    agent.set_attribute('cellcycle.mean_length', mean_length)
    sim.add_agent(agent)
    return sim, agent


def test_cell_cycle_carries_the_overshoot_into_the_next_cycle():
    # A 9.5-hour cycle updated hourly: without the carry-over every cycle would last 10 hours,
    # so the second division would happen only after 20 hours
    sim, _ = _cell_cycle_simulation((9.5, 'hour'))
    sim.run(time=(19, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert len(sim.agents) == 4


def test_waiting_division_does_not_advance_the_cell_cycle():
    # The space is full, so the completed cycle waits
    sim, agent = _cell_cycle_simulation((10, 'hour'), space_volume=(1, 'um3'))
    sim.run(time=(15, 'hour'), dt=(1, 'hour'), verbosity=0)
    expected = (1, True, 10 * 3600000.0)
    actual = (
        len(sim.agents),
        agent.get_attribute('space.division_requested'),
        agent.get_attribute('cellcycle.current_time'),
    )
    assert actual == expected


def test_new_cycle_after_division_keeps_an_inactive_cell_inactive():
    from lattics import Agent, UpdateContext
    from lattics.models import TimeBasedCellCycleModel

    model = TimeBasedCellCycleModel(distribution='fixed')
    agent = Agent()
    agent.set_attribute('cellcycle.mean_length', (10, 'hour'))
    model.initialize(agent, UpdateContext(time=0, dt=0))
    # divided at 5 hours, then stopped by another model (e.g. toxicity) before the cell cycle noticed it
    agent.set_attribute('space.last_division_time', 5 * 3600000.0)
    agent.set_attribute('cellcycle.is_active', False)
    model.update_agent(agent, UpdateContext(time=6 * 3600000.0, dt=3600000.0, last_update_time=5 * 3600000.0))
    assert agent.get_attribute('cellcycle.is_active') is False


def test_multi_phase_cell_cycle_stops_when_inactive():
    from lattics import Agent, UpdateContext
    from lattics.models import MultiPhaseTimeBasedCellCycleModel

    model = MultiPhaseTimeBasedCellCycleModel()
    agent = Agent()
    for phase in ('g1', 's', 'g2', 'm'):
        agent.set_attribute(f'cellcycle.phase_{phase}_mean', (1, 'hour'))
    model.initialize(agent, UpdateContext(time=0, dt=0))
    agent.set_attribute('cellcycle.is_active', False)
    model.update_agent(agent, UpdateContext(time=0, dt=5 * 3600000.0))
    expected = (0, False)
    actual = (agent.get_attribute('cellcycle.current_time'), agent.get_attribute('space.division_requested'))
    assert actual == expected
