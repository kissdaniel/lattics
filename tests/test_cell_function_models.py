import warnings


def test_time_based_cell_cycle_divides_in_homogeneous_space():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.add_space(HomogeneousSpace(sim, volume=10**9, dt_agent=(1, 'hour')))
    sim.add_model(TimeBasedCellCycleModel(update_interval=(1, 'hour'), distribution='fixed'))
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    agent.set_attribute('cellcycle.mean_length', (10, 'hour'))
    sim.add_agent(agent)
    sim.run(time=(3, 'day'), dt=(1, 'hour'), verbosity=0)
    # 72 hours with a 10-hour cycle: six or seven completed generations
    actual = len(sim.agents)
    assert 2**6 <= actual <= 2**7


def test_toxicity_stops_the_cell_cycle():
    from lattics import Agent, SubstrateInfo
    from lattics.models import ConcentrationDependentToxicityModel

    model = ConcentrationDependentToxicityModel(
        update_interval=(1, 'hour'), substrate_name='drug', max_rate=1e9, median_effective_concentration=1.0
    )
    model.update_info.increase_time(3600000)
    agent = Agent()
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', concentration=1e6)})
    agent.set_attribute('cellcycle.is_active', True)
    agent.set_attribute('space.division_requested', True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.update_attributes(agent)
    expected = ('necrotic', False, False)
    actual = (
        agent.get_attribute('toxicity.state'),
        agent.get_attribute('cellcycle.is_active'),
        agent.get_attribute('space.division_requested'),
    )
    assert actual == expected
