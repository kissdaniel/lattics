import math
import warnings

import numpy as np
import pytest

HOUR = 3600000.0


def _homogeneous_simulation():
    from lattics import HomogeneousSpace, Simulation

    sim = Simulation(seed=1)
    sim.set_space(HomogeneousSpace(volume=(1, 'mm3'), interval=(1, 'hour')))
    return sim


def test_first_order_decay_is_exact():
    from lattics.kinetics import FirstOrderDecay

    kinetics = FirstOrderDecay(rate=(0.5, '1/hour'))
    assert math.isclose(kinetics.apply(8.0, 2 * HOUR), 8.0 * math.exp(-1.0))


def test_second_order_decay_is_exact_and_handles_zero():
    from lattics.kinetics import SecondOrderDecay

    kinetics = SecondOrderDecay(rate=(0.1, '1/hour'))
    assert math.isclose(kinetics.apply(10.0, HOUR), 10.0 / (1 + 0.1 * 10.0)) and kinetics.apply(0.0, HOUR) == 0.0


def test_michaelis_menten_decay_stays_non_negative():
    from lattics.kinetics import MichaelisMentenDecay

    kinetics = MichaelisMentenDecay(vmax=(100.0, '1/hour'), km=1.0)
    assert 0 <= kinetics.apply(5.0, HOUR) < 5.0


def test_kinetics_work_on_arrays():
    from lattics.kinetics import FirstOrderDecay, MichaelisMentenDecay, SecondOrderDecay

    values = np.array([0.0, 1.0, 10.0])
    for kinetics in (
        FirstOrderDecay((1, '1/hour')),
        SecondOrderDecay((1, '1/hour')),
        MichaelisMentenDecay((1, '1/hour'), km=2.0),
    ):
        expected = [kinetics.apply(float(v), HOUR) for v in values]
        assert np.allclose(kinetics.apply(values, HOUR), expected)


def test_rates_must_carry_their_unit():
    from lattics.kinetics import FirstOrderDecay

    with pytest.raises(TypeError, match='1/hour'):
        FirstOrderDecay(rate=0.1)


def test_unit_conversions():
    from lattics import convert_diffusivity, convert_rate

    assert math.isclose(convert_rate(3.6, '1/hour', '1/sec'), 0.001)
    assert math.isclose(convert_diffusivity(1.0, 'mm2/sec', 'um2/ms'), 1000.0)
    with pytest.raises(ValueError):
        convert_rate(1.0, 'hour', '1/ms')


def test_substrate_applies_its_kinetics_in_order():
    from lattics import Substrate
    from lattics.kinetics import FirstOrderDecay, SecondOrderDecay

    first, second = FirstOrderDecay((1, '1/hour')), SecondOrderDecay((1, '1/hour'))
    substrate = Substrate('drug', kinetics=[first, second], diffusion_coefficient=(2.0, 'um2/sec'))
    expected = second.apply(first.apply(4.0, HOUR), HOUR)
    assert substrate.apply_kinetics(4.0, HOUR) == expected and substrate.diffusion_coefficient == 0.002


@pytest.mark.parametrize('space_first', [True, False])
def test_field_is_created_whether_the_space_is_set_before_or_after(space_first):
    from lattics import HomogeneousSpace, Simulation, Substrate

    sim = Simulation()
    if space_first:
        sim.set_space(HomogeneousSpace(volume=(1, 'mm3')))
    sim.add_substrate(Substrate('drug'))
    if not space_first:
        sim.set_space(HomogeneousSpace(volume=(1, 'mm3')))
    assert sim.substrate_field('drug').concentration == 0.0


def test_substrate_names_are_unique():
    from lattics import Substrate

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug'))
    with pytest.raises(ValueError):
        sim.add_substrate(Substrate('drug'))


def test_unknown_field_raises_a_clear_error():
    with pytest.raises(KeyError, match="'oxygen' is not added"):
        _homogeneous_simulation().substrate_field('oxygen')


def test_exchange_conserves_the_mass():
    from lattics import Agent, Substrate, SubstrateInfo, SubstrateNode

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug', interval=(1, 'min')))
    node = SubstrateNode(volume=(0.1, 'mm3'), transport={'drug': SubstrateInfo(type='flux', passive_rate=1.0)})
    node.get_attribute('substrate.transport')['drug'].concentration = 100.0
    sim.add_substrate_node(node)
    agent = Agent()
    agent.set_attribute('space.volume', 3000)
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', passive_rate=1e-3)})
    sim.add_agent(agent)
    sim.run(time=(2, 'hour'), dt=(1, 'min'), verbosity=0)

    def mass(item, volume):
        return item.get_attribute('substrate.transport')['drug'].concentration * volume

    total = sim.substrate_field('drug').concentration * 1e9 + mass(node, 1e8) + mass(agent, 3000)
    assert math.isclose(total, 100.0 * 1e8, rel_tol=1e-9)


def test_removed_agents_stop_exchanging():
    from lattics import Agent, Event, Substrate, SubstrateInfo, SubstrateNode

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug', interval=(1, 'min')))
    node = SubstrateNode(volume=(1, 'mm3'), transport={'drug': SubstrateInfo(type='fixed', concentration=10.0)})
    sim.add_substrate_node(node)
    agent = Agent()
    agent.set_attribute('space.volume', 3000)
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', uptake_rate=1e-3)})
    sim.add_agent(agent)
    sim.add_event(Event(time=(10, 'min'), handler=lambda simulation: simulation.remove_agent(agent)))
    sim.run(time=(10, 'min'), dt=(1, 'min'), verbosity=0)
    before = agent.get_attribute('substrate.transport')['drug'].concentration
    sim.run(time=(10, 'min'), dt=(1, 'min'), verbosity=0)
    assert before > 0 and agent.get_attribute('substrate.transport')['drug'].concentration == before


def test_models_reach_the_fields_through_the_context():
    from lattics import Agent, Substrate
    from lattics.models import BaseModel

    class Sensing(BaseModel):
        def __init__(self):
            super().__init__((1, 'hour'))
            self.seen = []

        def update_agent(self, agent, ctx):
            self.seen.append(ctx.substrate('drug').concentration_at(None))

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug'))
    model = Sensing()
    sim.add_model(model)
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    sim.add_agent(agent)
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert model.seen == [0.0]


def test_every_substrate_has_its_own_interval(capsys):
    from lattics import Substrate

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug', interval=(30, 'sec'), diffusion_coefficient=(1, 'um2/sec')))
    sim.add_substrate(Substrate('oxygen', interval=(1, 'hour')))
    sim.run(time=(1, 'hour'), dt=(30, 'sec'), verbosity=1)
    out = capsys.readouterr().out
    assert "Substrate 'drug'" in out and 'every 30 sec' in out and "Substrate 'oxygen'" in out
    assert "HomogeneousSpace: the diffusion coefficient of 'drug' is not used" in out


def test_exchange_with_an_undefined_substrate_is_reported():
    from lattics import Agent, Substrate, SubstrateInfo

    sim = _homogeneous_simulation()
    sim.add_substrate(Substrate('drug'))
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    agent.set_attribute('substrate.transport', {'drgu': SubstrateInfo(type='flux')})
    sim.add_agent(agent)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    messages = [str(w.message) for w in caught]
    assert messages == [
        "The substrate 'drgu' is exchanged by 1 agents, but it is not added to the simulation. "
        "Did you mean 'drug'?"
    ]


# --- two-dimensional lattice field ---------------------------------------------------------------------------


def _diffuse(concentration, r, steps=1):
    from lattics._kernels.diffusion import implicit_diffusion_2d, thomas_factors

    factors = (r, *thomas_factors(concentration.shape[0], r), *thomas_factors(concentration.shape[1], r))
    for _ in range(steps):
        concentration = implicit_diffusion_2d(concentration, *factors)
    return concentration


def test_diffusion_conserves_the_mass_and_keeps_a_uniform_field():
    rng = np.random.default_rng(0)
    field = rng.random((17, 23))
    diffused = _diffuse(field, r=3.0, steps=5)
    uniform = _diffuse(np.full((17, 23), 5.0), r=3.0)
    assert math.isclose(diffused.sum(), field.sum(), rel_tol=1e-12) and diffused.min() >= 0
    assert np.allclose(uniform, 5.0, rtol=1e-12, atol=0)


def test_diffusion_is_symmetric_and_spreads_with_the_expected_variance():
    # An implicit step on an unbounded line increases the variance by 2 r cells^2
    field = np.zeros((101, 101))
    field[50, 50] = 1.0
    diffused = _diffuse(field, r=0.5, steps=4)
    offsets = np.arange(101) - 50
    variance = (diffused.sum(axis=1) * offsets**2).sum()
    assert np.allclose(diffused, diffused[::-1, :]) and np.allclose(diffused, diffused[:, ::-1])
    assert np.allclose(diffused, diffused.T) and math.isclose(variance, 2 * 0.5 * 4, rel_tol=1e-6)


def _pk_scenario(space_class, with_agents=False):
    from lattics import Agent, Event, Simulation, Substrate, SubstrateInfo, SubstrateNode
    from lattics.kinetics import FirstOrderDecay, MichaelisMentenDecay

    sim = Simulation(seed=3)
    sim.set_space(space_class(size=((750, 'um'), (750, 'um')), grid_spacing=(15, 'um'), interval=(1, 'hour')))
    kinetics = [FirstOrderDecay((0.05, '1/hour')), MichaelisMentenDecay((10.0, '1/hour'), km=50.0)]
    sim.add_substrate(
        Substrate('drug', diffusion_coefficient=(2.0, 'um2/sec'), kinetics=kinetics, interval=(1, 'min'))
    )
    vessels = SubstrateNode(
        volume=(0.84375, 'mm3'), transport={'drug': SubstrateInfo(type='flux', passive_rate=0.5)}
    )
    organs = SubstrateNode(
        volume=(8.4375, 'mm3'), transport={'drug': SubstrateInfo(type='flux', release_rate=1.0, uptake_rate=5.0)}
    )
    sim.add_substrate_node(vessels)
    sim.add_substrate_node(organs)
    if with_agents:
        for _ in range(20):
            agent = Agent()
            agent.set_attribute('space.volume', 3000)
            agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', uptake_rate=1e-2)})
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sim.add_agent(agent)

    def dose(simulation):
        vessels.get_attribute('substrate.transport')['drug'].concentration = 1000.0

    sim.add_event(Event(time=(0, 'hour'), handler=dose))
    sim.run(time=(12, 'hour'), dt=(1, 'min'), verbosity=0)
    return sim, vessels, organs


def test_uniform_lattice_field_equals_the_homogeneous_field():
    from lattics import HomogeneousSpace, Lattice2DSpace

    results = []
    for space_class in (HomogeneousSpace, Lattice2DSpace):
        sim, vessels, organs = _pk_scenario(space_class)
        field = np.asarray(sim.substrate_field('drug').concentration)
        nodes = [n.get_attribute('substrate.transport')['drug'].concentration for n in (vessels, organs)]
        results.append((field.mean(), *nodes))
        if space_class is Lattice2DSpace:
            assert field.max() - field.min() <= 1e-9 * field.mean()
    homogeneous, lattice = results
    assert all(math.isclose(h, la, rel_tol=1e-9) for h, la in zip(homogeneous, lattice))


def test_agents_exchange_with_their_own_site():
    from lattics import Lattice2DSpace

    sim, _, _ = _pk_scenario(Lattice2DSpace, with_agents=True)
    field = sim.substrate_field('drug').concentration
    sites = [tuple(a.get_attribute('space.position')) for a in sim.agents]
    # the agents take up the drug, so their sites are below the mean of the field
    assert np.mean([field[s] for s in sites]) < field.mean()


def test_exchange_and_diffusion_conserve_the_mass_on_the_lattice():
    import lattics
    from lattics import Agent, Substrate, SubstrateInfo, SubstrateNode

    sim = lattics.Simulation(seed=1)
    sim.set_space(lattics.Lattice2DSpace(size=((200, 'um'), (200, 'um')), interval=(1, 'hour')))
    sim.add_substrate(Substrate('drug', diffusion_coefficient=(1.0, 'um2/sec'), interval=(1, 'min')))
    source = SubstrateNode(
        volume=(1e5, 'um3'), transport={'drug': SubstrateInfo(type='flux', passive_rate=1.0)}, position=(0, 0)
    )
    source.get_attribute('substrate.transport')['drug'].concentration = 100.0
    sim.add_substrate_node(source)
    agent = Agent()
    agent.set_attribute('space.volume', 3000)
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', uptake_rate=1e-2)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.add_agent(agent)
    sim.run(time=(3, 'hour'), dt=(1, 'min'), verbosity=0)
    field = sim.substrate_field('drug').concentration

    def mass(item, volume):
        return item.get_attribute('substrate.transport')['drug'].concentration * volume

    total = field.sum() * 20 * 20 * 20 + mass(source, 1e5) + mass(agent, 3000)
    assert math.isclose(total, 100.0 * 1e5, rel_tol=1e-9) and field[0, 0] > field[9, 9]


def test_fixed_node_sets_the_concentration_of_its_site():
    import lattics
    from lattics import Substrate, SubstrateInfo, SubstrateNode

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(size=((200, 'um'), (200, 'um'))))
    sim.add_substrate(Substrate('oxygen'))
    sim.add_substrate_node(
        SubstrateNode(transport={'oxygen': SubstrateInfo(type='fixed', concentration=7.0)}, position=(2, 3))
    )
    sim.run(time=(1, 'min'), dt=(1, 'min'), verbosity=0)
    field = sim.substrate_field('oxygen')
    assert field.concentration_at((2, 3)) == 7.0 and field.concentration.sum() == 7.0


def test_node_outside_the_lattice_is_rejected():
    import lattics
    from lattics import Substrate, SubstrateInfo, SubstrateNode

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(size=((200, 'um'), (200, 'um'))))
    sim.add_substrate(Substrate('oxygen'))
    sim.add_substrate_node(
        SubstrateNode(transport={'oxygen': SubstrateInfo(type='fixed', concentration=7.0)}, position=(10, 0))
    )
    with pytest.raises(ValueError, match='outside the lattice'):
        sim.run(time=(1, 'min'), dt=(1, 'min'), verbosity=0)


def test_summary_reports_nodes_spread_over_the_lattice(capsys):
    import lattics
    from lattics import Substrate, SubstrateInfo, SubstrateNode

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(size=((200, 'um'), (200, 'um'))))
    sim.add_substrate(Substrate('drug'))
    sim.add_substrate_node(SubstrateNode(volume=(1, 'mm3'), transport={'drug': SubstrateInfo(type='flux')}))
    sim.run(time=(1, 'min'), dt=(1, 'min'), verbosity=1)
    out = capsys.readouterr().out
    assert "Substrate 'drug': 1 substrate nodes without a position are spread uniformly" in out
