import math
import warnings

import pytest


def _simulation(strict=False):
    from lattics import HomogeneousSpace, Simulation

    sim = Simulation(strict=strict)
    sim.set_space(HomogeneousSpace(volume=(1, 'mm3'), interval=(1, 'hour')))
    return sim


def _add(sim, agent):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim.add_agent(agent)
    return [str(w.message) for w in caught]


def test_velocity_conversion():
    from lattics import convert_velocity

    assert math.isclose(convert_velocity(0.6, 'um/min', 'um/ms'), 1e-5)
    assert math.isclose(convert_velocity(1.0, 'mm/hour', 'um/sec'), 1 / 3.6)


def test_flow_rate_conversion():
    from lattics import convert_flow_rate

    assert math.isclose(convert_flow_rate(1.0, 'nL/sec', 'um3/ms'), 1000.0)
    with pytest.raises(ValueError):
        convert_flow_rate(1.0, 'um3', 'um3/ms')


def test_attributes_are_converted_to_the_internal_unit_when_the_agent_is_added():
    from lattics import Agent
    from lattics.attributes import internal_unit

    agent = Agent()
    agent.set_attribute('space.volume', (3, 'nL'))
    agent.set_attribute('mechanics.motility', (0.6, 'um/min'))
    messages = _add(_simulation(), agent)
    assert messages == [] and internal_unit('space.volume') == 'um3'
    assert agent.get_attribute('space.volume') == 3e6 and math.isclose(
        agent.get_attribute('mechanics.motility'), 1e-5
    )


def test_attribute_without_unit_is_taken_in_the_internal_unit_with_a_warning():
    from lattics import Agent

    agent = Agent()
    agent.set_attribute('space.volume', 3000)
    messages = _add(_simulation(), agent)
    assert agent.get_attribute('space.volume') == 3000
    assert messages == [
        "The agent attribute 'space.volume' is given without a unit, it is taken in um3. "
        "Give it as (value, unit), for example (1, 'um3')."
    ]


def test_strict_simulation_rejects_quantities_without_unit():
    from lattics import Agent

    agent = Agent()
    agent.set_attribute('space.volume', 3000)
    with pytest.raises(ValueError, match="'space.volume' is given without a unit"):
        _simulation(strict=True).add_agent(agent)


def test_daughters_do_not_warn():
    from lattics import Agent
    from lattics.models import TimeBasedCellCycleModel

    sim = _simulation()
    sim.add_model(TimeBasedCellCycleModel((1, 'hour'), distribution='fixed'))
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    agent.set_attribute('cellcycle.mean_length', (2, 'hour'))
    sim.add_agent(agent)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim.run(time=(5, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert len(sim.agents) > 1 and caught == []


def test_substrate_info_converts_its_rates():
    from lattics import SubstrateInfo

    info = SubstrateInfo(type='flux', passive_rate=(1.0, 'nL/sec'), uptake_rate=(60, 'um3/min'))
    assert (info.passive_rate, info.uptake_rate, info.release_rate, info.unitless_rates) == (
        1000.0,
        0.001,
        0.0,
        (),
    )


def test_substrate_info_without_units_warns_when_added():
    from lattics import Agent, Substrate, SubstrateInfo, SubstrateNode

    sim = _simulation()
    sim.add_substrate(Substrate('drug'))
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    agent.set_attribute('substrate.transport', {'drug': SubstrateInfo(type='flux', passive_rate=0.5)})
    messages = _add(sim, agent)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim.add_substrate_node(SubstrateNode(transport={'drug': SubstrateInfo(type='flux', release_rate=1.0)}))
    expected = (
        "The rates passive_rate of the SubstrateInfo of 'drug' are given without a unit, they are taken in um3/ms."
    )
    assert len(messages) == 1 and messages[0].startswith(expected)
    assert len(caught) == 1 and 'release_rate' in str(caught[0].message)


def test_model_rates_need_their_unit():
    from lattics.models import ConcentrationDependentToxicityModel, StochasticTransitionModel

    with pytest.raises(TypeError, match="1.0, '1/day'"):
        ConcentrationDependentToxicityModel((1, 'hour'), 'drug', max_rate=1.0, median_effective_concentration=1.0)
    with pytest.raises(TypeError):
        StochasticTransitionModel((1, 'hour'), condition=('a', 1), end_states={'b': 2}, rate=1.0)


def test_space_lengths_need_their_unit():
    import lattics

    with pytest.raises(TypeError, match='grid spacing'):
        lattics.create_simulation_2d(dimensions=((200, 'um'), (200, 'um')), grid_spacing=20)
