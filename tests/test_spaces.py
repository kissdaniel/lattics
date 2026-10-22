import math
import warnings

import numpy as np
import pytest

SPACE_ARGUMENTS = dict(size=((200, 'um'), (200, 'um')), thickness=(15, 'um'), grid_spacing=(20, 'um'))


def _cell_attributes():
    return {
        'space.volume': (3000, 'um3'),
        'cellcycle.phase_g1_mean': (10, 'hour'),
        'cellcycle.phase_s_mean': (6, 'hour'),
        'cellcycle.phase_g2_mean': (3, 'hour'),
        'cellcycle.phase_m_mean': (1, 'hour'),
        'cellcycle.inhibition_density_limit': 0.8,
        'mechanics.motility': (0, 'um/ms'),
        'mechanics.binding_affinity': 0,
        'mechanics.displacement_limit': 1,
    }


def _run_scenario(space_class):
    import lattics
    from lattics.models import MultiPhaseTimeBasedCellCycleModel

    sim = lattics.Simulation()
    sim.set_space(space_class(interval=(1, 'hour'), **SPACE_ARGUMENTS))
    sim.add_model(MultiPhaseTimeBasedCellCycleModel(update_interval=(1, 'hour')))
    lattics.populate_agents(sim, count=10, attributes=_cell_attributes())
    sim.run(time=(2, 'day'), dt=(1, 'hour'), verbosity=0)
    return sim


@pytest.mark.parametrize('space_name', ['HomogeneousSpace', 'Lattice2DSpace'])
def test_same_scenario_runs_in_both_spaces(space_name):
    import lattics

    sim = _run_scenario(getattr(lattics, space_name))
    densities = [a.get_attribute('space.local_density') for a in sim.agents]
    assert len(sim.agents) > 10 and all(0.0 <= d <= 1.0 for d in densities)


def test_spaces_compute_the_same_volume():
    from lattics import HomogeneousSpace, Lattice2DSpace

    expected = 200 * 200 * 15
    actual = [HomogeneousSpace(**SPACE_ARGUMENTS).volume, Lattice2DSpace(**SPACE_ARGUMENTS).volume]
    assert actual == [expected, expected]


def test_thickness_defaults_to_grid_spacing():
    from lattics import HomogeneousSpace

    space = HomogeneousSpace(size=((100, 'um'), (100, 'um')), grid_spacing=(10, 'um'))
    assert space.volume == 100 * 100 * 10


def test_lattice_size_must_be_a_multiple_of_the_grid_spacing():
    from lattics import Lattice2DSpace

    with pytest.raises(ValueError):
        Lattice2DSpace(size=((750, 'um'), (750, 'um')), grid_spacing=(20, 'um'))


def test_lattice_places_agents_without_position_on_free_sites():
    import lattics

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(**SPACE_ARGUMENTS))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=100, attributes={'space.volume': (3000, 'um3')})
    positions = {tuple(int(v) for v in a.get_attribute('space.position')) for a in sim.agents}
    assert len(positions) == 100 and all(0 <= x < 10 and 0 <= y < 10 for x, y in positions)


def test_lattice_raises_when_full():
    import lattics

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(size=((40, 'um'), (40, 'um')), grid_spacing=(20, 'um')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=4, attributes={'space.volume': (3000, 'um3')})
        with pytest.raises(ValueError):
            lattics.populate_agents(sim, count=1, attributes={'space.volume': (3000, 'um3')})


def test_agents_added_before_the_space_are_placed():
    import lattics

    sim = lattics.Simulation()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=3, attributes={'space.volume': (3000, 'um3')})
        sim.set_space(lattics.Lattice2DSpace(**SPACE_ARGUMENTS))
    assert all(a.has_attribute('space.position') for a in sim.agents)


def test_homogeneous_local_density_is_the_occupied_fraction_of_the_free_volume():
    import lattics

    sim = lattics.Simulation()
    sim.set_space(
        lattics.HomogeneousSpace(size=((10, 'um'), (10, 'um')), thickness=(10, 'um'), interval=(1, 'hour'))
    )
    lattics.populate_agents(sim, count=2, attributes={'space.volume': (100, 'um3')})
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    # (200 - 100) / (1000 - 100): the other agent's volume relative to the volume free for this agent
    expected = 1 / 9
    actual = sim.agents[0].get_attribute('space.local_density')
    assert math.isclose(actual, expected)


def test_spaces_require_the_agent_volume():
    import lattics

    for space_class in (lattics.HomogeneousSpace, lattics.Lattice2DSpace):
        sim = lattics.Simulation()
        sim.set_space(space_class(**SPACE_ARGUMENTS))
        with pytest.raises(ValueError):
            sim.add_agent(lattics.Agent())


def test_homogeneous_space_accepts_volume_instead_of_size():
    from lattics import HomogeneousSpace

    space = HomogeneousSpace(volume=(0.6, 'mm3'))
    assert space.volume == 0.6e9


def test_size_and_volume_together_are_rejected():
    from lattics import HomogeneousSpace

    with pytest.raises(ValueError):
        HomogeneousSpace(volume=(0.6, 'mm3'), **SPACE_ARGUMENTS)


def test_lattice_space_needs_size_not_volume():
    from lattics import Lattice2DSpace

    with pytest.raises(ValueError):
        Lattice2DSpace(volume=(0.6, 'mm3'))


def test_volume_conversion():
    from lattics import convert_volume

    expected = [1e9, 1.0, 1e3]
    actual = [convert_volume(1, 'mm3', 'um3'), convert_volume(1, 'uL', 'mm3'), convert_volume(1, 'mL', 'uL')]
    assert actual == expected


@pytest.mark.parametrize('space_name', ['HomogeneousSpace', 'Lattice2DSpace'])
def test_removal_comes_before_division(space_name):
    import lattics

    sim = lattics.Simulation()
    sim.set_space(getattr(lattics, space_name)(interval=(1, 'hour'), **SPACE_ARGUMENTS))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=3, attributes={'space.volume': (3000, 'um3')})
    doomed = sim.agents[0]
    doomed.set_attribute('space.removal_requested', True)
    doomed.set_attribute('space.division_requested', True)
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert len(sim.agents) == 2 and sim.removed_agents == [doomed]


@pytest.mark.parametrize('space_name', ['HomogeneousSpace', 'Lattice2DSpace'])
def test_division_sets_the_division_time_of_both_agents(space_name):
    import lattics

    sim = lattics.Simulation()
    sim.set_space(getattr(lattics, space_name)(interval=(1, 'hour'), **SPACE_ARGUMENTS))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=1, attributes={'space.volume': (3000, 'um3')})
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    mother = sim.agents[0]
    mother.set_attribute('space.division_requested', True)
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    expected = [(False, 3600000.0), (False, 3600000.0)]
    actual = [
        (a.get_attribute('space.division_requested'), a.get_attribute('space.last_division_time'))
        for a in sim.agents
    ]
    assert actual == expected


def _growing_lattice(limits):
    import lattics
    from lattics.models import MultiPhaseTimeBasedCellCycleModel

    sim = lattics.Simulation(seed=4)
    sim.set_space(lattics.Lattice2DSpace(size=((300, 'um'), (240, 'um')), interval=(30, 'min')))
    sim.add_model(MultiPhaseTimeBasedCellCycleModel(update_interval=(30, 'min')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for limit in limits:
            attributes = dict(_cell_attributes(), **{'mechanics.motility': (5e-6, 'um/ms')})
            attributes['mechanics.displacement_limit'] = limit
            attributes['space.volume'] = (300, 'um3')
            lattics.populate_agents(sim, count=8, attributes=attributes)
    sim.run(time=(2, 'day'), dt=(30, 'min'), verbosity=0)
    return sim


def test_occupied_sites_follow_the_agents():
    sim = _growing_lattice(limits=(1, 2))
    space = sim._space
    expected = np.not_equal(space._agent_layer, None)
    positions = sorted(tuple(int(v) for v in a.get_attribute('space.position')) for a in sim.agents)
    assert np.array_equal(space._occupied, expected) and positions == sorted(map(tuple, np.argwhere(expected)))


def test_local_density_equals_the_occupied_fraction_of_the_disk():
    sim = _growing_lattice(limits=(1, 2, 3))
    occupied = sim._space._occupied
    i, j = np.ogrid[: occupied.shape[0], : occupied.shape[1]]
    for agent in sim.agents:
        x, y = agent.get_attribute('space.position')
        limit = agent.get_attribute('mechanics.displacement_limit')
        disk = (i - x) ** 2 + (j - y) ** 2 <= limit**2
        expected = (np.sum(disk & occupied) - 1) / (np.sum(disk) - 1)
        assert agent.get_attribute('space.local_density') == expected
