import warnings

import pytest

HOUR = (1, 'hour')


def _simulation(models=(), strict=False, space=None, **agent_attributes):
    from lattics import Agent, HomogeneousSpace, Simulation

    sim = Simulation(strict=strict)
    sim.set_space(space or HomogeneousSpace(interval=HOUR))
    for model in models:
        sim.add_model(model)
    for _ in range(3):
        agent = Agent()
        agent.set_attribute('space.volume', (1, 'um3'))
        for name, value in agent_attributes.items():
            agent.set_attribute(name.replace('__', '.'), value)
        with warnings.catch_warnings():
            # the lattice warns about the mechanics defaults
            warnings.simplefilter('ignore')
            sim.add_agent(agent)
    return sim


def _run(sim, verbosity=0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim.run(time=HOUR, dt=HOUR, verbosity=verbosity)
    return [str(w.message) for w in caught]


def _reading_model(name):
    from lattics.models import BaseModel

    class Reader(BaseModel):
        reads = (name,)

        def update_agent(self, agent, ctx):
            pass

    return Reader()


def test_consistent_attributes_give_no_warning():
    from lattics.models import TimeBasedCellCycleModel

    sim = _simulation([TimeBasedCellCycleModel(HOUR)], cellcycle__mean_length=(10, 'hour'))
    assert _run(sim) == []


def test_attribute_read_but_never_written_is_reported():
    sim = _simulation([_reading_model('vessel.distance')])
    messages = _run(sim)
    assert len(messages) == 1 and "'vessel.distance' is read by Reader" in messages[0] and '3 of 3' in messages[0]


def test_missing_attribute_suggests_a_similar_agent_attribute():
    sim = _simulation([_reading_model('vessel.distance')], vessel__distnce=5)
    messages = _run(sim)
    # one message for the typo, not a second one for the unused attribute
    assert len(messages) == 1 and "Did you mean 'vessel.distnce'?" in messages[0]


def test_misspelled_agent_attribute_is_reported():
    from lattics.models import MultiPhaseTimeBasedCellCycleModel

    phases = {f'cellcycle__phase_{p}_mean': (1, 'hour') for p in ('g1', 's', 'g2', 'm')}
    sim = _simulation([MultiPhaseTimeBasedCellCycleModel(HOUR)], cellcycle__inhibition_density_limt=0.5, **phases)
    messages = _run(sim)
    assert len(messages) == 1 and "Did you mean 'cellcycle.inhibition_density_limit'?" in messages[0]


def test_renamed_attribute_is_reported():
    messages = _run(_simulation(motility=1))
    assert messages == [
        "The agent attribute 'motility' is not used any more, it was renamed to 'mechanics.motility'."
    ]


def test_strict_simulation_raises():
    sim = _simulation([_reading_model('vessel.distance')], strict=True)
    with pytest.raises(ValueError, match='vessel.distance'):
        sim.run(time=HOUR, dt=HOUR, verbosity=0)


def test_declared_attribute_is_intentional():
    sim = _simulation(experiment__labl='A')
    sim.declare('experiment.labl')
    assert _run(sim) == []


def test_declared_default_is_given_to_every_agent():
    from lattics import Agent

    sim = _simulation()
    sim.declare('experiment.doses', default=[])
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    sim.add_agent(agent)
    doses = [a.get_attribute('experiment.doses') for a in sim.agents]
    doses[0].append(1)
    assert len(doses) == 4 and doses[1:] == [[], [], []]


def test_built_in_attribute_cannot_be_declared():
    from lattics import Simulation

    with pytest.raises(ValueError):
        Simulation().declare('space.volume', default=1)


def test_summary_lists_the_attributes_the_space_ignores(capsys):
    sim = _simulation(mechanics__motility=0, experiment__label='A')
    sim.declare('experiment.label')
    _run(sim, verbosity=1)
    assert 'Agent attributes not used by any component: mechanics.motility\n' in capsys.readouterr().out


def test_summary_reports_random_placement(capsys):
    import lattics

    sim = _simulation(space=lattics.Lattice2DSpace(size=((200, 'um'), (200, 'um')), interval=HOUR))
    _run(sim, verbosity=1)
    _run(sim, verbosity=1)
    out = capsys.readouterr().out
    # reported once, at the first run after the placement
    assert out.count("3 agents without 'space.position' placed on random free sites") == 1


def test_missing_required_attribute_is_named():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.set_space(HomogeneousSpace())
    sim.add_model(TimeBasedCellCycleModel())
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    with pytest.raises(
        AttributeError, match="TimeBasedCellCycleModel requires the agent attribute.*'cellcycle.mean_length'"
    ):
        sim.add_agent(agent)
