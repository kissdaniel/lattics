import math
import warnings

import numpy as np
import pytest

HOUR = 3600000.0


def _simulation(count=4):
    from lattics import Agent, HomogeneousSpace, Simulation

    sim = Simulation(seed=1)
    sim.set_space(HomogeneousSpace(volume=(1, 'mm3'), interval=(1, 'hour')))
    agents = []
    for i in range(count):
        agent = Agent()
        agent.set_attribute('space.volume', (1, 'um3'))
        agent.set_attribute('experiment.group', 'a' if i % 2 == 0 else 'b')
        sim.add_agent(agent)
        agents.append(agent)
    sim.declare('experiment.group')
    return sim, agents


def _remove(ctx, agent):
    ctx.remove_agent(agent)


def test_agent_count_with_a_condition():
    from lattics.recorders import AgentCount

    sim, _ = _simulation()
    everyone = sim.add_recorder(AgentCount(interval=(1, 'hour')))
    group_a = sim.add_recorder(AgentCount(interval=(1, 'hour'), where=('experiment.group', 'a')))
    group_b = sim.add_recorder(
        AgentCount(interval=(1, 'hour'), where=lambda a: a.get_attribute('experiment.group') == 'b')
    )
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert (everyone.series.values, group_a.series.values, group_b.series.values) == ([4, 4], [2, 2], [2, 2])


def test_removed_agents_are_counted_even_between_records():
    from lattics import Event
    from lattics.recorders import AgentCount

    sim, agents = _simulation()
    counts = sim.add_recorder(AgentCount(interval=(4, 'hour'), where=('experiment.group', 'a')))
    for hour, agent in zip((1, 2, 3), agents[:3]):
        sim.add_event(Event(time=(hour, 'hour'), handler=_remove, agent=agent))
    sim.run(time=(4, 'hour'), dt=(1, 'hour'), verbosity=0)
    # agents 0 and 2 belong to group a; they were removed at 1 h and 3 h, between the records at 0 h and 4 h
    assert counts.series.values == [2, 0] and counts.removed.values == [0, 2]


def test_substrate_level_statistics_and_position():
    import lattics
    from lattics.events import SetConcentration
    from lattics.recorders import SubstrateLevel

    sim = lattics.Simulation()
    sim.set_space(lattics.Lattice2DSpace(size=((60, 'um'), (40, 'um'))))
    sim.add_substrate(lattics.Substrate('drug'))
    sim.add_event(SetConcentration(time=(0, 'hour'), substrate='drug', concentration=2.0))
    mean = sim.add_recorder(SubstrateLevel('drug', interval=(1, 'hour')))
    total = sim.add_recorder(SubstrateLevel('drug', interval=(1, 'hour'), statistic='sum'))
    corner = sim.add_recorder(SubstrateLevel('drug', interval=(1, 'hour'), position=(2, 1)))
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    # the event sets the field after the record at 0 h
    assert (
        mean.series.values == [0.0, 2.0] and total.series.values == [0.0, 12.0] and corner.series.values[-1] == 2.0
    )


def test_unknown_statistic_is_rejected():
    from lattics.recorders import SubstrateLevel

    with pytest.raises(ValueError, match='median'):
        SubstrateLevel('drug', statistic='median')


def test_snapshot_records_every_attribute_by_default():
    from lattics.recorders import Snapshot

    sim, agents = _simulation(count=2)
    full = sim.add_recorder(Snapshot(interval=(1, 'hour')))
    narrow = sim.add_recorder(Snapshot(interval=(1, 'hour'), attributes=['experiment.group']))
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    first = full.frames[0].agents[0]
    assert first['id'] == agents[0].id and set(agents[0]._attributes) <= set(first)
    assert narrow.frames[0].agents[0] == {'id': agents[0].id, 'experiment.group': 'a'}


def test_snapshot_frames_are_copies():
    from lattics.recorders import Snapshot

    sim, agents = _simulation(count=1)
    snapshot = sim.add_recorder(Snapshot(interval=(1, 'hour')))
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    agents[0].set_attribute('experiment.group', 'changed')
    assert snapshot.frame_at((0, 'hour')).agents[0]['experiment.group'] == 'a'


def test_snapshot_keeps_the_removed_agents_and_the_fields(tmp_path):
    import lattics
    from lattics import Event
    from lattics.events import SetConcentration
    from lattics.recorders import Snapshot

    sim = lattics.Simulation(seed=1)
    sim.set_space(lattics.Lattice2DSpace(size=((60, 'um'), (40, 'um')), interval=(1, 'hour')))
    sim.add_substrate(lattics.Substrate('drug'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=2, attributes={'space.volume': (100, 'um3')})
    doomed = sim.agents[0]
    sim.add_event(Event(time=(1, 'hour'), handler=_remove, agent=doomed))
    sim.add_event(SetConcentration(time=(1, 'hour'), substrate='drug', concentration=3.0))
    snapshot = sim.add_recorder(Snapshot(interval=(2, 'hour')))
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    snapshot.save(tmp_path / 'frames.pkl')
    loaded = lattics.recorders.Snapshot.load(tmp_path / 'frames.pkl')
    last = loaded.frame_at((2, 'hour'))
    assert loaded.dimensions == (3, 2) and [a['id'] for a in last.removed] == [doomed.id]
    assert len(last.agents) == 1 and np.all(last.fields['drug'] == 3.0)


def test_snapshot_load_rejects_other_files(tmp_path):
    import pickle

    from lattics.recorders import Snapshot

    path = tmp_path / 'other.pkl'
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        Snapshot.load(path)


def test_set_concentration_of_a_node():
    from lattics import Substrate, SubstrateInfo, SubstrateNode
    from lattics.events import SetConcentration

    sim, _ = _simulation(count=0)
    sim.add_substrate(Substrate('drug'))
    node = SubstrateNode(volume=(1, 'mm3'), transport={'drug': SubstrateInfo(type='flux')})
    sim.add_substrate_node(node)
    sim.add_event(SetConcentration(time=(30, 'min'), substrate='drug', concentration=5.0, node=node))
    sim.run(time=(1, 'hour'), dt=(30, 'min'), verbosity=0)
    assert node.get_attribute('substrate.transport')['drug'].concentration == 5.0


def test_event_handlers_receive_the_context_and_the_parameters():
    from lattics import Event

    seen = []

    def handler(ctx, label):
        seen.append((ctx.time, label, len(ctx.population)))

    sim, _ = _simulation()
    sim.add_event(Event(time=(1, 'hour'), handler=handler, label='x'))
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert seen == [(HOUR, 'x', 4)]


def test_rmse_uses_the_common_time_points():
    from lattics.utils.time_series import TimeSeries

    a = TimeSeries([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    b = TimeSeries([1, 3], [2.0, 6.0])
    assert math.isclose(a.rmse(b), math.sqrt((0**2 + 2**2) / 2))
    with pytest.raises(ValueError):
        a.rmse(TimeSeries([10], [1.0]))


def test_time_series_csv_round_trip(tmp_path):
    from lattics.utils.time_series import TimeSeries

    series = TimeSeries([0.0, HOUR], [1.5, 2.5])
    series.to_csv(tmp_path / 'series.csv', time_unit='hour')
    loaded = TimeSeries.from_csv(tmp_path / 'series.csv', time_unit='hour')
    assert loaded.get_timepoints('ms') == [0.0, HOUR] and loaded.values == [1.5, 2.5]


def test_visualization_without_matplotlib_explains_the_extra(monkeypatch):
    import sys

    from lattics.recorders import Snapshot
    from lattics.utils.visualize import show_agents_and_substrates_2d

    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    with pytest.raises(ImportError, match=r'lattics\[viz\]'):
        show_agents_and_substrates_2d(Snapshot())
