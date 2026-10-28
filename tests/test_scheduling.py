import math
import warnings

HOUR = 3600000.0
MINUTE = 60000.0


def _recording_simulation(update_interval):
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import BaseModel

    class Recording(BaseModel):
        def __init__(self, update_interval):
            super().__init__(update_interval)
            self.calls = []

        def update_agent(self, agent, ctx):
            self.calls.append((ctx.time, ctx.dt))

    sim = Simulation()
    sim.set_space(HomogeneousSpace(interval=update_interval))
    model = Recording(update_interval)
    sim.add_model(model)
    agent = Agent()
    agent.set_attribute('space.volume', (1, 'um3'))
    sim.add_agent(agent)
    return sim, model


def test_interval_is_rounded_up_to_a_multiple_of_the_time_step():
    sim, model = _recording_simulation((45, 'min'))
    sim.run(time=(3, 'hour'), dt=(30, 'min'), verbosity=0)
    expected = [(0, HOUR), (HOUR, HOUR), (2 * HOUR, HOUR)]
    assert model.calls == expected


def test_interval_shorter_than_the_time_step_uses_the_time_step():
    sim, model = _recording_simulation((10, 'min'))
    sim.run(time=(1, 'hour'), dt=(30, 'min'), verbosity=0)
    expected = [(0, 30 * MINUTE), (30 * MINUTE, 30 * MINUTE)]
    assert model.calls == expected


def test_no_interval_means_every_time_step():
    sim, model = _recording_simulation(None)
    sim.run(time=(1, 'hour'), dt=(20, 'min'), verbosity=0)
    expected = [(0, 20 * MINUTE), (20 * MINUTE, 20 * MINUTE), (40 * MINUTE, 20 * MINUTE)]
    assert model.calls == expected


def test_consecutive_runs_continue_the_schedule():
    sim_once, model_once = _recording_simulation((1, 'hour'))
    sim_once.run(time=(4, 'hour'), dt=(30, 'min'), verbosity=0)
    sim_twice, model_twice = _recording_simulation((1, 'hour'))
    sim_twice.run(time=(90, 'min'), dt=(30, 'min'), verbosity=0)
    sim_twice.run(time=(150, 'min'), dt=(30, 'min'), verbosity=0)
    assert model_twice.calls == model_once.calls and sim_twice.time == sim_once.time


def test_loop_jumps_over_time_points_where_nothing_is_due():
    sim, model = _recording_simulation((1, 'hour'))
    points = []
    original = sim._execute_events
    sim._execute_events = lambda now: (points.append(now), original(now))
    sim.run(time=(1, 'day'), dt=(1, 'sec'), verbosity=0)
    assert len(model.calls) == 24 and len(points) == 24


def test_events_run_at_their_exact_time():
    from lattics import Event

    sim, model = _recording_simulation((1, 'hour'))
    seen = []
    sim.add_event(Event(time=(45, 'min'), handler=lambda ctx: seen.append(ctx.time)))
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert seen == [45 * MINUTE] and model.calls == [(0, HOUR), (HOUR, HOUR)]


def test_events_with_the_same_time_keep_their_order():
    from lattics import Event

    sim, _ = _recording_simulation(None)
    seen = []
    for label in 'abc':
        sim.add_event(Event(time=(10, 'min'), handler=lambda ctx, label=label: seen.append(label)))
    sim.run(time=(1, 'hour'), dt=(30, 'min'), verbosity=0)
    assert seen == ['a', 'b', 'c']


def test_recorders_record_at_their_interval_and_at_the_end():
    from lattics.recorders import AgentCount

    sim, _ = _recording_simulation(None)
    counts = sim.add_recorder(AgentCount(interval=(1, 'hour')))
    sim.run(time=(3, 'hour'), dt=(30, 'min'), verbosity=0)
    expected = [0, HOUR, 2 * HOUR, 3 * HOUR]
    actual = counts.series.timepoints
    assert actual == expected


def test_cell_division_is_not_delayed():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    counts = []
    for hours in (9, 10):
        sim = Simulation()
        sim.set_space(HomogeneousSpace(interval=(1, 'hour')))
        sim.add_model(TimeBasedCellCycleModel(update_interval=(1, 'hour'), distribution='fixed'))
        agent = Agent()
        agent.set_attribute('space.volume', (1, 'um3'))
        agent.set_attribute('cellcycle.mean_length', (10, 'hour'))
        sim.add_agent(agent)
        sim.run(time=(hours, 'hour'), dt=(1, 'hour'), verbosity=0)
        counts.append(len(sim.agents))
    # a 10-hour cycle completes exactly after 10 hours
    assert counts == [1, 2]


def test_time_step_below_the_substrate_interval_does_not_change_the_result():
    from lattics import Event, HomogeneousSpace, Simulation, Substrate, SubstrateInfo, SubstrateNode
    from lattics.kinetics import FirstOrderDecay

    def run(dt):
        sim = Simulation()
        sim.set_space(HomogeneousSpace(volume=(1, 'mm3'), interval=(1, 'hour')))
        sim.add_substrate(Substrate('drug', kinetics=FirstOrderDecay(rate=(1e-7, '1/ms')), interval=(30, 'sec')))
        node = SubstrateNode(
            volume=(1e8, 'um3'), transport={'drug': SubstrateInfo(type='flux', passive_rate=(0.5, 'um3/ms'))}
        )
        sim.add_substrate_node(node)

        def dose(ctx):
            node.get_attribute('substrate.transport')['drug'].concentration = 1000.0

        sim.add_event(Event(time=(0, 'sec'), handler=dose))
        sim.run(time=(2, 'hour'), dt=dt, verbosity=0)
        return sim.substrate_field('drug').concentration

    assert run((10, 'sec')) == run((30, 'sec'))


def test_summary_shows_the_effective_intervals(capsys):
    sim, _ = _recording_simulation((45, 'min'))
    sim.run(time=(1, 'hour'), dt=(30, 'min'), verbosity=1)
    out = capsys.readouterr().out
    assert 'Recording' in out and 'every 1 hour (requested 45 min, rounded to the time step)' in out


def _free_walkers(count, motility, size=200):
    import lattics

    sim = lattics.Simulation(seed=8)
    sim.set_space(lattics.Lattice2DSpace(size=((20 * size, 'um'), (20 * size, 'um')), interval=(1, 'hour')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # without binding every move to a free site is accepted, so the agents perform random walks
        lattics.populate_agents(
            sim, count=count, attributes={'space.volume': (1000, 'um3'), 'mechanics.motility': motility}
        )
    return sim


def test_fast_motility_is_split_into_sweeps(capsys):
    # 1e-4 um/ms * 1 hour / 20 um = 18 expected trials per update
    sim = _free_walkers(5, (1e-4, 'um/ms'))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=1)
    assert math.isclose(1e-4 * HOUR / 20, 18)
    assert 'Lattice2DSpace: the agents move in 18 displacement sweeps per update' in capsys.readouterr().out


def test_sweeps_keep_the_expected_number_of_steps():
    import numpy as np

    # 4 expected trials in a single update: a random walk of 4 steps has a mean squared displacement of 4
    # (truncating the motility to one trial per update would give 1)
    sim = _free_walkers(400, (4 * 20 / HOUR, 'um/ms'))
    start = {a.id: np.array(a.get_attribute('space.position'), dtype=float) for a in sim.agents}
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    squared = [np.sum((a.get_attribute('space.position') - start[a.id]) ** 2) for a in sim.agents]
    assert 3.5 < np.mean(squared) < 4.5
