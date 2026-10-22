import warnings

import numpy as np
import pytest

from lattics import (
    Agent,
    Event,
    HomogeneousSpace,
    Lattice2DSpace,
    Simulation,
    Substrate,
    SubstrateInfo,
    SubstrateNode,
)
from lattics.events import SetConcentration
from lattics.kinetics import FirstOrderDecay
from lattics.models import (
    ConcentrationDependentToxicityModel,
    MultiPhaseTimeBasedCellCycleModel,
    StochasticTransitionModel,
)
from lattics.recorders import AgentCount, Snapshot, SubstrateLevel


def _lattice_simulation():
    sim = Simulation(seed=5)
    sim.set_space(Lattice2DSpace(size=((400, 'um'), (400, 'um')), interval=(30, 'min')))
    sim.add_model(MultiPhaseTimeBasedCellCycleModel(update_interval=(30, 'min')))
    sim.add_model(
        StochasticTransitionModel(
            update_interval=(30, 'min'),
            condition=('cellcycle.current_phase', 'G0'),
            end_states={'space.removal_requested': True},
            rate=(1.0, '1/day'),
        )
    )
    for _ in range(20):
        agent = Agent()
        for name, value in {
            'space.volume': (3000, 'um3'),
            'mechanics.motility': (5e-6, 'um/ms'),
            'mechanics.binding_affinity': 1.0,
            'mechanics.displacement_limit': 2,
            'cellcycle.phase_g1_mean': (6, 'hour'),
            'cellcycle.phase_g1_sd': (2, 'hour'),
            'cellcycle.phase_s_mean': (4, 'hour'),
            'cellcycle.phase_g2_mean': (2, 'hour'),
            'cellcycle.phase_m_mean': (1, 'hour'),
            'cellcycle.random_initial': True,
            'cellcycle.inhibition_density_limit': 0.6,
        }.items():
            agent.set_attribute(name, value)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sim.add_agent(agent)
    sim.add_recorder(AgentCount(interval=(5, 'hour')))
    sim.add_recorder(Snapshot(interval=(10, 'hour'), attributes=['space.position', 'cellcycle.current_phase']))
    return sim


def _homogeneous_simulation():
    sim = Simulation(seed=6)
    sim.set_space(HomogeneousSpace(volume=(1, 'mm3'), interval=(1, 'hour')))
    sim.add_substrate(Substrate('drug', kinetics=FirstOrderDecay((0.1, '1/hour')), interval=(1, 'min')))
    vessels = SubstrateNode(
        volume=(0.1, 'mm3'), transport={'drug': SubstrateInfo(type='flux', passive_rate=(1, 'um3/ms'))}
    )
    sim.add_substrate_node(vessels)
    sim.add_model(
        ConcentrationDependentToxicityModel(
            (1, 'hour'), 'drug', max_rate=(5.0, '1/day'), median_effective_concentration=10.0
        )
    )
    for _ in range(30):
        agent = Agent()
        agent.set_attribute('space.volume', (3000, 'um3'))
        agent.set_attribute(
            'substrate.transport', {'drug': SubstrateInfo(type='flux', uptake_rate=(1e-3, 'um3/ms'))}
        )
        sim.add_agent(agent)
    for hour in (0, 13, 30):
        sim.add_event(SetConcentration(time=(hour, 'hour'), substrate='drug', concentration=100.0, node=vessels))
    sim.add_recorder(SubstrateLevel('drug', interval=(1, 'hour')))
    sim.add_recorder(AgentCount(interval=(1, 'hour'), where=('toxicity.state', 'necrotic')))
    return sim


def _state(sim):
    recorders = []
    for recorder in sim._recorders:
        if isinstance(recorder, Snapshot):
            # positions are arrays, so the frames are compared through their text form
            recorders.append([(f.time, repr(f.agents), repr(f.removed)) for f in recorder.frames])
        else:
            recorders.append((recorder.series.timepoints, recorder.series.values))
    agents = sorted((a.id, repr(sorted(a._attributes.items(), key=lambda item: item[0]))) for a in sim.agents)
    return recorders, agents, sim.time, sim.rng.random()


@pytest.mark.parametrize('build', [_lattice_simulation, _homogeneous_simulation])
def test_saved_simulation_continues_exactly(build, tmp_path):
    uninterrupted = build()
    uninterrupted.run(time=(2, 'day'), dt=(30, 'min'), verbosity=0)

    first = build()
    first.run(time=(1, 'day'), dt=(30, 'min'), verbosity=0)
    first.save(tmp_path / 'sim.lattics')
    resumed = Simulation.load(tmp_path / 'sim.lattics')
    resumed.run(time=(1, 'day'), dt=(30, 'min'), verbosity=0)

    assert _state(resumed) == _state(uninterrupted)


def test_simulation_with_a_lambda_cannot_be_saved(tmp_path):
    sim = Simulation()
    sim.add_event(Event(time=(1, 'hour'), handler=lambda ctx: None))
    with pytest.raises(TypeError, match='module level'):
        sim.save(tmp_path / 'sim.lattics')


def _save_during_run(ctx, path):
    ctx._simulation.save(path)


def test_simulation_cannot_be_saved_during_a_run(tmp_path):
    sim = Simulation()
    sim.add_event(Event(time=(0, 'hour'), handler=_save_during_run, path=tmp_path / 'sim.lattics'))
    with pytest.raises(RuntimeError, match='between two runs'):
        sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)


def test_load_checks_the_file_and_the_version(tmp_path, monkeypatch):
    import pickle

    import lattics

    other = tmp_path / 'other.pkl'
    other.write_bytes(pickle.dumps({'something': 'else'}))
    with pytest.raises(ValueError, match='not a saved LattiCS simulation'):
        Simulation.load(other)

    Simulation(seed=1).save(tmp_path / 'sim.lattics')
    monkeypatch.setattr(lattics, '__version__', '99.0')
    with pytest.warns(UserWarning, match='saved with LattiCS'):
        loaded = Simulation.load(tmp_path / 'sim.lattics')
    assert loaded.seed == 1 and np.isfinite(loaded.rng.random())
