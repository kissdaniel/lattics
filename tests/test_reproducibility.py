import warnings

import numpy as np


def _run_lattice(seed=None):
    import lattics
    from lattics.models import MultiPhaseTimeBasedCellCycleModel, StochasticTransitionModel

    sim = lattics.Simulation(seed=seed)
    sim.set_space(lattics.Lattice2DSpace(size=((400, 'um'), (400, 'um')), interval=(30, 'min')))
    sim.add_model(MultiPhaseTimeBasedCellCycleModel(update_interval=(30, 'min')))
    sim.add_model(
        StochasticTransitionModel(
            update_interval=(30, 'min'),
            condition=('cellcycle.current_phase', 'G0'),
            end_states={'space.removal_requested': True},
            rate=1.0,
        )
    )
    attributes = {
        'space.volume': 3000,
        'mechanics.motility': 5e-6,
        'mechanics.binding_affinity': 1.0,
        'mechanics.displacement_limit': 2,
        'cellcycle.phase_g1_mean': (6, 'hour'),
        'cellcycle.phase_g1_sd': (2, 'hour'),
        'cellcycle.phase_s_mean': (4, 'hour'),
        'cellcycle.phase_g2_mean': (2, 'hour'),
        'cellcycle.phase_m_mean': (1, 'hour'),
        'cellcycle.random_initial': True,
        'cellcycle.inhibition_density_limit': 0.6,
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lattics.populate_agents(sim, count=20, attributes=attributes)
    sim.run(time=(1, 'day'), dt=(30, 'min'), verbosity=0)
    state = sorted(
        (tuple(int(v) for v in a.get_attribute('space.position')), a.get_attribute('cellcycle.current_time'))
        for a in sim.agents
    )
    return sim, state


def test_same_seed_gives_the_same_result():
    _, first = _run_lattice(seed=7)
    _, second = _run_lattice(seed=7)
    assert first == second


def test_different_seeds_give_different_results():
    _, first = _run_lattice(seed=7)
    _, second = _run_lattice(seed=8)
    assert first != second


def test_generated_seed_repeats_the_simulation():
    sim, first = _run_lattice()
    _, second = _run_lattice(seed=sim.seed)
    assert isinstance(sim.seed, int) and first == second


def test_simulation_does_not_use_the_global_numpy_generator():
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    _run_lattice(seed=7)
    actual = np.random.random()
    assert actual == expected


def test_context_without_simulation_has_a_generator():
    from lattics import UpdateContext

    assert isinstance(UpdateContext(time=0, dt=1).rng, np.random.Generator)


def test_summary_shows_the_seed(capsys):
    import lattics

    sim = lattics.Simulation(seed=42)
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=1)
    assert 'seed 42' in capsys.readouterr().out
