"""Reference runs: two seeded simulations whose results are stored in ``tests/reference/``.

They pin down the current behaviour of the simulation core, so that refactoring steps which must not change
the results can be verified. When a change is meant to alter the results, rewrite the stored files with
``pytest --regen-references`` and explain the difference in the commit message.
"""

import json
import math
import pathlib
import pickle

import numpy as np
import pytest

from lattics.core import Agent, Event, Simulation, SubstrateInfo, SubstrateNode
from lattics.core.spaces import HomogeneousSpace, Lattice2DSpace
from lattics.models.cellcycle import MultiPhaseTimeBasedCellCycleModel
from lattics.models.toxicity import ConcentrationDependentToxicityModel
from lattics.models.transition import StochasticTransitionModel

from .helpers import seed_everything

REFERENCE_DIR = pathlib.Path(__file__).parent / 'reference'
SEED = 12345
LATTICE_SIZE = 40


def run_lattice_growth() -> dict:
    """2D lattice: multi-phase cell cycle with contact inhibition, displacement trials, cell division, and
    removal of inhibited (G0) cells."""
    seed_everything(SEED)
    simulation = Simulation(id='reference_lattice_growth')
    space = Lattice2DSpace(
        simulation, dimensions=(LATTICE_SIZE, LATTICE_SIZE), grid_spacing=20, dt_agent=(30, 'min')
    )
    simulation.add_space(space)
    simulation.add_model(MultiPhaseTimeBasedCellCycleModel(update_interval=(30, 'min')))
    simulation.add_model(
        StochasticTransitionModel(
            update_interval=(30, 'min'),
            condition=('cellcycle.current_phase', 'G0'),
            end_states={'removal_needed': True},
            rate=1.0,
        )
    )
    for site in np.random.choice(LATTICE_SIZE**2, 60, replace=False):
        agent = Agent()
        attributes = {
            'position': np.array(divmod(int(site), LATTICE_SIZE), dtype=np.int32),
            'volume': 300,
            'motility': 5e-6,
            'binding_affinity': 1.0,
            'mechanics.displacement_limit': 3,
            'cellcycle.phase_g1_mean': (10, 'hour'),
            'cellcycle.phase_g1_sd': (2, 'hour'),
            'cellcycle.phase_s_mean': (6, 'hour'),
            'cellcycle.phase_g2_mean': (3, 'hour'),
            'cellcycle.phase_m_mean': (1, 'hour'),
            'cellcycle.random_initial': True,
            'cellcycle.inhibition_density_limit': 0.7,
        }
        for name, value in attributes.items():
            agent.set_attribute(name, value)
        simulation.add_agent(agent)
    simulation.run(time=(3, 'day'), dt=(30, 'min'), dt_history=(6, 'hour'), save_mode='never', verbosity=0)

    history = [pickle.loads(entry) for entry in simulation._history]
    return {
        'agent_counts': [len(state.agents) for state in history],
        'g0_counts': [
            sum(a.get_attribute('cellcycle.current_phase') == 'G0' for a in state.agents) for state in history
        ],
        'final_agents': sorted(
            [
                [int(v) for v in a.get_attribute('position')],
                a.get_attribute('cellcycle.current_phase'),
                float(a.get_attribute('cellcycle.current_time')),
                float(a.get_attribute('mechanics.local_density')),
            ]
            for a in simulation.agents
        ),
    }


def _set_node_concentration(simulation, node, name, concentration):
    node.get_attribute('substrate_info')[name].concentration = concentration


def run_homogeneous_drug_treatment() -> dict:
    """Homogeneous space: drug pharmacokinetics (two static nodes, two doses given by events), drug uptake by
    the agents, and concentration-dependent necrosis."""
    seed_everything(SEED)
    simulation = Simulation(id='reference_homogeneous_drug_treatment')
    space = HomogeneousSpace(simulation, volume=8.4375e6, dt_agent=(1, 'hour'), dt_substrate=(30, 'sec'))
    simulation.add_space(space)
    space.add_substrate(name='drug', diffusion_coefficient=2e-3, decay_coefficient=1e-8)
    field = space._substrates['drug']

    vessels = SubstrateNode()
    vessels.set_attribute('volume', 8.4375e5)
    vessels.get_attribute('substrate_info')['drug'] = SubstrateInfo(type='flux', passive_rate=0.5)
    field.add_static_substrate_node(vessels)
    organs = SubstrateNode()
    organs.set_attribute('volume', 8.4375e6)
    organs.get_attribute('substrate_info')['drug'] = SubstrateInfo(type='flux', release_rate=1.0, uptake_rate=5.0)
    field.add_static_substrate_node(organs)

    simulation.add_model(
        ConcentrationDependentToxicityModel(
            update_interval=(1, 'hour'), substrate_name='drug', max_rate=1.0, median_effective_concentration=1e4
        )
    )
    for _ in range(100):
        agent = Agent()
        agent.set_attribute('volume', 3375)
        agent.set_attribute('substrate_info', {'drug': SubstrateInfo(type='flux', passive_rate=1e-3)})
        simulation.add_agent(agent)
    for day, dose in [(0, 1e6), (1, 5e5)]:
        simulation.add_event(
            Event(
                time=(day, 'day'), handler=_set_node_concentration, node=vessels, name='drug', concentration=dose
            )
        )
    simulation.run(time=(2, 'day'), dt=(30, 'sec'), dt_history=(1, 'hour'), save_mode='never', verbosity=0)

    history = [pickle.loads(entry) for entry in simulation._history]
    return {
        'field_concentrations': [float(state._space._substrates['drug']._concentration) for state in history],
        'necrotic_counts': [
            sum(a.has_attribute('state') and a.get_attribute('state') == 'necrotic' for a in state.agents)
            for state in history
        ],
        'final_node_concentrations': [
            float(node.get_attribute('substrate_info')['drug'].concentration) for node in (vessels, organs)
        ],
        'final_agent_concentrations': [
            float(a.get_attribute('substrate_info')['drug'].concentration) for a in simulation.agents
        ],
    }


def _dumps(result: dict) -> str:
    """JSON with one record per line, so that a rewritten reference shows up as a readable diff."""
    blocks = []
    for key, values in result.items():
        rows = ',\n'.join(f'    {json.dumps(v)}' for v in values)
        blocks.append(f'  {json.dumps(key)}: [\n{rows}\n  ]')
    return '{\n' + ',\n'.join(blocks) + '\n}\n'


def _assert_matches(actual, expected, path):
    """Integers and strings must match exactly, floats up to a relative tolerance of 1e-9: their last digits
    may differ between platforms and library versions."""
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f'{path}: {actual!r} != {expected!r}'
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_matches(a, e, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f'{path}: {actual!r} != {expected!r}'
    else:
        assert actual == expected, f'{path}: {actual!r} != {expected!r}'


def _check_reference(name, result, request):
    path = REFERENCE_DIR / f'{name}.json'
    if request.config.getoption('--regen-references'):
        REFERENCE_DIR.mkdir(exist_ok=True)
        path.write_text(_dumps(result), encoding='utf-8')
        pytest.skip(f'reference rewritten: {path.name}')
    expected = json.loads(path.read_text(encoding='utf-8'))
    actual = json.loads(json.dumps(result))
    assert actual.keys() == expected.keys()
    for key in expected:
        _assert_matches(actual[key], expected[key], key)


def test_lattice_growth_matches_reference(request):
    _check_reference('lattice_growth', run_lattice_growth(), request)


def test_homogeneous_drug_treatment_matches_reference(request):
    _check_reference('homogeneous_drug_treatment', run_homogeneous_drug_treatment(), request)
