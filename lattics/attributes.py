"""Registry of the agent attributes used by LattiCS.

Attributes are referred to by plain names everywhere, for example ``agent.get_attribute('space.position')``.
Every built-in name carries the prefix of the subsystem (model) that defines its meaning, even if other models
read it. Agents accept any other name as well, so users can freely introduce their own attributes.

This module lists the built-in names together with their default values. Custom attributes can be declared with
:meth:`lattics.Simulation.declare`; the validation at the start of a run then treats them as intentional.
"""


class _Missing:
    def __repr__(self) -> str:
        return 'MISSING'


MISSING = _Missing()
"""Returned by :func:`default_of` for attributes without a default value."""

_REGISTRY = {
    # Space: placement, division and removal requests, spatial observables
    'space.position': MISSING,
    'space.volume': MISSING,
    'space.division_requested': False,
    'space.last_division_time': None,
    'space.removal_requested': False,
    'space.local_density': 0.0,
    # Mechanics of the lattice spaces
    'mechanics.motility': 0,
    'mechanics.binding_affinity': 0,
    'mechanics.displacement_limit': 1,
    # Substrate exchange between the agents and the substrate fields
    'substrate.transport': MISSING,
    # Cell cycle models
    'cellcycle.is_active': True,
    'cellcycle.mean_length': MISSING,
    'cellcycle.length': MISSING,
    'cellcycle.current_time': 0,
    'cellcycle.current_phase': 'G1',
    'cellcycle.random_initial': MISSING,
    'cellcycle.inhibition_density_limit': 1.0,
    'cellcycle.phase_g1_mean': MISSING,
    'cellcycle.phase_g1_sd': MISSING,
    'cellcycle.phase_g1_length': MISSING,
    'cellcycle.phase_s_mean': MISSING,
    'cellcycle.phase_s_sd': MISSING,
    'cellcycle.phase_s_length': MISSING,
    'cellcycle.phase_g2_mean': MISSING,
    'cellcycle.phase_g2_sd': MISSING,
    'cellcycle.phase_g2_length': MISSING,
    'cellcycle.phase_m_mean': MISSING,
    'cellcycle.phase_m_sd': MISSING,
    'cellcycle.phase_m_length': MISSING,
    # Toxicity model
    'toxicity.state': MISSING,
}


# Names used by earlier versions of LattiCS, reported by the validation at the start of a run
_RENAMED = {
    'position': 'space.position',
    'volume': 'space.volume',
    'duplication_needed': 'space.division_requested',
    'duplication_completed': 'space.last_division_time',
    'space.duplication_needed': 'space.division_requested',
    'space.duplication_completed': 'space.last_division_time',
    'space.division_completed': 'space.last_division_time',
    'removal_needed': 'space.removal_requested',
    'state': 'toxicity.state',
    'motility': 'mechanics.motility',
    'binding_affinity': 'mechanics.binding_affinity',
    'mechanics.local_density': 'space.local_density',
    'substrate_info': 'substrate.transport',
    'cellcycle_is_active': 'cellcycle.is_active',
}


def known_names() -> list[str]:
    """Returns the names of the built-in attributes.

    Returns
    -------
    list[str]
        The attribute names, in alphabetical order
    """
    return sorted(_REGISTRY)


def default_of(name: str):
    """Returns the default value of the given attribute.

    Parameters
    ----------
    name : str
        The name of the attribute

    Returns
    -------
    Any
        The default value, or :data:`MISSING` if the attribute has no default or is not a built-in attribute
    """
    return _REGISTRY.get(name, MISSING)
