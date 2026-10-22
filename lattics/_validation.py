import difflib
from collections import Counter
from collections.abc import Iterable

from .attributes import _RENAMED, known_names


def check_attributes(agents: list, components: Iterable, declared: Iterable[str]) -> tuple[list[str], list[str]]:
    """Compares the attributes of the agents with the attributes the components (models and the space) read and
    write.

    Parameters
    ----------
    agents : list[Agent]
        The agents of the simulation
    components : Iterable
        The models and the space; their ``reads``, ``optional_reads`` and ``writes`` are compared
    declared : Iterable[str]
        The names declared with :meth:`Simulation.declare`

    Returns
    -------
    tuple[list[str], list[str]]
        The problems found, and the names of the agent attributes that no component uses
    """
    readers = dict()
    used = set()
    written = set()
    for component in components:
        name = type(component).__name__
        for attribute in component.reads:
            readers.setdefault(attribute, []).append(name)
        used.update(component.reads)
        used.update(getattr(component, 'optional_reads', ()))
        written.update(component.writes)
    used |= written
    declared = set(declared)
    known = set(known_names())

    present = Counter()
    for agent in agents:
        present.update(agent._attributes.keys())
    unknown = sorted(name for name in present if name not in used and name not in declared)

    problems = []
    suggested = set()
    for attribute, names in readers.items():
        missing = len(agents) - present[attribute]
        if attribute in written or missing == 0:
            continue
        message = (
            f'The attribute \'{attribute}\' is read by {", ".join(names)}, but no component writes it and '
            f'{missing} of {len(agents)} agents do not have it.'
        )
        match = difflib.get_close_matches(attribute, unknown, n=1, cutoff=0.8)
        if match:
            message += f' Did you mean \'{match[0]}\'?'
            suggested.add(match[0])
        problems.append(message)

    unused = []
    for name in unknown:
        if name in suggested:
            continue
        if name in _RENAMED:
            problems.append(
                f'The agent attribute \'{name}\' is not used any more, it was renamed to \'{_RENAMED[name]}\'.'
            )
            continue
        match = None if name in known else difflib.get_close_matches(name, sorted(used | known | declared), 1, 0.8)
        if match:
            problems.append(
                f'The agent attribute \'{name}\' is not used by any component. Did you mean \'{match[0]}\'? '
                'If the name is intentional, declare it with Simulation.declare().'
            )
        else:
            unused.append(name)
    return problems, unused
