from ._version import get_versions

from .core import Agent, Event, Simulation
from .spaces import HomogeneousSpace, Lattice2DSpace
from .substrates import SubstrateInfo

__version__ = get_versions()["version"]
del get_versions
