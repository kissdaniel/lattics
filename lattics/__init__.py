from ._version import get_versions

from .core import Agent, Simulation, Event
from .spaces import HomogeneousSpace, Lattice2DSpace
from .substrates import *
from .models import *
from .utils import *

__version__ = get_versions()["version"]
del get_versions
