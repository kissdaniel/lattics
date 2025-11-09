from ._version import get_versions

from lattics.utils.create import create_agent
from lattics.utils.create import create_population
from lattics.utils.create import create_simulation

__version__ = get_versions()["version"]
del get_versions
