from ._version import get_versions

from lattics.core._simulation import create_simulation

__version__ = get_versions()["version"]
del get_versions
