import random

import numba
import numpy as np


@numba.njit
def _seed_numba(seed):
    np.random.seed(seed)


def seed_everything(seed: int) -> None:
    """Seeds every random number generator that LattiCS currently draws from: the global NumPy generator,
    the standard library's ``random`` module, and the separate generator of numba-compiled functions
    (``np.random.seed`` called from Python does not affect the latter).

    Parameters
    ----------
    seed : int
        The seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    _seed_numba(seed)
