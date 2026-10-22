"""Reaction kinetics of the substrates.

A kinetics object changes the concentration of a substrate over a time step. Every kinetics works on a single
concentration (homogeneous fields) and on arrays of concentrations (spatial fields) alike. A substrate may have
several kinetics; they are applied one after the other (operator splitting).

Rates are given as ``(value, unit)`` tuples with a ``'1/<time unit>'`` unit, for example ``(0.1, '1/hour')``.
Concentrations have no unit in LattiCS: they are given in the unit chosen by the user, which applies to every
concentration of the simulation.
"""

from abc import ABC, abstractmethod

import numpy as np

from lattics._units import convert_rate


class Kinetics(ABC):
    """Base class of the reaction kinetics. A custom kinetics implements :meth:`apply`."""

    @abstractmethod
    def apply(self, concentration, dt: float):
        """Returns the concentration after the time step.

        Parameters
        ----------
        concentration : float or numpy.ndarray
            The concentration at the start of the time step
        dt : float
            The time step, in milliseconds

        Returns
        -------
        float or numpy.ndarray
            The concentration at the end of the time step
        """
        raise NotImplementedError


class FirstOrderDecay(Kinetics):
    """First-order decay, ``dC/dt = -k C``, solved exactly.

    Parameters
    ----------
    rate : tuple[float, str]
        The rate constant ``k``, for example ``(0.1, '1/hour')``
    """

    def __init__(self, rate: tuple[float, str]) -> None:
        self.rate = _to_rate(rate, 'rate')

    def apply(self, concentration, dt: float):
        return concentration * np.exp(-self.rate * dt)


class SecondOrderDecay(Kinetics):
    """Second-order decay, ``dC/dt = -k C^2``, solved exactly.

    Parameters
    ----------
    rate : tuple[float, str]
        The rate constant ``k`` per unit concentration, for example ``(0.01, '1/hour')``
    """

    def __init__(self, rate: tuple[float, str]) -> None:
        self.rate = _to_rate(rate, 'rate')

    def apply(self, concentration, dt: float):
        return concentration / (1 + self.rate * concentration * dt)


class MichaelisMentenDecay(Kinetics):
    """Saturable (Michaelis-Menten) elimination, ``dC/dt = -Vmax C / (Km + C)``, solved with an implicit Euler
    step, which keeps the concentration non-negative.

    Parameters
    ----------
    vmax : tuple[float, str]
        The maximal elimination rate in concentration units per time, for example ``(5.0, '1/hour')``
    km : float
        The Michaelis constant, a concentration
    """

    def __init__(self, vmax: tuple[float, str], km: float) -> None:
        self.vmax = _to_rate(vmax, 'vmax')
        self.km = km

    def apply(self, concentration, dt: float):
        x = self.km - concentration + dt * self.vmax
        return (-x + (x**2 + 4 * self.km * concentration) ** 0.5) / 2


def _to_rate(value: tuple[float, str], name: str) -> float:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f'The {name} must be given as (value, unit), for example (0.1, \'1/hour\').')
    return convert_rate(value[0], value[1], '1/ms')
