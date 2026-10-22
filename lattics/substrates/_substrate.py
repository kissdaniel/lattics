from collections.abc import Sequence

from lattics._units import convert_diffusivity, convert_time
from lattics.kinetics import Kinetics


class Substrate:
    """The definition of a substrate (for example a drug or a nutrient). It is
    added to the simulation with :meth:`~lattics.Simulation.add_substrate`, and
    the space of the simulation creates its concentration field.

    Parameters
    ----------
    name : str
        The name of the substrate; the ``substrate.transport`` attribute of the
        agents and nodes refers to it
    diffusion_coefficient : tuple[float, str], optional
        For example ``(2.0, 'um2/sec')``. Spaces without spatial structure do
        not use it.
    kinetics : Kinetics or Sequence[Kinetics], optional
        The reactions of the substrate, see :mod:`lattics.kinetics`. Several
        kinetics are applied one after the other.
    interval : tuple[float, str], optional
        The time between two updates of the field, for example ``(30, 'sec')``.
        Defaults to every time step of the simulation.
    """

    def __init__(
        self,
        name: str,
        diffusion_coefficient: tuple[float, str] = None,
        kinetics: Kinetics | Sequence[Kinetics] = None,
        interval: tuple[float, str] = None,
    ) -> None:
        self.name = name
        if diffusion_coefficient is None:
            self.diffusion_coefficient = 0.0
        elif isinstance(diffusion_coefficient, tuple) and len(diffusion_coefficient) == 2:
            self.diffusion_coefficient = convert_diffusivity(
                diffusion_coefficient[0], diffusion_coefficient[1], 'um2/ms'
            )
        else:
            raise TypeError(
                'The diffusion coefficient must be given as (value, unit), for example (2.0, \'um2/sec\').'
            )
        if kinetics is None:
            self.kinetics = ()
        elif isinstance(kinetics, Kinetics):
            self.kinetics = (kinetics,)
        else:
            self.kinetics = tuple(kinetics)
        self.interval = convert_time(interval[0], interval[1], 'ms') if interval else None

    def apply_kinetics(self, concentration, dt: float):
        """Applies the kinetics of the substrate one after the other.

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
        for kinetics in self.kinetics:
            concentration = kinetics.apply(concentration, dt)
        return concentration
