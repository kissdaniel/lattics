import numpy as np

from lattics._context import UpdateContext
from lattics._kernels.diffusion import implicit_diffusion_2d, thomas_factors

from ._base import BaseSubstrateField, exchange


class Lattice2DSubstrateField(BaseSubstrateField):
    """The concentration field of a substrate on a two-dimensional lattice:
    one concentration per lattice site. Created by
    :class:`~lattics.Lattice2DSpace`.

    An update performs, in this order, the exchange with the substrate nodes
    and the agents, the diffusion (implicit, zero-flux boundaries) and the
    kinetics of the substrate. Every site has the volume ``dx^2 * thickness``.

    Agents and nodes with a ``space.position`` exchange with their own site.
    Nodes without a position exchange with the whole field, as in a
    homogeneous space: the exchange uses the mean concentration, the mass the
    node releases is spread uniformly, and the mass it takes up is taken in
    proportion to the local concentrations. A uniform field therefore gives
    the same result as a homogeneous space of the same volume.
    """

    def __init__(self, substrate, space, nodes) -> None:
        super().__init__(substrate, space, nodes)
        self._cell_volume = self._space._site_volume
        self._dx = self._space._grid_spacing
        self._concentration = np.zeros(self._space._dimensions, dtype=np.float64)
        self._factors = None

    @property
    def concentration(self) -> np.ndarray:
        return self._concentration

    def concentration_at(self, position=None) -> float:
        return float(self._concentration[tuple(position)])

    def collect_notes(self) -> list[str]:
        name = self._substrate.name
        spread = sum(
            1
            for n in self._nodes
            if name in n.get_attribute('substrate.transport') and not n.has_attribute('space.position')
        )
        if spread:
            return [f'{spread} substrate nodes without a position are spread uniformly over the field']
        return []

    def update(self, ctx: UpdateContext) -> None:
        dt = ctx.dt
        self._exchange(dt)
        self._diffuse(dt)
        self._concentration = self._substrate.apply_kinetics(self._concentration, dt)

    def _exchange(self, dt: float) -> None:
        name = self._substrate.name
        fixed_spread = []
        for n in self.exchanging_nodes():
            info = n.get_attribute('substrate.transport')[name]
            if n.has_attribute('space.position'):
                site = self._site_of(n)
                if info.type == 'flux':
                    v_n = n.get_attribute('space.volume')
                    c_n, c_f = exchange(info, v_n, self._concentration[site], self._cell_volume, dt)
                    info.concentration = c_n
                    self._concentration[site] = c_f
                elif info.type == 'fixed':
                    self._concentration[site] = info.concentration
            elif info.type == 'flux':
                self._exchange_spread(info, n.get_attribute('space.volume'), dt)
            elif info.type == 'fixed':
                fixed_spread.append(info.concentration)
        if fixed_spread:
            self._concentration[:] = sum(fixed_spread) / len(fixed_spread)

    def _exchange_spread(self, info, v_n: float, dt: float) -> None:
        field_volume = self._cell_volume * self._concentration.size
        mass = self._concentration.sum() * self._cell_volume
        c_n, c_f = exchange(info, v_n, mass / field_volume, field_volume, dt)
        info.concentration = c_n
        new_mass = c_f * field_volume
        if new_mass >= mass:
            self._concentration += (new_mass - mass) / field_volume
        else:
            # Taken in proportion to the local concentrations, which keeps them non-negative
            self._concentration *= new_mass / mass

    def _diffuse(self, dt: float) -> None:
        D = self._substrate.diffusion_coefficient
        if D == 0:
            return
        r = D * dt / self._dx**2
        if self._factors is None or self._factors[0] != r:
            n0, n1 = self._concentration.shape
            self._factors = (r, *thomas_factors(n0, r), *thomas_factors(n1, r))
        self._concentration = implicit_diffusion_2d(self._concentration, *self._factors)

    def _site_of(self, node) -> tuple[int, int]:
        site = tuple(int(i) for i in node.get_attribute('space.position'))
        if len(site) != 2 or not all(0 <= i < n for i, n in zip(site, self._concentration.shape)):
            raise ValueError(f'The position {site} of a substrate node is outside the lattice.')
        return site
