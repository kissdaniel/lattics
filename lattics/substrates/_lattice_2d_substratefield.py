import numpy as np

from lattics._context import UpdateContext
from lattics._kernels.diffusion import diffusion_solver_lod_2d

from ._base import BaseSubstrateField


# Not used by Lattice2DSpace yet: the field is reworked and connected in step 5.2
class Lattice2DSubstrateField(BaseSubstrateField):
    def __init__(self, substrate, space, nodes) -> None:
        super().__init__(substrate, space, nodes)
        self._dim_x = self._space._dimensions[0]
        self._dim_y = self._space._dimensions[1]
        self._dx = self._space._grid_spacing
        self._concentration = np.zeros((self._dim_x, self._dim_y), dtype=np.float32)

    @property
    def concentration(self) -> np.ndarray:
        return self._concentration

    def concentration_at(self, position=None) -> float:
        return self._concentration[tuple(position)]

    def update(self, ctx: UpdateContext) -> None:
        self.update_nodes(ctx.dt)
        self.diffusion_decay(ctx.dt)

    def update_nodes(self, dt: int) -> None:
        v_f = self._dx**2
        for n in self.exchanging_nodes():
            info = n.get_attribute('substrate.transport')[self._substrate.name]
            pos = n.get_attribute('space.position')
            if info.type == 'flux':
                k_p = info.passive_rate
                k_u = info.uptake_rate
                k_r = info.release_rate
                c_n_cur = info.concentration
                v_n = n.get_attribute('space.volume')
                c_f_cur = self._concentration[tuple(pos)]
                dn = (k_p * (c_f_cur - c_n_cur) + k_u * c_f_cur - k_r * c_n_cur) * dt
                c_n_new = c_n_cur + dn / v_n
                c_f_new = c_f_cur - dn / v_f
                info.concentration = c_n_new
                self._concentration[tuple(pos)] = c_f_new
            elif info.type == 'fixed':
                self._concentration[tuple(pos)] = info.concentrattion

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration
        D = self._substrate.diffusion_coefficient
        dx = self._dx
        C = diffusion_solver_lod_2d(C, np.float32(D), np.float32(0.0), np.float32(dt), np.float32(dx))
        self._concentration = self._substrate.apply_kinetics(C, dt).astype(np.float32)
