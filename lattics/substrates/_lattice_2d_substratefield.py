import numpy as np

from lattics._kernels.diffusion import diffusion_solver_lod_2d

from ._base import BaseSubstrateField


class Lattice2DSubstrateField(BaseSubstrateField):
    def __init__(
        self, domain, substrate_name: str, diffusion_coefficient: float = 0.0, decay_coefficient: float = 0.0
    ) -> None:
        super().__init__(
            domain=domain,
            substrate_name=substrate_name,
            diffusion_coefficient=diffusion_coefficient,
            decay_coefficient=decay_coefficient,
        )
        self._dim_x = self._domain._dimensions[0]
        self._dim_y = self._domain._dimensions[1]
        self._dx = self._domain._grid_spacing
        self._concentration = np.zeros((self._dim_x, self._dim_y), dtype=np.float32)

    def get_concentration(self, position) -> float:
        return self._concentration[position]

    def update(self, dt: int) -> None:
        self.update_nodes(dt)
        self.diffusion_decay(dt)

    def update_nodes(self, dt: int) -> None:
        v_f = self._dx**2
        all_nodes = self._static_nodes + self._dynamic_nodes
        for n in all_nodes:
            info = n.get_attribute('substrate_info')[self._substrate_name]
            pos = n.get_attribute('position')
            if info.type == 'flux':
                k_p = info.passive_rate
                k_u = info.uptake_rate
                k_r = info.release_rate
                c_n_cur = info.concentration
                v_n = n.get_attribute('volume')
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
        D = self._diffusion_coefficient
        d = self._decay_coefficient
        dx = self._dx
        self._concentration = diffusion_solver_lod_2d(C, D, d, np.float32(dt), dx)
