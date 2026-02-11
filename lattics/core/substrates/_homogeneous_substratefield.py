from ._base import BaseSubstrateField

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class HomogeneousSubstrateField(BaseSubstrateField):
    def __init__(self,
                 domain,
                 substrate_name: str,
                 diffusion_coefficient: float = 0.0,
                 decay_coefficient: float = 0.0,
                 decay_kinetics: str = 'first-order',
                 mm_constant: float = None
                 ) -> None:
        super().__init__(domain=domain,
                         substrate_name=substrate_name,
                         diffusion_coefficient=diffusion_coefficient,
                         decay_coefficient=decay_coefficient
                         )
        self._concentration = 0.0
        if decay_kinetics == 'first-order':
            self._decay_function = self._decay_first_order
        if decay_kinetics == 'second-order':
            self._decay_function = self._decay_second_order
        if decay_kinetics == 'michaelis-menten':
            self._decay_function = self._decay_michaelis_menten
            self._mm_constant = mm_constant

    def get_concentration(self, position=None) -> float:
        return self._concentration

    def update(self, dt: int) -> None:
        self.update_nodes(dt)
        self.diffusion_decay(dt)

    def update_nodes(self, dt: int) -> None:
        count_fixed = 0
        sum_fixed = 0
        v_f = self._domain._volume
        all_nodes = self._static_nodes + self._dynamic_nodes
        for n in all_nodes:
            info = n.get_attribute('substrate_info')[self._substrate_name]
            if info.type == 'flux':
                k_p = info.passive_rate
                k_u = info.uptake_rate
                k_r = info.release_rate
                c_n_cur = info.concentration
                v_n = n.get_attribute('volume')
                c_f_cur = self._concentration
                m_t = c_n_cur * v_n + c_f_cur * v_f
                c_n_new = (c_n_cur + (dt / v_n) * (m_t / v_f) * (k_p + k_u)) / (1 + (dt / v_n) * ((1 + (v_n / v_f)) * k_p + (v_n / v_f) * k_u + k_r))
                c_f_new = (m_t - c_n_new * v_n) / v_f
                info.concentration = c_n_new
                self._concentration = c_f_new
            elif info.type == 'fixed':
                sum_fixed = sum_fixed + info.concentration
                count_fixed = count_fixed + 1
        if 0 < count_fixed:
            self._concentration = sum_fixed / count_fixed

    def diffusion_decay(self, dt: int) -> None:
        self._decay_function(dt)

    def _decay_first_order(self, dt: int) -> None:
        C = self._concentration
        d = self._decay_coefficient
        self._concentration = C * np.exp(-d * dt)

    def _decay_second_order(self, dt: int) -> None:
        C = self._concentration
        d = self._decay_coefficient
        self._concentration = 1 / ((1 / C) + d * dt)

    def _decay_michaelis_menten(self, dt: int) -> None:
        C = self._concentration
        v_max = self._decay_coefficient
        k_M = self._mm_constant
        x = k_M - C + dt * v_max
        self._concentration = (-x + (x**2 + 4 * k_M * C)**0.5) / 2
