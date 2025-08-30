from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any
from .numba_functions import diffusion_solver_lod_2d


@dataclass
class SubstrateInfo:
    type: str
    concentration: float = 0.0
    passive_rate: float = 0.0
    uptake_rate: float = 0.0
    release_rate: float = 0.0


class SubstrateNode:
    def __init__(self) -> None:
        """Constructor method.
        """
        self._attributes = dict()
        self.set_attribute('substrate_info', dict())

    def set_attribute(self, name: str, value: Any) -> None:
        self._attributes[name] = value

    def get_attribute(self, name: str) -> Any:
        return self._attributes[name]


class BaseSubstrateField(ABC):
    def __init__(self,
                 domain,
                 substrate_name: str,
                 diffusion_coefficient: float = 0.0,
                 decay_coefficient: float = 0.0
                 ) -> None:
        self._domain = domain
        self._substrate_name = substrate_name
        self._diffusion_coefficient = diffusion_coefficient
        self._decay_coefficient = decay_coefficient
        self._static_nodes = list()
        self._dynamic_nodes = list()

    def add_static_substrate_node(self, substrate_node: SubstrateNode) -> None:
        self._static_nodes.append(substrate_node)

    def add_dynamic_substrate_node(self, substrate_node: SubstrateNode) -> None:
        self._dynamic_nodes.append(substrate_node)

    def clear_dynamic_nodes(self) -> None:
        self._dynamic_nodes.clear()

    @abstractmethod
    def get_concentration(self) -> float:
        pass

    @abstractmethod
    def update(self, dt) -> None:
        pass


class HomogeneousSubstrateField(BaseSubstrateField):
    def __init__(self,
                 domain,
                 substrate_name: str,
                 diffusion_coefficient: float = 0.0,
                 decay_coefficient: float = 0.0
                 ) -> None:
        super().__init__(domain=domain,
                         substrate_name=substrate_name,
                         diffusion_coefficient=diffusion_coefficient,
                         decay_coefficient=decay_coefficient
                         )
        self._concentration = 0.0

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
                dn = (k_p * (c_f_cur - c_n_cur) + k_u * c_f_cur - k_r * c_n_cur) * dt
                c_n_new = c_n_cur + dn / v_n
                c_f_new = c_f_cur - dn / v_f
                info.concentration = c_n_new
                self._concentration = c_f_new
            elif info.type == 'fixed':
                sum_fixed = sum_fixed + info.concentration
                count_fixed = count_fixed + 1
        if 0 < count_fixed:
            self._concentration = sum_fixed / count_fixed

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration
        d = self._decay_coefficient
        self._concentration = C * np.exp(-d * dt)


class Lattice2DSubstrateField(BaseSubstrateField):
    def __init__(self,
                 domain,
                 substrate_name: str,
                 diffusion_coefficient: float = 0.0,
                 decay_coefficient: float = 0.0
                 ) -> None:
        super().__init__(domain=domain,
                         substrate_name=substrate_name,
                         diffusion_coefficient=diffusion_coefficient,
                         decay_coefficient=decay_coefficient
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
                c_f_cur = self._concentration[*pos]
                dn = (k_p * (c_f_cur - c_n_cur) + k_u * c_f_cur - k_r * c_n_cur) * dt
                c_n_new = c_n_cur + dn / v_n
                c_f_new = c_f_cur - dn / v_f
                info.concentration = c_n_new
                self._concentration[*pos] = c_f_new
            elif info.type == 'fixed':
                self._concentration[*pos] = info.concentrattion

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration
        D = self._diffusion_coefficient
        d = self._decay_coefficient
        dx = self._dx
        self._concentration = diffusion_solver_lod_2d(C, D, d, np.float32(dt), dx)
