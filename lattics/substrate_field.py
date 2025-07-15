import numpy as np
import copy
import numba
from numba import void, float32, int32, boolean, types
from . import numba_functions

class SubstrateField:
    """Base class for different substrate fields.
    """
    def __init__(self, domain) -> None:
        self._domain = domain
        self._substrate_data = None
        self._substrate_nodes = list()
        self.initialize()

    def initialize(self) -> None:
        pass

    def set_substrate_data(self, substrate_data: dict) -> None:
        self._substrate_data = substrate_data

    def add_node(self, node) -> None:
        self._substrate_nodes.append(node)

    def update(self, dt: int) -> None:
        pass


class SubstrateFieldNoDimension(SubstrateField):
    def __init__(self, domain, volume: int) -> None:
        self._volume = volume
        self._concentration_layer = None
        super().__init__(domain)

    def initialize(self) -> None:
        self._concentration_layer = 0

    def update(self, dt: int) -> None:
        self.update_nodes(dt)

    def update_nodes(self, dt: int) -> None:
        for n in self._substrate_nodes:
            typ = n['type']
            if typ == 'flux':
                r_u = n['uptake_rate']
                r_r = n['release_rate']
                c_n_cur = n['concentration']
                c_n_max = n['maximal_concentration']
                c_f_cur = self._concentration_layer
                dc = (r_u * c_f_cur * ((c_n_max - c_n_cur) / c_n_max) - r_r * c_n_cur) * dt
                c_n_new = c_n_cur + dc
                c_f_new = c_f_cur - dc
                n['concentration'] = c_n_new
                self._concentration_layer = c_f_new
            elif typ == 'fixed':
                self._concentration_layer = n['concentration'] / self._volume


class SubstrateField2D(SubstrateField):
    def __init__(self, domain, dim_x: int, dim_y: int, dx: int) -> None:
        self._concentration_layer = None
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._dx = dx
        super().__init__(domain)

    def initialize(self):
        self._concentration_layer = np.zeros((self._dim_x + 2, self._dim_y + 2), dtype=np.float32)

    def update(self, dt: int) -> None:
        self.update_nodes(dt)
        self.diffusion_decay(dt)

    def update_nodes(self, dt: int) -> None:
        for n in self._substrate_nodes:
            pos = n['position']
            typ = n['type']
            if typ == 'flux':
                r_u = n['uptake_rate']
                r_r = n['release_rate']
                c_n_cur = n['concentration']
                c_n_max = n['maximal_concentration']
                c_f_cur = self._concentration_layer[pos]
                dc = (r_u * c_f_cur * ((c_n_max - c_n_cur) / c_n_max) - r_r * c_n_cur) * dt
                c_n_new = c_n_cur + dc
                c_f_new = c_f_cur - dc
                n['concentration'] = c_n_new
                self._concentration_layer[pos] = c_f_new
            elif typ == 'fixed':
                self._concentration_layer[pos] = n['concentration']

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration_layer
        shape_x = self._concentration_layer.shape[0]
        shape_y = self._concentration_layer.shape[1]
        D = self._substrate_data['diffusion_coefficient']
        d = self._substrate_data['decay_coefficient']
        dx = self._dx
        # numba_functions.diffusion_solver_ftcs_2d(C, shape_x, shape_y, D, d, dt, dx)
        numba_functions.diffusion_solver_pr_adi_2d(C, shape_x, shape_y, D, d, dt, dx)
