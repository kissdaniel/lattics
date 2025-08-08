import numpy as np
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

    def get_concentration(self, position) -> float:
        pass

    def update(self, dt: int) -> None:
        pass


class SubstrateFieldNoDimension(SubstrateField):
    def __init__(self, domain, volume: int) -> None:
        self._volume = volume
        self._concentration_layer = None
        super().__init__(domain)

    def initialize(self) -> None:
        self._concentration_layer = 0

    def get_concentration(self, position) -> float:
        return self._concentration_layer

    def update(self, dt: int) -> None:
        self.update_nodes(dt)
        self.diffusion_decay(dt)

    def update_nodes(self, dt: int) -> None:
        count_fixed = 0
        sum_fixed = 0
        v_f = self._volume
        for n in self._substrate_nodes:
            typ = n['type']
            if typ == 'flux':
                k_p = n['passive_rate']
                k_u = n['uptake_rate']
                k_r = n['release_rate']
                c_n_cur = n['concentration']
                v_n = n['volume']
                c_f_cur = self._concentration_layer
                dn = (k_p * (c_f_cur - c_n_cur) + k_u * c_f_cur - k_r * c_n_cur) * dt
                c_n_new = c_n_cur + dn / v_n
                c_f_new = c_f_cur - dn / v_f
                n['concentration'] = c_n_new
                self._concentration_layer = c_f_new
            elif typ == 'fixed':
                sum_fixed = sum_fixed + n['concentration']
                count_fixed = count_fixed + 1
        if 0 < count_fixed:
            self._concentration_layer = sum_fixed / count_fixed

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration_layer
        d = self._substrate_data['decay_coefficient']
        self._concentration_layer = C * np.exp(-d * dt)


class SubstrateField2D(SubstrateField):
    def __init__(self, domain, dim_x: int, dim_y: int, dx: int) -> None:
        self._concentration_layer = None
        self._dim_x = dim_x
        self._dim_y = dim_y
        self._dx = dx
        super().__init__(domain)

    def initialize(self):
        self._concentration_layer = np.zeros((self._dim_x, self._dim_y), dtype=np.float32)

    def get_concentration(self, position) -> float:
        return self._concentration_layer[position]

    def update(self, dt: int) -> None:
        self.update_nodes(dt)
        self.diffusion_decay(dt)

    def update_nodes(self, dt: int) -> None:
        v_f = self._dx**2
        for n in self._substrate_nodes:
            pos = n['position']
            typ = n['type']
            if typ == 'flux':
                k_p = n['passive_rate']
                k_u = n['uptake_rate']
                k_r = n['release_rate']
                c_n_cur = n['concentration']
                v_n = n['volume']
                c_f_cur = self._concentration_layer[pos]
                dn = (k_p * (c_f_cur - c_n_cur) + k_u * c_f_cur - k_r * c_n_cur) * dt
                c_n_new = c_n_cur + dn / v_n
                c_f_new = c_f_cur - dn / v_f
                n['concentration'] = c_n_new
                self._concentration_layer[pos] = c_f_new
            elif typ == 'fixed':
                self._concentration_layer[pos] = n['concentration']

    def diffusion_decay(self, dt: int) -> None:
        C = self._concentration_layer
        D = self._substrate_data['diffusion_coefficient']
        d = self._substrate_data['decay_coefficient']
        dx = self._dx
        self._concentration_layer = numba_functions.diffusion_solver_lod_2d(C, D, d, np.float32(dt), dx)
