from lattics._context import UpdateContext

from ._base import BaseSubstrateField


class HomogeneousSubstrateField(BaseSubstrateField):
    """The concentration field of a substrate in a perfectly mixed space: a
    single concentration for the whole domain. Created by
    :class:`~lattics.HomogeneousSpace`.
    """

    def __init__(self, substrate, space, nodes) -> None:
        super().__init__(substrate, space, nodes)
        self._concentration = 0.0

    @property
    def concentration(self) -> float:
        return self._concentration

    def concentration_at(self, position=None) -> float:
        return self._concentration

    def update(self, ctx: UpdateContext) -> None:
        self._exchange(ctx.dt)
        self._concentration = self._substrate.apply_kinetics(self._concentration, ctx.dt)

    def _exchange(self, dt: float) -> None:
        name = self._substrate.name
        count_fixed = 0
        sum_fixed = 0
        v_f = self._space.volume
        for n in self.exchanging_nodes():
            info = n.get_attribute('substrate.transport')[name]
            if info.type == 'flux':
                k_p = info.passive_rate
                k_u = info.uptake_rate
                k_r = info.release_rate
                c_n_cur = info.concentration
                v_n = n.get_attribute('space.volume')
                c_f_cur = self._concentration
                m_t = c_n_cur * v_n + c_f_cur * v_f
                c_n_new = (c_n_cur + (dt / v_n) * (m_t / v_f) * (k_p + k_u)) / (
                    1 + (dt / v_n) * ((1 + (v_n / v_f)) * k_p + (v_n / v_f) * k_u + k_r)
                )
                c_f_new = (m_t - c_n_new * v_n) / v_f
                info.concentration = c_n_new
                self._concentration = c_f_new
            elif info.type == 'fixed':
                sum_fixed = sum_fixed + info.concentration
                count_fixed = count_fixed + 1
        if 0 < count_fixed:
            self._concentration = sum_fixed / count_fixed
