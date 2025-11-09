import numpy as np

from ._base import BaseModel
from lattics.core._agent import Agent
from lattics.core._convert import convert_time


class ConcentrationDependentToxicityModel(BaseModel):
    def __init__(self,
                 update_interval: tuple[float, str],
                 substrate_name: str,
                 max_rate: float,
                 median_effective_concentration: float
                 ) -> None:
        super().__init__(update_interval)
        self._substrate_name = substrate_name
        self._max_rate = max_rate / convert_time(1, 'day', 'ms')
        self._ed50 = median_effective_concentration

    def update_attributes(self, agent: Agent) -> None:
        concentration = agent.get_attribute('substrate_info')[self._substrate_name].concentration
        rate = self._mm_fun(x=concentration, vmax=self._max_rate, ed50=self._ed50)
        dt = self.update_info.elapsed_time
        prob = 1 - np.exp(-rate * dt)
        if np.random.random() < prob:
            agent.set_attribute('state', 'necrotic')
            agent.set_attribute('cellcycle_is_active', False)
            agent.set_attribute('division_pending', False)
            agent.set_attribute('motility', 0)
            agent.set_attribute('binding_affinity', 0)

    def _initialize_required_attributes(self, agent: Agent) -> None:
        if not agent.has_attribute('substrate_info'):
            raise AttributeError()

    def _mm_fun(self, x, vmax, ed50):
        return vmax * (x / (ed50 + x))
