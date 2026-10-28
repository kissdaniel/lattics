import numpy as np

from lattics._agent import Agent
from lattics._context import UpdateContext
from lattics._units import convert_rate, from_quantity

from ._base import BaseModel


class ConcentrationDependentToxicityModel(BaseModel):
    """Necrosis with a rate that depends on the concentration of a substrate
    inside the agent: ``max_rate * c / (median_effective_concentration + c)``.

    Parameters
    ----------
    update_interval : tuple[float, str]
        The time between two updates of the model
    substrate_name : str
        The name of the substrate
    max_rate : tuple[float, str]
        The maximal rate of necrosis, for example ``(1.0, '1/day')``
    median_effective_concentration : float
        The concentration at which the rate is half of the maximum
    """

    reads = ('substrate.transport',)
    writes = (
        'toxicity.state',
        'cellcycle.is_active',
        'space.division_requested',
        'mechanics.motility',
        'mechanics.binding_affinity',
    )

    def __init__(
        self,
        update_interval: tuple[float, str],
        substrate_name: str,
        max_rate: tuple[float, str],
        median_effective_concentration: float,
    ) -> None:
        super().__init__(update_interval)
        self._substrate_name = substrate_name
        self._max_rate = from_quantity(max_rate, convert_rate, '1/ms', 'max_rate', "(1.0, '1/day')")
        self._ed50 = median_effective_concentration

    def update_agent(self, agent: Agent, ctx: UpdateContext) -> None:
        concentration = agent.get_attribute('substrate.transport')[self._substrate_name].concentration
        rate = self._mm_fun(x=concentration, vmax=self._max_rate, ed50=self._ed50)
        dt = ctx.dt
        prob = 1 - np.exp(-rate * dt)
        if ctx.rng.random() < prob:
            agent.set_attribute('toxicity.state', 'necrotic')
            agent.set_attribute('cellcycle.is_active', False)
            agent.set_attribute('space.division_requested', False)
            agent.set_attribute('mechanics.motility', 0)
            agent.set_attribute('mechanics.binding_affinity', 0)

    def _initialize_required_attributes(self, agent: Agent, ctx: UpdateContext) -> None:
        self._require_attributes(agent, 'substrate.transport')

    def _mm_fun(self, x, vmax, ed50):
        return vmax * (x / (ed50 + x))
