
from .agent import Agent
from .simulation import Simulation
import numpy as np
import warnings


class SimulationDomain:
    def __init__(self,
                 simulation: Simulation
                 ) -> None:
        self._simulation = simulation

    def initialize(self) -> None:
        pass

    def add_agent(self, agent: Agent) -> None:
        pass

    def remove_agent(self, agent: Agent) -> None:
        pass

    def update(self, dt: int) -> None:
        pass


class UnstructuredSimulationDomain(SimulationDomain):
    def __init__(self,
                 simulation: Simulation,
                 capacity: int = None
                 ) -> None:
        super().__init__(simulation)
        self._capacity = capacity
        self._agents = list()

    def initialize(self) -> None:
        pass

    def add_agent(self, agent: Agent) -> None:
        self._agents.append(agent)

    def remove_agent(self, agent: Agent) -> None:
        self._agents.remove(agent)

    def update(self, dt: int) -> None:
        for a in self._agents:
            if a.get_status_flag('division_pending'):
                self._perform_cell_division(a)

    def _perform_cell_division(self, agent: Agent) -> None:
        if np.random.random() < self._get_division_probability():
            agent.set_status_flag('division_pending', False)
            agent.set_status_flag('division_completed', True)
            new_agent = agent.clone()
            self._simulation.add_agent(new_agent)

    def _get_division_probability(self) -> float:
        if self._capacity:
            return 1.0 - len(self._agents) / self._capacity
        else:
            return 1.0


class Structured2DSimulationDomain(SimulationDomain):
    pass
