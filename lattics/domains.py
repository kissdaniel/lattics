
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
    pass


class Structured2DSimulationDomain(SimulationDomain):
    pass
