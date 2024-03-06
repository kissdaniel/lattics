"""The main component of the LattiCS framework containing functionalities to set up and execute a simulation.
"""

from . import agent

class Simulation:
    def __init__(self):        
        self._agents = list()

    def add_agent(self, agent):
        self._agents.append(agent)

    def initialize(self):
        pass

    def run(self):
        for t in range(100):
            for a in self._agents:
                a.update_models(1)

    